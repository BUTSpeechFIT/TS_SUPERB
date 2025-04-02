# Copyright (c) 2025 Nippon Telegraph and Telephone Corporation (NTT)
# Copyright (c) 2025 Brno University of Technology (BUT)
# Modifications made by NTT & BUT are licensed under the NTT License (see LICENSE.NTT).
#
# The original code is:
#   Copyright (c) Johns Hopkins University.
#   Licensed under the Apache License, Version 2.0.
#   See the LICENSE.Apache file or http://www.apache.org/licenses/LICENSE-2.0
#
# Summary of Modifications by NTT & BUT:
# - Enable training of a downstream model with SparseLibri2Mix and enrollment speech.
#
# Original source: https://github.com/s3prl/s3prl/blob/v0.4.17/s3prl/downstream/diarization/expert.py

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler

from ..mhfa import MHFA
from .dataset import DiarizationDataset
from .model import Model


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, upstream_rate, downstream_expert, expdir, upstream_dim_spk, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.upstream_rate = upstream_rate
        self.upstream_dim_spk = upstream_dim_spk
        self.datarc = downstream_expert["datarc"]

        config_frame_shift = self.datarc.get("frame_shift")
        if isinstance(config_frame_shift, int):
            logging.warning(
                f"Diarization label frame shfit: {config_frame_shift}. "
                "It is set in the config field. You don't need to set this config field if "
                "you are training new downstream models. This module will then automatically "
                "use upstream's downsample rate as the training label frame shift. This "
                "'if condition' is designed only to inference the already trained downstream "
                "checkpoints with the command: python3 run_downstream.py -m evaluate -e [ckpt]. "
                "The checkpoint contains the frame_shift used for its training, and the same "
                "frame_shift should be prepared for the inference."
            )
            frame_shift = config_frame_shift
        else:
            logging.warning(
                f"Diarization label frame shfit: {upstream_rate}. It is automatically set as "
                "upstream's downsample rate to best align the representation v.s. labels for training. "
                "This frame_shift information will be saved in the checkpoint for future inference."
            )
            frame_shift = upstream_rate

        self.datarc["frame_shift"] = frame_shift
        with (Path(expdir) / "frame_shift").open("w") as file:
            print(frame_shift, file=file)

        self.loaderrc = downstream_expert["loaderrc"]
        self.modelrc = downstream_expert["modelrc"]

        self.train_batch_size = self.loaderrc["train_batchsize"]
        self.eval_batch_size = self.loaderrc["eval_batchsize"]

        self.expdir = expdir

        project_dim = self.modelrc.get("project_dim", 0)
        if project_dim > 0:
            self.projector = nn.Linear(upstream_dim, project_dim)
            idim = project_dim
        else:
            self.projector = None
            idim = self.upstream_dim

        self.model = Model(
            input_dim=idim,
            output_class_num=3,  # tss, ntss, ns
            **self.modelrc,
        )
        self.objective = nn.CrossEntropyLoss()

        if self.modelrc["spk_extractor"] == "MHFA":
            self.spk_extractor = MHFA(
                head_nb=8, inputs_dim=self.upstream_dim_spk, outputs_dim=self.modelrc["hidden_size"]
            )
        else:
            raise ValueError("Speaker extractor type not defined.")

        self.softmax = nn.Softmax(dim=1)  # for mAP
        self.logging = os.path.join(expdir, "log.log")
        self.register_buffer("best_score", torch.zeros(1))

    # Interface
    def get_dataloader(self, mode):
        """
        Args:
            mode: string
                'train', 'dev' or 'test'
        Return:
            a torch.utils.data.DataLoader returning each batch in the format of:
            [wav1, wav2, ...], your_other_contents1, your_other_contents2, ...
            where wav1, wav2 ... are in variable length
            each wav is torch.FloatTensor in cpu with:
                1. dim() == 1
                2. sample_rate == 16000
                3. directly loaded by torchaudio
        """
        if not hasattr(self, f"{mode}_dataset"):
            if self.datarc["corpus"] == "sparselibrimix":
                dataset = DiarizationDataset(
                    mode,
                    self.loaderrc[f"{mode}_dir"],
                    self.loaderrc[f"{mode}_dir"],
                    **self.datarc,
                )
            elif self.datarc["corpus"] == "librimix":
                dataset = DiarizationDataset(
                    mode,
                    self.loaderrc[f"{mode}_dir"],
                    self.loaderrc[f"{mode}_enr_dir"],
                    **self.datarc,
                )

            else:
                raise ValueError("Invalid datarc.corpus")
            setattr(self, f"{mode}_dataset", dataset)

        if mode == "train":
            return self._get_train_dataloader(self.train_dataset)
        elif mode == "dev":
            return self._get_dev_dataloader(self.dev_dataset)
        elif mode == "test":
            return self._get_test_dataloader(self.test_dataset)

    """
    Datalaoder Specs:
        Each dataloader should output in the following format:

        [[wav1, wav2, ...], your_other_contents1, your_other_contents2, ...]

        where wav1, wav2 ... are in variable length
        each wav is torch.FloatTensor in cpu with dim()==1 and sample_rate==16000
    """

    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.loaderrc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

    def _get_dev_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.loaderrc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

    def _get_test_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.loaderrc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

    def _match_length(self, features, frame_lengths):
        """
        Since the upstream extraction process can sometimes cause a mismatch
        between the seq lenth of inputs and labels:
        - if len(inputs) > len(labels), we truncate the final few timestamp of inputs to match the length of labels
        - if len(inputs) < len(labels), we duplicate the last timestep of inputs to match the length of labels
        Note that the length of labels should never be changed.
        """
        assert len(features) == len(
            frame_lengths
        ), f"The difference of batch size of {len(features)} and {len(frame_lengths)}"

        matched_features = []
        for idx in range(len(features)):
            feature, frame_len = features[idx], frame_lengths[idx].item()
            feature_len = feature.size(0)

            factor = int(round(frame_len / feature_len))
            assert (
                np.abs(factor) <= 1
            ), f"The length mismatch between feature and label is too long or too short ({feature_len}, {frame_len})"

            if feature_len > frame_len:
                feature = feature[:frame_len, :]
            elif feature_len < frame_len:
                pad_vec = feature[-1, :].unsqueeze(0)  # (1, feature_dim)
                feature = torch.cat(
                    (feature, pad_vec.repeat(frame_len - feature_len, 1)), dim=0
                )  # (batch_size, seq_len, feature_dim), where seq_len == labels.size(-1)
            matched_features.append(feature)
        return matched_features

    # Interface
    def forward(
        self,
        mode,
        features,
        features_k,
        features_v,
        labels,
        frame_lengths,
        rec_ids,
        enr_ids,
        enroll_wav,
        records,
        **kwargs,
    ):
        """
        Args:
            mode: string
                'train', 'dev' or 'test' for this forward step

            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args

            labels:
                the frame-wise speaker labels

            rec_id:
                related recording id, use for inference

            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step

        Return:
            loss:
                the loss to be optimized, should not be detached
        """

        # mixture and labels
        bs = len(features)
        labels = [torch.from_numpy(label) for label in labels]
        frame_lengths = torch.LongTensor(frame_lengths)
        features = self._match_length(features, frame_lengths)
        features = pad_sequence(features, batch_first=True)
        if self.projector is not None:
            features = self.projector(features)
        labels = pad_sequence(labels, batch_first=True, padding_value=0).long().to(features.device)  # (B, T)

        if self.modelrc["spk_extractor"] == "MHFA":
            features_k = pad_sequence(features_k, batch_first=True).to(device=features.device)
            features_v = pad_sequence(features_v, batch_first=True).to(device=features.device)
            spk_outemb = self.spk_extractor(features_k, features_v).unsqueeze(1)  # (B, 1, F)
        predicted = self.model(features, frame_lengths, spk_outemb)

        loss = 0
        for b in range(bs):
            loss += self.objective(predicted[b][: frame_lengths[b]], labels[b][: frame_lengths[b]])
        loss /= bs

        # logging
        preds = predicted.detach().clone()  # (B, T, 3)
        labs = labels.detach().clone()  # (B, T)
        n_correct = 0
        n_frames = 0
        outputs = []
        targets = []
        for b in range(bs):
            # for acc
            classes = torch.argmax(preds[b][: frame_lengths[b]], dim=1)
            n_frames += frame_lengths[b]
            n_correct += torch.sum(classes == labs[b][: frame_lengths[b]]).item()
            # for AP
            p = self.softmax(preds[b][: frame_lengths[b]])
            outputs.append(p.cpu().numpy())
            targets.append(labs[b][: frame_lengths[b]].cpu().numpy())
        records["loss"].append(loss.item())
        records["predict"] += outputs
        records["truth"] += targets
        records["filename"] += rec_ids
        records["filename_enr"] += enr_ids
        records["n_frames"].append(n_frames)
        records["n_correct"].append(n_correct)
        return loss

    def calc_pvad_metrics(self, n_correct, n_frames, outputs, targets):
        acc = 100.0 * sum(n_correct) / sum(n_frames)
        # AP
        outputs_np = np.concatenate(outputs)
        targets_np = np.concatenate(targets)
        targets_oh = np.eye(3)[targets_np]
        tss_AP, ntss_AP, ns_AP = average_precision_score(targets_oh, outputs_np, average=None)
        mAP = average_precision_score(targets_oh, outputs_np, average="micro")
        return acc, tss_AP, ntss_AP, ns_AP, mAP

    # interface
    def log_records(self, mode, records, logger, global_step, batch_ids, total_batch_num, **kwargs):
        """
        Args:
            mode: string
                'train':
                    records and batchids contain contents for `log_step` batches
                    `log_step` is defined in your downstream config
                    eg. downstream/example/config.yaml
                'dev' or 'test' :
                    records and batchids contain contents for the entire evaluation dataset

            records:
                defaultdict(list), contents already appended

            logger:
                Tensorboard SummaryWriter
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

            global_step:
                The global_step when training, which is helpful for Tensorboard logging

            batch_ids:
                The batches contained in records when enumerating over the dataloader

            total_batch_num:
                The total amount of batches in the dataloader

        Return:
            a list of string
                Each string is a filename we wish to use to save the current model
                according to the evaluation result, like the best.ckpt on the dev set
                You can return nothing or an empty list when no need to save the checkpoint

        """
        loss = torch.FloatTensor(records["loss"]).mean().item()
        acc, tss_AP, ntss_AP, ns_AP, mAP = self.calc_pvad_metrics(
            records["n_correct"], records["n_frames"], records["predict"], records["truth"]
        )

        logger.add_scalar(f"pvad/{mode}-loss", loss, global_step=global_step)
        logger.add_scalar(f"pvad/{mode}-acc", acc, global_step=global_step)
        logger.add_scalar(f"pvad/{mode}-tss_AP", tss_AP, global_step=global_step)
        logger.add_scalar(f"pvad/{mode}-ntss_AP", ntss_AP, global_step=global_step)
        logger.add_scalar(f"pvad/{mode}-ns_AP", ns_AP, global_step=global_step)
        logger.add_scalar(f"pvad/{mode}-mAP", mAP, global_step=global_step)

        print(f"{mode} loss: {loss}")
        print(f"{mode} acc: {acc}")
        print(f"{mode} tss_AP: {tss_AP}")
        print(f"{mode} ntss_AP: {ntss_AP}")
        print(f"{mode} ns_AP: {ns_AP}")
        print(f"{mode} mAP: {mAP}")

        save_ckpt = []
        if mode == "dev" and mAP > self.best_score:
            self.best_score = torch.ones(1) * mAP
            save_ckpt.append(f"best-states-{mode}.ckpt")

        # for debugging
        #     predict_file = os.path.join(self.expdir, f"{mode}_predict.npy")
        #     truth_file = os.path.join(self.expdir, f"{mode}_truth.npy")
        #     line = {f"{a}_{b}": c for a, b, c in zip(records["filename"], records["filename_enr"], records["predict"])}
        #     np.save(predict_file, line)
        #     line = {f"{a}_{b}": c for a, b, c in zip(records["filename"], records["filename_enr"], records["truth"])}
        #     np.save(truth_file, line)
        # if mode == "test":
        #     predict_file = os.path.join(self.expdir, f"{mode}_predict.npy")
        #     truth_file = os.path.join(self.expdir, f"{mode}_truth.npy")
        #     line = {f"{a}_{b}": c for a, b, c in zip(records["filename"], records["filename_enr"], records["predict"])}
        #     np.save(predict_file, line)
        #     line = {f"{a}_{b}": c for a, b, c in zip(records["filename"], records["filename_enr"], records["truth"])}
        #     np.save(truth_file, line)

        return save_ckpt
