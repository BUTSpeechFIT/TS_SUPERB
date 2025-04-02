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
# - Enable training of a downstream model with Libri2Mix and enrollment speech.
#
# Original source: https://github.com/s3prl/s3prl/blob/v0.4.17/s3prl/downstream/separation_stft/expert.py

import os
import pickle
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from asteroid.losses import PITLossWrapper
from asteroid.metrics import get_metrics
from asteroid.utils.torch_utils import pad_x_to_y
from asteroid_filterbanks import make_enc_dec
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from torch.utils.data import DataLoader

from ..mhfa import MHFA
from ..separation_stft.loss import MSELoss, PairwiseNegSDR, SISDRLoss
from .dataset import SeparationDataset
from .model import SepRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ENC_DEC_KERNEL_SIZE = 1024
ENC_DEC_N_FILTERS = 512


def match_length(feat_list, length_list):
    assert len(feat_list) == len(length_list)
    bs = len(length_list)
    new_feat_list = []
    for i in range(bs):
        assert abs(feat_list[i].size(0) - length_list[i]) < 20, f"{feat_list[i].size(0)}, {length_list[i]}"
        # assert abs(feat_list[i].size(0) - length_list[i]) < 5
        if feat_list[i].size(0) == length_list[i]:
            new_feat_list.append(feat_list[i])
        elif feat_list[i].size(0) > length_list[i]:
            new_feat_list.append(feat_list[i][: length_list[i], :])
        else:
            new_feat = torch.zeros(length_list[i], feat_list[i].size(1)).to(feat_list[i].device)
            new_feat[: feat_list[i].size(0), :] = feat_list[i]
            new_feat_list.append(new_feat)
    return new_feat_list


class TimeSISDRLoss(object):
    def __init__(self):
        self.loss = PITLossWrapper(PairwiseNegSDR("sisdr"), pit_from="pw_mtx")

    def compute_loss(self, est_targets, wav_length, target_wav_list):
        # est_target is [B, 1, wav_length]
        est_targets = est_targets.to(device)
        targets = torch.stack(target_wav_list, dim=1).to(device)
        loss = self.loss(est_targets, targets, length=wav_length)
        return loss


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
        self.loaderrc = downstream_expert["loaderrc"]
        self.modelrc = downstream_expert["modelrc"]
        self.expdir = expdir

        idim = upstream_dim

        self.train_dataset = SeparationDataset(
            csv_dir=self.loaderrc["train_dir"],
            rate=self.datarc["rate"],
            n_fft=self.datarc["n_fft"],
            hop_length=self.upstream_rate,
            win_length=self.datarc["win_length"],
            window=self.datarc["window"],
            center=self.datarc["center"],
            test=False,
        )

        self.dev_dataset = SeparationDataset(
            csv_dir=self.loaderrc["dev_dir"],
            rate=self.datarc["rate"],
            n_fft=self.datarc["n_fft"],
            hop_length=self.upstream_rate,
            win_length=self.datarc["win_length"],
            window=self.datarc["window"],
            center=self.datarc["center"],
            test=False,
        )

        self.test_dataset = SeparationDataset(
            csv_dir=self.loaderrc["test_dir"],
            rate=self.datarc["rate"],
            n_fft=self.datarc["n_fft"],
            hop_length=self.upstream_rate,
            win_length=self.datarc["win_length"],
            window=self.datarc["window"],
            center=self.datarc["center"],
            test=True,
        )

        if self.modelrc["model"] == "SepRNN":
            self.model = SepRNN(
                input_dim=idim,
                num_bins=int(self.datarc["n_fft"] / 2),
                rnn=self.modelrc["rnn"],
                num_spks=self.datarc["num_speakers"],
                num_layers=self.modelrc["rnn_layers"],
                hidden_size=self.modelrc["hidden_size"],
                dropout=self.modelrc["dropout"],
                non_linear=self.modelrc["non_linear"],
                bidirectional=self.modelrc["bidirectional"],
            )

            if self.modelrc["spk_extractor"] == "MHFA":
                self.spk_extractor = MHFA(
                    head_nb=8, inputs_dim=self.upstream_dim_spk, outputs_dim=self.modelrc["hidden_size"] * 2
                )
            else:
                raise ValueError("Invalid speaker extractor type.")

            self.encoder, self.decoder = make_enc_dec(
                "free",
                kernel_size=ENC_DEC_KERNEL_SIZE,
                n_filters=ENC_DEC_N_FILTERS,
                stride=self.upstream_rate,
                sample_rate=self.datarc["rate"],
            )
        else:
            raise ValueError("Invalid model type.")

        self.loss_type = self.modelrc["loss_type"]
        if self.loss_type == "MSE":
            self.objective = MSELoss(self.datarc["num_speakers"], self.modelrc["mask_type"])
        elif self.loss_type == "SISDR":
            self.objective = SISDRLoss(
                self.datarc["num_speakers"],
                n_fft=self.datarc["n_fft"],
                hop_length=self.upstream_rate,
                win_length=self.datarc["win_length"],
                window=self.datarc["window"],
                center=self.datarc["center"],
            )
        elif self.loss_type == "TimeSISDR":
            self.objective = TimeSISDRLoss()
        else:
            raise ValueError("Invalid loss type.")

        self.register_buffer("best_score", torch.ones(1) * -10000)

    def _get_train_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.loaderrc["train_batchsize"],
            shuffle=True,
            num_workers=self.loaderrc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.loaderrc["eval_batchsize"],
            shuffle=False,
            num_workers=self.loaderrc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

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
        if mode == "train":
            return self._get_train_dataloader(self.train_dataset)
        elif mode == "dev":
            return self._get_eval_dataloader(self.dev_dataset)
        elif mode == "test":
            return self._get_eval_dataloader(self.test_dataset)

    def forward(
        self,
        mode,
        features,
        features_k,
        features_v,
        uttname_list,
        source_attr,
        source_wav,
        target_attr,
        target_wav_list,
        feat_length,
        wav_length,
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

            features_k: [spk_enroll]
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args

            features_v: [spk_enroll]
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args

            uttname_list:
                list of utterance names

            source_attr:
                source_attr is a dict containing the STFT information
                for the mixture. source_attr['magnitude'] stores the STFT
                magnitude, source_attr['phase'] stores the STFT phase and
                source_attr['stft'] stores the raw STFT feature. The shape
                is [bs, max_length, feat_dim]

            source_wav:
                source_wav contains the raw waveform for the mixture,
                and it has the shape of [bs, max_wav_length]

            target_attr:
                similar to source_attr, it contains the STFT information
                for individual sources. It only has two keys ('magnitude' and 'phase')
                target_attr['magnitude'] is a list of length n_srcs, and
                target_attr['magnitude'][i] has the shape [bs, max_length, feat_dim]

            target_wav_list:
                target_wav_list contains the raw waveform for the individual
                sources, and it is a list of length n_srcs. target_wav_list[0]
                has the shape [bs, max_wav_length]

            feat_length:
                length of STFT features

            wav_length:
                length of raw waveform

            enroll_features_list:
                enroll utternaces

            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step

        Return:
            loss:
                the loss to be optimized, should not be detached
        """

        # match the feature length to STFT feature length

        source_wav_list = [torch.FloatTensor(wav).cuda() for wav in source_attr["source_wav_list"]]
        encoder_outputs = [
            self.encoder(wav_lst.unsqueeze(0).unsqueeze(0)).squeeze(0).transpose(0, 1) for wav_lst in source_wav_list
        ]
        # encoder_outputs: List [frame_len, Fdim]
        feat_length = torch.tensor([encoder_output.size(0) for encoder_output in encoder_outputs])

        encoder_outputs = pad_sequence(encoder_outputs, batch_first=True)
        features_k = torch.stack(features_k, dim=0)
        features_v = torch.stack(features_v, dim=0)
        spk_outemb = self.spk_extractor(features_k, features_v).unsqueeze(1)
        features = match_length(features, feat_length)
        features = pack_sequence(features)
        mask_list = self.model(features, spkinfo=spk_outemb)
        predict_tf = mask_list[0] * encoder_outputs
        wav_pred = self.decoder(predict_tf.transpose(1, 2))
        reconstructed = pad_x_to_y(wav_pred, source_wav)

        # evaluate the separation quality of predict sources
        if mode in ["dev", "test"]:
            if mode == "dev":
                COMPUTE_METRICS = ["si_sdr"]
            elif mode == "test":
                COMPUTE_METRICS = ["si_sdr", "stoi", "pesq"]

            assert len(wav_length) == 1
            predict_srcs_np = np.stack(reconstructed.squeeze(1).data.cpu().numpy(), 0)
            gt_srcs_np = torch.cat(target_wav_list, 0).data.cpu().numpy()
            mix_np = source_wav.data.cpu().numpy()

            utt_metrics = get_metrics(
                mix_np,
                gt_srcs_np,
                predict_srcs_np,
                sample_rate=self.datarc["rate"],
                metrics_list=COMPUTE_METRICS,
                compute_permutation=True,
            )

            for metric in COMPUTE_METRICS:
                input_metric = "input_" + metric
                assert metric in utt_metrics and input_metric in utt_metrics
                imp = utt_metrics[metric] - utt_metrics[input_metric]
                if metric not in records:
                    records[metric] = []
                if metric == "si_sdr":
                    records[metric].append(imp)
                elif metric == "stoi" or metric == "pesq":
                    records[metric].append(utt_metrics[metric])
                else:
                    raise ValueError("Metric type not defined.")

            assert "batch_id" in kwargs
            if kwargs["batch_id"] % 1000 == 0:  # Save the prediction every 1000 examples
                records["mix"].append(mix_np)
                records["hypo"].append(predict_srcs_np)
                records["ref"].append(gt_srcs_np)
                records["uttname"].append(uttname_list[0])

        if self.loss_type == "MSE":  # mean square loss
            loss = self.objective.compute_loss(mask_list, feat_length, source_attr, target_attr)
        elif self.loss_type == "SISDR":  # end-to-end SI-SNR loss
            loss = self.objective.compute_loss(mask_list, feat_length, source_attr, wav_length, target_wav_list)
        elif self.loss_type == "TimeSISDR":
            loss = self.objective.compute_loss(reconstructed, wav_length, target_wav_list)
        else:
            raise ValueError("Loss type not defined.")

        records["loss"].append(loss.item())
        return loss

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
        if mode == "train":
            avg_loss = np.mean(records["loss"])
            logger.add_scalar(f"tse_pse/{mode}-loss", avg_loss, global_step=global_step)
            return []
        else:
            if mode == "dev":
                COMPUTE_METRICS = ["si_sdr"]
            elif mode == "test":
                COMPUTE_METRICS = ["si_sdr", "stoi", "pesq"]
            avg_loss = np.mean(records["loss"])
            logger.add_scalar(f"tse_pse/{mode}-loss", avg_loss, global_step=global_step)
            if mode == "test":
                file_sc = open(os.path.join(self.expdir, "si-sinr-imp-list.pickle"), "wb")
                pickle.dump(records["si_sdr"], file_sc)
                file_sc.close()

            with (Path(self.expdir) / f"{mode}_metrics.txt").open("w") as output:
                for metric in COMPUTE_METRICS:
                    avg_metric = np.mean(records[metric])
                    if mode in ["test", "dev"]:
                        print("Average {} of {} utts: {:.4f}".format(metric, len(records[metric]), avg_metric))
                        print(metric, avg_metric, file=output)

                    logger.add_scalar(f"tse_pse/{mode}-" + metric, avg_metric, global_step=global_step)

            save_ckpt = []
            assert "si_sdr" in records
            if mode == "dev" and np.mean(records["si_sdr"]) > self.best_score:
                self.best_score = torch.ones(1) * np.mean(records["si_sdr"])
                save_ckpt.append(f"best-states-{mode}.ckpt")

            for s in ["mix", "ref", "hypo", "uttname"]:
                assert s in records
            for i in range(len(records["uttname"])):
                utt = records["uttname"][i]
                mix_wav = records["mix"][i][0, :]
                mix_wav = librosa.util.normalize(mix_wav, norm=np.inf, axis=None)
                logger.add_audio(
                    "step{:06d}_{}_mix.wav".format(global_step, utt),
                    mix_wav,
                    global_step=global_step,
                    sample_rate=self.datarc["rate"],
                )

                for j in range(records["ref"][i].shape[0]):
                    ref_wav = records["ref"][i][j, :]
                    hypo_wav = records["hypo"][i][j, :]
                    ref_wav = librosa.util.normalize(ref_wav, norm=np.inf, axis=None)
                    hypo_wav = librosa.util.normalize(hypo_wav, norm=np.inf, axis=None)
                    logger.add_audio(
                        "step{:06d}_{}_ref_s{}.wav".format(global_step, utt, j + 1),
                        ref_wav,
                        global_step=global_step,
                        sample_rate=self.datarc["rate"],
                    )
                    logger.add_audio(
                        "step{:06d}_{}_hypo_s{}.wav".format(global_step, utt, j + 1),
                        hypo_wav,
                        global_step=global_step,
                        sample_rate=self.datarc["rate"],
                    )
            return save_ckpt
