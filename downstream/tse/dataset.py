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
# - Enable loading of enrollment speech.
#
# Original source: https://github.com/s3prl/s3prl/blob/v0.4.17/s3prl/downstream/separation_stft/dataset.py

import random
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from asteroid.data import LibriMix
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset

N_SRC = 2
SEGMENT = 3
SEGMENT_AUX = 3


def read_enrollment_csv(csv_path):
    data = defaultdict(dict)
    with open(csv_path, "r") as f:
        f.readline()  # csv header

        for line in f:
            mix_id, utt_id, *aux = line.strip().split(",")
            aux_it = iter(aux)
            aux = [(auxpath, int(float(length))) for auxpath, length in zip(aux_it, aux_it)]
            data[(mix_id, utt_id)] = aux
    return data


class SeparationDataset(Dataset):
    def __init__(
        self,
        csv_dir,
        rate=16000,
        n_fft=512,
        hop_length=320,
        win_length=512,
        window="hann",
        center=True,
        test=False,
    ):
        super(SeparationDataset, self).__init__()
        """
        Args:
            data_dir (str):
                prepared data directory

            rate (int):
                audio sample rate

            src and tgt (list(str)):
                the input and desired output.
                LibriMix offeres different options for the users. For
                clean source separation, src=['mix_clean'] and tgt=['s1', 's2'].
                Please see https://github.com/JorisCos/LibriMix for details

            n_fft (int):
                length of the windowed signal after padding with zeros.

            hop_length (int):
                number of audio samples between adjacent STFT columns.

            win_length (int):
                length of window for each frame

            window (str):
                type of window function, only support Hann window now

            center (bool):
                whether to pad input on both sides so that the
                t-th frame is centered at time t * hop_length

            The STFT related parameters are the same as librosa.
        """

        self.csv_dir = csv_dir
        self.rate = rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.n_srcs = 1  # this means target speech enhancement, not separation

        if test:
            segment = None
            segment_aux = None
        else:
            segment = 3
            segment_aux = 3

        # MIX data
        self.base_dataset = LibriMix(self.csv_dir, "sep_clean", self.rate, N_SRC, segment)
        data_aux = read_enrollment_csv(Path(self.csv_dir) / "mixture2enrollment.csv")

        # data filtering
        if segment_aux is not None:
            max_len = np.sum([len(aux) for aux in data_aux.values()])
            self.seg_len_aux = int(segment_aux * self.rate)
            self.data_aux = {}
            for k, v in data_aux.items():
                items = [(path, length) for path, length in v if length >= self.seg_len_aux]
                if len(items) == 0:
                    continue
                else:
                    self.data_aux[k] = items
            new_len = np.sum([len(aux) for aux in self.data_aux.values()])
            print(f"Drop {max_len - new_len} utterances from {max_len} " f"(shorter than {segment_aux} seconds)")
        else:
            self.seg_len_aux = None
            self.data_aux = data_aux

        self.data_aux_list = list(self.data_aux.keys())
        print(f"The number of enrollment speechs: {len(self.data_aux_list)}")

    def __len__(self):
        return len(self.data_aux_list)

    def _get_segment_start_stop(self, seg_len, length):
        if seg_len is not None:
            start = random.randint(0, length - seg_len)
            stop = start + seg_len
        else:
            start = 0
            stop = None
        return start, stop

    def __getitem__(self, i):
        mix_id, utt_id = self.data_aux_list[i]
        reco = mix_id
        row = self.base_dataset.df[self.base_dataset.df["mixture_ID"] == mix_id].squeeze()

        mixture_path = row["mixture_path"]
        tgt_spk_idx = mix_id.split("_").index(utt_id)

        # read mixture
        seg_len = None
        start, stop = self._get_segment_start_stop(seg_len, row["length"])
        src_samp, _ = sf.read(mixture_path, dtype="float32", start=start, stop=stop)
        src_feat = np.transpose(
            librosa.stft(
                src_samp,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=self.center,
            )
        )

        # read source
        tgt_samp_list, tgt_feat_list = [], []
        source_path = row[f"source_{tgt_spk_idx+1}_path"]
        tgt_samp, _ = sf.read(source_path, dtype="float32", start=start, stop=stop)

        tgt_feat = np.transpose(
            librosa.stft(
                tgt_samp,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=self.center,
            )
        )
        tgt_samp_list.append(tgt_samp)
        tgt_feat_list.append(tgt_feat)

        # enroll source
        enroll_path, enroll_length = random.choice(self.data_aux[(mix_id, utt_id)])
        start_e, stop_e = self._get_segment_start_stop(self.seg_len_aux, enroll_length)
        enroll_wave, _ = sf.read(enroll_path, dtype="float32", start=start_e, stop=stop_e)

        """
        reco (str):
            name of the utterance

        src_sample (ndarray):
            audio samples for the source [T, ]

        src_feat (ndarray):
            the STFT feature map for the source with shape [T1, D]

        tgt_samp_list (list(ndarray)):
            list of audio samples for the targets

        tgt_feat_list (list(ndarray)):
            list of STFT feature map for the targets
        """
        return reco, src_samp, src_feat, tgt_samp_list, tgt_feat_list, enroll_wave

    def collate_fn(self, batch):
        sorted_batch = sorted(batch, key=lambda x: -x[1].shape[0])
        bs = len(sorted_batch)
        uttname_list = [sorted_batch[i][0] for i in range(bs)]

        # Store the magnitude, phase for the mixture in source_attr
        source_attr = {}
        mix_magnitude_list = [torch.from_numpy(np.abs(sorted_batch[i][2])) for i in range(bs)]
        mix_phase_list = [torch.from_numpy(np.angle(sorted_batch[i][2])) for i in range(bs)]
        mix_stft_list = [torch.from_numpy(sorted_batch[i][2]) for i in range(bs)]
        mix_magnitude = pad_sequence(mix_magnitude_list, batch_first=True)
        mix_phase = pad_sequence(mix_phase_list, batch_first=True)
        mix_stft = pad_sequence(mix_stft_list, batch_first=True)
        source_attr["magnitude"] = mix_magnitude
        source_attr["phase"] = mix_phase
        source_attr["stft"] = mix_stft

        target_attr = {}
        target_attr["magnitude"] = []
        target_attr["phase"] = []
        for j in range(self.n_srcs):
            tgt_magnitude_list = [torch.from_numpy(np.abs(sorted_batch[i][4][j])) for i in range(bs)]
            tgt_phase_list = [torch.from_numpy(np.angle(sorted_batch[i][4][j])) for i in range(bs)]
            tgt_magnitude = pad_sequence(tgt_magnitude_list, batch_first=True)
            tgt_phase = pad_sequence(tgt_phase_list, batch_first=True)
            target_attr["magnitude"].append(tgt_magnitude)
            target_attr["phase"].append(tgt_phase)

        wav_length = torch.from_numpy(np.array([len(sorted_batch[i][1]) for i in range(bs)]))
        source_wav_list = [torch.from_numpy(sorted_batch[i][1]) for i in range(bs)]
        source_wav = pad_sequence(source_wav_list, batch_first=True)
        source_attr["source_wav_list"] = source_wav_list

        target_wav_list = []
        for j in range(self.n_srcs):
            target_wav_list.append(
                pad_sequence([torch.from_numpy(sorted_batch[i][3][j]) for i in range(bs)], batch_first=True)
            )

        feat_length = torch.from_numpy(np.array([stft.size(0) for stft in mix_stft_list]))

        enroll_wav_list = [torch.from_numpy(sorted_batch[i][5]) for i in range(bs)]
        enroll_wav = pad_sequence(enroll_wav_list, batch_first=True)

        """
        source_wav_list (list(tensor)):
            list of audio samples for the source

        uttname_list (list(str)):
            list of utterance names

        source_attr (dict):
            dictionary containing magnitude and phase information for the sources

        source_wav (tensor):
            padded version of source_wav_list, with size [bs, max_T]

        target_attr (dict):
            dictionary containing magnitude and phase information for the targets

        feat_length (tensor):
            length of the STFT feature for each utterance

        wav_length (tensor):
            number of samples in each utterance
        """
        return (
            source_wav_list,
            uttname_list,
            source_attr,
            source_wav,
            target_attr,
            target_wav_list,
            feat_length,
            wav_length,
            enroll_wav,
        )
