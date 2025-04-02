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
# - Enable loading SparseLibriMix dataset and enrollment speech.
#
# Original source: https://github.com/s3prl/s3prl/blob/v0.4.17/s3prl/downstream/diarization/dataset.py

import io
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from asteroid.data import LibriMix
from torch.utils.data.dataset import Dataset

from ..diarization.dataset import KaldiData

SAMPLE_RATE = 16000
SEGMENT = 3
N_SRC = 2


def read_enrollment_csv(csv_path, mixture_ids):
    data = {}
    with open(csv_path, "r") as f:
        f.readline()  # csv header

        for line in f:
            mix_id, utt_id, *aux = line.strip().split(",")
            if mixture_ids is not None and mix_id not in mixture_ids:
                continue
            aux_it = iter(aux)
            aux = [(auxpath, int(float(length))) for auxpath, length in zip(aux_it, aux_it)]
            data[(mix_id, utt_id)] = aux
    return data


class DiarizationDataset(Dataset):
    def __init__(
        self,
        mode,
        data_dir,
        enroll_dir,
        dtype=np.float32,
        frame_shift=256,
        subsampling=1,
        corpus="sparselibrimix",
    ):
        super(DiarizationDataset, self).__init__()

        self.mode = mode
        self.data_dir = data_dir
        self.enroll_dir = enroll_dir
        self.dtype = dtype
        self.frame_shift = frame_shift
        self.subsampling = subsampling
        self.corpus = corpus

        if mode == "test":
            segment = None
            segment_aux = None
        else:
            segment = SEGMENT
            segment_aux = SEGMENT  # sec

        # load mixture data
        if self.corpus == "librimix":
            self.data = KaldiData(self.data_dir)
            use_mids = None
        elif self.corpus == "sparselibrimix":
            self.data = JsonData(self.data_dir, segment)
            use_mids = self.data.base_dataset.df["mixture_ID"].tolist()
        print(f"The number of mixed speechs: {len(self.data.wavs)}")

        # load enrollment data
        data_aux = read_enrollment_csv(Path(self.enroll_dir) / "mixture2enrollment.csv", use_mids)
        # data filtering
        if segment_aux is not None:
            max_len = np.sum([len(aux) for aux in data_aux.values()])
            self.seg_len_aux = int(segment_aux * SAMPLE_RATE)
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

    def __getitem__(self, i):
        # enroll
        mix_id, utt_id = self.data_aux_list[i]
        enroll_path, enroll_length = random.choice(self.data_aux[(mix_id, utt_id)])

        start_e, stop_e = self._get_segment_start_stop(self.seg_len_aux, enroll_length)
        enroll_wave, _ = sf.read(enroll_path, dtype="float32", start=start_e, stop=stop_e)

        # mix and label
        tgt_spk_id = utt_id.split("-")[0]
        Y, T = self._get_labeled_speech(mix_id, tgt_spk_id)

        return Y, T, mix_id, utt_id, enroll_wave

    def _get_segment_start_stop(self, seg_len, length):
        if seg_len is not None:
            start = random.randint(0, length - seg_len)
            stop = start + int(seg_len)
        else:
            start = 0
            stop = None
        return start, stop

    def _get_labeled_speech(self, rec, tgt_spk_id):
        """Extracts speech chunks and corresponding labels

        Extracts speech chunks and corresponding diarization labels for
        given recording id and start/end times

        Args:
            rec (str): recording id
            start (int): start frame index
            end (int): end frame index
            n_speakers (int): number of speakers
                if None, the value is given from data
        Returns:
            data: speech chunk
                (n_samples)
            T: label
                (n_frmaes, n_speakers)-shaped np.int32 array.
        """
        data, rate = self.data.load_wav(rec)
        frame_num = int(data.shape[0] / self.frame_shift)
        filtered_segments = self.data.segments[rec]
        T = np.full(frame_num, 2, dtype=np.int32)  # ns

        # tss, ntss
        tss_frames = []
        for seg in filtered_segments:
            start_frame = np.rint(seg["st"] * rate / self.frame_shift).astype(int)
            end_frame = np.rint(seg["et"] * rate / self.frame_shift).astype(int)
            if self.data.utt2spk[seg["utt"]] == tgt_spk_id:
                tss_frames.append((start_frame, end_frame))
            else:
                T[start_frame:end_frame] = 1  # ntss
        # finally insert tss frames
        assert len(tss_frames) >= 1, f"not appear the target speaker in the mixed speech. ({rec}, {tgt_spk_id})"
        for start_frame, end_frame in tss_frames:
            T[start_frame:end_frame] = 0  # tss
        return data, T

    def collate_fn(self, batch):
        batch_size = len(batch)
        frame_len_list = [len(batch[i][1]) for i in range(batch_size)]
        wav = []
        label = []
        mix_id = []
        utt_id = []
        enroll_wav = []
        for i in range(batch_size):
            wav.append(batch[i][0].astype(np.float32))
            label.append(batch[i][1].astype(np.int32))
            mix_id.append(batch[i][2])
            utt_id.append(batch[i][3])
            enroll_wav.append(batch[i][4].astype(np.float32))
        frame_length = np.array(frame_len_list)
        return wav, label, frame_length, mix_id, utt_id, enroll_wav


class JsonData:
    def __init__(self, data_dir, segment):
        json_path = os.path.join(data_dir, "metadata.json")
        with open(json_path, "r") as f:
            json_data = json.load(f)

        self.base_dataset = LibriMix(data_dir, "sep_noisy", SAMPLE_RATE, N_SRC, segment)
        self.wavs = dict(zip(self.base_dataset.df["mixture_ID"], self.base_dataset.df["mixture_path"]))
        self.segments = {}
        self.utt2spk = {}
        for entry in json_data:
            if entry["mixture_name"] not in self.segments:
                self.segments[entry["mixture_name"]] = []
            for sid in ["s1", "s2"]:
                for each_seg in entry[sid]:
                    utt_id = each_seg["file"].split("/")[-1].replace(".flac", "")
                    self.segments[entry["mixture_name"]].append(
                        {"utt": f"{sid}-{utt_id}", "st": float(each_seg["start"]), "et": float(each_seg["stop"])}
                    )
                    self.utt2spk[f"{sid}-{utt_id}"] = sid

    def load_wav(self, recid, start=0, end=None):
        """Load wavfile given recid, start time and end time."""
        data, rate = self._load_wav(self.wavs[recid], start, end)
        return data, rate

    def _load_wav(self, wav_rxfilename, start=0, end=None):
        """This function reads audio file and return data in numpy.float32 array.
        "lru_cache" holds recently loaded audio so that can be called
        many times on the same audio file.
        OPTIMIZE: controls lru_cache size for random access,
        considering memory size
        """
        if wav_rxfilename.endswith("|"):
            # input piped command
            p = subprocess.Popen(
                wav_rxfilename[:-1],
                shell=True,
                stdout=subprocess.PIPE,
            )
            data, samplerate = sf.read(
                io.BytesIO(p.stdout.read()),
                dtype="float32",
            )
            # cannot seek
            data = data[start:end]
        elif wav_rxfilename == "-":
            # stdin
            data, samplerate = sf.read(sys.stdin, dtype="float32")
            # cannot seek
            data = data[start:end]
        else:
            # normal wav file
            data, samplerate = sf.read(wav_rxfilename, start=start, stop=end)
        assert SAMPLE_RATE == samplerate, "Wrong sampling rate of dataset"
        return data, samplerate
