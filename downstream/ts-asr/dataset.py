# Copyright (c) 2025 Nippon Telegraph and Telephone Corporation (NTT)
# Copyright (c) 2025 Brno University of Technology (BUT)
# Modifications made by NTT & BUT are licensed under the NTT License (see LICENSE.NTT).
#
# The original code is:
#   Copyright (c) Speech Lab, NTU, Taiwan.
#   Licensed under the Apache License, Version 2.0.
#   See the LICENSE.Apache file or http://www.apache.org/licenses/LICENSE-2.0
#
# Summary of Modifications by NTT & BUT:
# - Enable loading Libri2Mix and enrollment speech.
#
# Original source: https://github.com/s3prl/s3prl/blob/v0.4.17/s3prl/downstream/asr/dataset.py

import os
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from asteroid.data import LibriMix
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from ..asr.dictionary import Dictionary

SAMPLE_RATE = 16000
HALF_BATCHSIZE_TIME = 2000
N_SRC = 2
SEGMENT = 3
SEGMENT_AUX = 3

LIBRI_SUBDIR = {
    "train": "train-clean-100",
    "dev": "dev-clean",
    "test": "test-clean",
}


def read_enrollment_csv(csv_path):
    data = {}
    with open(csv_path, "r") as f:
        f.readline()  # csv header
        for line in f:
            mix_id, utt_id, *aux = line.strip().split(",")
            aux_it = iter(aux)
            aux = [(auxpath, int(float(length))) for auxpath, length in zip(aux_it, aux_it)]
            data[(mix_id, utt_id)] = aux
    return data


class SequenceDataset(Dataset):
    def __init__(self, split, bucket_size, dictionary, libri_root, **kwargs):
        super(SequenceDataset, self).__init__()

        self.dictionary = dictionary
        self.libri_root = os.path.join(libri_root, LIBRI_SUBDIR[split])
        self.sample_rate = SAMPLE_RATE
        self.csv_dir = kwargs[f"{split}_dir"]
        self.use_spkcode = kwargs.get("use_spkcode", False)
        self.use_noisemix = kwargs.get("use_noisemix", False)
        if self.use_spkcode:
            print("Use speaker code as enrollment speech")

        if self.use_spkcode:
            self.spk_dict = {}
            with open(str(Path(__file__).parent / "spkdict.txt"), "r") as d:
                for line in d:
                    lspl = line.strip().split(" ")
                    self.spk_dict[int(lspl[0])] = int(lspl[1])

        if split == "test":
            segment = None
            segment_aux = None
        else:
            segment = SEGMENT  # sec
            segment_aux = SEGMENT_AUX  # sec

        # Enrollments
        data_aux = read_enrollment_csv(Path(self.csv_dir) / "mixture2enrollment.csv")

        # data filtering
        if segment_aux is not None:
            max_len = np.sum([len(aux) for aux in data_aux.values()])
            self.seg_len_aux = int(segment_aux * self.sample_rate)
            self.Enr = {}
            for k, v in data_aux.items():
                items = [(path, length) for path, length in v if length >= self.seg_len_aux]
                if len(items) == 0:
                    continue
                else:
                    self.Enr[k] = items
            new_len = np.sum([len(aux) for aux in self.Enr.values()])
            print(f"Drop {max_len - new_len} utterances from {max_len} " f"(shorter than {segment_aux} seconds)")
        else:
            self.seg_len_aux = None
            self.Enr = data_aux

        # MIX data
        if self.use_noisemix:
            table_list = LibriMix(self.csv_dir, "sep_noisy", self.sample_rate, N_SRC, segment)
        else:
            table_list = LibriMix(self.csv_dir, "sep_clean", self.sample_rate, N_SRC, segment)
        # sort
        table_list = table_list.df.sort_values(by=["length"], ascending=False)

        use_single_source = kwargs.get("use_single_source", False)
        if use_single_source:  # evaluate the upper-bound score
            X1 = [item for item in table_list["source_1_path"].tolist()]
            X2 = [item for item in table_list["source_2_path"].tolist()]
            X = [item for pair in zip(X1, X2) for item in pair]
        else:
            X = [item for item in table_list["mixture_path"].tolist() for _ in range(N_SRC)]
        assert len(X) != 0, f"0 data found for {split}"
        X_lens = [item for item in table_list["length"].tolist() for _ in range(N_SRC)]
        X_mixid = [item for item in table_list["mixture_ID"].tolist() for _ in range(N_SRC)]
        X_uttid = []
        for i, x in enumerate(X_mixid):
            if i % 2 == 0:
                X_uttid.append(x.split("_")[0])
            else:
                X_uttid.append(x.split("_")[1])

        # Use bucketing to allow different batch sizes at run time
        self.X = []
        batch_x, batch_len = [], []
        counter = 0

        for x, x_len, x_mixid, x_uttid in tqdm(
            zip(X, X_lens, X_mixid, X_uttid), total=len(X), desc=f"ASR dataset {split}", dynamic_ncols=True
        ):
            if (x_mixid, x_uttid) not in self.Enr.keys():
                counter += 1
                continue
            # assert (x_mixid, x_uttid) in self.Enr.keys(), f"Not exist {(x_mixid, x_uttid)} in the enrollment dict"
            batch_x.append((x, x_mixid, x_uttid))
            batch_len.append(x_len)

            # Fill in batch_x until batch is full
            if len(batch_x) == bucket_size:
                # Half the batch size if seq too long
                if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME):
                    self.X.append(batch_x[: bucket_size // 2])
                    self.X.append(batch_x[bucket_size // 2 :])
                else:
                    self.X.append(batch_x)
                batch_x, batch_len = [], []

        # Gather the last batch
        if len(batch_x) > 1:
            self.X.append(batch_x)

        if counter > 0:
            print(f"Drop {str(counter)} samples")

        # Transcripts
        Y = self._load_transcript(X_uttid)
        self.Y = {k: self.dictionary.encode_line(v, line_tokenizer=lambda x: x.split()).long() for k, v in Y.items()}

    def _parse_x_name(self, x):
        return x.split("/")[-1].split(".")[0]

    def _load_wav(self, wav_path):
        wav, sr = torchaudio.load(wav_path)
        assert sr == self.sample_rate, f"Sample rate mismatch: real {sr}, config {self.sample_rate}"
        return wav.view(-1)

    def _get_segment_start_stop(self, seg_len, length):
        if seg_len is not None:
            start = random.randint(0, length - seg_len)
            stop = start + int(seg_len)
        else:
            start = 0
            stop = None
        return start, stop

    def _load_enrollment(self, enrolls):
        enroll_path, enroll_length = random.choice(enrolls)
        if self.use_spkcode:
            spkidx = int(enroll_path.split("/")[-2].replace("s", "")) - 1  # s1 or s2
            spkid = int(enroll_path.split("/")[-1].split("_")[spkidx].split("-")[0])
            return torch.LongTensor([self.spk_dict[spkid]])
        start_e, stop_e = self._get_segment_start_stop(self.seg_len_aux, enroll_length)
        enroll_wave = self._load_wav(enroll_path)[start_e:stop_e]
        return enroll_wave

    def _load_transcript(self, x_list):
        """Load the transcripts for Librispeech"""

        def process_trans(transcript):
            # TODO: support character / bpe
            transcript = transcript.upper()
            return " ".join(list(transcript.replace(" ", "|"))) + " |"

        trsp_sequences = {}
        for dir in x_list:
            parts = dir.split("-")
            trans_path = f"{parts[0]}-{parts[1]}.trans.txt"
            path = os.path.join(self.libri_root, parts[0], parts[1], trans_path)
            assert os.path.exists(path), f"Not exisit: {path}"

            find_transcript = False
            with open(path, "r") as trans_f:
                for line in trans_f:
                    lst = line.strip().split()
                    if lst[0] == dir:
                        trsp_sequences[lst[0]] = process_trans(" ".join(lst[1:]))
                        find_transcript = True
                        break
            assert find_transcript, f"Not found transcript: {dir}"

        return trsp_sequences

    def _build_dictionary(self, transcripts, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        d = Dictionary()
        transcript_list = list(transcripts.values())
        Dictionary.add_transcripts_to_dictionary(transcript_list, d, workers)
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Load acoustic feature and pad
        wav_batch = [self._load_wav(x_file[0]).numpy() for x_file in self.X[index]]
        label_batch = [self.Y[x_file[2]].numpy() for x_file in self.X[index]]
        enroll_batch = [self._load_enrollment(self.Enr[(x_file[1], x_file[2])]).numpy() for x_file in self.X[index]]
        filename_batch = [Path(x_file[0]).stem for x_file in self.X[index]]
        return wav_batch, label_batch, filename_batch, enroll_batch  # bucketing, return ((wavs, labels))

    def collate_fn(self, items):
        assert len(items) == 1
        return items[0][0], items[0][1], items[0][2], items[0][3]  # hack bucketing, return (wavs, labels, filenames)
