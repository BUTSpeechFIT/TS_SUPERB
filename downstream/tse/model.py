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
# - Enable accepting and conditioning on enrollment speech.
#
# Original source: https://github.com/s3prl/s3prl/blob/v0.4.17/s3prl/downstream/separation_stft/model.py

import torch
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence


class SepRNN(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        num_bins,
        rnn="lstm",
        num_spks=2,
        num_layers=3,
        hidden_size=896,
        dropout=0.0,
        non_linear="relu",
        bidirectional=True,
    ):
        super(SepRNN, self).__init__()

        if non_linear not in ["relu", "sigmoid", "tanh"]:
            raise ValueError("Unsupported non-linear type:{}".format(non_linear))
        self.num_spks = num_spks
        rnn = rnn.upper()
        if rnn not in ["RNN", "LSTM", "GRU"]:
            raise ValueError("Unsupported rnn type: {}".format(rnn))
        self.rnn1 = getattr(torch.nn, rnn)(
            input_dim, hidden_size, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional
        )
        self.rnn2 = getattr(torch.nn, rnn)(
            hidden_size * 2,
            hidden_size,
            num_layers - 1,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.drops = torch.nn.Dropout(p=dropout)
        self.linear = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size * 2 if bidirectional else hidden_size, num_bins) for _ in range(self.num_spks)]
        )
        self.non_linear = {
            "relu": torch.nn.functional.relu,
            "sigmoid": torch.nn.functional.sigmoid,
            "tanh": torch.nn.functional.tanh,
        }[non_linear]
        self.num_bins = num_bins

    def forward(self, x, train=True, spkinfo=None):
        assert isinstance(x, PackedSequence)
        x, _ = self.rnn1(x)
        x, len_x = pad_packed_sequence(x, batch_first=True)
        x = x * spkinfo
        x = pack_padded_sequence(x, len_x, batch_first=True)
        x, _ = self.rnn2(x)
        x, len_x = pad_packed_sequence(x, batch_first=True)
        x = self.drops(x)
        m = []
        for linear in self.linear:
            y = linear(x)
            y = self.non_linear(y)
            if not train:
                y = y.view(-1, self.num_bins)
            m.append(y)
        return m
