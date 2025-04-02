# Copyright (c) 2025 Nippon Telegraph and Telephone Corporation (NTT)
# Copyright (c) 2025 Brno University of Technology (BUT)
# This software is licensed under the NTT License (see LICENSE.NTT).

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Model(nn.Module):
    def __init__(self, input_dim, rnn_layers, hidden_size, bidirectional=False, additional_linear=False, **kwargs):
        super(Model, self).__init__()

        self.additional_linear = additional_linear

        if bidirectional:
            lstm_hidden_size = hidden_size // 2
        else:
            lstm_hidden_size = hidden_size

        self.rnn1 = nn.LSTM(input_dim, lstm_hidden_size, num_layers=1, batch_first=True, bidirectional=bidirectional)

        self.rnn2 = nn.LSTM(
            lstm_hidden_size * 2,
            lstm_hidden_size,
            num_layers=rnn_layers - 1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        if self.additional_linear:
            self.fc = nn.Linear(hidden_size, hidden_size)
            self.tanh = nn.Tanh()
        self.linear = nn.Linear(hidden_size, 3)  # tss, ntss, ns

    def forward(self, input_x, x_len, spkinfo):
        input_x = input_x.float()
        input_x = pack_padded_sequence(input_x, x_len, batch_first=True, enforce_sorted=False)
        output, _ = self.rnn1(input_x)
        output, x_len = pad_packed_sequence(output, batch_first=True)

        output = output * spkinfo
        output = pack_padded_sequence(output, x_len, batch_first=True, enforce_sorted=False)
        output, _ = self.rnn2(output)
        output, x_len = pad_packed_sequence(output, batch_first=True)

        if self.additional_linear:
            output = self.fc(output)
            output = self.tanh(output)

        output = self.linear(output)
        return output
