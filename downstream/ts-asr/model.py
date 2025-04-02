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
# - Enable accepting and conditioning on enrollment speech.
#
# Original source: https://github.com/s3prl/s3prl/blob/v0.4.17/s3prl/downstream/asr/model.py

import torch.nn as nn

from ..asr.model import RNNLayer, downsample


class TSRNNs(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        upstream_rate,
        module,
        bidirection,
        dim,
        dropout,
        layer_norm,
        proj,
        sample_rate,
        sample_style,
        spk_conditioning_layer=1,
        total_rate=320,
    ):
        super(TSRNNs, self).__init__()
        latest_size = input_size

        self.sample_rate = 1 if total_rate == -1 else round(total_rate / upstream_rate)
        self.sample_style = sample_style
        if sample_style == "concat":
            latest_size *= self.sample_rate

        self.rnns = nn.ModuleList()
        for i in range(len(dim)):
            rnn_layer = RNNLayer(
                latest_size,
                module,
                bidirection,
                dim[i],
                dropout[i],
                layer_norm[i],
                sample_rate[i],
                proj[i],
            )
            self.rnns.append(rnn_layer)
            latest_size = rnn_layer.out_dim

        self.linear = nn.Linear(latest_size, output_size)
        self.spk_conditioning_layer = spk_conditioning_layer

    def forward(self, x, x_len, spkinfo):
        r"""
        Args:
            x (torch.Tensor): Tensor of dimension (batch_size, input_length, num_features).
            x_len (torch.IntTensor): Tensor of dimension (batch_size).
            spkinfo: Tensor of dimension (batch_size, 1, num_features).
        Returns:
            Tensor: Predictor tensor of dimension (batch_size, input_length, number_of_classes).
        """
        # Perform Downsampling
        if self.sample_rate > 1:
            x, x_len = downsample(x, x_len, self.sample_rate, self.sample_style)

        for i, rnn in enumerate(self.rnns):
            if self.spk_conditioning_layer == i:
                x = x * spkinfo
            x, x_len = rnn(x, x_len)

        logits = self.linear(x)
        return logits, x_len
