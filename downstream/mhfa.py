# Copyright (c) 2025 Nippon Telegraph and Telephone Corporation (NTT)
# Copyright (c) 2025 Brno University of Technology (BUT)
# This software is licensed under the NTT License (see LICENSE.NTT).

import torch
import torch.nn as nn


# multi-head factorized attentive pooling (https://arxiv.org/abs/2210.01273)
class MHFA(nn.Module):
    def __init__(self, head_nb=8, inputs_dim=768, compression_dim=128, outputs_dim=256):
        super(MHFA, self).__init__()
        self.head_nb = head_nb
        self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.ous_dim = outputs_dim
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim)
        self.att_head = nn.Linear(self.cmp_dim, self.head_nb)
        self.pooling_fc = nn.Linear(self.head_nb * self.cmp_dim, self.ous_dim)

    def forward(self, k, v):
        k = self.cmp_linear_k(k)
        v = self.cmp_linear_v(v)
        att_k = self.att_head(k)
        v = v.unsqueeze(-2)
        pooling_outs = torch.sum(v.mul(nn.functional.softmax(att_k, dim=1).unsqueeze(-1)), dim=1)
        b, h, f = pooling_outs.shape
        pooling_outs = pooling_outs.reshape(b, -1)
        outs = self.pooling_fc(pooling_outs)
        return outs
