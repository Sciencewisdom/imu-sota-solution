#!/usr/bin/env python3
from __future__ import annotations

"""
Minimal MS-TCN2 model definition for inference.

This is vendored from the MS-TCN2 repo (model.py) but stripped down to avoid
training-time dependencies (loguru/eval/etc.). Only the network modules needed
to run forward() and load_state_dict() are included.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class MS_TCN2(nn.Module):
    def __init__(self, num_layers_PG: int, num_layers_R: int, num_R: int, num_f_maps: int, dim: int, num_classes: int):
        super().__init__()
        self.PG = Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes)
        self.Rs = nn.ModuleList(
            [copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes, num_classes)) for _ in range(int(num_R))]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.PG(x)
        outputs = out.unsqueeze(0)
        for R in self.Rs:
            out = R(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class Prediction_Generation(nn.Module):
    def __init__(self, num_layers: int, num_f_maps: int, dim: int, num_classes: int):
        super().__init__()
        self.num_layers = int(num_layers)
        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1 = nn.ModuleList(
            (
                nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2 ** (self.num_layers - 1 - i), dilation=2 ** (self.num_layers - 1 - i))
                for i in range(self.num_layers)
            )
        )
        self.conv_dilated_2 = nn.ModuleList(
            (nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**i, dilation=2**i) for i in range(self.num_layers))
        )
        self.conv_fusion = nn.ModuleList((nn.Conv1d(2 * num_f_maps, num_f_maps, 1) for _ in range(self.num_layers)))
        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.conv_1x1_in(x)
        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in
        return self.conv_out(f)


class Refinement(nn.Module):
    def __init__(self, num_layers: int, num_f_maps: int, dim: int, num_classes: int):
        super().__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps)) for i in range(int(num_layers))])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        return self.conv_out(out)


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation: int, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=int(dilation), dilation=int(dilation))
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out

