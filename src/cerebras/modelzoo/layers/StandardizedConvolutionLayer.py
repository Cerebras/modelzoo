# Copyright 2022 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.utils import _pair, _single, _triple


class StdConv1d(nn.Conv1d):
    def forward(self, inputs: Tensor):
        w = self.weight
        std, mean = torch.std_mean(w, dim=[1, 2], keepdim=True, unbiased=False)
        w = (w - mean) / (std + 1e-6)

        if self.padding_mode != "zeros":
            return F.conv1d(
                F.pad(
                    inputs,
                    self._reversed_padding_repeated_twice,
                    mode=self.padding_mode,
                ),
                w,
                self.bias,
                self.stride,
                _single(0),
                self.dilation,
                self.groups,
            )

        return F.conv1d(
            inputs,
            w,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class StdConv2d(nn.Conv2d):
    def forward(self, inputs: Tensor):
        w = self.weight
        std, mean = torch.std_mean(
            w, dim=[1, 2, 3], keepdim=True, unbiased=False
        )
        w = (w - mean) / (std + 1e-6)

        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(
                    inputs,
                    self._reversed_padding_repeated_twice,
                    mode=self.padding_mode,
                ),
                w,
                self.bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )

        return F.conv2d(
            inputs,
            w,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class StdConv3d(nn.Conv3d):
    def forward(self, inputs: Tensor):
        w = self.weight
        std, mean = torch.std_mean(
            w, dim=[1, 2, 3, 4], keepdim=True, unbiased=False
        )
        w = (w - mean) / (std + 1e-6)

        if self.padding_mode != "zeros":
            return F.conv3d(
                F.pad(
                    inputs,
                    self._reversed_padding_repeated_twice,
                    mode=self.padding_mode,
                ),
                w,
                self.bias,
                self.stride,
                _triple(0),
                self.dilation,
                self.groups,
            )

        return F.conv3d(
            inputs,
            w,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
