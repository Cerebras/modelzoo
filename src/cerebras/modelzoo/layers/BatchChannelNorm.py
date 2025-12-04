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

from functools import partial

import torch
import torch.nn as nn

from cerebras.modelzoo.layers.utils import ModuleWrapperClass


class BatchChannelNorm2D(nn.Module):
    """
    Implements Batch Channel Normalization
    proposed in `Micro-Batch Training with Batch-Channel
    Normalization and Weight Standardization`
    <https://arxiv.org/abs/1903.10520>

    Args:
        num_groups (int): number of groups to separate the channels into.
        num_channels (int): number of channels. `C` from an expected input of size (N, C, H, W).
        eps (float): a value added to the denominator for numerical stability. Default: 1e-5.
        momentum (float): The `Update rate` value used for the
            `running_mean` and `running_var` computation. Default: 0.1.
        device (torch.device): Device to place the learnable parameters.
        dtype (torch.dtype): Data type of learnable parameters.

    Shape:
        input: `(N, C, H, W)`
        output: `(N, C, H, W)` (same shape as input)

    """

    def __init__(
        self,
        num_groups,
        num_channels,
        eps=1.0e-5,
        momentum=0.1,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BatchChannelNorm2D, self).__init__()
        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')

        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(
            torch.empty(1, num_groups, 1, **factory_kwargs)
        )
        self.bias = nn.Parameter(
            torch.empty(1, num_groups, 1, **factory_kwargs)
        )

        self.batchnorm = nn.BatchNorm2d(
            num_channels,
            eps=self.eps,
            momentum=self.momentum,
            affine=True,
            **factory_kwargs,
        )

        cn = partial(
            nn.functional.batch_norm,
            running_mean=None,
            running_var=None,
            weight=None,
            bias=None,
            training=self.training,
            momentum=0,
            eps=self.eps,
        )
        self.channelnorm = ModuleWrapperClass(
            fcn=cn,
            name='ChannelNorm',
            kwargs={
                "momentum": 0,
                "eps": self.eps,
                "affine": False,
                "track_running_stats": False,
            },
        )
        self.reset_parameters()

    def forward(self, input):
        out = self.batchnorm(input)
        out = out.reshape(1, input.shape[0] * self.num_groups, -1)
        out = self.channelnorm(out)
        out = out.reshape(input.shape[0], self.num_groups, -1)
        out = self.weight * out + self.bias
        out = out.reshape(input.shape)
        return out

    def reset_parameters(self) -> None:
        nn.init.ones_(self.batchnorm.weight.data)
        nn.init.zeros_(self.batchnorm.bias.data)
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
