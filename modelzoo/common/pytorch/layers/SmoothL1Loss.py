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

from modelzoo.common.pytorch.layers.utils import apply_loss_reduction


# Adapted from https://github.com/pytorch/pytorch/blob/f96d96a7fcaa5bb06829d2c7de1992d6ab6e9235/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp#L608
class SmoothL1Loss(nn.Module):
    def __init__(self, reduction='mean', beta=1.0):
        super(SmoothL1Loss, self).__init__()
        assert (
            beta >= 0
        ), "SmoothL1Loss only supports non-negative values for beta."
        self.reduction = reduction
        self.beta = beta

    def forward(self, input, target):
        z = torch.abs(input - target)
        loss = torch.where(
            z < self.beta, 0.5 * z * z / self.beta, z - 0.5 * self.beta
        )
        return apply_loss_reduction(loss, self.reduction)
