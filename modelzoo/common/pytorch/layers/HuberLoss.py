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


# Adapted from https://github.com/pytorch/pytorch/blob/473b733bae7009945cc5712699d346678e8a40ff/torch/_decomp/decompositions.py#L349
class HuberLoss(nn.Module):
    def __init__(self, reduction='mean', delta=1.0):
        super(HuberLoss, self).__init__()
        self.reduction = reduction
        assert delta > 0, "HuberLoss only supports positive values for delta."
        self.delta = delta

    def forward(self, input, target):
        z = (input - target).abs()
        loss = torch.where(
            z < self.delta, 0.5 * z * z, self.delta * (z - 0.5 * self.delta)
        )
        return apply_loss_reduction(loss, self.reduction)
