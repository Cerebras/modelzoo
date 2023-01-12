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


class MultiMarginLoss(nn.Module):
    def __init__(self, p=1, margin=1.0, weight=None, reduction='mean'):
        super(MultiMarginLoss, self).__init__()
        assert p in [1, 2], "MultiMarginLoss only supports p being 1 or 2."
        self.p = p
        self.margin = margin
        self.weight = weight
        self.reduction = reduction

    def forward(self, x, y):
        # generate identity matrix
        eye = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
        loss = (
            self.margin
            # exclude i == y, use eye + index_select, since one_hot can only have i64 output
            * (1 - eye.index_select(0, y))
            # - x[y]
            - torch.gather(x, -1, y.unsqueeze(-1))
            + x
        )

        if self.weight:
            # * w[y]
            loss = loss * torch.gather(self.weight, 0, y).unsqueeze(-1)

        loss = loss.clamp(min=0)

        if self.p == 2:
            # pow(loss, p)
            loss = loss * loss

        if self.reduction == "sum":
            loss = torch.div(loss, x.shape[-1])
        return apply_loss_reduction(loss, self.reduction)
