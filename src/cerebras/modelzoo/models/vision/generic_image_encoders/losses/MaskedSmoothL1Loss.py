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

from typing import Literal

import torch.nn as nn

from cerebras.modelzoo.config import ModelConfig
from cerebras.pytorch.nn import SmoothL1Loss


class MaskedSmoothL1LossConfig(ModelConfig):
    name: Literal["MaskedSmoothL1Loss"]

    reduction: Literal["mean", "none", "sum"] = "mean"

    beta: float = 1.0

    @property
    def __model_cls__(self):
        return MaskedSmoothL1Loss


class MaskedSmoothL1Loss(nn.Module):
    def __init__(self, config: MaskedSmoothL1LossConfig):
        if isinstance(config, dict):
            config = MaskedSmoothL1LossConfig(**config)

        super().__init__()

        self.reduction = config.reduction
        self.beta = config.beta
        self.loss_fn = SmoothL1Loss(reduction="none", beta=self.beta)

    def forward(self, input, target, mask):
        loss = self.loss_fn(input, target)
        mask = mask.broadcast_to(loss.shape)
        loss = loss * mask

        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.sum() / mask.sum()
        else:
            return loss
