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
from torch import nn

from cerebras.modelzoo.common.half_dtype import maybe_to_half_dtype


class ClipContrastiveLoss(nn.Module):
    def __init__(
        self,
    ):
        super(ClipContrastiveLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def generate_labels(self, logits):
        return torch.arange(len(logits), dtype=torch.int, device=logits.device)

    def forward(self, logits):
        labels = self.generate_labels(logits)
        if logits.device.type in ["lazy", "cuda"]:
            logits = maybe_to_half_dtype(logits)
        loss = (
            self.loss_fn(logits, labels.view(-1).long())
            + self.loss_fn(logits.T, labels.view(-1).long())
        ) / 2.0
        return loss
