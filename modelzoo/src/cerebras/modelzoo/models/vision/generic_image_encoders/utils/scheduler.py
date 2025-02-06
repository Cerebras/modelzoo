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

import math

import torch
from torch import nn


class LinearWarmupCosineDecayScheduler(nn.Module):
    def __init__(
        self,
        base_value,
        final_value,
        total_steps,
        warmup_steps=0,
        start_warmup_value=0,
        freeze_steps=0,
    ):
        super().__init__()
        self.base_value = base_value
        self.final_value = final_value
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.start_warmup_value = start_warmup_value
        self.freeze_steps = freeze_steps

    def forward(self, step):
        if self.warmup_steps > 1:
            linear_part = (self.base_value - self.start_warmup_value) / (
                self.warmup_steps - 1
            )
            warmup_temp = self.start_warmup_value + linear_part * (
                step - self.freeze_steps
            )
        else:
            warmup_temp = self.start_warmup_value
        cos_temp = self.final_value + 0.5 * (
            self.base_value - self.final_value
        ) * (
            1
            + torch.cos(
                math.pi
                * (step - self.warmup_steps - self.freeze_steps)
                / (self.total_steps - self.warmup_steps - self.freeze_steps)
            )
        )
        temp = torch.where(
            step < self.freeze_steps,
            0,
            torch.where(
                step < (self.warmup_steps + self.freeze_steps),
                warmup_temp,
                cos_temp,
            ),
        )
        return temp


class LinearWarmupConstantScheduler(nn.Module):
    def __init__(
        self, base_value, total_steps, warmup_steps=0, start_warmup_value=0
    ):
        super().__init__()
        self.base_value = base_value
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.start_warmup_value = start_warmup_value

    def forward(self, step):
        if self.warmup_steps > 1:
            linear_part = (self.base_value - self.start_warmup_value) / (
                self.warmup_steps - 1
            )
            warmup_temp = self.start_warmup_value + linear_part * step
        else:
            warmup_temp = self.start_warmup_value
        temp = torch.where(
            step < self.warmup_steps, warmup_temp, self.base_value
        )
        return temp
