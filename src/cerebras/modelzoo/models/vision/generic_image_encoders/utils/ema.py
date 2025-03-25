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


def create_momentum_scheduler(scheduler_name, **scheduler_params):
    # Returns a fcn lambda to compute decay every step
    dict_map = {"linear": linear_scheduler, "cosine": cosine_scheduler}

    return dict_map[scheduler_name](**scheduler_params)


def linear_scheduler(ema_decay_start, ema_decay_end, total_steps):
    def compute_decay(step):
        decay = ema_decay_start + step * (ema_decay_end - ema_decay_start) / (
            total_steps
        )
        return decay

    return compute_decay


def cosine_scheduler(ema_decay_start, ema_decay_end, total_steps):
    def compute_decay(step):
        decay = ema_decay_end + 0.5 * (ema_decay_start - ema_decay_end) * (
            1
            + torch.cos(
                torch.pi * torch.tensor(step, dtype=torch.int64) / total_steps
            )
        )
        return decay

    return compute_decay


class EMAWrapper:
    def __init__(self, src_model, tgt_model, scheduler):
        self.src_model = src_model  # nn.Module
        self.tgt_model = tgt_model  # nn.Module

        self.scheduler = scheduler  # torch.Tensor

    def apply_ema(self, step):
        # tgt_model = alpha * tgt_model + (1- alpha) * src_model
        # Do not apply EMA at initialization

        with torch.no_grad():
            decay = torch.where(step > 0, self.scheduler(step), 1.0)
            for param_src, param_tgt in zip(
                self.src_model.parameters(), self.tgt_model.parameters()
            ):
                param_tgt.mul_(decay).add_((1.0 - decay) * param_src.detach())
