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

import torch
from torch import nn

from cerebras.modelzoo.config import ModelConfig


class KoLeoLossConfig(ModelConfig):
    name: Literal["KoLeoLoss"]

    @property
    def __model_cls__(self):
        return KoLeoLoss


class KoLeoLoss(nn.Module):
    def __init__(self, config: KoLeoLossConfig):
        if isinstance(config, dict):
            config = KoLeoLossConfig(**config)

        super().__init__()

    def forward(self, student_global_cls_tokens):
        """
        Args:
            student_global_output: Output from student model with global views as input
            Shape: (bsz, num_global_views, input_size).
        """
        if student_global_cls_tokens.shape[0] < 2:
            raise ValueError("KoLeoLoss requires batch_size >= 2")

        eps = 1e-8

        norm = torch.sqrt(
            torch.sum(
                student_global_cls_tokens * student_global_cls_tokens,
                dim=-1,
                keepdim=True,
            )
            + eps
        )
        student_global_cls_tokens = student_global_cls_tokens / norm

        # we don't apply koleo loss between cls tokens of a same image
        student_global_cls_tokens = student_global_cls_tokens.chunk(
            student_global_cls_tokens.shape[1], dim=1
        )
        loss = 0
        for student_out in student_global_cls_tokens:
            student_output = student_out.squeeze(1)  # (bsz, input_size)
            dots = student_output @ student_output.T  # (bsz, bsz)

            # exclude self-to-self distance and find the nearest neighbor
            dots = dots.fill_diagonal_(-1)
            I = dots.argmax(dim=1)

            # compute the distance between the student_output and its nearest neighbor
            distances = (student_output - student_output[I]).norm(p=2, dim=-1)
            loss += -torch.log(distances + eps).mean()
        return loss
