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
import torch.nn as nn
import torch.nn.functional as F

from cerebras.modelzoo.config import ModelConfig
from cerebras.modelzoo.models.vision.generic_image_encoders.utils.scheduler import (
    LinearWarmupConstantScheduler,
    LinearWarmupCosineDecayScheduler,
)


class DinoDistillationLossConfig(ModelConfig):
    name: Literal["DinoDistillationLoss"]

    input_size: int = ...
    "Size of input tensor in last dimension. Used to create teacher center buffer"
    warmup_teacher_temp: float = ...
    "Teacher warmup temperature value"
    teacher_temp: float = ...
    "Final teacher temperature value"
    warmup_teacher_temp_steps: int = ...
    "number of steps to raise temperature value from `warmup_teacher_temp` to `teacher_temp`"
    total_steps: int = ...
    "Total steps in teacher temperature scheduler"
    student_temp: float = 0.1
    "Constant value to use as temperature for input tensor from student model"
    center_momentum: float = 0.9
    """Value to update batch center. 
    center = center * center_momentum + (1-center_momentum) * current_batch_center"""
    teacher_temp_scheduler: Literal["linear_constant", "linear_cosine"] = (
        "linear_constant"
    )
    "Scheduler to use for teacher temperature param"

    @property
    def __model_cls__(self):
        return DinoDistillationLoss


class DinoDistillationLoss(nn.Module):
    def __init__(self, config: DinoDistillationLossConfig):
        if isinstance(config, dict):
            config = DinoDistillationLossConfig(**config)

        super().__init__()

        self.student_temp = config.student_temp
        self.center_momentum = config.center_momentum
        self.register_buffer("center", torch.zeros(config.input_size))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        # dino v2 uses a cosine scheduler over a const one so provide that as an option
        if config.teacher_temp_scheduler == "linear_constant":
            self.teacher_temp_schedule = LinearWarmupConstantScheduler(
                base_value=config.teacher_temp,
                start_warmup_value=config.warmup_teacher_temp,
                warmup_steps=config.warmup_teacher_temp_steps,
                total_steps=config.total_steps,
            )
        elif config.teacher_temp_scheduler == "linear_cosine":
            self.teacher_temp_schedule = LinearWarmupCosineDecayScheduler(
                base_value=config.teacher_temp,
                final_value=config.teacher_temp,
                total_steps=config.total_steps,
                warmup_steps=config.warmup_teacher_temp_steps,
                start_warmup_value=config.warmup_teacher_temp,
            )
        self.register_buffer("step", torch.zeros(1, dtype=torch.int32))

    def forward(
        self,
        student_global_output,
        teacher_global_output,
        student_local_output=None,
    ):
        """
        Args:
            student_global_output: Output from student model with global views as input
            Shape: (bsz, num_global_views, input_size).

            teacher_global_output: Output from teacher model with global views as input
            Shape: (bsz, num_global_views, input_size)

            student_local_output: Output from student model with local views as input
            Shape: (bsz, num_local_views, input_size)
        """

        loss_global_global, n_loss_global_global = self.forward_crops(
            student_global_output, teacher_global_output, ignore_same_index=True
        )
        loss_local_global, n_loss_local_global = (
            torch.tensor(0.0, dtype=loss_global_global.dtype),
            0,
        )
        if student_local_output is not None:
            loss_local_global, n_loss_local_global = self.forward_crops(
                student_local_output,
                teacher_global_output,
                ignore_same_index=False,
            )

        total_sum_loss = loss_global_global + loss_local_global
        n_loss_terms = n_loss_global_global + n_loss_local_global

        total_loss = total_sum_loss / n_loss_terms

        self.update_center(teacher_global_output)
        self.step += 1

        return total_loss

    def forward_crops(self, student_output, teacher_output, ignore_same_index):

        student_out = student_output / self.student_temp
        student_out = student_out.chunk(student_output.shape[1], dim=1)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule(self.step)
        # Note: Keep self.center in f32 type, so entire loss computation is in f32.
        # To align with the public repo implementation:
        # https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/loss/ibot_patch_loss.py#L50
        teacher_out = F.softmax(
            (
                teacher_output
                - self.center[None, None, :].broadcast_to(teacher_output.shape)
            )
            / temp,
            dim=-1,
        )
        teacher_out = teacher_out.detach().chunk(teacher_output.shape[1], dim=1)

        total_loss = 0
        n_loss_terms = 0

        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if ignore_same_index and v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(
                    -q.squeeze(1)
                    * F.log_softmax(student_out[v].squeeze(1), dim=-1),
                    dim=-1,
                )
                total_loss += loss.mean()
                n_loss_terms += 1
        return total_loss, n_loss_terms

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=[0, 1], keepdim=False)
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )
