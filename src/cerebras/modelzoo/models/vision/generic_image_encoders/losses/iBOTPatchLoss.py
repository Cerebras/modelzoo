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
import torch.nn.functional as F
from torch import nn

from cerebras.modelzoo.config import ModelConfig
from cerebras.modelzoo.models.vision.generic_image_encoders.utils.scheduler import (
    LinearWarmupConstantScheduler,
    LinearWarmupCosineDecayScheduler,
)


class iBOTPatchLossConfig(ModelConfig):
    name: Literal["iBOTPatchLoss"]

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
    teacher_temp_scheduler: Literal[
        "linear_constant",
        "linear_cosine",
    ] = "linear_cosine"
    "Scheduler to use for teacher temperature param"
    no_slice: bool = False

    @property
    def __model_cls__(self):
        return iBOTPatchLoss


class iBOTPatchLoss(nn.Module):
    def __init__(self, config: iBOTPatchLossConfig):
        if isinstance(config, dict):
            config = iBOTPatchLossConfig(**config)

        super().__init__()

        self.student_temp = config.student_temp
        self.center_momentum = config.center_momentum
        self.register_buffer("center", torch.zeros(config.input_size))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_patch_tokens = None
        self.no_slice = config.no_slice

        # set these values from: https://github.com/facebookresearch/dinov2/blob/main/dinov2/train/train.py#L84C1-L90C6
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

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_patch_tokens):
        # teacher centering and sharpening
        # Note: Keep self.center in f32 type, so entire loss computation is in f32.
        # To align with the public repo implementation:
        # https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/loss/ibot_patch_loss.py#L50
        center_reshaped = self.center[None, None, :].broadcast_to(
            teacher_patch_tokens.shape
        )

        teacher_temp = self.teacher_temp_schedule(self.step)

        self.step += 1

        return F.softmax(
            (teacher_patch_tokens - center_reshaped) / teacher_temp,
            dim=-1,
        )

    def forward(
        self,
        teacher_global_output,
        student_global_output,
        student_masks,
    ):
        student_masks = student_masks.to(student_global_output.dtype)
        if self.no_slice:
            student_global_patch_tokens = student_global_output
            teacher_global_patch_tokens = teacher_global_output
            mask_cls_token = torch.zeros(
                student_masks.shape[0],
                student_masks.shape[1],
                1,
                dtype=student_masks.dtype,
                device=student_masks.device,
            )
            student_masks = torch.cat([mask_cls_token, student_masks], dim=2)
        else:
            # ViT trunk sends outputs of shape [batch, n_img, n_patches+1, h]
            # we're only interested in the last n_patches tokens and not the CLS
            student_global_patch_tokens = student_global_output[:, :, 1:]
            teacher_global_patch_tokens = teacher_global_output[:, :, 1:]

        original_mask_shape = student_masks.shape
        bsz, n_imgs = original_mask_shape[0], original_mask_shape[1]
        n_loss_terms = bsz * n_imgs  # need to divide by total images

        student_global_patch_tokens = (
            student_global_patch_tokens / self.student_temp
        )

        student_global_patch_tokens = student_global_patch_tokens.reshape(
            student_global_patch_tokens.shape[0],
            student_global_patch_tokens.shape[1]
            * student_global_patch_tokens.shape[2],
            student_global_patch_tokens.shape[3],
        )

        teacher_global_patch_tokens = teacher_global_patch_tokens.reshape(
            teacher_global_patch_tokens.shape[0],
            teacher_global_patch_tokens.shape[1]
            * teacher_global_patch_tokens.shape[2],
            teacher_global_patch_tokens.shape[3],
        )

        student_masks = student_masks.reshape(
            student_masks.shape[0],
            student_masks.shape[1] * student_masks.shape[2],
        )
        new_mask_shape = student_masks.shape
        # teacher centering and sharpening
        teacher_softmax_out = self.softmax_center_teacher(
            teacher_global_patch_tokens
        )

        # we are interested in cases where student and teacher operate on the same view
        patch_loss = torch.sum(
            -teacher_softmax_out
            * F.log_softmax(student_global_patch_tokens, dim=-1),
            dim=-1,
        )
        patch_loss = patch_loss * student_masks
        mask_weight = (
            torch.sum(
                student_masks.reshape(original_mask_shape), dim=-1, keepdim=True
            )
            .clamp(min=1)
            .broadcast_to(original_mask_shape)
        ).reshape(new_mask_shape)
        patch_loss = patch_loss / mask_weight
        patch_loss = torch.sum(patch_loss, dim=-1)
        total_loss = patch_loss.sum()
        self.update_center(teacher_global_patch_tokens, student_masks)
        ibot_loss = total_loss / n_loss_terms
        return ibot_loss

    @torch.no_grad()
    def update_center(self, teacher_patch_tokens, student_masks):
        students_masks_casted = student_masks.to(teacher_patch_tokens.dtype)

        total_masks = torch.sum(students_masks_casted)
        batch_center_sum = torch.sum(
            teacher_patch_tokens * students_masks_casted.unsqueeze(-1),
            dim=[0, 1],
        )
        batch_center = batch_center_sum / total_masks
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )
