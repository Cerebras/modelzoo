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

"""Contains the MixedPrecision callback class for mixed precision training."""

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Literal, Optional, Union
from warnings import warn

import torch

import cerebras.pytorch as cstorch
from cerebras.modelzoo.trainer.callbacks import Callback


class Precision(Callback, ABC):
    """Base precision class for implementing custom backwards pass and
    optimization step to handle different precision types.
    """

    @abstractmethod
    def autocast_context_manager(self):
        """Returns the context manager that performs autocasting for the
        forward pass.
        """

    @abstractmethod
    def backward(self, loss: torch.Tensor):
        """Performs the backward pass.

        Args:
            loss: Loss tensor.
        """

    @abstractmethod
    def clip_gradients(self, optimizer: cstorch.optim.Optimizer):
        """Clips the gradients before the optimization step.

        Args:
            optimizer: The optimizer to step.
        """

    @abstractmethod
    def optimizer_step(self, optimizer: cstorch.optim.Optimizer):
        """Performs the optimization step.

        Args:
            optimizer: The optimizer to step.
        """


class MixedPrecision(Precision):
    """
    Callback class that handles mixed precision training.
    """

    def __init__(
        self,
        enabled: bool = True,
        fp16_type: Literal["float16", "bfloat16", "cbfloat16"] = "float16",
        precision_opt_level: Optional[int] = None,
        loss_scaling_factor: Union[float, Literal["dynamic"]] = 1.0,
        initial_loss_scale: Optional[float] = None,
        steps_per_increase: int = 2000,
        min_loss_scale: Optional[float] = None,
        max_loss_scale: Optional[float] = None,
        max_gradient_norm: Optional[float] = None,
        max_gradient_value: Optional[float] = None,
        log_loss_scale: bool = False,
    ):
        """
        Args:
            enabled: If True, enables mixed precision training.
            fp16_type: Half precision type. One of "float16", "bfloat16", "cbfloat16".
            precision_opt_level: Precision optimization level. If not None, sets the
                global precision optimization level.
            loss_scaling_factor: Initial loss scaling factor.
            initial_loss_scale: Initial loss scale.
            steps_per_increase: Number of steps before increasing the loss scale.
            min_loss_scale: Minimum loss scale.
            max_loss_scale: Maximum loss scale.
            max_gradient_norm: Maximum gradient norm for gradient clipping.
            max_gradient_value: Maximum gradient value for gradient clipping.
            log_loss_scale: If True, log the gradient scaler's loss scale.
        """
        self.scaler = None

        self.enabled = enabled
        self.fp16_type = fp16_type
        self.precision_opt_level = precision_opt_level
        self.loss_scaling_factor = loss_scaling_factor
        self.initial_loss_scale = initial_loss_scale
        self.steps_per_increase = steps_per_increase
        self.min_loss_scale = min_loss_scale
        self.max_loss_scale = max_loss_scale
        self.max_gradient_norm = max_gradient_norm
        self.max_gradient_value = max_gradient_value
        self.log_loss_scale = log_loss_scale

        if precision_opt_level is not None:
            cstorch.backends.csx.precision.optimization_level = (
                precision_opt_level
            )

    def setup(self, trainer):
        # pylint: disable=attribute-defined-outside-init
        self.backend = trainer.backend

        if not self.enabled:
            if self.backend.is_csx:
                warn(
                    "Mixed precision must be enabled for CSX. Setting enabled = True"
                )
                self.enabled = True
            else:
                return

        valid_dtypes = ["float16", "bfloat16", "cbfloat16"]
        if self.fp16_type not in valid_dtypes:
            raise ValueError(
                f"Invalid value for \"fp16_type\". Expected one of {valid_dtypes}. "
                f"Got: {self.fp16_type}."
            )

        if self.fp16_type == "cbfloat16":
            if not self.backend.is_csx:
                self.fp16_type = torch.bfloat16
                warn(
                    "cbfloat16 is only supported on CSX. Setting half dtype to bfloat16."
                )
            elif self.loss_scaling_factor != "dynamic":
                raise ValueError(
                    f"In order to use cbfloat16, dynamic loss scaling must be enabled. "
                    f"Otherwise, gradients might underflow/overflow in the middle of "
                    f"training and cause NaNs. Please set `loss_scaling_factor` to "
                    f"`dynamic` to use cbfloat16."
                )
        elif self.backend.is_cpu and self.fp16_type != "bfloat16":
            self.fp16_type = torch.bfloat16
            warn(
                "Mixed precision on CPU is only supported with bfloat16. "
                "Setting half dtype to bfloat16."
            )

        cstorch.amp.set_half_dtype(self.fp16_type)

        if self.fp16_type == "bfloat16":
            if self.loss_scaling_factor == "dynamic":
                trainer.logger.info(
                    f"No need to use DLS for loss when half dtype is bfloat16. "
                    f"Disabling gradient scaling."
                )
            self.scaler = None
        elif self.backend.is_csx:
            self.scaler = cstorch.amp.GradScaler(
                loss_scale=self.loss_scaling_factor,
                init_scale=self.initial_loss_scale,
                steps_per_increase=self.steps_per_increase,
                min_loss_scale=self.min_loss_scale,
                max_loss_scale=self.max_loss_scale,
            )
        else:
            if self.loss_scaling_factor == "dynamic":
                self.scaler = torch.cuda.amp.GradScaler(
                    init_scale=self.initial_loss_scale,
                    growth_interval=self.steps_per_increase,
                )
            else:
                self.scaler = torch.cuda.amp.GradScaler(
                    init_scale=self.loss_scaling_factor,
                    growth_interval=2**63 - 1,
                )

        max_gradient_norm = self.max_gradient_norm
        max_gradient_value = self.max_gradient_value

        if max_gradient_norm is not None and max_gradient_norm <= 0.0:
            raise ValueError(
                f"max_gradient_norm has to be a positive float. Got "
                f"{max_gradient_norm}"
            )
        if max_gradient_value is not None and max_gradient_value <= 0.0:
            raise ValueError(
                f"max_gradient_value has to be a positive float. Got "
                f"{max_gradient_value}"
            )
        if max_gradient_norm is not None and max_gradient_value is not None:
            raise ValueError(
                f"Gradients can be clipped by norm(={max_gradient_norm}) or by "
                f"value(={max_gradient_value}), but not both. "
                f"Do not set both `max_gradient_norm` and `max_gradient_value`."
            )

    def autocast_context_manager(self):
        if not self.enabled or self.backend.is_csx:
            return nullcontext()
        else:
            return cstorch.amp.autocast()

    def backward(self, loss):
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def on_before_optimizer_step(self, trainer, model, optimizer):
        """Unscales the gradients and performs gradient clipping."""
        if self.scaler:
            self.scaler.unscale_(optimizer)

    def clip_gradients(self, optimizer):
        params = (
            p
            for param_group in optimizer.param_groups
            for p in param_group["params"]
        )
        if self.max_gradient_norm is not None:
            torch.nn.utils.clip_grad_norm_(list(params), self.max_gradient_norm)
        elif self.max_gradient_value is not None:
            torch.nn.utils.clip_grad_value_(
                list(params), self.max_gradient_value
            )

    def optimizer_step(self, optimizer):
        if self.scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if self.scaler and self.log_loss_scale:
            trainer.log_metrics(loss_scale=self.scaler.get_scale())

    def on_save_checkpoint(self, trainer, state_dict):
        if self.scaler:
            state_dict["grad_scaler"] = self.scaler.state_dict()

    def on_load_checkpoint(self, trainer, state_dict):
        if self.scaler:
            # TODO: handle conversion between CUDA grad scaler and cstorch grad scaler
            if "grad_scaler" in state_dict:
                self.scaler.load_state_dict(state_dict["grad_scaler"])

                trainer.logger.info(
                    f"Gradient scaler state found in checkpoint and loaded successfully."
                )
            else:
                trainer.logger.info(
                    "Gradient scaler state not found in the checkpoint. "
                    "Using default initialized state."
                )
