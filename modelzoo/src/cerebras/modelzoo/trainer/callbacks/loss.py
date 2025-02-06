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

"""This module contains the CheckLoss callback."""

from math import prod

import torch

import cerebras.pytorch as cstorch
from cerebras.modelzoo.common.half_dtype import cb16_to_fp32
from cerebras.modelzoo.trainer.callbacks import Callback


class CheckLoss(Callback):
    """Callback class that checks for NaN or inf loss values.

    It also checks whether the model output contains a scalar loss value.
    """

    def on_after_forward(self, trainer, model, outputs, batch):
        if "loss" in outputs:
            loss = outputs["loss"]

            if not isinstance(loss, torch.Tensor):
                raise TypeError(
                    f"Expected loss to be a scalar torch.Tensor, "
                    f"but got {type(loss)} instead."
                )
            elif prod(loss.shape) > 1:
                raise TypeError(
                    f"Expected loss to be a scalar torch.Tensor, "
                    f"but got tensor with shape {loss.shape} instead."
                )

    @cstorch.step_closure
    def check_loss(self, loss: torch.Tensor):  # pylint: disable=no-self-use
        """Checks for NaN or inf loss values.

        Args:
            loss: Scalar loss tensor.
        """
        msg_postfix = (
            "This could potentially be due to selected hyperparameters "
            "such as the learning rate, batch size, etc. or it could due "
            "an internal error. Please try with different set of "
            "hyperparameters and contact Cerebras Support if the issue "
            "persists."
        )

        from cerebras.appliance.errors import ApplianceNanError

        if torch.isnan(loss).any().item():
            raise ApplianceNanError(f"NaN loss detected. {msg_postfix}")
        if torch.isinf(loss).any().item():
            raise ApplianceNanError(f"inf loss detected. {msg_postfix}")

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if "loss" in outputs:
            loss = cb16_to_fp32(outputs["loss"])
            self.check_loss(loss)

    def on_validate_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if "loss" in outputs:
            loss = cb16_to_fp32(outputs["loss"])
            self.check_loss(loss)
