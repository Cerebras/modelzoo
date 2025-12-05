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
from cerebras.modelzoo.trainer.callbacks import Callback, CoreCallback


class LossAccumulationCallback(CoreCallback):
    """Callback class that accumulates loss values for training and validation.

    This callback will log the loss metric at the end of each training batch and
    validation batch. It will also log the average validation loss at the end of
    each validation epoch.
    """

    def __init__(self):
        self.accum_loss = 0
        self.total_eval_steps = 0
        self.avg_eval_loss = None

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if "loss" in outputs:
            self.accumulate_train_loss(trainer, cb16_to_fp32(outputs["loss"]))

    @cstorch.step_closure
    def accumulate_train_loss(self, trainer, loss):
        """Accumulates training loss and logs the loss metric."""
        if trainer.should_run_optimizer_step:
            loss = self.accum_loss + loss.item()
            self.accum_loss = 0

            trainer.log_metrics(loss=loss)
        else:
            # accumulate loss for gradient accumulation
            self.accum_loss += loss.item()

    def on_validate_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if "loss" in outputs:
            self.accumulate_val_loss(trainer, cb16_to_fp32(outputs["loss"]))

    def on_validate_end(self, trainer, model, loop):
        if self.avg_eval_loss is not None:
            trainer.logger.info(f"Avg Eval Loss: {self.avg_eval_loss}")
            self.avg_eval_loss = None

    @cstorch.step_closure
    def accumulate_val_loss(self, trainer, loss):
        """Accumulates validation loss and logs the average validation loss."""
        self.accum_loss += loss.item()
        self.total_eval_steps += 1

        # Need to log the intermediate eval loss differently
        # from the final loss
        trainer.log_metrics(val_loss=loss.item())

        if trainer.is_final_iteration:
            self.avg_eval_loss = self.accum_loss / self.total_eval_steps

            trainer.log_metrics(loss=self.avg_eval_loss)

            self.accum_loss = 0
            self.total_eval_steps = 0


class CheckLoss(Callback):
    """Callback class that checks for NaN or inf loss values.

    It also checks whether the model output contains a scalar loss value.
    """

    def on_train_step_end(self, trainer, model, outputs, batch):
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

            self.check_loss(cb16_to_fp32(outputs["loss"]))

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

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass
