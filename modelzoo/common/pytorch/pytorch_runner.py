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

import logging
import os
import warnings
from datetime import datetime
from typing import Optional, Union

import torch
from torch.cuda.amp import GradScaler, autocast

from modelzoo.common.pytorch.pytorch_base_runner import PyTorchBaseRunner


class PyTorchRunner(PyTorchBaseRunner):
    """Class for running PyTorch models on CPU/GPU."""

    def __init__(self, device, model, params):
        self._device = device
        self._scaler = None
        self._loss_scaling_factor = params["optimizer"].get(
            "loss_scaling_factor", 1.0
        )
        self._initial_loss_scale = params["optimizer"].get(
            "initial_loss_scale", 2 ** 15
        )
        self._steps_per_increase = params["optimizer"].get(
            "steps_per_increase", 2000
        )

        if (
            params["model"]["mixed_precision"]
            and device is not None
            and device.type != "cuda"
        ):
            warnings.warn(
                "Mixed-Precision training is only supported on GPU... "
                "Autocast has no effect!"
            )
        self.use_bfloat16 = params["model"].get("use_bfloat16", False)

        super().__init__(model=model, params=params)

    ##################################################################
    #                         Training Hooks                         #
    ##################################################################

    def on_train_start(self):
        if (
            self._device.type == "cuda"
            and self._mixed_precision
            and self._loss_scaling_factor != 1.0
        ):
            if self._loss_scaling_factor == "dynamic":
                self._scaler = GradScaler(
                    init_scale=self._initial_loss_scale,
                    growth_interval=self._steps_per_increase,
                )
            else:
                # Emulating static loss scaling in PyTorch
                self._scaler = GradScaler(
                    init_scale=self._loss_scaling_factor,
                    growth_interval=2 ** 63 - 1,
                )

    def on_train_end(self, early_exit: bool):
        logging.info("Training Completed Successfully!")

    def on_train_batch_start(self, data):
        return self._to_device(data)

    def train_forward(self, data):

        with autocast(
            dtype=torch.bfloat16 if self.use_bfloat16 else torch.float16,
            enabled=self._mixed_precision,
        ):
            # Normalize loss to account for gradient accumulation
            return super().train_forward(data) / self._grad_accum_steps

    def eval_forward(self, data):

        with autocast(
            dtype=torch.bfloat16 if self.use_bfloat16 else torch.float16,
            enabled=self._mixed_precision,
        ):
            return super().eval_forward(data)

    ##################################################################
    #                        Evaluation Hooks                        #
    ##################################################################

    def on_eval_end(self, early_exit: bool):
        logging.info("Evaluation Completed Successfully!")

    def on_eval_batch_start(self, data):
        return self._to_device(data)

    ##################################################################
    #                   Override Abstract Methods                    #
    ##################################################################

    def _increment_global_step(self):
        self._global_step += 1

    def _write_log(self, loss, step):
        update_data = [
            'Device={}'.format(self._device.type),
            'Step={}'.format(step),
            'Loss={:.5f}'.format(loss),
            'Time={}'.format(datetime.now().strftime('%H:%M:%S')),
        ]
        logging.info(
            f"| {self._active_mode.title()} "
            f"{', '.join(item for item in update_data if item)}"
        )

        if self._scaler:
            self._writer.add_scalar(
                "loss_scale", self._scaler.get_scale(), step
            )
        if self._lr_scheduler:
            # self._lr_scheduler.get_last_lr() return a list of LRs for
            # different param groups of the optimizer. Here, we are assuming
            # that we only have one param group in the optimizer
            self._writer.add_scalar(
                "lr", self._lr_scheduler.get_last_lr()[0], step
            )

    def _maybe_load_checkpoint(self, checkpoint_path: Optional[str], mode: str):
        state_dict = super()._maybe_load_checkpoint(checkpoint_path, mode)
        if state_dict:
            scaler_state = state_dict.get("scaler")
            if self._scaler and scaler_state:
                self._scaler.load_state_dict(scaler_state)
        return state_dict

    def _save_checkpoint(self, step):
        logging.info(f"Saving checkpoint at step : {step}.")
        file_name = os.path.join(self._model_dir, f"checkpoint_{step}.mdl")
        model_state = self._model.get_state()
        model_state["global_step"] = step
        if self._scaler:
            model_state["scaler"] = self._scaler.state_dict()

        torch.save(model_state, file_name)

    def _to_device(self, data: Union[dict, list, tuple], non_blocking=False):
        device_data = None
        if isinstance(data, dict):
            device_data = {
                name: tensor.to(device=self._device, non_blocking=non_blocking)
                for name, tensor in data.items()
            }
        elif isinstance(data, (list, tuple)):
            device_data = type(data)(
                tensor.to(device=self._device, non_blocking=non_blocking)
                for tensor in data
            )
        else:
            raise RuntimeError(
                f"Data should be either a List or Dict of tensors. It was {type(data)} instead."
            )

        return device_data
