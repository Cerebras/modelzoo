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
from collections import UserDict
from contextlib import nullcontext
from datetime import datetime
from typing import Optional, Union

import torch
from torch.cuda.amp import GradScaler

from modelzoo.common.pytorch.pytorch_base_runner import PyTorchBaseRunner
from modelzoo.common.pytorch.sparsity.finalizer import finalize_cs2_sparsity


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

        super().__init__(model=model, params=params)

        if self._mixed_precision:
            use_bfloat16 = params["model"].get("use_bfloat16", False)
            dev_type = self._device.type if self._device is not None else "cuda"

            if dev_type == "cpu" and not use_bfloat16:
                raise ValueError(
                    "Mixed precision on CPU is only supported with bfloat16. "
                    "Please set use_bfloat16 to True in the model config."
                )

            dtype = torch.bfloat16 if use_bfloat16 else torch.float16

            self._autocast_ctx = torch.autocast(dev_type, dtype=dtype)
        else:
            self._autocast_ctx = nullcontext()

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
        with self._autocast_ctx:
            # Normalize loss to account for gradient accumulation
            return super().train_forward(data) / self._grad_accum_steps

    def eval_forward(self, data):
        with self._autocast_ctx:
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

    def _maybe_load_checkpoint(self, checkpoint_path: Optional[str], mode: str):
        state_dict = super()._maybe_load_checkpoint(checkpoint_path, mode)
        if state_dict:
            if self._params.get("sparsity", {}).get("type") == "sideband":
                # CS2 training sideband sparsity represents pruned weights with
                # NaN. Set those all to zero after loading a checkpoint.
                logging.warning(
                    "FINALIZING SPARSITY. ALL FUTURE TRAINING WILL BE DENSE."
                )
                finalize_cs2_sparsity(self._model.model, self._model.optimizer)
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

        self.on_checkpoint_saved(file_name, step)

    def _to_device(
        self, data: Union[dict, UserDict, list, tuple], non_blocking=False
    ):
        device_data = None
        if isinstance(data, (dict, UserDict)):
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
