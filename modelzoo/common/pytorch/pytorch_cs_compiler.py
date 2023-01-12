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

"""Contains the CS Compiler"""

# pylint: disable=attribute-defined-outside-init

import logging
from typing import Tuple

import torch

from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch import cbtorch
from modelzoo.common.pytorch.loss_utils import extract_loss
from modelzoo.common.pytorch.pytorch_base_cs_runner import PyTorchBaseCSRunner

COMPILE_ONLY_MSG = "Compiling the model. This may take a few minutes."


class PyTorchCSCompiler(PyTorchBaseCSRunner):
    """Class for compiling PyTorch models for Cerebras hardware."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # irrelevant config options for compile only
        self._save_initial_checkpoint = False
        self._save_losses = False

        self._validate_only = self._runconfig.get("validate_only", False)

    ##################################################################
    #                         Training Hooks                         #
    ##################################################################

    def on_train_start(self):
        if self._model.grad_scaler:
            self._scaler = self._model.grad_scaler
        cm.set_run_config(1, 0, 0)

    def on_train_epoch_end(self, early_exit: bool):
        logging.info(COMPILE_ONLY_MSG)

        assert cbtorch.compile(
            fabric_config_file=self._fabric_config_file,
            validate_only=self._validate_only,
        ), "Compile failed"

        logging.info("Compile for training completed successfully!")

    def on_train_batch_end(self, *args, **kwargs):
        pass  # noop

    ##################################################################
    #                        Evaluation Hooks                        #
    ##################################################################

    def on_eval_start(self):
        cm.set_run_config(1, 0, 0)

    def eval_forward(self, data):
        outputs = super().eval_forward(data)

        # Need to track eval model outputs to compile
        loss = extract_loss(outputs)
        cbtorch.state().track_object({"loss": loss})
        cbtorch.state().track_object(outputs)

        return outputs

    def on_eval_epoch_end(self, early_exit: bool):
        logging.info(COMPILE_ONLY_MSG)

        assert cbtorch.compile(
            fabric_config_file=self._fabric_config_file,
            validate_only=self._validate_only,
        ), "Compile Failed"

        logging.info("Compile for evaluation completed successfully!")

    def on_eval_batch_end(self, *args, **kwargs):
        pass  # noop

    def compute_eval_metrics(self):
        pass  # noop

    ##################################################################
    #                   Override Abstract Methods                    #
    ##################################################################

    def train(
        self, dataloader: torch.utils.data.DataLoader
    ):  # pylint: disable=arguments-renamed
        dataloader = cbtorch.dataloader(dataloader)
        super().train(dataloader)

    def evaluate(
        self, dataloader: cbtorch.data.DataLoader
    ):  # pylint: disable=arguments-renamed
        dataloader = cbtorch.dataloader(dataloader)
        super().evaluate(dataloader)

    def _should_stop(self, epoch_step: int, mode: str) -> Tuple[bool, bool]:
        return True, True

    def _configure_run_steps(self, dataloader, mode: str):
        self._num_epochs = 1
        self._total_steps = 1
        self._checkpoint_steps = 0
        self._fetch_steps = 0

    def _increment_global_step(self):
        self._global_step += 1

    def _write_log(self, loss, global_step):
        pass  # noop

    def _save_checkpoint(self, *args, **kwargs):
        # Should never reach here
        raise RuntimeError("Should not be saving checkpoint in compile only")
