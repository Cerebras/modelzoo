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

"""Contains the CS PyTorch Runner"""

import logging
import time

import torch

from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch import cbtorch, modes
from modelzoo.common.pytorch.perf_utils import save_perf
from modelzoo.common.pytorch.pytorch_base_cs_runner import PyTorchBaseCSRunner

COMPILE_MSG = (
    "Compiling the model and programming onto fabric. "
    "This may take a few minutes."
)


class PyTorchCSRunner(PyTorchBaseCSRunner):
    """Class for running PyTorch models on Cerebras hardware."""

    ##################################################################
    #                         Training Hooks                         #
    ##################################################################

    def on_train_start(self):
        if self._model.grad_scaler:
            self._scaler = self._model.grad_scaler

        cm.write_to_summary(
            self._writer,
            0,
            dict_to_write={"TensorboardStartTimestamp": time.time()},
        )
        cm.set_run_config(
            self._total_steps, self._checkpoint_steps, self._fetch_steps
        )

    def on_train_end(self, early_exit: bool):
        save_perf(self._perf_dir)

        if self._show_debug_metrics:
            cm.print_metrics_report()

        super().on_train_end(early_exit)

    def on_train_epoch_end(self, early_exit: bool):
        if early_exit:
            cm.mark_step()  # required to complete execution

    def optimizer_step(self):
        super().optimizer_step()

        if self._global_step == self._initial_step:
            logging.info(COMPILE_MSG)

    ##################################################################
    #                        Evaluation Hooks                        #
    ##################################################################

    def on_eval_start(self):
        cm.write_to_summary(
            self._writer,
            0,
            dict_to_write={"TensorboardStartTimestamp": time.time()},
        )
        cm.set_run_config(self._total_steps, 0, 1)

        logging.info(COMPILE_MSG)

    def on_eval_end(self, early_exit: bool):
        save_perf(self._perf_dir)

        if self._show_debug_metrics:
            cm.print_metrics_report()

        super().on_eval_end(early_exit)

    def on_eval_epoch_end(self, early_exit: bool):
        if early_exit:
            cm.mark_step()  # required to complete execution

    ##################################################################
    #                   Override Abstract Methods                    #
    ##################################################################

    def train(self, dataloader: torch.utils.data.DataLoader) -> None:
        dataloader = cbtorch.dataloader(dataloader)

        with cbtorch.Session(dataloader, modes.TRAIN):
            super().train(dataloader)

    def evaluate(self, dataloader: cbtorch.data.DataLoader):
        dataloader = cbtorch.dataloader(dataloader)

        with cbtorch.Session(dataloader, modes.EVAL):
            super().evaluate(dataloader)
