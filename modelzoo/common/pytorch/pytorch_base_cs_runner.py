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

"""Module containing the Base PyTorch CS Runner"""

import abc
import logging
import os
import warnings

import torch

from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch import cbtorch, modes, perf_utils
from modelzoo.common.pytorch.pytorch_base_runner import PyTorchBaseRunner

COMPILE_MSG = (
    "Compiling the model and programming onto fabric. "
    "This may take a few minutes."
)


class PyTorchBaseCSRunner(PyTorchBaseRunner, metaclass=abc.ABCMeta):
    """Base Class containing common elements between CS runner and CS compiler"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._device = cm.device()

        # batch size to be inferred on first iteration
        self._batch_size = None

        # Number of replicas to use for multireplica
        # 1 replica meaning no multireplica and -1 meaning
        # choose optimal number of replicas
        num_replicas = self._runconfig.get("num_replicas", 1)
        if num_replicas == 1 and self._runconfig.get("multireplica"):
            num_replicas = -1

        if num_replicas != 1 and not self._model.allow_multireplica:
            warnings.warn(
                "Multireplica not allowed for current model. Disabling multireplica."
            )
        else:
            cm.set_target_num_replicas(num_replicas)

    ##################################################################
    #                   Override Abstract Methods                    #
    ##################################################################

    def on_train_batch_start(self, data):
        cm.set_state_names(self._model.get_state())
        return data

    def on_eval_batch_start(self, data):
        cm.set_state_names(self._model.get_state())
        return data

    def train_and_eval(
        self,
        train_data_loader: torch.utils.data.DataLoader,
        eval_data_loader: torch.utils.data.DataLoader,
    ):  # pylint: disable=arguments-renamed
        raise RuntimeError(
            "Training with Eval on CS is not currently supported."
        )

    def on_train_end(self, early_exit: bool):
        rate_tracker = cbtorch.state().rate_tracker
        if rate_tracker is None:
            logging.info(f"Training Complete.")
            return

        pd = perf_utils.collect_perf_data(rate_tracker)
        logging.info(
            f"Training Complete. Completed {int(pd.total_samples)} sample(s)"
            f" in {pd.total_time} seconds."
        )

    def on_eval_end(self, early_exit: bool):
        rate_tracker = cbtorch.state().rate_tracker
        if rate_tracker is None:
            logging.info(f"Evaluation Complete.")
            return

        pd = perf_utils.collect_perf_data(rate_tracker)
        logging.info(
            f"Evaluation Complete. Completed {int(pd.total_samples)} sample(s)"
            f" in {pd.total_time} seconds."
        )

    @property
    def _perf_dir(self) -> str:
        """Return the directory to use for saving performance metrics."""
        return os.path.join(self._model_dir, "performance")

    @property
    def world_global_step(self):
        return self._global_step * cm.num_receivers()

    def _increment_global_step(self):
        self._global_step += cm.get_run_step() - self._run_step

    def _save_stream(self, data_loader, mode: str):
        if mode == modes.TRAIN_AND_EVAL:
            train_data_loader, eval_data_loader = data_loader
            self._save_stream(train_data_loader, modes.TRAIN)
            self._save_stream(eval_data_loader, modes.EVAL)
            return

        assert isinstance(
            data_loader, cbtorch.utils.data.dataloader.DataLoader
        ), f"DataLoader type: {type(data_loader)}"
        # Use non parallel loader to save stream
        # pylint: disable=protected-access
        super()._save_stream(data_loader._loader, mode)

    @cm.step_closure
    def _write_log(self, loss, global_step):
        """Print training update to screen.

        Args:
            loss: The loss tensor.
        """
        cm.print_update(
            self._device,
            global_step,
            loss.item(),
            self._active_mode,
            summary_writer=self._writer,
        )

    @cm.step_closure
    def _save_checkpoint(self, step):  # pylint: disable=arguments-differ
        """Conditionally add a step closure to save checkpoint."""
        file_name = os.path.join(self._model_dir, f"checkpoint_{step}.mdl")

        state_dict = self._model.get_state()
        state_dict["global_step"] = state_dict.get("global_step", step)

        def post_transfer_callback(state_dict):
            if "optimizer" in state_dict:
                state_dict[
                    "optimizer"
                ] = self._optimizer.convert_state_dict_for_checkpoint(
                    state_dict["optimizer"]
                )
            return state_dict

        cm.save(
            state_dict,
            file_name,
            master_only=True,
            post_transfer_callback=post_transfer_callback,
        )
        logging.info(f"Saved checkpoint at global step: {step}")
