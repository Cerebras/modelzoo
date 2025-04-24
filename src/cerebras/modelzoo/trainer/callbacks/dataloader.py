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

"""
This module contains the implementation of the DataLoader callback class.
"""

from contextlib import contextmanager

from cerebras.modelzoo.trainer.callbacks import Callback, CoreCallback
from cerebras.pytorch.utils.nest import visit_torch_tensors


class DataLoaderCallback(CoreCallback):
    """
    Callback class that handles saving and loading dataloader state
    to the checkpoint.
    """

    def __init__(self):
        """
        Attributes:
            dataloader: The training dataloader object to save to the checkpoint.
        """

        self.dataloader = None

    def on_enter_fit(
        self, trainer, stack, train_dataloader, val_dataloader, loop
    ):
        @contextmanager
        def store_dataloader():
            # pylint: disable=attribute-defined-outside-init
            try:
                self.dataloader = train_dataloader
                yield
            finally:
                self.dataloader = None

        stack.enter_context(store_dataloader())

    def on_save_checkpoint(self, trainer, state_dict):
        if self.dataloader is not None and self.dataloader.is_restartable:
            state_dict["dataloader"] = self.dataloader.state_dict()

    def on_load_checkpoint(self, trainer, state_dict):
        if self.dataloader is not None and self.dataloader.is_restartable:
            if "dataloader" in state_dict:
                self.dataloader.load_state_dict(
                    state_dict["dataloader"],
                    strict=not trainer.checkpoint.disable_strict_checkpoint_loading,
                )

                trainer.logger.info(
                    f"Dataloader state found in checkpoint and loaded successfully."
                )
            else:
                trainer.logger.info(
                    "Dataloader state not found in the checkpoint. "
                    "DataLoaders will yield samples from the beginning."
                )


class LogInputSummaries(Callback):
    """Callback class that logs the batches produced by the dataloader."""

    def log_input_summaries(  # pylint: disable=no-self-use
        self, trainer, batch
    ):
        """Logs the input summaries."""
        trainer.log_metrics(
            **{
                ".".join(map(str, scope)): tensor
                for scope, tensor in visit_torch_tensors(batch, scope=["input"])
            }
        )

    def on_before_forward(self, trainer, model, batch, args, kwargs):
        self.log_input_summaries(trainer, batch)

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass
