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

"""Model Callback class."""

from typing import Callable, Union
from warnings import warn

import torch

import cerebras.pytorch as cstorch
from cerebras.modelzoo.trainer.callbacks import CoreCallback


class ModelCallback(CoreCallback):
    """Callback class that handles setting up and compiling the model."""

    def __init__(
        self,
        model: Union[Callable[[], torch.nn.Module], torch.nn.Module],
    ):
        """
        Args:
            model: The model to train. It must be one of the following:
                - If a callable is passed, it is assumed to be a function that
                  takes in no arguments returns a torch.nn.Module.

                - If a torch.nn.Module is passed, it is used as is.
        """
        self.model = model

    def setup(self, trainer):
        if callable(self.model) and not isinstance(self.model, torch.nn.Module):
            with trainer.backend.device:
                trainer.model = self.model()
        elif isinstance(self.model, torch.nn.Module):
            trainer.model = self.model
        else:
            raise ValueError(
                f"Expected model to be a torch.nn.Module or a callable that "
                f"returns a torch.nn.Module, but got {type(self.model)}."
            )

        trainer.compiled_model = cstorch.compile(trainer.model, trainer.backend)

    def on_train_start(self, trainer, model, train_dataloader, loop, loop_idx):
        model.train()

    def on_validate_start(self, trainer, model, val_dataloader, loop):
        model.eval()

    def on_before_backward(self, trainer, model, outputs):
        if "loss" not in outputs:
            raise ValueError(
                f"Expected the {model.__class__.__name__}'s forward call to "
                f"return a dictionary with a 'loss' key. "
                f"Got: {sorted(outputs.keys())}"
            )

    def on_save_checkpoint(self, trainer, state_dict):
        state_dict["model"] = trainer.model.state_dict()

    def on_load_checkpoint(self, trainer, state_dict):
        if "model" not in state_dict:
            warn(
                f"Checkpoint does not contain a model state dict. "
                f"Model state was not loaded"
            )
        else:
            # This check is required for backward compatibility with checkpoints
            # saved with older versions of ModelZoo (pre rel-2.0.0)
            # We check that the model state dict keys start with "model."
            # and if they don't, we load the state dict into the model's model
            if hasattr(trainer.model, "model") and not all(
                k.startswith("model.") for k in state_dict["model"].keys()
            ):
                trainer.model.model.load_state_dict(
                    state_dict["model"],
                    strict=not trainer.checkpoint.disable_strict_checkpoint_loading,
                )

            # This should be the case that is used for all checkpoints saved
            # post rel-2.0.0
            else:
                trainer.model.load_state_dict(
                    state_dict["model"],
                    strict=not trainer.checkpoint.disable_strict_checkpoint_loading,
                )

            trainer.logger.info(
                f"Model state found in checkpoint and loaded successfully."
            )
