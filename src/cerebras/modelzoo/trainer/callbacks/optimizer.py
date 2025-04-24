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

"""Module containing the OptimizerCallback class."""

from collections.abc import Iterable
from typing import Callable
from typing import Iterable as IterableType
from typing import Union

import torch

import cerebras.pytorch as cstorch
from cerebras.modelzoo.trainer.callbacks import Callback, CoreCallback


class OptimizerCallback(CoreCallback):
    """Callback to setup the optimizer for the Trainer."""

    def __init__(
        self,
        optimizer: Union[
            cstorch.optim.Optimizer,
            Callable[[torch.nn.Module], cstorch.optim.Optimizer],
            None,
        ] = None,
    ):
        """
        Args:
            optimizer: Optimizer to be used for training. It can be a
                an instance of ``cstorch.optim.Optimizer`` or a callable
                that takes a ``torch.nn.Module`` as input and returns an
                instance of ``cstorch.optim.Optimizer``.
                If None, the optimizer will not be set up by this callback.
        """
        self.optimizer = optimizer

    def setup(self, trainer):
        if self.optimizer is None:
            trainer.optimizer = None
        elif isinstance(self.optimizer, cstorch.optim.Optimizer):
            trainer.optimizer = self.optimizer
        else:
            trainer.optimizer = self.optimizer(trainer.model)

    def on_fit_start(self, trainer, train_dataloader, val_dataloader, loop):
        if trainer.optimizer is None:
            raise RuntimeError(
                "Optimizer is not defined. Please provide an optimizer "
                "to the Trainer in order to run fit."
            )

    def on_save_checkpoint(self, trainer, state_dict):
        if trainer.optimizer:
            state_dict["optimizer"] = trainer.optimizer.state_dict()

    def on_load_checkpoint(self, trainer, state_dict):
        if trainer.optimizer:
            if "optimizer" in state_dict:
                trainer.optimizer.load_state_dict(state_dict["optimizer"])

                trainer.logger.info(
                    f"Optimizer state found in checkpoint and loaded successfully."
                )
            else:
                trainer.logger.info(
                    "optimizer state not found in the checkpoint. "
                    "Using default preinitialized state."
                )


class LogOptimizerParamGroup(Callback):
    """Logs specific param group keys the optimizer used in the most recent step."""

    def __init__(self, keys: Union[str, IterableType[str]]):
        """
        Args:
            keys: A string or an iterable of strings representing the keys in
                the param group to log.
        """
        if isinstance(keys, str):
            self.keys = [keys]
        elif isinstance(keys, Iterable):
            for key in keys:
                if not isinstance(key, str):
                    raise ValueError(
                        f"Invalid value for key in `keys`. Expected a string, got {type(key)}"
                    )
            self.keys = keys
        else:
            raise ValueError(
                f"Invalid value for `keys`. Expected a string or Iterable, got {type(keys)}"
            )

    def setup(self, trainer):
        if trainer.optimizer is None:
            return
        for key in self.keys:
            for param_group in trainer.optimizer.param_groups:
                if key not in param_group:
                    raise ValueError(
                        f"Key {key} not found in optimizer param_groups."
                    )

    def on_after_optimizer_step(self, trainer, model, optimizer):
        for key in self.keys:
            trainer.log_metrics(
                **{
                    f"{key}/{group}": val[key]
                    for group, val in enumerate(optimizer.param_groups)
                }
            )

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass
