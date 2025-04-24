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

"""Contains utilities for summarizing scalars and tensors."""

from contextlib import contextmanager
from warnings import warn

import torch

from cerebras.modelzoo.trainer.callbacks import (
    Callback,
    register_global_callback,
)


class _LogSummaries(Callback):
    """Callback class caches the trainer instance for summarizing through methods below."""

    def __init__(self):
        self._trainer_stack = []

    @property
    def trainer(self):
        """Return the current trainer instance."""
        if self._trainer_stack:
            return self._trainer_stack[-1]
        return None

    def on_enter_train(
        self,
        trainer,
        stack,
        train_dataloader,
        loop,
        loop_idx,
    ):
        stack.enter_context(self._cache_trainer(trainer))

    def on_enter_validate(self, trainer, stack, val_dataloader, loop):
        stack.enter_context(self._cache_trainer(trainer))

    def on_enter_validate_all(
        self, trainer, stack, val_dataloaders, loop, ckpt_paths
    ):
        stack.enter_context(self._cache_trainer(trainer))

    @contextmanager
    def _cache_trainer(self, trainer):
        try:
            self._trainer_stack.append(trainer)
            yield
        finally:
            self._trainer_stack.pop()

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass


_GLOBAL_SUMMARIES = _LogSummaries()
register_global_callback(_GLOBAL_SUMMARIES)


def summarize_scalar(name: str, value: torch.Tensor):
    """Log scalar values to the trainer loggers.

    Args:
        name: The name of the metric.
        value: Scalar value of the metric to log.
    """
    if not _GLOBAL_SUMMARIES.trainer:
        raise RuntimeError(
            "\"summarize_scalar()\" must only be called in the context of a run using the "
            "\"Trainer\" class."
        )

    if isinstance(value, torch.Tensor) and (
        value.numel() > 1 or value.ndim > 0
    ):
        raise ValueError(
            f"Expected a scalar tensor for metric '{name}', "
            f"but got tensor with shape {value.shape}.\n"
            f"To summarize tensors, use 'summarize_tensor()' instead, e.g.\n\n"
            f"\tfrom cerebras.modelzoo.trainer import summarize_tensor\n"
            f"\tsummarize_tensor('{name}', tensor)\n\n"
        )

    _GLOBAL_SUMMARIES.trainer.log_metrics(**{name: value})


def summarize_tensor(name: str, value: torch.Tensor):
    """Log tensor values to the trainer loggers.

    Args:
        name: The name of the metric.
        value: Tensor value of the metric to log.
    """
    if not _GLOBAL_SUMMARIES.trainer:
        raise RuntimeError(
            "\"summarize_scalar()\" must only be called in the context of a run using the "
            "\"Trainer\" class."
        )

    if not isinstance(value, torch.Tensor):
        raise ValueError(
            f"Expected a tensor for metric '{name}', but got '{type(value)}'."
        )

    if value.numel() == 1 and value.ndim == 0:
        warn(
            f"summarize_tensor got a scalar tensor for {name}. "
            f"The scalar tensor will be reshaped to a 1D tensor so that "
            f"it can be logged as a tensor. If you want to log the scalar "
            f"value as a scalar, use 'summarize_scalar()' instead."
        )
        value = value.reshape(-1)

    _GLOBAL_SUMMARIES.trainer.log_metrics(**{name: value})
