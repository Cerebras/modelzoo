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

"""Callback to implement early stopping in the Trainer."""

from typing import OrderedDict

from torch.utils import hooks

from cerebras.modelzoo.trainer.callbacks import CoreCallback


class EarlyExit(CoreCallback):

    def __init__(self):
        self._early_exit_fns = OrderedDict()
        self._should_exit = False
        self._cleanup_registered = False

    def reset_exit_fns(self):
        """Clear the early exit functions dict and reset exit flag."""
        self._should_exit = False
        self._early_exit_fns.clear()

    def register(self, fn, *args, **kwargs) -> hooks.RemovableHandle:
        """Register an early exit function and return its handle."""
        if not callable(fn):
            raise TypeError(
                f"Early exit function must be callable, got {type(fn)}"
            )

        handle = hooks.RemovableHandle(self._early_exit_fns)

        self._early_exit_fns[handle.id] = (fn, args, kwargs)

        return handle

    @property
    def should_exit(self) -> bool:
        """Check if any of the early exit functions returns True."""
        if not self._should_exit:
            self._should_exit = any(
                fn(*args, **kwargs)
                for fn, args, kwargs in self._early_exit_fns.values()
            )
        return self._should_exit

    # Register resets for the early exit functions
    def on_enter_fit(
        self, trainer, stack, train_dataloader, val_dataloader, loop
    ):
        if not self._cleanup_registered:
            stack.callback(self.reset_exit_fns)
            self._cleanup_registered = True

    def on_enter_validate(self, trainer, stack, val_dataloader, loop):
        if not self._cleanup_registered:
            stack.callback(self.reset_exit_fns)
            self._cleanup_registered = True

    def on_enter_validate_all(
        self, trainer, stack, val_dataloaders, loop, ckpt_paths
    ):
        if not self._cleanup_registered:
            stack.callback(self.reset_exit_fns)
            self._cleanup_registered = True
