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

"""Callbacks related to perf/debug flags in cstorch.backends."""

from copy import deepcopy

import cerebras.pytorch as cstorch
from cerebras.appliance.utils.debug_args import get_debug_args
from cerebras.modelzoo.trainer.callbacks import Callback

_MISSING = object()


class GlobalFlags(Callback):
    """Callback to set global perf/debug flags with no scoping.

    This has side effect on all runs. To scope to a given run, use scoped flags instead.
    """

    def __init__(self, **flags):
        """
        Args:
            flags: Dictionary of debug/performance flags to set
                The keys must be the full path to the flag after cstorch.backends,
                e.g. "csx.debug.debug_args".
        """
        self.flags = deepcopy(flags)

    def pre_setup(self, trainer):
        for k, v in self.flags.items():
            _set_flag(k, v)

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass


class _ScopedFlags(Callback):
    """
    Callback to set perf/debug flags in the global scope.
    """

    def __init__(self, **flags):
        """
        Args:
            flags: Dictionary of debug/performance flags to set
                The keys must be the full path to the flag after cstorch.backends,
                e.g. "csx.debug.debug_args".
        """
        self.flags = deepcopy(flags)
        self.original_flags = {}

    @staticmethod
    def get_flag(flag):
        """
        Set a flag in the global perf/debug flags.
        """
        flag_class = cstorch.backends

        if "." in flag:
            scope, attr = flag.rsplit(".", 1)
            for s in scope.split("."):
                flag_class = getattr(flag_class, s)
        else:
            attr = flag

        return getattr(flag_class, attr, _MISSING)

    @staticmethod
    def set_flag(flag, value):
        """
        Set a flag in the global perf/debug flags.
        """
        flag_class = cstorch.backends

        if "." in flag:
            scope, attr = flag.rsplit(".", 1)
            for s in scope.split("."):
                flag_class = getattr(flag_class, s)
        else:
            attr = flag

        if value is _MISSING:
            delattr(flag_class, attr)
        else:
            setattr(flag_class, attr, value)

    def _set_all_flags(self):
        """Set all the flags passed in the constructor."""
        if self.original_flags:
            return

        for k, v in self.flags.items():
            # Save the original flag to be able to restore it later
            self.original_flags[k] = _get_flag(k)
            # Set the new flag value
            _set_flag(k, v)

    def _restore_all_flags(self):
        """Restore all the flags to their original values."""
        for k, v in self.original_flags.items():
            # Restore the original flag value
            _set_flag(k, v)

        self.original_flags.clear()

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass


class ScopedTrainFlags(_ScopedFlags):
    """
    Callback to set global perf/debug flags within the training scope.

    The overwritten flags are restored after training is complete
    """

    def on_train_start(self, trainer, model, train_dataloader, loop, loop_idx):
        self._set_all_flags()

    def on_train_end(self, trainer, model, loop, loop_idx):
        self._restore_all_flags()


class ScopedValidateFlags(_ScopedFlags):
    """
    Callback to set global perf/debug flags within the validation scope.

    The overwritten flags are restored after validation is complete
    """

    def on_validate_start(self, trainer, model, val_dataloader, loop):
        self._set_all_flags()

    def on_validate_end(self, trainer, model, loop):
        self._restore_all_flags()


class DebugArgsPath(Callback):
    """
    Callback to load debug args from a file.
    """

    def __init__(self, debug_args_path: str):
        """
        Args:
            debug_args_path: Path to the debug args file.
        """
        self.debug_args_path = debug_args_path

    def setup(self, trainer):
        debug_args = get_debug_args(self.debug_args_path)
        cstorch.backends.csx.debug.debug_args.MergeFrom(debug_args)

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass


def _get_flag(flag):
    """Set a flag in the global perf/debug flags."""
    flag_class = cstorch.backends

    if "." in flag:
        scope, attr = flag.rsplit(".", 1)
        for s in scope.split("."):
            flag_class = getattr(flag_class, s)
    else:
        attr = flag

    return getattr(flag_class, attr, _MISSING)


def _set_flag(flag, value):
    """Set a flag in the global perf/debug flags."""
    flag_class = cstorch.backends

    if "." in flag:
        scope, attr = flag.rsplit(".", 1)
        for s in scope.split("."):
            flag_class = getattr(flag_class, s)
    else:
        attr = flag

    if value is _MISSING:
        delattr(flag_class, attr)
    else:
        setattr(flag_class, attr, value)
