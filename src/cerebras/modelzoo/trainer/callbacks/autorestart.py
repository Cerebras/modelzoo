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

"""Implementation of AutoRestart core callback class."""

import os
from typing import Optional

from cerebras.modelzoo.trainer.callbacks import Callback, CoreCallback

TRAINER_STATE_FILENAME = "trainer_state_{step}.mdl"


class AutoRestart(CoreCallback):

    def __init__(
        self,
        max_num_restarts: Optional[int] = None,
        min_num_csx: int = 1,
    ):
        """
        Args:
            max_num_restarts: The max number of attempts at restarting, without progress,
                before the run is considered as failed.
            min_num_csx: The minimum number of CSX systems with which to perform a restart. If the
                user-specified `num_csx` is greater than the number of available systems in the
                cluster, the restart logic will continue to restart with fewer systems until this
                threshold is hit.
        """
        if max_num_restarts is not None and (
            not isinstance(max_num_restarts, int) or max_num_restarts < 0
        ):
            raise ValueError(
                f"AutoRestart `max_num_restarts` must be non-negative integer, got {max_num_restarts}"
            )

        if not isinstance(min_num_csx, int) or min_num_csx <= 0:
            raise ValueError(
                f"AutoRestart `min_num_csx` must be an integer greater than 0, got {min_num_csx}"
            )

        self.max_num_restarts = max_num_restarts
        self.min_num_csx = min_num_csx
        self.trainer_state_filename = TRAINER_STATE_FILENAME
        self.trainer_state_file = None

        self._prev_trainer_state_file = None

    def setup(self, trainer):
        if not self.enabled:
            return

        # Check that all callbacks implement on_save_trainer_state and on_load_trainer_state
        # when autorestart is enabled
        for callback in trainer.all_callbacks:
            if isinstance(callback, CoreCallback):
                continue

            method_names = {"on_save_trainer_state", "on_load_trainer_state"}
            if not_overriden := {
                name
                for name in method_names
                if getattr(type(callback), name) is getattr(Callback, name)
            }:
                method_names_log = '\n\t'.join(method_names)
                not_overriden_log = '\n\t'.join(not_overriden)

                raise ValueError(
                    f"Autorestart is enabled but the {type(callback).__name__} callback "
                    f"does not implement all methods for restartability. Expected all "
                    f"callbacks to implement the following methods (even if they are no-op): "
                    f"\n\t{method_names_log}\nBut the following methods were not overriden: "
                    f"\n\t{not_overriden_log}\n"
                )

    @property
    def enabled(self):
        return self.max_num_restarts is not None and self.max_num_restarts > 0

    def on_save_trainer_state(self, trainer, state_dict):
        self.trainer_state_file = (
            trainer.artifact_dir
            / self.trainer_state_filename.format(step=trainer.global_step)
        )

    def on_after_save_trainer_state(self, trainer, trainer_state_file):
        # Delete previously saved trainer state file (if any)
        if self._prev_trainer_state_file is not None and os.path.exists(
            self._prev_trainer_state_file
        ):
            os.remove(self._prev_trainer_state_file)

        self._prev_trainer_state_file = trainer_state_file
