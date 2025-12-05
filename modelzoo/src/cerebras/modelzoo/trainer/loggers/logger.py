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

"""Logger interface for logging metrics to different backends."""

from abc import ABC, abstractmethod

from cerebras.modelzoo.trainer.callbacks import Callback


class Logger(Callback, ABC):
    """
    Base class for logging metrics to different backends.

    It is a simple subclass of Callback that features one additional
    abstract method `log` which needs to be implemented by the derived
    classes.
    """

    @property
    def name(self):
        return self.__class__.__name__

    def flush(self):
        """Manually flush the logger."""

    @abstractmethod
    def log_metrics(self, metrics: dict, step: int):
        """
        Logs the metrics to the backend at the given step.

        Args:
            metrics: Dictionary containing the metrics to be logged.
            step: The current step number.
        """

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass
