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
Contains the TelemetryLogger class for providing an interface to 
log telemetry data to TelemetryClient.
"""

from contextlib import contextmanager

from cerebras.appliance.cluster.client import TelemetryClient
from cerebras.modelzoo.trainer.loggers import Logger


class TelemetryLogger(Logger):
    """Logger class that logs telemetry data to TelemetryClient."""

    def __init__(
        self, max_buffer_size: int = 128, max_buffer_seconds: int = 150
    ):
        """
        Args:
            max_buffer_size: Maximum number of metrics the TelemetryClient's
                buffer can hold before flushing. Defaults to 128.
            max_buffer_seconds: Maximum waiting time for TelemetryClient's
                buffer before flushing. Defaults to 150 seconds.
        """

        self.max_buffer_size = max_buffer_size
        self.max_buffer_seconds = max_buffer_seconds

        self.client = None

    def pre_setup(self, trainer):
        # TelemetryLogger only needs to log telemetry data, hence
        # we remove it from the set of default logger destinations.
        trainer.remove_logger_destination(self.name)

    @contextmanager
    def telemetry_context(self, backend):
        """Return a context manager for telemetry logging if the backend is CSX."""

        if not backend.is_csx:
            yield
            return

        try:
            self._client = TelemetryClient(
                backend.cluster.client,
                self.max_buffer_size,
                self.max_buffer_seconds,
            )
            with self._client:
                yield
        finally:
            self._client = None

    def on_enter_train(self, trainer, stack, train_dataloader, loop, loop_idx):
        stack.enter_context(self.telemetry_context(trainer.backend))

    def on_enter_validate(self, trainer, stack, val_dataloader, loop):
        stack.enter_context(self.telemetry_context(trainer.backend))

    def log_metrics(self, metrics, step):
        if hasattr(self, "_client") and self._client is not None:
            self._client.push(metrics, labels={"step": str(step)})
