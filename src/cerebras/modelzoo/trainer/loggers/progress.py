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

"""Contains the Logging class that handles setting up and logging to the standard Python logger."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, List

import cerebras.pytorch as cstorch
from cerebras.modelzoo.trainer.callbacks.profiler import TimeRemaining
from cerebras.modelzoo.trainer.loggers import Logger
from cerebras.pytorch.utils.data.utils import infer_batch_size

if TYPE_CHECKING:
    from ..trainer import Trainer


class ProgressLogger(Logger):
    """
    Callback that handles setting up and logging to the standard Python logger.
    """

    def __init__(self):
        """Sets up the rate tracker and total samples tracker."""

        self.metrics = defaultdict(dict)
        # Keep track of the total samples processed
        # across all stages
        self.total_samples = cstorch.utils.tracker.RateTracker()

    def log_metrics(self, metrics, step):
        self.metrics[step].update(
            {k.split("/")[-1]: v for k, v in metrics.items()}
        )

    @staticmethod
    def format_rate(rate: float) -> str:
        """Format the rate for logging.

        Use two significant digits if the rate is less than 1.0, otherwise
        use two decimal places.

        Args:
            rate: Rate to format.
        """
        if rate < 1.0:
            return f"{rate:.3g} samples/sec"
        return f"{rate:.2f} samples/sec"

    @staticmethod
    def format_time_remaining(time_remaining: TimeRemaining) -> str:
        equals_or_gt = (
            ">"
            if isinstance(time_remaining, TimeRemaining)
            and time_remaining.stages_missing_duration
            else "="
        )
        return f"{equals_or_gt}{time_remaining}"

    def format_metrics(self, step: int) -> List[str]:
        """Returns the formatted metrics to append to the progress message."""
        formatted = []

        metrics = self.metrics.pop(step, None)
        if metrics is None:
            return formatted

        if (loss := metrics.get("loss")) is not None:
            formatted.append(f"Loss={loss:.5f}")
        elif (loss := metrics.get("val_loss")) is not None:
            formatted.append(f"Loss={loss:.5f}")

        if (rate := metrics.get("local_samples_per_sec")) is not None:
            formatted.append(f"Rate={self.format_rate(rate)}")
        if (global_rate := metrics.get("avg_samples_per_sec")) is not None:
            formatted.append(f"GlobalRate={self.format_rate(global_rate)}")

        if (t := metrics.get("loop_time_remaining")) is not None:
            formatted.append(
                f"LoopTimeRemaining{self.format_time_remaining(t)}"
            )
        if (t := metrics.get("time_remaining")) is not None:
            formatted.append(f"TimeRemaining{self.format_time_remaining(t)}")

        return formatted

    def on_fit_start(self, trainer, train_dataloader, val_dataloader, loop):
        self.total_samples.reset()

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        batch_size = infer_batch_size(batch)
        if batch_size:
            self.total_samples.add(batch_size)
        self.print_training_progress(trainer)

    @cstorch.step_closure
    def print_training_progress(self, trainer: Trainer):
        """Print training progress and log metrics."""
        # Only log progress if we are running the optimizer step
        if trainer.should_run_optimizer_step and trainer.is_log_step:
            progress_msg = [
                f"| Train Device={trainer.backend.device}",
                f"Step={trainer.global_step}",
                *self.format_metrics(trainer.global_step),
            ]
            trainer.logger.info(", ".join(progress_msg))

    def on_train_end(self, trainer, model, loop, loop_idx):
        trainer.logger.info("Training completed successfully!")

    def on_validate_batch_end(self, trainer, model, outputs, batch, batch_idx):
        self.print_validation_progress(trainer, batch_idx)

    @cstorch.step_closure
    def print_validation_progress(
        self, trainer: Trainer, batch_idx: int, prefix: str = "Eval"
    ):
        """Print validation progress and log metrics."""
        if trainer.is_log_step:
            progress_msg = [
                f"| {prefix} Device={trainer.backend.device}",
                f"GlobalStep={trainer.global_step}",
                f"Batch={batch_idx + 1}",  # batch_idx is 0-indexed
                *self.format_metrics(trainer.global_step),
            ]
            trainer.logger.info(", ".join(progress_msg))

    def on_validate_end(self, trainer, model, loop):
        trainer.logger.info("Evaluation completed successfully!")

    def on_fit_end(self, trainer, loop):
        # pylint: disable=protected-access
        if trainer.backend._impl.is_e2e_execution:
            trainer.logger.info(
                f"Processed {int(self.total_samples.total_count)} training sample(s) "
                f"in {self.total_samples.elapsed_seconds()} seconds."
            )

    def on_fit_exception(self, trainer, exception):
        self.on_fit_end(trainer, None)
