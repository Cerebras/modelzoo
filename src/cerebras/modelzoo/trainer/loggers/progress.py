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

from typing import TYPE_CHECKING, List, Optional

import torch

import cerebras.pytorch as cstorch
from cerebras.modelzoo.trainer.callbacks import RateProfiler
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

        self.rate_profiler = None
        # Keep track of the total samples processed
        # across all stages
        self.total_samples = cstorch.utils.tracker.RateTracker()

        self.accum_loss = None

    def setup(self, trainer):
        self.rate_profiler = trainer.get_callback(RateProfiler)

    @staticmethod
    def format_rate(rate: float):
        """Format the rate for logging.

        Use two significant digits if the rate is less than 1.0, otherwise
        use two decimal places.

        Args:
            rate: Rate to format.
        """
        if rate < 1.0:
            return f"{rate:.2g} samples/sec"
        return f"{rate:.2f} samples/sec"

    @property
    def postfix(self) -> List[str]:
        """Returns the postfix to append to the progress message."""
        if self.rate_profiler is not None:
            rate = self.rate_profiler.rate
            global_rate = self.rate_profiler.global_rate

            return [
                f"Rate={self.format_rate(rate)}",
                f"GlobalRate={self.format_rate(global_rate)}",
            ]

        return []

    def on_fit_start(self, trainer, train_dataloader, val_dataloader, loop):
        self.total_samples.reset()

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        batch_size = infer_batch_size(batch)
        if batch_size:
            self.total_samples.add(batch_size)

        self.print_training_progress(trainer, outputs["loss"], batch_size)

    @cstorch.step_closure
    def print_training_progress(
        self, trainer: Trainer, loss: torch.Tensor, batch_size: Optional[int]
    ):
        """Print training progress and log metrics."""
        if trainer.should_run_optimizer_step:
            if self.accum_loss is not None:
                loss = self.accum_loss + loss.item()
                self.accum_loss = None
            else:
                loss = loss.item()

            if trainer.is_log_step:
                progress_msg = [
                    f"| Train Device={trainer.backend.device}",
                    f"Step={trainer.global_step}",
                    f"Loss={loss:.5f}",
                    *self.postfix,
                ]
                trainer.logger.info(", ".join(progress_msg))

                self._log_loss_rate_metrics(trainer, loss, batch_size)
        else:
            # accumulate loss for gradient accumulation
            if self.accum_loss is None:
                self.accum_loss = loss.item()
            else:
                self.accum_loss += loss.item()

    def on_train_end(self, trainer, model, loop, loop_idx):
        trainer.logger.info("Training completed successfully!")

    def on_validate_start(self, trainer, model, val_dataloader, loop):
        # pylint: disable=attribute-defined-outside-init
        self.total_eval_loss = 0
        self.total_eval_steps = 0

    def on_validate_batch_end(self, trainer, model, outputs, batch, batch_idx):
        self.print_validation_progress(
            trainer, outputs["loss"], infer_batch_size(batch)
        )

    @cstorch.step_closure
    def print_validation_progress(
        self,
        trainer: Trainer,
        loss: torch.Tensor,
        batch_size: Optional[int],
    ):
        """Print validation progress and log metrics."""
        self.total_eval_loss += loss.item()
        self.total_eval_steps += 1

        if trainer.is_log_step:
            progress_msg = [
                f"| Eval Device={trainer.backend.device}",
                f"GlobalStep={trainer.global_step}",
                f"Batch={trainer.executor.user_iteration}",
                f"Loss={loss.item():.5f}",
                *self.postfix,
            ]
            trainer.logger.info(", ".join(progress_msg))

            if trainer.is_final_iteration:
                avg_eval_loss = self.total_eval_loss / self.total_eval_steps
                trainer.logger.info(f"Avg Eval Loss: {avg_eval_loss}")

                self._log_loss_rate_metrics(trainer, avg_eval_loss, batch_size)

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

    def log_metrics(self, metrics, step):
        pass

    def _log_loss_rate_metrics(
        self, trainer: Trainer, loss: float, batch_size: Optional[int]
    ):
        trainer.log_metrics(loss=loss)

        if self.rate_profiler:
            metrics = {
                "local_samples_per_sec": self.rate_profiler.rate,
                "avg_samples_per_sec": self.rate_profiler.global_rate,
            }

            if batch_size:
                metrics["avg_steps_per_sec"] = (
                    self.rate_profiler.global_rate / batch_size
                )

            trainer.log_metrics(**metrics)
