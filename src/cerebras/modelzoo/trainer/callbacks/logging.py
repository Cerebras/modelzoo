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

from contextlib import contextmanager
from math import gcd
from typing import Optional
from weakref import ref

from cerebras.appliance.log import ClassLogger, named_class_logger
from cerebras.modelzoo.trainer.callbacks import Callback
from cerebras.pytorch import _generating_docs

if _generating_docs:

    class ClassLogger:  # noqa: D101  # pylint: disable=function-redefined,missing-class-docstring
        pass


@named_class_logger("Logging")
class Logging(Callback, ClassLogger):
    """
    Callback that handles setting up the Trainer's logger
    as well as facilitates the cadence of logging.
    """

    def __init__(
        self,
        log_steps: int,
        log_level: str = "INFO",
        wsc_log_level: Optional[dict] = None,
        enable_act_frequency: bool = False,
    ):
        """
        Args:
            log_steps: Number of steps after which to log.
            log_level: Logging level for the Python logger.
            wsc_log_level: Specifes the logging level for particular
                Wafer-Scale Cluster servers or tasks.
            enable_act_frequency: If True, set the activation steps to be the
                log steps.
        """
        super().__init__()

        self.log_steps = log_steps
        self.enable_act_frequency = enable_act_frequency

        self.log_level = log_level
        self.wsc_log_level = wsc_log_level or {}

        self.initial_global_step = 0
        self.total_steps = 0
        self.is_training = True

        self.trainer = None

    def pre_setup(self, trainer):
        # TODO: Move setup_logging into this file
        from cerebras.modelzoo.common.pytorch_utils import setup_logging

        # Set up logging level
        setup_logging(
            self.log_level,
            streamer_logging_level=None,
            logging_dir=trainer.artifact_dir,
            model_dir=trainer.model_dir,
        )

        # TODO: Move set_wsc_log_level into this file
        from cerebras.modelzoo.common.run_utils import set_wsc_log_level

        set_wsc_log_level(self.wsc_log_level)

        self.trainer = ref(trainer)

    @contextmanager
    def flush_logs(self, trainer):
        """Context manager to flush all loggers after the context is exited."""
        try:
            yield
        finally:
            for logger in trainer.loggers:
                logger.flush()

    def on_enter_fit(
        self, trainer, stack, train_dataloader, val_dataloader, loop
    ):
        stack.enter_context(self.flush_logs(trainer))

    def on_fit_start(self, trainer, train_dataloader, val_dataloader, loop):
        self.initial_global_step = trainer.global_step
        self.total_steps = loop.total_steps

    def on_enter_train(self, trainer, stack, train_dataloader, loop, loop_idx):
        self.is_training = True

        stack.enter_context(self.flush_logs(trainer))

    def on_train_start(self, trainer, model, train_dataloader, loop, loop_idx):
        if self.log_steps and self.enable_act_frequency:
            # Error out if activation step is not a multiple of eval frequency
            if loop.eval_frequency is not None and loop.eval_frequency != 0:
                if (
                    loop.eval_frequency % self.log_steps != 0
                    and self.log_steps % loop.eval_frequency != 0
                ):
                    raise ValueError(
                        f"When activation frequency is enabled, log steps "
                        f"({self.log_steps}) must be a multiple or factor of "
                        f"eval frequency ({loop.eval_frequency})."
                    )

            if trainer.activation_steps is not None:
                trainer.activation_steps = gcd(
                    trainer.activation_steps, self.log_steps
                )
            else:
                trainer.activation_steps = self.log_steps

            trainer.activation_steps = min(
                trainer.activation_steps, loop.train_steps
            )
        else:
            trainer.activation_steps = None

    def on_train_end(self, trainer, model, loop, loop_idx):
        if self.log_steps:
            trainer.activation_steps = None

    def on_enter_validate(self, trainer, stack, val_dataloader, loop):
        self.is_training = False

        stack.enter_context(self.flush_logs(trainer))

    def is_log_step(self, trainer) -> bool:
        """Return True if the current step should be logged."""

        if self.is_training:
            fit_step = trainer.global_step - self.initial_global_step
            return fit_step >= self.total_steps or (
                self.log_steps and fit_step % self.log_steps == 0
            )
        else:
            return trainer.is_final_iteration or (
                self.log_steps
                and trainer.executor
                and trainer.executor.user_iteration % self.log_steps == 0
            )
