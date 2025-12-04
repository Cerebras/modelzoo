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

import logging
import os
import sys
import traceback
from contextlib import contextmanager
from functools import partial
from math import gcd
from typing import Optional
from weakref import ref

import cerebras.pytorch.distributed as dist
from cerebras.appliance.log import (
    ClassLogger,
    collect_wsc_log_settings,
    get_level_name,
    named_class_logger,
    wsc_logger,
)
from cerebras.appliance.utils.file import create_symlink
from cerebras.modelzoo.trainer.callbacks import CoreCallback
from cerebras.pytorch import _generating_docs
from cerebras.pytorch.utils.call_once import call_once

if _generating_docs:

    class ClassLogger:  # noqa: D101  # pylint: disable=function-redefined,missing-class-docstring
        pass


@named_class_logger("Logging")
class Logging(CoreCallback, ClassLogger):
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
        self.setup_logging(trainer)
        self.set_wsc_log_level()
        self.trainer = ref(trainer)

    def setup_logging(self, trainer):
        """Configure default logging format."""

        def block_filter(record, handler_type):
            return getattr(record, "block", None) != handler_type

        handlers = []
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_CustomFormatter())
        handler.addFilter(partial(block_filter, handler_type="console"))
        handlers.append(handler)

        logging_file = trainer.artifact_dir / "run.log"
        handler = logging.FileHandler(logging_file)
        handler.setFormatter(_CustomFormatter())
        handler.addFilter(partial(block_filter, handler_type="file"))
        handlers.append(handler)

        # set up run log symlink in summary dir
        create_symlink(
            trainer.summary_dir / "run.log",
            os.path.relpath(logging_file, trainer.summary_dir),
        )

        # set up latest run log symlink in model dir
        create_symlink(
            trainer.model_dir / "latest_run.log",
            logging_file.relative_to(trainer.model_dir),
        )

        if dist.is_master_ordinal():
            level = get_level_name(self.log_level)
        else:
            level = get_level_name("error")

        # Remove any handlers that may have been inadvertently set before
        # keep any handlers marked with the `is_sticky` attribute
        handlers.extend(
            handler
            for handler in logging.getLogger().handlers
            if getattr(handler, "is_sticky", False)
        )
        logging.getLogger().handlers.clear()
        logging.basicConfig(level=level, handlers=handlers)

        self.setup_logging_excepthook()

        # Begin by logging the command that is running to the run log
        logging.info(f"Current Working Directory: {os.getcwd()}")
        logging.info(f"Running command: {' '.join(sys.argv)}")

    @staticmethod
    @call_once()
    def setup_logging_excepthook():
        """Setup a logging hook that runs whenever an exception is raised that
        catches and logs the exception to ensure that the full traceback is printed
        in the log file.
        """
        original_hook = sys.excepthook

        def cerebras_logging_hook(exc_type, exc_value, exc_traceback):
            """Pipe uncaught exceptions through logger."""
            msg = "".join(
                traceback.format_exception(exc_type, exc_value, exc_traceback)
            )
            # Block console logging to avoid duplicate messages since exceptions
            # are logged by python interpreter by default anyways.
            logging.error(
                f"Uncaught exception:\n{msg}", extra={"block": "console"}
            )

            # Run the original except hook which prints the exception to stderr
            original_hook(exc_type, exc_value, exc_traceback)

        sys.excepthook = cerebras_logging_hook

    def set_wsc_log_level(self):
        """Assert the list of log levels is valid."""
        if isinstance(self.wsc_log_level, dict):
            for task, level in self.wsc_log_level.items():
                level = int(level) if level.isdigit() else get_level_name(level)
                if task:
                    wsc_logger.getChild(task).setLevel(level)
                else:
                    wsc_logger.setLevel(level)
        else:
            raise ValueError("Invalid log levels. Input must be a dict.")

        # validate log level setting
        collect_wsc_log_settings()

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
                trainer.activation_steps, trainer.schedule.train_steps
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


class _CustomFormatter(logging.Formatter):
    """Cerebras Preferred Log Formatting."""

    def __init__(self):
        """Set up the formatter."""

        ordinal = dist.get_ordinal()
        num_tasks = dist.num_tasks() - 1

        if num_tasks > 1 and dist.is_streamer():
            ordinal_msg = f"[{ordinal}/{num_tasks}]"
        else:
            ordinal_msg = ""

        fmt = f"%(asctime)s %(levelname)s: {ordinal_msg}  %(message)s"
        super().__init__(fmt=fmt)

        self.info_formatter = None
        # Only enable shorter info logging depending on environment variable
        # This is so that we have the option to experiment with this in the future
        if "USE_SHORT_INFO_LOGGING" in os.environ:
            fmt = "{}%(message)s".format(
                f"{ordinal_msg}:  " if ordinal > 0 else ""
            )
            self.info_formatter = logging.Formatter(fmt)

    def format(self, record):
        if self.info_formatter and record.levelno == logging.INFO:
            return logging.Formatter.format(self.info_formatter, record)

        return super().format(record)
