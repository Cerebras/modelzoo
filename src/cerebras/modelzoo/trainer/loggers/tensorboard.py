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

"""Contains the TensorboardLogger class for logging to Tensorboard."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from warnings import warn

import torch

from cerebras.modelzoo.trainer.loggers import Logger


class TensorBoardLogger(Logger):
    """Logger class that logs metrics to Tensorboard."""

    def __init__(
        self,
        summary_dir: Optional[str] = None,
        legacy_event_dirs: bool = False,
    ):
        """
        Args:
            summary_dir: Directory to save the Tensorboard logs.
                If None, use the trainer's model_dir.
            legacy_event_dirs: If True, use the legacy directory
                structure for event files. This option exists
                to maintain some backwards compatibility and should
                *not* be set to True or relied on if at all possible.
        """

        self.summary_dir = summary_dir
        if self.summary_dir is not None:
            self.summary_dir = Path(summary_dir)
            self.summary_dir.mkdir(parents=True, exist_ok=True)

        self.legacy_event_dirs = legacy_event_dirs
        if self.legacy_event_dirs:
            warn("Passing legacy_event_dirs=True is deprecated")
            self.writers = {}

        self.writer = None

    def flush(self):
        if self.legacy_event_dirs:
            for writer in self.writers.values():
                writer.flush()
            return

        if self.writer is not None:
            self.writer.flush()

    def setup_writer(self, trainer, mode=None):
        """Set up the writer in a context manager.

        The writer gets flushed on exit.
        """
        from cerebras.pytorch.utils.tensorboard import SummaryWriter

        if self.legacy_event_dirs:
            if mode not in self.writers:
                self.writers[mode] = SummaryWriter(
                    log_dir=str(trainer.model_dir / mode)
                )

            self.writer = self.writers[mode]
            return

        if self.writer is None:
            if self.summary_dir is None:
                self.summary_dir = trainer.model_dir / "events"

            self.writer = SummaryWriter(log_dir=str(self.summary_dir.resolve()))

            # Use absolute path since TB event file write will resolve symlinks too
            summary_dir = trainer.summary_dir.resolve()

            # pylint: disable=protected-access
            (summary_dir / "tf_events").symlink_to(
                os.path.relpath(
                    os.path.realpath(
                        self.writer._get_file_writer().event_writer._file_name
                    ),
                    summary_dir,
                )
            )
            (summary_dir / "cs_events").symlink_to(
                os.path.relpath(
                    os.path.realpath(self.writer.cs_events_dir),
                    summary_dir,
                )
            )

    def setup(self, trainer):
        if not self.legacy_event_dirs:
            self.setup_writer(trainer)

    def on_enter_train(self, trainer, stack, train_dataloader, loop, loop_idx):
        self.setup_writer(trainer, "train")

    def on_enter_validate(self, trainer, stack, val_dataloader, loop):
        self.setup_writer(trainer, "eval")

    def log_metrics(self, metrics, step):
        if self.writer is None:
            return

        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                # If a tensor has a single element, but has more than 0
                # dimensions, it is not a scalar tensor. We log it as a tensor.
                if value.numel() == 1 and value.ndim == 0:
                    self.writer.add_scalar(name, value.item(), step)
                else:
                    self.writer.add_tensor(name, value, step)
            elif isinstance(value, (int, float)):
                self.writer.add_scalar(name, value, step)
            elif isinstance(value, str):
                self.writer.add_text(name, value, step)
            else:
                try:
                    import pandas as pd

                    if isinstance(value, pd.DataFrame):
                        self.writer.add_text(
                            name, value.to_markdown(tablefmt="github"), step
                        )
                        continue
                except ImportError:
                    pass

                warn(
                    f"Attempting to log a {type(value)} for {name}. "
                    f"TensorBoard Logger does not support logging {type(value)}"
                )
