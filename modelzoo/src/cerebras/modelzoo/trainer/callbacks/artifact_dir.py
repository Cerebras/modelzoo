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
Contains callback that handles setting up the artifact directory.
"""

import json
import os
import time
from dataclasses import asdict

from cerebras.modelzoo.trainer.callbacks import CoreCallback
from cerebras.pytorch.utils.data.utils import Schedule


class ArtifactDirCallback(CoreCallback):
    """
    Sets up the artifact directory and write metadata to executor artifact dir
    with some information about the run.
    """

    def __init__(self):
        """
        Attributes:
            loop: The loop object from which to extract metadata.
        """
        self.loop = None

    def pre_setup(self, trainer):
        from cerebras.appliance.utils.file import create_symlink

        cerebras_logs_path = trainer.model_dir / "cerebras_logs"

        def _create():
            artifact_dir = cerebras_logs_path / time.strftime("%Y%m%d_%H%M%S")
            artifact_dir.mkdir(parents=True)
            return artifact_dir

        # Method for adding additional logs to the artifact directory.
        # For now we are only adding a log for environment variables
        def _create_support_logs():
            # get the user enviroment variables and log them
            env_vars_log = trainer.artifact_dir / "env_vars.txt"
            with open(env_vars_log, "w") as f:
                for key, value in sorted(os.environ.items()):
                    f.write(f'{key}={value}\n')
            return

        # CPU runs could potentially finish very fast, so back-to-back runs
        # may end up getting the same timestamp and we'd fail in creating
        # the duplicate directory. In case of directory already existing,
        # sleep for more than 1 second and try again. If we fail again,
        # then throw.
        try:
            trainer.artifact_dir = _create()
        except FileExistsError:
            time.sleep(1.5)
            try:
                trainer.artifact_dir = _create()
            except Exception as e:
                raise e from None

        # Create a symlink to the artifact_dir so that it's easy to find the latest run.
        # The symlink needs to be at the same level as the subdirectories.
        latest = cerebras_logs_path.joinpath("latest")
        # symlink to relative path
        create_symlink(
            latest,
            trainer.artifact_dir.relative_to(cerebras_logs_path),
            target_is_directory=True,
        )
        trainer.summary_dir = trainer.model_dir / trainer.artifact_dir.stem
        trainer.summary_dir.mkdir(parents=True, exist_ok=True)

        # Create a support directory for logging other info about user runs
        _create_support_logs()

    def _save_metadata(  # pylint: disable=no-self-use
        self, artifact_dir, metadata
    ):
        """Save the metadata to a json file inside the artifact directory."""
        with (artifact_dir / "metadata.json").open("w") as f:
            json.dump(metadata, f)

    def on_train_start(self, trainer, model, train_dataloader, loop, loop_idx):
        self.loop = loop

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        if self.loop is not None:
            checkpoint_steps = self.loop.checkpoint_steps
            if isinstance(checkpoint_steps, Schedule):
                checkpoint_steps = asdict(checkpoint_steps)

            self._save_metadata(
                trainer.executor.artifact_dir,
                {
                    "stage": "train",
                    "num_steps": self.loop.train_steps,
                    "checkpoint_steps": checkpoint_steps,
                },
            )
            self.loop = None

    def on_validate_start(self, trainer, model, val_dataloader, loop):
        self.loop = loop

    def on_validate_batch_start(self, trainer, model, batch, batch_idx):
        if self.loop is not None:
            self._save_metadata(
                trainer.executor.artifact_dir,
                {
                    "stage": "validate",
                    "num_steps": self.loop.eval_steps,
                },
            )
            self.loop = None
