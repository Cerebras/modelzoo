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
from dataclasses import asdict
from pathlib import Path

from cerebras.modelzoo.trainer.callbacks import Callback
from cerebras.pytorch.utils.data.utils import Schedule


class ArtifactDirCallback(Callback):
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
        # TODO: Move setup_artifact_dir logic into this callback
        from cerebras.modelzoo.common.pytorch_utils import setup_artifact_dir

        trainer.artifact_dir = Path(
            setup_artifact_dir(trainer.model_dir, mode="")
        )
        trainer.summary_dir = trainer.model_dir / trainer.artifact_dir.stem
        trainer.summary_dir.mkdir(parents=True, exist_ok=True)

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
                    # TODO: Handle properly exporting Schedule objects
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
