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

"""Various utility functions related to checkpoint saving and loading."""
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union
from warnings import warn


class CkptInfo:
    """Class to manage checkpoints created within one run."""

    def __init__(self, ckpt_dir: str):
        """Initializes CkptInfo.

        Args:
            ckpt_dir: directory where checkpoints are to be saved
        """
        self._ckpt_dir = os.path.abspath(ckpt_dir)
        self._ckpt_info = []

    def update(self, ckpt_path: str, max_store: Optional[int] = None):
        """Save ckpt_info if last checkpoint path changed.

        Args:
            ckpt_path: path to the last checkpoint
            max_store: maximum number of checkpoints to store
        """
        ckpt_path = os.path.relpath(ckpt_path, self._ckpt_dir)
        self._ckpt_info.append(ckpt_path)
        while max_store and len(self._ckpt_info) > max_store:
            drop_ckpt = self._ckpt_info.pop(0)
            drop_ckpt = os.path.join(self._ckpt_dir, drop_ckpt)
            if os.path.exists(drop_ckpt):
                logging.info(
                    f"Erasing {drop_ckpt} to maintain "
                    f"{max_store} checkpoints."
                )
                try:
                    os.remove(drop_ckpt)
                except OSError as e:
                    warn(
                        f"Failed to clean up old checkpoint {drop_ckpt} due to error: {e}"
                    )


def get_all_checkpoints(model_dir: str) -> List[str]:
    """Return the path to all available checkpoints"""
    ckpts = []
    for checkpoint in Path(model_dir).glob("checkpoint_*.mdl"):
        match = re.match(
            r"checkpoint_(?P<step>\d+)(?:_(?P<timestamp>\d{8}_\d{6}))?.mdl",
            checkpoint.name,
        )
        if not match:
            continue

        step = int(match.group("step"))
        timestamp = match.group("timestamp")
        if timestamp is not None:
            try:
                date = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
            except ValueError:
                continue
        else:
            date = datetime.min

        ckpts.append((checkpoint, step, date))

    # sort by step and then by timestamp
    return (
        [ckpt[0] for ckpt in sorted(ckpts, key=lambda x: (x[1], x[2]))]
        if ckpts
        else []
    )


def get_latest_checkpoint(model_dir: str) -> Union[str, None]:
    """Get the path to the checkpoint with the highest global step"""
    ckpts = get_all_checkpoints(model_dir)
    return ckpts[-1] if ckpts else None
