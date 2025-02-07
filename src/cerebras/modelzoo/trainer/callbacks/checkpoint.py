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

"""Checkpointing callback that aids in saving and loading model states."""

import os
import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from string import Formatter
from typing import Any, Dict, List, Optional, Union
from warnings import warn

from packaging.version import parse

import cerebras.pytorch as cstorch
from cerebras.modelzoo.trainer.callbacks import Callback, CoreCallback


class Checkpoint(CoreCallback):
    """A callback that handles standard checkpointing logic."""

    def __init__(
        self,
        steps: Optional[int] = None,
        autoload_last_checkpoint: bool = True,
        disable_strict_checkpoint_loading: bool = False,
        save_initial_checkpoint: bool = False,
        checkpoint_name: str = "checkpoint_{step}.mdl",
    ):
        """
        Args:
            steps: The frequency at which to save a checkpoint. If None, no
                checkpoints will be saved. Defaults to None.
            autoload_last_checkpoint: Whether to autoload the last
                checkpoint in the model directory. Defaults to True.
            disable_strict_checkpoint_loading: Whether to disable
                strict checkpoint loading. If True, the model will not raise an
                error if the checkpoint contains keys that are not present in the
                model. Defaults to False.
            save_initial_checkpoint: Whether to save the initial
                checkpoint at the start of training. Defaults to False.
            checkpoint_name: The unformatted name of the checkpoint file. The string will be
                formatted with the following keys: `step`
        """
        super().__init__()

        if not (steps is None or (isinstance(steps, int) and steps >= 0)):
            raise ValueError(
                f"Checkpoint steps must be None or a non-negative integer, "
                f"but got {steps}"
            )

        self.steps = steps
        self.autoload_last_checkpoint = autoload_last_checkpoint
        self.disable_strict_checkpoint_loading = (
            disable_strict_checkpoint_loading
        )
        self.save_initial_checkpoint = save_initial_checkpoint
        self.checkpoint_name = checkpoint_name
        self.model_dir = None

        self._stack_size = 0

        keys = set(
            fname
            for _, fname, _, _ in Formatter().parse(self.checkpoint_name)
            if fname
        )
        expected_keys = {"step"}
        if keys != expected_keys:
            raise ValueError(
                f"Found invalid keys in checkpoint_name format string. "
                f"Expected keys: {expected_keys}. Got: {keys}"
            )

    @contextmanager
    def _on_enter(self):
        try:
            self._stack_size += 1
            yield
        finally:
            self._stack_size -= 1

    def on_enter_fit(
        self, trainer, stack, train_dataloader, val_dataloader, loop
    ):
        stack.enter_context(self._on_enter())

    def on_enter_validate_all(self, trainer, stack, val_dataloaders, loop):
        stack.enter_context(self._on_enter())

    def on_enter_validate(self, trainer, stack, val_dataloader, loop):
        stack.enter_context(self._on_enter())

    def on_train_start(self, trainer, model, train_dataloader, loop, loop_idx):
        if (
            loop_idx == 0
            and self.save_initial_checkpoint
            and trainer.backend.is_e2e_execution
        ):
            trainer.save_checkpoint()

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if self.steps:
            # Call checkpoint closure every iteration and let the
            # closure handle calling it at the correct interval
            trainer.save_checkpoint()

    def on_before_load_checkpoint(self, trainer, ckpt_path):
        if ckpt_path is None:
            if self._stack_size <= 1:
                trainer.logger.info(
                    f"No checkpoint was provided. "
                    "Using randomly initialized model parameters."
                )
        else:
            trainer.logger.info(f"Loading weights from checkpoint {ckpt_path}")

    @staticmethod
    def check_compatibility(state_dict: Dict[str, Any]):
        """Checks that the checkpoint is compatible with the current version of modelzoo."""
        import cerebras.modelzoo as mz
        import cerebras.modelzoo.tools.convert_checkpoint as convert_ckpt

        if "__metadata__" in state_dict:
            # extract the last item in the list as this is the most recent metadata
            checkpoint_version = state_dict["__metadata__"][-1].get(
                "version", ""
            )
            if not checkpoint_version:
                return
            checkpoint_version = parse(checkpoint_version)
            if checkpoint_version.local is None:
                current_version = parse(cstorch.__version__)
                if (
                    checkpoint_version.major != current_version.major
                    or checkpoint_version.minor != current_version.minor
                ):
                    converter_path = os.path.relpath(
                        convert_ckpt.__file__,
                        os.path.dirname(mz.__file__),
                    )
                    warn(
                        f"Checkpoint version may be incompatible with Modelzoo version. Got "
                        f"checkpoint version {checkpoint_version} but Modelzoo version "
                        f"is {current_version}. You may need to run {converter_path} on the "
                        f"incompatible checkpoint."
                    )

    def preprocess_checkpoint(self, trainer, state_dict):
        self.check_compatibility(state_dict)

    def on_save_checkpoint(self, trainer, state_dict):
        trainer.logger.info(f"Saving checkpoint at step {trainer.global_step}")

    def on_after_save_checkpoint(self, trainer, ckpt_path):
        trainer.logger.info(f"Saved checkpoint {ckpt_path}")

    def get_checkpoint_path(self, ckpt_dir: str, step: int) -> Path:
        """Construct a path to the checkpoint file.

        If a checkpoint already exists inside the given checkpoint directory at
        the given step, append a timestamp to the filename.

        Args:
            ckpt_dir: The directory where the checkpoint will be saved.
            step: The step at which the checkpoint is saved.

        Returns:
            A path to which the checkpoint can be saved
        """
        # Keep in sync with self.get_all_checkpoints().
        ckpt_dir = Path(ckpt_dir)
        ckpt_path = ckpt_dir / self.checkpoint_name.format(step=step)

        if ckpt_path.exists():
            ckpt_path = ckpt_dir / self.checkpoint_name.format(
                step=f"{step}_{datetime.now():%Y%m%d_%H%M%S}"
            )

        return ckpt_path

    def get_latest_checkpoint(self, trainer):
        """Return the path to the latest checkpoint."""

        trainer.logger.info(
            f"Checkpoint autoloading is enabled. Looking for latest checkpoint "
            f"in \"{trainer.model_dir}\" directory with the following naming "
            f"convention: `checkpoint_(step)(_timestamp)?.mdl`."
        )
        ckpts = self.get_all_checkpoints(trainer.model_dir)
        ckpt_path = ckpts[-1] if ckpts else None
        if ckpt_path:
            trainer.logger.info(f"Found latest checkpoint at \"{ckpt_path}\".")
        else:
            trainer.logger.info(
                f"No checkpoints were found in \"{trainer.model_dir}\"."
            )
        return ckpt_path

    def get_all_checkpoints(self, model_dir: str) -> List[str]:
        """Return the path to all available checkpoints.

        Args:
            model_dir: The directory where the checkpoints are located.
        """
        ckpts = []

        # Keep in sync with self.get_checkpoint_path().
        pattern = re.compile(
            self.checkpoint_name.format(
                step=r"(?P<step>\d+)(?:_(?P<timestamp>\d{8}_\d{6}))?"
            )
        )

        for checkpoint in Path(model_dir).glob("*"):
            match = pattern.fullmatch(checkpoint.name)
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


class LoadCheckpointStates(Callback):
    """Callback to load specific states of the model from the checkpoint."""

    def __init__(
        self,
        load_checkpoint_states: Union[str, List[str]] = "all",
    ):
        """
        Args:
            load_checkpoint_states: The list of state names to load from the checkpoint.
        """
        states = load_checkpoint_states
        if isinstance(states, str) and states.lower() == "all":
            self.load_checkpoint_states = states
        else:
            if isinstance(states, str):
                states = states.split(",")

            if isinstance(states, (list, tuple, set)) and all(
                isinstance(s, str) for s in states
            ):
                self.load_checkpoint_states = set(states)
            else:
                raise TypeError(
                    f"Expected `load_checkpoint_states` to be one of the following: "
                    f"\n\t1. \"all\" to load all checkpoint states."
                    f"\n\t2. A comma-separated string of checkpoint states to load."
                    f"\n\t3. List of checkpoint state names to load."
                    f"\nBut got type \"{type(load_checkpoint_states)}\" with value "
                    f"{load_checkpoint_states}."
                )

    def preprocess_checkpoint(self, trainer, state_dict):
        if self.load_checkpoint_states == "all":
            # Load all states, nothing to do
            return

        checkpoint_states = set(state_dict)
        # Check that the specified states are valid checkpoint states
        if not self.load_checkpoint_states.issubset(checkpoint_states):
            raise KeyError(
                "Unexpected keys specified via `load_checkpoint_states`: "
                f"{_format_keys(self.load_checkpoint_states - checkpoint_states)}.\n"
                "Only the keys in the following list are accepted: "
                f"{_format_keys(checkpoint_states)}."
            )

        if keys := (checkpoint_states - self.load_checkpoint_states):
            trainer.logger.info(
                f"Opting out of loading the following state(s) as they are "
                f"not included in \"load_checkpoint_states\": {', '.join(sorted(keys))}"
            )

            for key in keys:
                state_dict.pop(key, None)


class SaveCheckpointState(Callback):
    """
    Callback to save an alternative checkpoint file that contains a subset of states and is not
    affected by deletion policies.
    """

    def __init__(
        self,
        k: int,
        checkpoint_states: Union[str, List[str]] = "model",
        checkpoint_name: str = "{checkpoint_states}_{ckpt_name}",
    ):
        """
        Args:
            k: Cadence at which alternative checkpoint is saved.
                Specifes after how many checkpoints saved an alternative checkpoint is saved.
                For example, if a full checkpoint is taken every 100 steps and k=5, then an
                alternative checkpoint is saved every 500 steps.
            checkpoint_states: List of valid checkpoint states to save. Can be a single state or
                list of states or 'all' (all states).
            checkpoint_name: Prefix to add to the alternative checkpoint file name.
                The name will be formatted with the following keys:
                * ``checkpoint_states``: ``_`` separated list of checkpoint states
                * ``ckpt_name``: original checkpoint file name
        """
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"Expected k to be a positive integer. Got: {k}")

        self.k = k
        self.count = 0

        if isinstance(checkpoint_states, str):
            if checkpoint_states.lower() == "all":
                self.checkpoint_states = ["all"]
            else:
                self.checkpoint_states = list(checkpoint_states.split(","))
        elif isinstance(checkpoint_states, (list, tuple, set)):
            self.checkpoint_states = list(checkpoint_states)
        else:
            raise TypeError(
                "Expected `checkpoint_states` to be a string, list, tuple, or set. "
                f"Got: {type(checkpoint_states)}"
            )

        self.checkpoint_name = checkpoint_name

        keys = set(
            fname
            for _, fname, _, _ in Formatter().parse(self.checkpoint_name)
            if fname
        )
        expected_keys = {"checkpoint_states", "ckpt_name"}
        if keys != expected_keys:
            raise ValueError(
                f"Found invalid keys in checkpoint_name format string. "
                f"Expected keys: {expected_keys}. Got: {keys}"
            )

        self._is_last_loop = False
        self._is_last_step = False

    def on_train_start(self, trainer, model, train_dataloader, loop, loop_idx):
        self._is_last_loop = loop_idx == loop.num_trains - 1

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        self._is_last_step = (
            self._is_last_loop and trainer.executor.on_final_iteration
        )

    def on_after_save_checkpoint(self, trainer, ckpt_path):
        self.count += 1
        if self.count >= self.k or self._is_last_step:
            state_dict = cstorch.load(ckpt_path)

            ckpt_keys = set(state_dict)

            if self.checkpoint_states == ["all"]:
                checkpoint_states = ckpt_keys
            else:
                checkpoint_states = set(self.checkpoint_states)

            # Check that the specified states are valid checkpoint states
            if not checkpoint_states.issubset(ckpt_keys):
                raise KeyError(
                    "Unexpected keys specified via `checkpoint_states`: "
                    f"{_format_keys(checkpoint_states - ckpt_keys)}.\n"
                    "Only the keys in the following list are accepted: "
                    f"{_format_keys(ckpt_keys)}."
                )

            # Store all keys specified and all "metadata"-like keys that begin with "__"
            subset_dict = {
                key: state_dict[key]
                for key in ckpt_keys
                if key in checkpoint_states or key.startswith("__")
            }
            ckpt_name = self.checkpoint_name.format(
                checkpoint_states="_".join(self.checkpoint_states),
                ckpt_name=ckpt_path.name,
            )
            cstorch.save(subset_dict, ckpt_path.parent / ckpt_name)
            self.count = 0


class KeepNCheckpoints(Callback):
    """Callback to regulate the maximum number of checkpoints retained."""

    def __init__(self, n: Optional[int] = None):
        """
        Args:
            n: Number of checkpoint files to keep. If the number of checkpoint files saved exceeds
                this number, checkpoint files are deleted starting with the oldest one. Does not
                affect checkpoints taken from previous runs.
                If n is None, no checkpoints are deleted.
        """
        if n is None:
            n = float("inf")
        elif not isinstance(n, int) or n < 1:
            raise ValueError(f"Expected n to be a positive integer. Got: {n}")

        self.n = n
        self.ckpt_paths = []

    def on_after_save_checkpoint(self, trainer, ckpt_path):
        self.ckpt_paths.append(ckpt_path)
        if len(self.ckpt_paths) > self.n:
            ckpt_path = self.ckpt_paths.pop(0)
            ckpt_path.unlink(missing_ok=True)


def _format_keys(keys: List[str]) -> str:
    """Format a string of keys."""
    return ", ".join(f"\"{key}\"" for key in keys)
