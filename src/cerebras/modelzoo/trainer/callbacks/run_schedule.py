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

import copy
import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from logging import Logger
from typing import List, Optional, Union

import cerebras.pytorch as cstorch
from cerebras.appliance.log import ClassLogger, named_class_logger
from cerebras.modelzoo.trainer.callbacks import (
    CoreCallback,
    ValidationCallback,
    ValidationLoop,
)
from cerebras.pytorch.utils.data import Schedule


class StageType(Enum):
    TRAIN_DATALOADER = auto()
    VAL_DATALOADER = auto()
    VAL_CALLBACK = auto()


@dataclass
class Stage:
    """Represents one stage/loop in a run, e.g. one iteration of a train loop."""

    type: StageType
    start_step: int  # The global step at which the stage starts.
    name: str = field(default="", compare=False)
    steps: Optional[int] = None
    every_n_vals: Optional[int] = (
        None  # Some validation callbacks only run once every N validations.
    )


@named_class_logger("RunSchedule")
class RunSchedule(CoreCallback, ClassLogger):
    def __init__(self):
        self.trainer = None
        self.reset_schedule()

    def reset_schedule(self):
        # List of stages for the entire run. Saved and loaded in trainer state.
        self.stages: Optional[List[Stage]] = None
        # How many epochs of training we need to run. Read by the trainer.
        # An epoch is one train loop followed by optional val loop(s).
        self.epochs: Optional[int] = None
        # The global step at which training started or restarted.
        self.initial_global_step: Optional[int] = None
        # Cached copy of global step so we give consistent responses within one step.
        self.global_step: Optional[int] = None
        # The index of the current stage.
        self.stage_idx: Optional[int] = None
        # Cached loop index, used to print our progress through the run.
        self.loop_idx: Optional[int] = None
        # Cached val loop, needed for extracting eval_steps after val starts.
        self.val_loop: Optional[ValidationLoop] = None
        # Cached index of the batch we're currently processing.
        self.batch_idx: Optional[int] = None
        # Checkpoint schedule for the whole run.
        self.checkpoint_schedule: Optional[Schedule.Range] = None
        # Checkpoint schedule for the current epoch. Read by the trainer.
        self.checkpoint_steps: Optional[Schedule] = None

    def setup(self, trainer):
        self.trainer = trainer

    def on_enter_fit(
        self, trainer, stack, train_dataloader, val_dataloader, loop
    ):
        stack.callback(self.reset_schedule)

    def on_fit_start(self, trainer, train_dataloader, val_dataloader, loop):
        if self.stages is not None:
            # The only time we already have a schedule is during a restart,
            # when our schedule state is loaded.
            return
        self.stage_idx = 0
        self.initial_global_step = trainer.global_step
        self.stages = _build_train_schedule(
            self.logger,
            self.initial_global_step,
            loop.total_steps,
            loop.eval_frequency,
            val_dataloader,
            trainer.validation_callbacks,
        )
        self.epochs = self.epochs_remaining

        if checkpoint_steps := trainer.checkpoint.steps:
            checkpoint_steps = min(checkpoint_steps, loop.total_steps)
            self.checkpoint_schedule = Schedule.Range(
                start=self.initial_global_step + checkpoint_steps - 1,
                step=checkpoint_steps,
                end=self.initial_global_step + loop.total_steps,
                include_last=True,
            )

    def on_enter_validate_all(
        self, trainer, stack, val_dataloaders, loop, ckpt_paths
    ):
        if self.stages is not None:
            # The only time we already have a schedule is during a restart,
            # when our schedule state is loaded.
            return
        stack.callback(self.reset_schedule)
        self.stage_idx = 0
        self.stages = _build_val_schedule(
            trainer.global_step,
            val_dataloaders,
            trainer.validation_callbacks,
            iterations=len(ckpt_paths),
        )
        self.epochs = 1
        self._print_validate_all_summary()

    def on_enter_validate(self, trainer, stack, val_dataloader, loop):
        setattr(self, loop.on_start_hook, self.on_validate_start)
        setattr(self, loop.on_end_hook, self.on_validate_end)
        setattr(self, loop.on_batch_start_hook, self.on_validate_batch_start)
        if self.stages is not None:
            # The only time we already have a schedule is during a restart,
            # when our schedule state is loaded. If this validate is running
            # in the context of a fit or validate_all, the schedule was already
            # built in on_fit_start or on_enter_validate_all.
            return
        stack.callback(self.reset_schedule)
        self.stage_idx = 0
        self.stages = _build_val_schedule(
            trainer.global_step,
            val_dataloader,
            val_callbacks=None,
            iterations=1,
        )
        self.epochs = 1
        self._print_validate_summary()

    def on_train_start(self, trainer, model, train_dataloader, loop, loop_idx):
        if loop_idx >= self.epochs:
            raise RuntimeError(
                f"Number of training runs {loop_idx + 1} exceeds the number of "
                f"expected runs {self.epochs}."
            )
        self.loop_idx = loop_idx

        if loop_idx == 0:
            self._print_fit_summary()
        self._print_train_loop_update()

        self.checkpoint_steps = None
        if self.checkpoint_schedule is None:
            return

        ckpt_steps = self.checkpoint_schedule.step
        global_steps_elapsed = trainer.global_step - self.initial_global_step
        start = (ckpt_steps - global_steps_elapsed % ckpt_steps) - 1
        end = self.train_steps
        is_final_loop = loop_idx == self.epochs - 1
        if start < end:
            self.checkpoint_steps = Schedule(
                [
                    Schedule.Range(
                        start=start,
                        step=ckpt_steps,
                        end=end,
                        include_last=is_final_loop,
                    )
                ]
            )
        elif is_final_loop:
            # If checkpointing is enabled, we always want a checkpoint at
            # the final step regardless.
            self.checkpoint_steps = end

    def on_validate_start(self, trainer, model, val_dataloader, loop):
        self.val_loop = loop
        if self.loop_idx is None:
            self.loop_idx = 0

    def on_train_end(self, trainer, model, loop, loop_idx):
        self.stage_idx += 1

    def on_validate_end(self, trainer, model, loop):
        self.stage_idx += 1

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        self.global_step = trainer.global_step
        self.batch_idx = batch_idx

    def on_validate_batch_start(self, trainer, model, batch, batch_idx):
        self.global_step = trainer.global_step
        self.batch_idx = batch_idx

        if batch_idx > 0:
            return

        # We check loop.eval_steps here because our on_validate_start gets called
        # before ValidationLoop's on_validate_start which is where eval_steps is set.
        for stage in self.stages:
            if stage.name == self.current_stage.name:
                if stage.steps:
                    # We've already set the steps for this val loop.
                    break
                stage.steps = self.val_loop.eval_steps

        self._print_val_loop_update()

    def on_save_trainer_state(self, trainer, state_dict):
        state_dict[trainer.get_callback_name(self)] = {
            "stages": self.stages,
            "initial_global_step": self.initial_global_step,
            "checkpoint_schedule": self.checkpoint_schedule,
        }

    def on_load_trainer_state(self, trainer, state_dict):
        key = trainer.get_callback_name(self)
        if key not in state_dict:
            raise RuntimeError(f"{key} state not found in trainer state.")
        self.reset_schedule()
        self.stages = state_dict[key]["stages"]
        self.initial_global_step = state_dict[key]["initial_global_step"]
        self.checkpoint_schedule = state_dict[key]["checkpoint_schedule"]

    def on_load_checkpoint(self, trainer, state_dict):
        if self.stages is None or self.stage_idx is not None:
            # If stages is None, we don't have a schedule to resume.
            # If stage_idx is not None, we already know where we are.
            return

        # We figure out where to resume in the schedule in on_load_checkpoint
        # so we can rely on trainer.global_step having been loaded already by
        # the loop callback.
        corrected_step = trainer.global_step - self.initial_global_step
        steps = 0
        for stage_idx, stage in enumerate(self.stages):
            if stage.type != StageType.TRAIN_DATALOADER:
                # We always want to restart on a train step so that we reserve
                # resources that will work for future val steps. Starting on a
                # val step may lead to us securing less resources than are
                # necessary for the following train.
                continue
            if corrected_step < steps + stage.steps:
                self.stage_idx = stage_idx
                next_eval = steps + stage.steps
                stage.steps = next_eval - corrected_step
                break
            steps += stage.steps
        else:
            raise RuntimeError(
                f"Loaded global step {trainer.global_step} exceeds total number "
                f"of training steps {steps}."
            )
        self.epochs = self.epochs_remaining

    @property
    def epochs_remaining(self) -> int:
        """Number of trains left in the run."""
        if self.stages is None or self.stage_idx is None:
            raise RuntimeError(
                "epochs_remaining requested but schedule hasn't been built yet."
            )
        return sum(
            stage.type == StageType.TRAIN_DATALOADER
            for stage in self.stages[self.stage_idx :]
        )

    @property
    def stages_remaining(self) -> List[Stage]:
        """
        Stages remaining in the run.

        Does not include the current stage, which is available through the
        current_stage property.
        """
        if self.stages is None or self.stage_idx is None:
            raise RuntimeError(
                "stages_remaining requested but schedule hasn't been built yet."
            )
        return self.stages[self.stage_idx + 1 :]

    @property
    def current_stage(self) -> Stage:
        """Stage that is currently in progress."""
        if self.stages is None or self.stage_idx is None:
            raise RuntimeError(
                "current_stage requested but schedule hasn't been built yet."
            )
        return self.stages[self.stage_idx]

    @property
    def train_steps(self) -> int:
        """
        Steps in the current train stage. Only available during training.
        """
        if self.current_stage.type != StageType.TRAIN_DATALOADER:
            raise RuntimeError("train_steps is only available during training.")
        return self.current_stage.steps

    @property
    def batches_remaining(self) -> int:
        """Batches remaining in the current stage."""
        if self.stages is None or self.stage_idx is None:
            raise RuntimeError(
                "batches_remaining requested but schedule hasn't been built yet."
            )
        return self.current_stage.steps - self.batch_idx - 1

    def _print_fit_summary(self):
        val_stage_every_n_vals = {
            stage.name: stage.every_n_vals or 1
            for stage in self.stages_remaining
            if stage.type != StageType.TRAIN_DATALOADER
        }
        lines = [
            _ascii_break("="),
            "Trainer Fit Summary",
            _ascii_break("-"),
            (
                f"Trainer will run {_pluralize(self.epochs, 'train loop')}"
                f"{', interleaving validation as follows:' if val_stage_every_n_vals else '.'}"
            ),
            *(
                f"* {stage_name} after every {_pluralize(every_n_vals, 'train loop', drop_one=True)}"
                for stage_name, every_n_vals in val_stage_every_n_vals.items()
            ),
            "",
            "Train steps per train loop:",
            *[
                f"* {_pluralize(loops, 'loop')} of {_pluralize(steps, 'step')}"
                for loops, steps in self._train_steps_per_loop()
            ],
            f"for a total of {_pluralize(self.trainer.loop.total_steps, 'train step')}.",
        ]
        if self.checkpoint_schedule:
            checkpoint_step = self.checkpoint_schedule.step
            schedule_copy = copy.copy(self.checkpoint_schedule)
            schedule_copy.start = schedule_copy.next_immediate_step(
                self.trainer.global_step
            )
            checkpoints_remaining = len(schedule_copy)
            last_step_checkpoint_message = (
                f" and on the last step,"
                if len(self.checkpoint_schedule)
                != len(self.checkpoint_schedule.range())
                else ""
            )
            lines.extend(
                [
                    "",
                    (
                        "Checkpoints will be taken every "
                        f"{_pluralize(checkpoint_step, 'step', drop_one=True)}"
                        f",{last_step_checkpoint_message} for a total of "
                        f"{_pluralize(checkpoints_remaining, 'checkpoint')}."
                    ),
                ]
            )
        lines.extend(
            [
                "",
                _progress_logger_summary(self.trainer.logging.log_steps),
                _ascii_break("="),
            ]
        )
        for line in lines:
            self.logger.info(line)

    def _train_steps_per_loop(self):
        """
        All train stages have the same number of steps except possibly the first
        due to restarts or the last due to the number of train loops not cleanly
        dividing the total steps. This helper function returns a list of tuples
        (loops, steps) which describes how many loops occur with the given number
        of steps. The list of returned tuples will have length between 1 and 3
        depending on whether the first and/or last train stage have a different
        number of steps than the rest of the stages.
        """
        train_stage_steps = [
            stage.steps
            for stage in [self.current_stage] + self.stages_remaining
            if stage.type == StageType.TRAIN_DATALOADER
        ]
        assert 1 <= len(set(train_stage_steps)) <= 3
        loops_and_steps = []
        previous_stage_steps = train_stage_steps[0]
        loops = 0
        for stage_steps in train_stage_steps:
            if stage_steps == previous_stage_steps:
                loops += 1
                continue
            loops_and_steps.append((loops, previous_stage_steps))
            previous_stage_steps = stage_steps
            loops = 1
        loops_and_steps.append((loops, previous_stage_steps))
        return loops_and_steps

    def _print_validate_summary(self):
        assert (
            len(self.stages) == 1
        ), "Top-level validate runs should only have one stage"
        for line in [
            _ascii_break("="),
            "Trainer Validate Summary",
            _ascii_break("-"),
            f"Trainer will run one loop of {self.stages[0].name}.",
            "",
            _progress_logger_summary(self.trainer.logging.log_steps),
            _ascii_break("="),
        ]:
            self.logger.info(line)

    def _print_validate_all_summary(self):
        # In validate_all, all stages run the same number of loops. We use a
        # Counter to get the count of each stage by retrieving the most common
        # stage (which should match all other stages), and also so that we get
        # the unique stage names in order (because dicts are ordered).
        val_stage_counts = Counter(
            stage.name for stage in [self.current_stage] + self.stages_remaining
        )
        _, num_loops_per_stage = val_stage_counts.most_common(1)[0]
        for line in [
            _ascii_break("="),
            "Trainer Validate All Summary",
            _ascii_break("-"),
            f"For each of {num_loops_per_stage} checkpoints, trainer will run:",
            *(f"* 1 loop of {k}" for k in val_stage_counts),
            "",
            _progress_logger_summary(self.trainer.logging.log_steps),
            _ascii_break("="),
        ]:
            self.logger.info(line)

    def _print_train_loop_update(self):
        for line in [
            _ascii_break("-"),
            (
                f"Starting train loop {self.loop_idx + 1} of {self.epochs}, "
                f"from global step {self.trainer.global_step + 1} to "
                f"{self.trainer.global_step + self.train_steps} "
                f"({_pluralize(self.train_steps, 'step')})"
            ),
            _ascii_break("-"),
        ]:
            self.logger.info(line)

    def _print_val_loop_update(self):
        every_n_vals = self.current_stage.every_n_vals or 1
        for line in [
            _ascii_break("-"),
            (
                f"Starting {self.current_stage.name} loop "
                f"{math.ceil((self.loop_idx + 1) / every_n_vals)} of "
                f"{math.ceil(self.epochs / every_n_vals)}, with "
                f"{_pluralize(self.current_stage.steps, 'step')}"
            ),
            _ascii_break("-"),
        ]:
            self.logger.info(line)


def _build_train_schedule(
    logger: Logger,
    start_step: int,
    total_steps: int,
    eval_frequency: Union[int, float, None],
    val_dataloaders: Union[
        cstorch.utils.data.DataLoader,
        List[cstorch.utils.data.DataLoader],
        None,
    ],
    val_callbacks: List[ValidationCallback],
) -> List[Stage]:
    """Builds a training schedule based on the provided configuration."""
    if not eval_frequency or (val_dataloaders is None and not val_callbacks):
        if val_dataloaders:
            logger.warning(
                f"A validation dataloader was provided but `eval_frequency` "
                f"is {eval_frequency}. The trainer will not run "
                f"validation during training."
            )
        elif val_callbacks:
            logger.warning(
                f"A validation callback was provided but `eval_frequency` "
                f"is {eval_frequency}. The trainer will not run "
                f"validation during training."
            )
        return [
            Stage(
                type=StageType.TRAIN_DATALOADER,
                name="train",
                start_step=start_step,
                steps=total_steps,
            )
        ]

    error_msg = (
        f"`eval_frequency` must be a positive integer "
        f"or a float in the range (0.0, 1.0]. "
        f"Got {eval_frequency} with type {type(eval_frequency)}. "
        f"To disable validation during training, set `eval_frequency` to None"
    )
    if isinstance(eval_frequency, float):
        if not 0.0 < eval_frequency <= 1.0:
            raise ValueError(error_msg)
        train_steps = math.ceil(eval_frequency * total_steps)

    elif isinstance(eval_frequency, int):
        if eval_frequency <= 0:
            raise ValueError(error_msg)
        train_steps = min(eval_frequency, total_steps)
    else:
        raise TypeError(error_msg)

    stages = []
    epochs = math.ceil(total_steps / train_steps)
    for epoch_idx in range(epochs):
        is_last_epoch = epoch_idx == epochs - 1
        train_stage_steps = (
            train_steps
            if not is_last_epoch
            else total_steps - (epoch_idx * train_steps)
        )
        stages.append(
            Stage(
                type=StageType.TRAIN_DATALOADER,
                name="train",
                start_step=start_step,
                steps=train_stage_steps,
            )
        )
        start_step += train_stage_steps
        _add_val_dataloaders_to_schedule(stages, val_dataloaders, start_step)
        _add_val_callbacks_to_schedule(
            stages, val_callbacks, epoch_idx, is_last_epoch, start_step
        )
    return stages


def _build_val_schedule(
    start_step: int,
    val_dataloaders: Union[
        cstorch.utils.data.DataLoader,
        List[cstorch.utils.data.DataLoader],
        None,
    ],
    val_callbacks: List[ValidationCallback],
    iterations: int,
) -> List[Stage]:
    """Builds a validation schedule based on the provided configuration."""
    stages = []
    for _ in range(iterations):
        _add_val_dataloaders_to_schedule(stages, val_dataloaders, start_step)
        _add_val_callbacks_to_schedule(
            stages,
            val_callbacks,
            epoch_idx=0,
            is_last_epoch=True,
            start_step=start_step,
        )
    return stages


def _add_val_callbacks_to_schedule(
    stages: List[Stage],
    val_callbacks: List[ValidationCallback],
    epoch_idx: int,
    is_last_epoch: bool,
    start_step: int,
) -> None:
    stages.extend(
        Stage(
            type=StageType.VAL_CALLBACK,
            name=f"val_callback_{callback.name_scope}_{val_loop_idx}",
            start_step=start_step,
            every_n_vals=callback.every_n_vals,
        )
        for callback in val_callbacks or []
        if is_last_epoch or (epoch_idx + 1) % callback.every_n_vals == 0
        for val_loop_idx in range(callback.num_validate_loops)
    )


def _add_val_dataloaders_to_schedule(
    stages: List[Stage],
    val_dataloaders: Union[
        cstorch.utils.data.DataLoader,
        List[cstorch.utils.data.DataLoader],
        None,
    ],
    start_step: int,
) -> None:
    """Adds the provided dataloaders to the schedule."""
    if val_dataloaders is None:
        return
    if not isinstance(val_dataloaders, (list, tuple)):
        val_dataloaders = [val_dataloaders]
    stages.extend(
        Stage(
            type=StageType.VAL_DATALOADER,
            name=f"val_dataloader_{val_dataloader.id}",
            start_step=start_step,
        )
        for val_dataloader in val_dataloaders
    )


def _pluralize(num, string, drop_one=False):
    num_str = f"{num} "
    if drop_one and num == 1:
        num_str = ""
    return f"{num_str}{string}{'s' if num > 1 else ''}"


def _ascii_break(character):
    return character * 75


def _progress_logger_summary(log_steps):
    return f"Progress will be logged every {_pluralize(log_steps, 'step', drop_one=True)}."
