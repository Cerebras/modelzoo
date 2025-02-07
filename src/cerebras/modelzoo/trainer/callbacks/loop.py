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
This module provides the base LoopCallback class and its subclasses,
TrainingLoop and ValidationLoop, which are used to manage the
training and validation loops in the Trainer.
"""

import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from math import ceil
from typing import Optional, Union
from warnings import warn

import cerebras.pytorch as cstorch
from cerebras.modelzoo.trainer.callbacks import CoreCallback


class LoopCallback(CoreCallback, ABC):
    """
    Base class for all loop callbacks.

    This class should not be instantiated directly. Only subclasses of
    LoopCallback should be used.

    The loop callback owns the global step and is responsible for incrementing
    it after each training step.
    """

    @abstractmethod
    def __init__(self):
        pass

    def setup(self, trainer):
        trainer.global_step = 0

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if trainer.should_run_optimizer_step:
            # Only increment the global step if the optimizer step was run
            trainer.global_step += 1

    def on_save_checkpoint(self, trainer, state_dict):
        state_dict["global_step"] = trainer.global_step

    def on_load_checkpoint(self, trainer, state_dict):
        if "global_step" in state_dict:
            trainer.global_step = state_dict["global_step"]


class TrainingLoop(LoopCallback):
    """Callback class that manages the training loop."""

    def __init__(
        self,
        num_steps: Optional[int] = None,
        max_steps: Optional[int] = None,
        num_epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        eval_frequency: Union[int, float, None] = 1.0,
        eval_steps: Optional[int] = None,
        grad_accum_steps: int = 1,
    ):
        """
        Args:
            num_steps: The total number of training steps to perform.
                This will take precedence over max_steps.
            max_steps: The maximum number of training steps to perform.
                max_steps if provided will take the global step into account.
                That is, providing max_steps is equivalent to setting
                ``num_steps = max_steps - global_step``.
            num_epochs: The number of epochs to train for.
                This argument is mutually exclusive with num_steps.
            steps_per_epoch: Number of steps to train for in each epoch.
            eval_frequency: Frequency of evaluation during training. It can be:
                - a positive integer which specifies the number of
                  training steps between evaluations.
                - a float in the range [0.0, 1.0] which specifies
                  the fraction of training steps between evaluations.
                  i.e. if `eval_frequency=0.5`, evaluation will be performed
                  once after half of the training steps have completed
                  and once more at the end of training.
                - If None or zero, no evaluation is performed during training.
            eval_steps: The number of validation steps to perform.
            grad_accum_steps: Number of steps to accumulate gradients before
                performing an optimizer step. This is only relevant for CPU/GPU
                runs.
        """
        super().__init__()
        self.num_steps = num_steps
        self.max_steps = max_steps
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.eval_frequency = eval_frequency
        self.grad_accum_steps = grad_accum_steps

        self.val_loop = ValidationLoop(eval_steps)

        self.train_steps = None
        self.checkpoint_steps = None

    def on_enter_fit(
        self, trainer, stack, train_dataloader, val_dataloader, loop
    ):
        if loop is self:
            stack.enter_context(trainer.on_exception("fit"))

    def on_fit_start(self, trainer, train_dataloader, val_dataloader, loop):
        if loop is not self:
            return

        if not isinstance(train_dataloader, cstorch.utils.data.DataLoader):
            raise TypeError(
                f"train_dataloader must be an instance of cstorch.utils.data.DataLoader. "
                f"Got {type(train_dataloader)}"
            )

        # pylint: disable=attribute-defined-outside-init
        if not trainer.backend.is_e2e_execution:
            self.total_steps = 1
        else:
            self.total_steps = cstorch.utils.data.compute_num_steps(
                train_dataloader,
                initial_step=trainer.global_step,
                num_steps=self.num_steps,
                max_steps=self.max_steps,
                num_epochs=self.num_epochs,
                steps_per_epoch=self.steps_per_epoch,
                grad_accum_steps=self.grad_accum_steps,
            )

        if (
            self.eval_frequency is None
            or self.eval_frequency == 0
            or (val_dataloader is None and not trainer.validation_callbacks)
        ):
            if val_dataloader is not None:
                warn(
                    f"A validation dataloader was provided but `eval_frequency` "
                    f"is {self.eval_frequency}. The trainer will not run "
                    f"validation during training."
                )
            elif trainer.validation_callbacks:
                warn(
                    f"A validation callback was provided but `eval_frequency` "
                    f"is {self.eval_frequency}. The trainer will not run "
                    f"validation during training."
                )

            self.num_trains = 1
            self.train_steps = self.total_steps
            return

        error_msg = (
            f"`eval_frequency` must be a positive integer "
            f"or a float in the range (0.0, 1.0]. "
            f"Got {self.eval_frequency} with type {type(self.eval_frequency)}. "
            f"To disable validation during training, set `eval_frequency` to None"
        )

        if isinstance(self.eval_frequency, float):
            if not 0.0 < self.eval_frequency <= 1.0:
                raise ValueError(error_msg)

            self.train_steps = ceil(self.eval_frequency * self.total_steps)

        elif isinstance(self.eval_frequency, int):
            if self.eval_frequency <= 0:
                raise ValueError(error_msg)

            self.train_steps = min(self.eval_frequency, self.total_steps)

        else:
            raise TypeError(error_msg)

        self.num_trains = ceil(self.total_steps / self.train_steps)

    def on_enter_train(self, trainer, stack, train_dataloader, loop, loop_idx):
        if loop is self:
            stack.enter_context(trainer.on_exception("train"))

    def on_train_start(self, trainer, model, train_dataloader, loop, loop_idx):
        if loop is not self:
            return

        if self.train_steps is None:
            raise RuntimeError(
                "Detected that TrainingLoop.on_fit_start was called "
                "before TrainingLoop.on_train_start."
            )

        curr_step = self.train_steps * loop_idx

        if loop_idx == self.num_trains - 1:
            self.train_steps = self.total_steps - self.train_steps * loop_idx
        elif loop_idx >= self.num_trains:
            raise RuntimeError(
                "Number of training runs exceeds the number of expected runs."
            )

        if trainer.checkpoint and (ckpt_steps := trainer.checkpoint.steps):
            ckpt_steps = min(ckpt_steps, self.total_steps)

            start = (ckpt_steps - curr_step % ckpt_steps) - 1
            end = self.train_steps
            final_loop = loop_idx == self.num_trains - 1

            if start < end:
                self.checkpoint_steps = cstorch.utils.data.Schedule(
                    [
                        cstorch.utils.data.Schedule.Range(
                            start=start,
                            step=ckpt_steps,
                            end=end,
                            include_last=final_loop,
                        )
                    ]
                )
            elif final_loop:
                # If checkpointing is enabled, we always want a checkpoint at
                # the final step regardless.
                self.checkpoint_steps = end
            else:
                self.checkpoint_steps = None
        else:
            self.checkpoint_steps = None

        logging.info(
            f"Starting training loop {loop_idx + 1}, from global step {trainer.global_step} to "
            f"{trainer.global_step + self.train_steps}"
        )


class ValidationLoop(LoopCallback):
    """Callback class that manages the validation loop."""

    def __init__(self, eval_steps: Optional[int] = None, hook="validate"):
        """
        Args:
            eval_steps: The number of validation steps to perform.
            hook: The base name of the validation hooks to run.
                Default: "validate".
        """
        super().__init__()
        self._eval_steps = None
        self._max_eval_steps = eval_steps

        self.on_start_hook = f"on_{hook}_start"
        self.on_end_hook = f"on_{hook}_end"
        self.on_batch_start_hook = f"on_{hook}_batch_start"
        self.on_batch_end_hook = f"on_{hook}_batch_end"

    def on_enter_validate(self, trainer, stack, val_dataloader, loop):
        if loop is not self:
            return

        stack.enter_context(trainer.on_exception("validate"))

        @contextmanager
        def register_non_standard_hooks():
            # Whitelist the non-standard hooks for the current validation run
            non_standard_hooks = {
                self.on_start_hook,
                self.on_end_hook,
                self.on_batch_start_hook,
                self.on_batch_end_hook,
            } - trainer.non_standard_hooks_whitelist

            if not any(
                hasattr(callback, hook_name)
                for callback in trainer.all_callbacks
                for hook_name in non_standard_hooks
            ):
                warn(
                    f"No callbacks found that implement any of the specified "
                    f"non-standard hooks: {','.join(non_standard_hooks)}. "
                    "This means that no validation hooks will be run."
                )

            try:
                trainer.non_standard_hooks_whitelist |= non_standard_hooks
                yield
            finally:
                trainer.non_standard_hooks_whitelist -= non_standard_hooks

        stack.enter_context(register_non_standard_hooks())

    @property
    def eval_steps(self) -> int:
        """Returns the number of validation steps to perform."""

        if self._eval_steps is None:
            raise RuntimeError(
                f"Detected that {self.__class__.__name__}.on_validate_start was not called."
            )

        return self._eval_steps

    def on_validate_start(self, trainer, model, val_dataloader, loop):
        if loop is not self:
            return

        if (
            val_dataloader is not None
            and not isinstance(val_dataloader, cstorch.utils.data.DataLoader)
            and (
                not isinstance(val_dataloader, (list, tuple))
                or not all(
                    isinstance(d, cstorch.utils.data.DataLoader)
                    for d in val_dataloader
                )
            )
        ):
            raise TypeError(
                f"val_dataloader must be an instance or list of cstorch.utils.data.DataLoader. "
                f"Got {type(val_dataloader)}"
            )

        if not trainer.backend.is_e2e_execution:
            self._eval_steps = 1
        else:
            self._eval_steps = cstorch.utils.data.compute_num_steps(
                val_dataloader,
                num_steps=self._max_eval_steps,
                num_epochs=1 if self._max_eval_steps is None else None,
            )
