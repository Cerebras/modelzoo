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

"""Contains a callback that handles setting up and stepping all schedulers for the run."""

from collections import defaultdict
from collections.abc import Iterable
from itertools import combinations
from typing import Callable, List, Union

import cerebras.pytorch as cstorch
from cerebras.modelzoo.trainer.callbacks import CoreCallback

SchedulerInput = Union[
    Callable[
        [cstorch.optim.Optimizer],
        cstorch.optim.scheduler.Scheduler,
    ],
    cstorch.optim.scheduler.Scheduler,
    None,
]

SchedulersInput = Union[
    SchedulerInput,
    List[SchedulerInput],
    None,
]


class SchedulersCallback(CoreCallback):
    """Callback that sets up all the schedulers for the Trainer."""

    def __init__(
        self,
        schedulers: SchedulersInput = None,
    ):
        """
        Args:
            schedulers: The set of optimizer schedulers to be used. Common schedulers include LR
                schedulers. It must be a list of these items:
                - If a cstorch.optim.scheduler.Scheduler is passed, it is used as is.

                - A callable that is assumed to be a function that
                  takes in a cstorch.optim.Optimizer and returns a
                  cstorch.optim.scheduler.Scheduler.

                - If None, there is no optimizer param group scheduling.
        """
        if schedulers is None:
            self.schedulers = []
        elif isinstance(schedulers, Iterable):
            self.schedulers = schedulers
        else:
            self.schedulers = [schedulers]

        self.name_scope = None

    def setup(self, trainer):
        trainer.schedulers = []
        if trainer.optimizer is None:
            return
        if self.schedulers:
            for scheduler in self.schedulers:
                if isinstance(scheduler, cstorch.optim.scheduler.Scheduler):
                    trainer.schedulers.append(scheduler)
                else:
                    trainer.schedulers.append(scheduler(trainer.optimizer))

        for s1, s2 in combinations(trainer.schedulers, 2):
            if s1.param_group_key == s2.param_group_key:
                if s1.param_group_tags is None or s2.param_group_tags is None:
                    raise RuntimeError(
                        f"Schedulers with the same `param_group_key` cannot share "
                        f"`param_group_tags`. But got at least one scheduler for param group "
                        f"key {s1.param_group_key} that targets all param group tags."
                    )
                if intersect := s1.param_group_tags.intersection(
                    s2.param_group_tags
                ):
                    raise RuntimeError(
                        f"Schedulers with the same `param_group_key` cannot share "
                        f"`param_group_tags`. But got two schedulers targeting param group key "
                        f"{s1.param_group_key} which share the following param group tags: "
                        f"{intersect}."
                    )

        for param_group in trainer.optimizer.param_groups:
            matches = defaultdict(int)
            for _scheduler in trainer.schedulers:
                param_group_tags = _scheduler.param_group_tags
                param_group_key = _scheduler.param_group_key
                if p_tags := param_group.get("tags", None):
                    if (
                        param_group_tags is None
                        or param_group_tags.intersection(p_tags)
                    ):
                        matches[param_group_key] += 1
            for param_group_key, count in matches.items():
                if count > 1:
                    raise RuntimeError(
                        f"Multiple schedulers are targeting the same param group with "
                        f"key {param_group_key}. Schedulers that update the same "
                        f"param group must have a different `param_group_key`, otherwise "
                        f"the behavior is unexpected."
                    )

    def on_before_scheduler_step(self, trainer, model, optimizer, scheduler):
        # pylint: disable=no-member
        self.name_scope = cstorch.name_scope(scheduler.param_group_key)
        self.name_scope.__enter__()

    def on_after_scheduler_step(self, trainer, model, optimizer, scheduler):
        # pylint: disable=no-member
        self.name_scope.__exit__(None, None, None)
        self.name_scope = None

    def on_save_checkpoint(self, trainer, state_dict):
        if trainer.schedulers:
            state_dict["schedulers"] = [
                scheduler.state_dict() for scheduler in trainer.schedulers
            ]

    def on_load_checkpoint(self, trainer, state_dict):
        if trainer.schedulers:
            if "schedulers" in state_dict:
                if "lr_scheduler" in state_dict:
                    raise RuntimeError(
                        "Only one of `schedulers` and `lr_scheduler` should be in `state_dict`. "
                        "Multiple schedulers are now supported so new checkpoints should only "
                        "contain the `schedulers` key."
                    )
                for sd, scheduler in zip(
                    state_dict["schedulers"], trainer.schedulers
                ):
                    scheduler.load_state_dict(sd)

                trainer.logger.info(
                    f"Scheduler state found in checkpoint and loaded successfully."
                )
            elif "lr_scheduler" in state_dict:
                lr_scheduler_count = 0
                for scheduler in trainer.schedulers:
                    if isinstance(
                        scheduler, cstorch.optim.lr_scheduler.LRScheduler
                    ):
                        lr_scheduler_count += 1
                        if lr_scheduler_count > 1:
                            raise RuntimeError(
                                "Multiple LR schedulers found in the trainer but only one found in "
                                "the checkpoint. The checkpoint has the `lr_scheduler` key which "
                                "means it is from when only a single scheduler was supported."
                            )
                        scheduler.load_state_dict(state_dict["lr_scheduler"])
                        trainer.logger.info(
                            f"Scheduler state found in checkpoint and loaded successfully."
                        )
                if lr_scheduler_count == 0:
                    trainer.logger.info(
                        "Scheduler state not found in the checkpoint. "
                        "Using default initialized state."
                    )
            else:
                trainer.logger.info(
                    "Scheduler state not found in the checkpoint. "
                    "Using default initialized state."
                )
