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

import abc
import math
import warnings
from bisect import bisect_right
from typing import List

import torch

from modelzoo.common.pytorch import cb_model as cm


class LRScheduler(torch.optim.lr_scheduler.LambdaLR, abc.ABC):
    """
    Cerebras specific learning rate scheduler base class.

    The learning rate schedulers implemented in this file are specifically
    designed to be run on a Cerebras system. This means that there are certain
    caveats to these custom schedulers that differ from a typical LR scheduler
    found in core PyTorch.

    The learning rate schedulers here are intended to be stepped at every
    iteration. This means `lr_scheduler.step()` should be called after every
    `optimizer.step()`. Hence, the learning rate schedulers operate on a
    step-by-step basis. Having said that, there are some variables used such
    as `last_epoch` that might indicate otherwise. The only reason these
    variables are used is to match what is used in core PyTorch. It does *not*
    indicate that things are operating on an epoch-by-epoch basis.

    Also, note that the above means that our LR schedulers are incompatible with
    the LR schedulers found in core PyTorch. The state cannot simply be transferred
    between the two. So, one of the LR schedulers defined here must be used in
    order to have LR scheduling on the Cerebras system.
    """

    global_start_step = 0
    initial_epoch = 0

    def __init__(
        self,
        optimizer,
        decay_steps: int = None,
        disable_lr_steps_reset: bool = False,
    ):
        self.decay_steps = decay_steps
        self.disable_lr_steps_reset = disable_lr_steps_reset

        self.start_step = LRScheduler.global_start_step
        if decay_steps is not None:
            LRScheduler.global_start_step += decay_steps

        self.cb_scheduler = None

        # Cerebras specific learning rate scheduler configuration
        if cm.use_cs():
            from modelzoo.common.pytorch import cbtorch

            if not isinstance(optimizer, cbtorch.optim.Optimizer):
                raise TypeError(
                    f"Expected a Cerebras Optimizer. Got: {type(optimizer)}"
                )

            self.cb_scheduler = self._configure_cerebras_lrs(optimizer)

            super().__init__(optimizer._optimizer, lr_lambda=self.lr_function)

            self._last_lr = 0.1
        else:
            super().__init__(optimizer, lr_lambda=self.lr_function)

        LRScheduler.initial_epoch = self.last_epoch

    def _configure_cerebras_lrs(self, optimizer):
        raise NotImplementedError(
            f"Cerebras LR scheduler configuration is not implemented for: {self}"
        )

    def lr_function(self, global_step):
        if cm.use_cs():
            return self._lr_function(global_step)
        else:
            return self._lr_function(torch.tensor(global_step)).item()

    @abc.abstractmethod
    def _lr_function(self, global_step):
        raise NotImplementedError(
            f"_lr_function is not implemented for: {self}"
        )

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "to get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )

        # note, different from the parent class,
        # we ignore the base learning rate entirely
        return [lmbda(self.last_epoch) for lmbda in self.lr_lambdas]

    def state_dict(self):
        state = super().state_dict()
        return {
            key: val
            for key, val in state.items()
            if key not in ("cb_scheduler",)
        }

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        LRScheduler.initial_epoch = self.last_epoch

        # Make sure the learning rate schedules are set properly
        if not cm.use_cs():
            self._step_count = 0
            self.last_epoch -= 1
            super().step()

    def step(self, *args, **kwargs):
        """
        Steps the scheduler and computes the latest learning rate

        Only sets the last_epoch if running on CS
        """
        if cm.use_cs():
            if self.last_epoch == -1:
                self.last_epoch = 0
            else:
                self.last_epoch = cm.get_run_step() + LRScheduler.initial_epoch
        else:
            super().step(*args, **kwargs)


class Constant(LRScheduler):
    """
    Constant update

    Args:
        optimizer: The optimizer to schedule
        val: The actual learning_rate value
        decay_steps: The number of steps to decay for
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        val: int,
        decay_steps: int = None,
        disable_lr_steps_reset: bool = False,
    ):
        self.val = val
        super().__init__(optimizer, decay_steps, disable_lr_steps_reset)

    def _configure_cerebras_lrs(self, optimizer):
        from modelzoo.common.pytorch import cbtorch

        return cbtorch.optim.lr_scheduler.Constant(
            optimizer, self.val, self.decay_steps, self.disable_lr_steps_reset
        )

    def _lr_function(self, global_step):
        return torch.tensor(self.val, device=global_step.device)


class Polynomial(LRScheduler):
    """
    Polynomial Decay

    Args:
        optimizer: The optimizer to schedule
        learning_rate: The initial learning rate.
        end_learning_rate: The final learning rate
        decay_steps: Number of steps to perform the decay
        power: Exponent to apply to "x" (as in y=mx+b),
            which is ratio of step completion (1 for linear)
            Default: 1.0 (only Linear supported at the moment)
        cycle: Whether to cycle
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        end_learning_rate: float,
        decay_steps: int,
        power: float = 1.0,
        cycle: bool = False,
        disable_lr_steps_reset: bool = False,
    ):
        self.learning_rate = learning_rate
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle
        super().__init__(optimizer, decay_steps, disable_lr_steps_reset)

    def _configure_cerebras_lrs(self, optimizer):
        from modelzoo.common.pytorch import cbtorch

        return cbtorch.optim.lr_scheduler.Polynomial(
            optimizer,
            self.learning_rate,
            self.end_learning_rate,
            self.decay_steps,
            self.power,
            self.cycle,
            self.disable_lr_steps_reset,
        )

    def _lr_function(self, global_step):
        lr_diff = self.learning_rate - self.end_learning_rate
        alpha = torch.tensor(
            1.0, dtype=torch.float32, device=global_step.device
        )
        if self.cycle:
            alpha = torch.add(global_step, 1).div(self.decay_steps).ceil()

        return torch.where(
            global_step >= self.decay_steps,
            torch.tensor(
                self.end_learning_rate,
                dtype=torch.float32,
                device=global_step.device,
            ),
            torch.sub(
                1, torch.div(global_step, torch.mul(self.decay_steps, alpha)),
            )
            .pow(self.power)
            .mul(lr_diff)
            .add(self.end_learning_rate)
            .float(),
        )


class Exponential(LRScheduler):
    """
    Exponential Decay

    Args:
        optimizer: The optimizer to schedule
        learning_rate: The initial learning rate.
        decay_steps: Number of steps to perform the decay
        decay_rate: The decay rate
        staircase: If True decay the learning rate at discrete intervals
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        decay_steps: int,
        decay_rate: int,
        staircase: bool = False,
        disable_lr_steps_reset: bool = False,
    ):
        self.learning_rate = float(learning_rate)
        self.decay_rate = decay_rate
        self.staircase = staircase
        super().__init__(optimizer, decay_steps, disable_lr_steps_reset)

    def _configure_cerebras_lrs(self, optimizer):
        from modelzoo.common.pytorch import cbtorch

        return cbtorch.optim.lr_scheduler.Exponential(
            optimizer,
            self.learning_rate,
            self.decay_steps,
            self.decay_rate,
            self.staircase,
            self.disable_lr_steps_reset,
        )

    def _lr_function(self, global_step):
        power = torch.div(global_step, self.decay_steps)
        if self.staircase:
            power.floor_()
        return torch.pow(self.decay_rate, power).mul(self.learning_rate)


class InverseExponentialTimeDecay(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        step_exponent: int,
        decay_steps: int,
        decay_rate: int,
        staircase: bool = False,
        disable_lr_steps_reset: bool = False,
    ):
        self.learning_rate = learning_rate
        self.step_exponent = step_exponent
        self.decay_rate = decay_rate
        self.staircase = staircase
        super().__init__(optimizer, decay_steps, disable_lr_steps_reset)

    def _configure_cerebras_lrs(self, optimizer):
        from modelzoo.common.pytorch import cbtorch

        return cbtorch.optim.lr_scheduler.InverseExponentialTimeDecay(
            optimizer,
            self.learning_rate,
            self.step_exponent,
            self.decay_steps,
            self.decay_rate,
            self.staircase,
            self.disable_lr_steps_reset,
        )

    def _lr_function(self, global_step):
        alpha = torch.div(
            torch.pow(global_step.float(), self.step_exponent), self.decay_steps
        )
        if self.staircase:
            alpha.floor_()
        return torch.div(
            torch.tensor(
                self.learning_rate,
                dtype=torch.float32,
                device=global_step.device,
            ),
            torch.mul(alpha, self.decay_rate).add(1.0),
        )


class CosineDecay(LRScheduler):
    """
    Cosine Decay

    Args:
        optimizer: The optimizer to schedule
        learning_rate: The initial learning rate.
        end_learning_rate: The final learning rate
        decay_steps: Number of steps to perform the decay
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        end_learning_rate: float,
        decay_steps: int,
        disable_lr_steps_reset: bool = False,
    ):
        self.learning_rate = learning_rate
        self.end_learning_rate = end_learning_rate

        super().__init__(optimizer, decay_steps, disable_lr_steps_reset)

    def _configure_cerebras_lrs(self, optimizer):
        from modelzoo.common.pytorch import cbtorch

        return cbtorch.optim.lr_scheduler.CosineDecay(
            optimizer,
            self.learning_rate,
            self.end_learning_rate,
            self.decay_steps,
            self.disable_lr_steps_reset,
        )

    def _lr_function(self, global_step):
        lr_diff = self.learning_rate - self.end_learning_rate
        # clip the steps to be at most decay_steps

        step = torch.minimum(
            torch.tensor(
                self.decay_steps, dtype=torch.float32, device=global_step.device
            ),
            global_step,
        )
        # where we are at the cosine curve
        progress = (
            torch.div(math.pi, self.decay_steps).mul(step).cos().add(1).mul(0.5)
        )
        return torch.mul(progress, lr_diff).add(self.end_learning_rate)


class SequentialLR(torch.optim.lr_scheduler.SequentialLR):
    def __init__(self, optimizer, *args, **kwargs):
        if cm.use_cs():
            from modelzoo.common.pytorch import cbtorch

            if not isinstance(optimizer, cbtorch.optim.Optimizer):
                raise TypeError(
                    f"Expected a Cerebras Optimizer. Got: {type(optimizer)}"
                )

            optimizer = optimizer._optimizer

        super().__init__(optimizer, *args, **kwargs)
        LRScheduler.initial_epoch = self.last_epoch
        self._init_step()

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
        LRScheduler.initial_epoch = self.last_epoch
        self._init_step()

    def _init_step(self):
        # Step the current schedule once more in order to
        # make sure the learning rate is set properly
        if not cm.use_cs():
            idx = bisect_right(self._milestones, self.last_epoch)
            scheduler = self._schedulers[idx]
            if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
                scheduler.last_epoch = -1
                scheduler._step_count = 0
            else:
                scheduler.last_epoch -= 1
                scheduler._step_count -= 1

            scheduler.step()

            self._last_lr = scheduler.get_last_lr()

    def lr_function(self, global_step):
        if cm.use_cs():
            return self._lr_function(global_step)
        else:
            return self._lr_function(torch.tensor(global_step)).item()

    # Iterate through all milestones and select the correct LR scheduler
    # based on global step.
    def _lr_function(self, global_step):
        new_lr = self._schedulers[0].lr_function(global_step)
        for idx, milestone in enumerate(self._milestones):
            # If current global step is equal or greater than
            # the 'milestone', we will choose the corresponding
            # LR scheduler which is indexed 'idx+1' in 'self._schedulers`.
            # Otherwise, we will use the LR scheduler from previous iteration.
            res = torch.where(
                global_step < milestone,
                new_lr,
                self._schedulers[idx + 1].lr_function(global_step),
            )
            new_lr = res
        return new_lr

    def step(self):
        if cm.use_cs():
            self.last_epoch = cm.get_run_step() + LRScheduler.initial_epoch
        else:
            self.last_epoch += 1

        idx = bisect_right(self._milestones, self.last_epoch)
        scheduler = self._schedulers[idx]
        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            scheduler.last_epoch = 0

        scheduler.step()

        if not cm.use_cs():
            self._last_lr = scheduler.get_last_lr()


class PiecewiseConstant(SequentialLR):
    def __init__(
        self,
        optimizer,
        learning_rates: List[float],
        milestones: List[int],
        disable_lr_steps_reset: bool = False,
    ):
        schedulers = []
        boundaries = [0]
        boundaries.extend(milestones)
        for lr, b1, b2 in zip(learning_rates, boundaries[:-1], boundaries[1:]):
            schedulers.append(
                Constant(optimizer, lr, b2 - b1, disable_lr_steps_reset)
            )
        # Final learning rate
        schedulers.append(
            Constant(
                optimizer,
                learning_rates[-1],
                disable_lr_steps_reset=disable_lr_steps_reset,
            )
        )

        super().__init__(optimizer, schedulers, milestones)


class MultiStep(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        gamma: float,
        milestones: List[int],
        disable_lr_steps_reset: bool = False,
    ):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.milestones = milestones
        super().__init__(
            optimizer, decay_steps=None, disable_lr_steps_reset=False
        )

    def _configure_cerebras_lrs(self, optimizer):
        from modelzoo.common.pytorch import cbtorch

        return cbtorch.optim.lr_scheduler.MultiStep(
            optimizer,
            self.learning_rate,
            self.gamma,
            self.milestones,
            self.disable_lr_steps_reset,
        )

    def _lr_function(self, global_step):
        new_lr = torch.tensor(
            self.learning_rate, dtype=torch.float32, device=global_step.device
        )
        for milestone in self.milestones:
            res = torch.where(
                global_step < milestone,
                new_lr,
                torch.mul(
                    torch.tensor(
                        self.gamma,
                        dtype=torch.float32,
                        device=global_step.device,
                    ),
                    new_lr,
                ),
            )
            new_lr = res
        return new_lr


class Step(LRScheduler):
    """
    Step Decay

    Args:
        optimizer: The optimizer to schedule
        learning_rate: The initial learning rate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        step_size: int,
        gamma: int,
        disable_lr_steps_reset: bool = False,
    ):
        self.learning_rate = float(learning_rate)
        self.gamma = gamma
        self.step_size = step_size
        super().__init__(
            optimizer, decay_steps=None, disable_lr_steps_reset=False
        )

    def _configure_cerebras_lrs(self, optimizer):
        from modelzoo.common.pytorch import cbtorch

        return cbtorch.optim.lr_scheduler.Step(
            optimizer,
            self.learning_rate,
            self.gamma,
            self.step_size,
            self.disable_lr_steps_reset,
        )

    def _lr_function(self, global_step):
        return torch.mul(
            torch.pow(
                torch.tensor(
                    self.gamma, dtype=torch.float32, device=global_step.device
                ),
                torch.div(global_step, self.step_size).floor_(),
            ),
            self.learning_rate,
        )


class CosineAnnealing(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        T_max: int,
        eta_min: float,
        disable_lr_steps_reset: bool = False,
    ):
        self.learning_rate = float(learning_rate)
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(
            optimizer, decay_steps=None, disable_lr_steps_reset=False
        )

    def _configure_cerebras_lrs(self, optimizer):
        from modelzoo.common.pytorch import cbtorch

        return cbtorch.optim.lr_scheduler.CosineAnnealing(
            optimizer,
            self.learning_rate,
            self.T_max,
            self.eta_min,
            self.disable_lr_steps_reset,
        )

    def _lr_function(self, global_step):
        lr_diff = self.learning_rate - self.eta_min
        a = torch.div(
            torch.mul(torch.div(global_step, self.T_max), math.pi).cos().add(1),
            2,
        )
        return torch.add(torch.mul(a, lr_diff), self.eta_min)


class Lambda(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        disable_lr_steps_reset: bool = False,
    ):
        self.learning_rate = learning_rate
        super().__init__(
            optimizer, decay_steps=None, disable_lr_steps_reset=False
        )

    def set_lr_lambda(self):
        lambda1 = lambda epoch: torch.div(epoch, 30)
        lambda2 = lambda epoch: torch.pow(
            torch.tensor(0.95, dtype=torch.float32, device=epoch.device), epoch,
        )
        lr_lambda = [lambda1, lambda2]
        return lr_lambda

    def _configure_cerebras_lrs(self, optimizer):
        from modelzoo.common.pytorch import cbtorch

        return cbtorch.optim.lr_scheduler.Lambda(
            optimizer, self.learning_rate, self.disable_lr_steps_reset,
        )

    def _lr_function(self, global_step):
        new_lr = None
        lr_lambda = self.set_lr_lambda()
        tmp_var = global_step
        for lr in lr_lambda:
            new_lr = torch.mul(
                torch.tensor(
                    self.learning_rate,
                    dtype=torch.float32,
                    device=global_step.device,
                ),
                lr(tmp_var),
            )
        return torch.where(
            tmp_var > 0,
            new_lr,
            torch.tensor(
                self.learning_rate,
                dtype=torch.float32,
                device=global_step.device,
            ),
        )
