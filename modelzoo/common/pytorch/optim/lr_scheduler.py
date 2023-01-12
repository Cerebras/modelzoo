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
import string
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


class ConstantLR(LRScheduler):
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
        val: float,
        decay_steps: int = None,
        disable_lr_steps_reset: bool = False,
    ):
        self.learning_rate = val
        super().__init__(optimizer, decay_steps, disable_lr_steps_reset)

    def _configure_cerebras_lrs(self, optimizer):
        from modelzoo.common.pytorch import cbtorch

        return cbtorch.optim.lr_scheduler.Constant(
            optimizer,
            self.learning_rate,
            self.decay_steps,
            self.disable_lr_steps_reset,
        )

    def _lr_function(self, global_step):
        return torch.tensor(self.learning_rate, device=global_step.device)


class PolynomialLR(LRScheduler):
    """
    Polynomial Decay

    Args:
        optimizer: The optimizer to schedule
        initial_learning_rate: The initial learning rate.
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
        initial_learning_rate: float,
        end_learning_rate: float,
        decay_steps: int,
        power: float = 1.0,
        cycle: bool = False,
        disable_lr_steps_reset: bool = False,
    ):
        self.initial_learning_rate = initial_learning_rate
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle
        super().__init__(optimizer, decay_steps, disable_lr_steps_reset)

    def _configure_cerebras_lrs(self, optimizer):
        from modelzoo.common.pytorch import cbtorch

        return cbtorch.optim.lr_scheduler.Polynomial(
            optimizer,
            self.initial_learning_rate,
            self.end_learning_rate,
            self.decay_steps,
            self.power,
            self.cycle,
            self.disable_lr_steps_reset,
        )

    def _lr_function(self, global_step):
        lr_diff = self.initial_learning_rate - self.end_learning_rate
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


class ExponentialLR(LRScheduler):
    """
    Exponential Decay

    Args:
        optimizer: The optimizer to schedule
        initial_learning_rate: The initial learning rate.
        decay_steps: Number of steps to perform the decay
        decay_rate: The decay rate
        staircase: If True decay the learning rate at discrete intervals
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_learning_rate: float,
        decay_steps: int,
        decay_rate: float,
        staircase: bool = False,
        disable_lr_steps_reset: bool = False,
    ):
        self.initial_learning_rate = float(initial_learning_rate)
        self.decay_rate = decay_rate
        self.staircase = staircase
        super().__init__(optimizer, decay_steps, disable_lr_steps_reset)

    def _configure_cerebras_lrs(self, optimizer):
        from modelzoo.common.pytorch import cbtorch

        return cbtorch.optim.lr_scheduler.Exponential(
            optimizer,
            self.initial_learning_rate,
            self.decay_steps,
            self.decay_rate,
            self.staircase,
            self.disable_lr_steps_reset,
        )

    def _lr_function(self, global_step):
        power = torch.div(global_step, self.decay_steps)
        if self.staircase:
            power.floor_()
        return torch.pow(self.decay_rate, power).mul(self.initial_learning_rate)


class InverseExponentialTimeDecayLR(LRScheduler):
    """
    InverseExponentialTimeDecay

    Args:
        optimizer: The optimizer to schedule
        initial_learning_rate: The initial learning rate.
        step_exponent: Exponential value.
        decay_steps: Number of steps to perform the decay.
        decay_rate: The decay rate.
        staircase: If True decay the learning rate at discrete intervals.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_learning_rate: float,
        step_exponent: int,
        decay_steps: int,
        decay_rate: float,
        staircase: bool = False,
        disable_lr_steps_reset: bool = False,
    ):
        self.initial_learning_rate = initial_learning_rate
        self.step_exponent = step_exponent
        self.decay_rate = decay_rate
        self.staircase = staircase
        super().__init__(optimizer, decay_steps, disable_lr_steps_reset)

    def _configure_cerebras_lrs(self, optimizer):
        from modelzoo.common.pytorch import cbtorch

        return cbtorch.optim.lr_scheduler.InverseExponentialTimeDecay(
            optimizer,
            self.initial_learning_rate,
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
                self.initial_learning_rate,
                dtype=torch.float32,
                device=global_step.device,
            ),
            torch.mul(alpha, self.decay_rate).add(1.0),
        )


class InverseSquareRootDecayLR(LRScheduler):
    """
    InverseSquareRootDecay

    Args:
        optimizer: The optimizer to schedule
        initial_learning_rate: The initial learning rate.
        scale: Multiplicative factor to scale the result.
        warmup_steps: use initial_learning_rate for the first warmup_steps.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_learning_rate: float,
        scale: float,
        warmup_steps: int,
        disable_lr_steps_reset: bool = False,
    ):
        self.initial_learning_rate = initial_learning_rate
        self.scale = scale
        self.warmup_steps = warmup_steps
        super().__init__(
            optimizer,
            decay_steps=None,
            disable_lr_steps_reset=disable_lr_steps_reset,
        )

    def _configure_cerebras_lrs(self, optimizer):
        from modelzoo.common.pytorch import cbtorch

        return cbtorch.optim.lr_scheduler.InverseSquareRootDecay(
            optimizer,
            self.initial_learning_rate,
            self.scale,
            self.warmup_steps,
            self.disable_lr_steps_reset,
        )

    def _lr_function(self, global_step):
        return torch.div(
            torch.tensor(
                self.scale, dtype=torch.float32, device=global_step.device
            ),
            torch.sqrt(
                torch.max(
                    torch.tensor(
                        self.warmup_steps,
                        dtype=torch.float32,
                        device=global_step.device,
                    ),
                    global_step,
                )
            ),
        ).mul(self.initial_learning_rate)


class CosineDecayLR(LRScheduler):
    """
    Cosine Decay

    Args:
        optimizer: The optimizer to schedule
        initial_learning_rate: The initial learning rate.
        end_learning_rate: The final learning rate
        decay_steps: Number of steps to perform the decay
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_learning_rate: float,
        end_learning_rate: float,
        decay_steps: int,
        disable_lr_steps_reset: bool = False,
    ):
        self.initial_learning_rate = initial_learning_rate
        self.end_learning_rate = end_learning_rate

        super().__init__(optimizer, decay_steps, disable_lr_steps_reset)

    def _configure_cerebras_lrs(self, optimizer):
        from modelzoo.common.pytorch import cbtorch

        return cbtorch.optim.lr_scheduler.CosineDecay(
            optimizer,
            self.initial_learning_rate,
            self.end_learning_rate,
            self.decay_steps,
            self.disable_lr_steps_reset,
        )

    def _lr_function(self, global_step):
        lr_diff = self.initial_learning_rate - self.end_learning_rate
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
                self._schedulers[idx + 1].lr_function(global_step - milestone),
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
            scheduler.last_epoch = -1

        scheduler.step()

        if not cm.use_cs():
            self._last_lr = scheduler.get_last_lr()


class PiecewiseConstantLR(SequentialLR):
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
                ConstantLR(optimizer, lr, b2 - b1, disable_lr_steps_reset)
            )
        # Final learning rate
        schedulers.append(
            ConstantLR(
                optimizer,
                learning_rates[-1],
                disable_lr_steps_reset=disable_lr_steps_reset,
            )
        )

        super().__init__(optimizer, schedulers, milestones)


class MultiStepLR(LRScheduler):
    """
    MultiStep

    Args:
        optimizer: The optimizer to schedule
        initial_learning_rate: The initial learning rate.
        gamma: Multiplicative factor of learning rate decay.
        milestones: List of step indices. Must be increasing.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_learning_rate: float,
        gamma: float,
        milestones: List[int],
        disable_lr_steps_reset: bool = False,
    ):
        self.initial_learning_rate = initial_learning_rate
        self.gamma = gamma
        self.milestones = milestones
        super().__init__(
            optimizer, decay_steps=None, disable_lr_steps_reset=False
        )

    def _configure_cerebras_lrs(self, optimizer):
        from modelzoo.common.pytorch import cbtorch

        return cbtorch.optim.lr_scheduler.MultiStep(
            optimizer,
            self.initial_learning_rate,
            self.gamma,
            self.milestones,
            self.disable_lr_steps_reset,
        )

    def _lr_function(self, global_step):
        new_lr = torch.tensor(
            self.initial_learning_rate,
            dtype=torch.float32,
            device=global_step.device,
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


class StepLR(LRScheduler):
    """
    Step Decay

    Args:
        optimizer: The optimizer to schedule
        initial_learning_rate: The initial learning rate.
        step_size: Period of learning rate decay.
        gamma: Multiplicative factor of learning rate decay.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_learning_rate: float,
        step_size: int,
        gamma: float,
        disable_lr_steps_reset: bool = False,
    ):
        self.initial_learning_rate = float(initial_learning_rate)
        self.gamma = gamma
        self.step_size = step_size
        super().__init__(
            optimizer, decay_steps=None, disable_lr_steps_reset=False
        )

    def _configure_cerebras_lrs(self, optimizer):
        from modelzoo.common.pytorch import cbtorch

        return cbtorch.optim.lr_scheduler.Step(
            optimizer,
            self.initial_learning_rate,
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
            self.initial_learning_rate,
        )


class CosineAnnealingLR(LRScheduler):
    """
    CosineAnnealing

    Args:
        optimizer: The optimizer to schedule
        initial_learning_rate: The initial learning rate.
        T_max: Maximum number of iterations.
        eta_min: Minimum learning rate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_learning_rate: float,
        T_max: int,
        eta_min: float,
        disable_lr_steps_reset: bool = False,
    ):
        self.initial_learning_rate = float(initial_learning_rate)
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(
            optimizer, decay_steps=None, disable_lr_steps_reset=False
        )

    def _configure_cerebras_lrs(self, optimizer):
        from modelzoo.common.pytorch import cbtorch

        return cbtorch.optim.lr_scheduler.CosineAnnealing(
            optimizer,
            self.initial_learning_rate,
            self.T_max,
            self.eta_min,
            self.disable_lr_steps_reset,
        )

    def _lr_function(self, global_step):
        lr_diff = self.initial_learning_rate - self.eta_min
        a = torch.div(
            torch.mul(torch.div(global_step, self.T_max), math.pi).cos().add(1),
            2,
        )
        return torch.add(torch.mul(a, lr_diff), self.eta_min)


class LambdaLR(LRScheduler):
    """
    CosineAnnealing

    Args:
        optimizer: The optimizer to schedule
        initial_learning_rate: The initial learning rate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_learning_rate: float,
        disable_lr_steps_reset: bool = False,
    ):
        self.initial_learning_rate = initial_learning_rate
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
            optimizer, self.initial_learning_rate, self.disable_lr_steps_reset,
        )

    def _lr_function(self, global_step):
        new_lr = torch.tensor(
            1.0, dtype=torch.float32, device=global_step.device,
        )
        lr_lambda = self.set_lr_lambda()
        for lr in lr_lambda:
            new_lr = torch.mul(
                torch.mul(
                    torch.tensor(
                        self.initial_learning_rate,
                        dtype=torch.float32,
                        device=global_step.device,
                    ),
                    lr(global_step),
                ),
                new_lr,
            )
        return new_lr


class CosineAnnealingWarmRestarts(LRScheduler):
    """
    CosineAnnealingWarmRestarts

    Args:
        optimizer: The optimizer to schedule
        initial_learning_rate: The initial learning rate.
        T_0: Number of iterations for the first restart.
        T_mult: A factor increases Ti after a restart.
        eta_min: Minimum learning rate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_learning_rate: float,
        T_0: int,
        T_mult: int,
        eta_min: float,
        disable_lr_steps_reset: bool = False,
    ):
        if T_mult != 1.0:
            raise ValueError(
                f"Unsupported value of Parameters 'T_mult' for LR scheduler type CosineAnnealingWarmRestarts, "
                f"Only supported default T_mult value: 1.0 (SW-76459). "
            )
        self.initial_learning_rate = float(initial_learning_rate)
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        super().__init__(
            optimizer, decay_steps=None, disable_lr_steps_reset=False
        )

    def _configure_cerebras_lrs(self, optimizer):
        from modelzoo.common.pytorch import cbtorch

        return cbtorch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            self.initial_learning_rate,
            self.T_0,
            self.T_mult,
            self.eta_min,
            self.disable_lr_steps_reset,
        )

    def _lr_function(self, global_step):
        tensor_t_i_1 = torch.tensor(
            self.T_0, dtype=torch.float32, device=global_step.device
        )

        tensor_t_cur_1 = global_step.float()
        tensor_t_cur_2 = torch.sub(
            torch.torch.mul(
                torch.div(global_step, self.T_0).floor_(), self.T_0
            ),
            self.T_0,
        )

        tensor_t_mul = torch.tensor(
            self.T_mult, dtype=torch.float32, device=global_step.device
        )
        nn = torch.mul(
            torch.div(global_step, self.T_0), tensor_t_mul.sub(1)
        ).add(1)
        n = torch.div(torch.log(nn), torch.log(tensor_t_mul)).floor_()

        tensor_t_i_3 = torch.pow(tensor_t_mul, n).mul(self.T_0)
        tensor_t_cur_3 = torch.sub(
            global_step,
            torch.div(
                torch.pow(tensor_t_mul, n).sub(1), tensor_t_mul.sub(1)
            ).mul(self.T_0),
        ).float()

        T_i = torch.where(tensor_t_mul == 1, tensor_t_i_1, tensor_t_i_3)
        T_cur = torch.where(
            global_step < self.T_0,
            tensor_t_cur_1,
            torch.where(tensor_t_mul == 1, tensor_t_cur_2, tensor_t_cur_3),
        )
        lr_diff = self.initial_learning_rate - self.eta_min
        a = torch.div(
            torch.mul(torch.div(T_cur, T_i), math.pi).cos().add(1), 2,
        )
        return torch.add(torch.mul(a, lr_diff), self.eta_min)


class MultiplicativeLR(LRScheduler):
    """
    Multiplicative

    Args:
        optimizer: The optimizer to schedule
        initial_learning_rate: The initial learning rate.
        coefficient: Multiplicative factor of learning rate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_learning_rate: float,
        coefficient: float,
        disable_lr_steps_reset: bool = False,
    ):
        self.initial_learning_rate = initial_learning_rate
        self.coefficient = coefficient
        super().__init__(
            optimizer, decay_steps=None, disable_lr_steps_reset=False
        )

    def set_lr_lambda(self):
        lr_lambda = lambda epoch: self.coefficient
        return lr_lambda

    def _configure_cerebras_lrs(self, optimizer):
        from modelzoo.common.pytorch import cbtorch

        return cbtorch.optim.lr_scheduler.Multiplicative(
            optimizer,
            self.initial_learning_rate,
            self.coefficient,
            self.disable_lr_steps_reset,
        )

    def _lr_function(self, global_step):
        new_lr = None
        lr_lambda = self.set_lr_lambda()
        new_lr = torch.mul(
            torch.pow(
                torch.tensor(
                    lr_lambda(global_step),
                    dtype=torch.float32,
                    device=global_step.device,
                ),
                global_step,
            ),
            self.initial_learning_rate,
        )
        return new_lr


class ChainedScheduler(torch.optim.lr_scheduler.ChainedScheduler):
    """
    ChainedScheduler

    Chains list of learning rate schedulers.
    It takes a list of chainable learning rate schedulers and
    performs consecutive step() functions belonging to them by just one call.
    """

    def __init__(self, *args, **kwargs):
        if cm.use_cs():
            pass

        super().__init__(*args, **kwargs)
        self.optimizer = self._schedulers[0].optimizer
        self._init_step()

    def _init_step(self):
        if not cm.use_cs():
            scheduler = self._schedulers
            scheduler[0].last_epoch = -1
            scheduler[0]._step_count = -1
            scheduler[0].step()
            lrs = scheduler[0].get_last_lr()
            for sched in scheduler[1:]:
                sched.last_epoch = -1
                sched._step_count = 0
                sched.step()
                for idx, lr in enumerate(lrs):
                    lrs[idx] *= (
                        sched.get_last_lr()[idx] / sched.initial_learning_rate
                    )
            self._last_lr = lrs
            # Save the new rescaled lr into optimizer
            for idx, group in enumerate(
                self._schedulers[-1].optimizer.param_groups
            ):
                group['lr'] = self._last_lr[idx]

    def lr_function(self, global_step):
        if cm.use_cs():
            return self._lr_function(global_step)
        else:
            return self._lr_function(torch.tensor(global_step)).item()

    def _lr_function(self, global_step):
        new_lr = self._schedulers[0].lr_function(global_step)
        for scheduler in self._schedulers[1:]:
            new_lr = torch.mul(
                new_lr,
                torch.div(
                    scheduler.lr_function(global_step),
                    scheduler.initial_learning_rate,
                ),
            )
        return new_lr

    def step(self):
        if not cm.use_cs():
            scheduler = self._schedulers
            scheduler[0].step()
            lrs = scheduler[0].get_last_lr()
            for sched in scheduler[1:]:
                sched.step()
                for idx, lr in enumerate(lrs):
                    lrs[idx] *= (
                        sched.get_last_lr()[idx] / sched.initial_learning_rate
                    )

            self._last_lr = lrs
            # Save the new rescaled lr into optimizer
            for idx, group in enumerate(
                self._schedulers[-1].optimizer.param_groups
            ):
                group['lr'] = self._last_lr[idx]


class CyclicLR(LRScheduler):
    """
    Cyclic

    Args:
        optimizer: The optimizer to schedule.
        base_lr: Initial learning rate which is the lower boundary in the cycle.
        max_lr: Upper learning rate boundaries in the cycle.
        step_size_up: Number of training iterations in the increasing half of a cycle.
        step_size_down: Number of training iterations in the decreasing half of a cycle.
        mode: One of {triangular, triangular2, exp_range}.
        gamma: Constant in 'exp_range' scaling function: gamma**(cycle iterations).
        scale_mode: Defines whether scale_fn is evaluated on cycle number or cycle iterations.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        max_lr: float,
        step_size_up: int,
        step_size_down: int,
        mode: string,
        gamma: float,
        scale_mode: string,
        disable_lr_steps_reset: bool = False,
    ):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down
        self.mode = mode
        self.gamma = gamma
        self.scale_mode = scale_mode
        if self.step_size_down == None:
            self.step_size_down = step_size_up
        super().__init__(
            optimizer, decay_steps=None, disable_lr_steps_reset=False
        )

    def _triangular_scale_fn(self, x):
        return 1.0

    def _triangular2_scale_fn(self, x):
        return torch.div(
            torch.tensor(1, dtype=torch.float32, device=x.device),
            torch.pow(
                torch.tensor(2, dtype=torch.float32, device=x.device),
                torch.sub(x, 1),
            ),
        )

    def _exp_range_scale_fn(self, x):
        return torch.pow(
            torch.tensor(self.gamma, dtype=torch.float32, device=x.device), x
        )

    def _configure_cerebras_lrs(self, optimizer):
        from modelzoo.common.pytorch import cbtorch

        return cbtorch.optim.lr_scheduler.CyclicLR(
            optimizer,
            self.base_lr,
            self.max_lr,
            self.step_size_up,
            self.step_size_down,
            self.mode,
            self.gamma,
            self.scale_mode,
            self.disable_lr_steps_reset,
        )

    def set_scale_fn(self):
        scale_fn = None
        if self.mode == 'triangular':
            scale_fn = self._triangular_scale_fn
            self.scale_mode = 'cycle'
        elif self.mode == 'triangular2':
            scale_fn = self._triangular2_scale_fn
            self.scale_mode = 'cycle'
        else:
            scale_fn = self._exp_range_scale_fn
            self.scale_mode = 'iterations'
        return scale_fn

    def _lr_function(self, global_step):
        scale_fn = self.set_scale_fn()
        total_size = self.step_size_up + self.step_size_down
        step_ratio = self.step_size_up / total_size
        cycle = torch.floor(torch.div(global_step, total_size).add(1))
        x = torch.sub(torch.div(global_step, total_size), cycle).add(1)
        scale_factor = torch.where(
            x <= step_ratio,
            torch.div(x, step_ratio),
            torch.div(torch.sub(x, 1), torch.sub(step_ratio, 1)),
        )

        base_height = torch.mul((scale_factor), (self.max_lr - self.base_lr))
        if self.scale_mode == "cycle":
            return torch.add(
                torch.mul(base_height, scale_fn(cycle)), self.base_lr
            )
        else:
            return torch.add(
                torch.mul(base_height, scale_fn(global_step)), self.base_lr,
            )


class OneCycleLR(LRScheduler):
    """
    OneCycle

    Args:
        optimizer: The optimizer to schedule
        initial_learning_rate: Initial learning rate. Compared with PyTorch, this is equivalent to max_lr / div_factor.
        max_lr: Upper learning rate boundaries in the cycle.
        total_steps: The total number of steps in the cycle.
        pct_start: The percentage of the cycle (in number of steps) spent increasing the learning rate.
        final_div_factor: Determines the minimum learning rate via min_lr = initial_lr/final_div_factor.
        three_phase: If True, use a third phase of the schedule to annihilate the learning rate
        anneal_strategy: Specifies the annealing strategy: “cos” for cosine annealing, “linear” for linear annealing.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_learning_rate: float,
        max_lr: float,
        total_steps: int,
        pct_start: float,
        final_div_factor: float,
        three_phase: bool,
        anneal_strategy: string,
        disable_lr_steps_reset: bool = False,
    ):
        self.initial_learning_rate = initial_learning_rate
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.final_div_factor = final_div_factor
        self.three_phase = three_phase
        self.anneal_strategy = anneal_strategy
        super().__init__(
            optimizer, decay_steps=None, disable_lr_steps_reset=False
        )

    def _configure_cerebras_lrs(self, optimizer):
        from modelzoo.common.pytorch import cbtorch

        return cbtorch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.initial_learning_rate,
            self.max_lr,
            self.total_steps,
            self.pct_start,
            self.final_div_factor,
            self.three_phase,
            self.anneal_strategy,
            self.disable_lr_steps_reset,
        )

    def _annealing_cos(self, start, end, pct):
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        cos_out = torch.mul(pct, math.pi).cos().add(1)
        return torch.add(torch.mul(cos_out, ((start - end) / 2.0)), end)

    def _annealing_linear(self, start, end, pct):
        "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        return torch.add(torch.mul(pct, (end - start)), start)

    def _lr_function(self, global_step):

        min_lr = self.initial_learning_rate / self.final_div_factor
        if self.three_phase:
            milestones = [
                self.pct_start * self.total_steps - 1,
                2 * self.pct_start * self.total_steps - 2,
                self.total_steps - 1,
            ]
            lr_start = [
                self.initial_learning_rate,
                self.max_lr,
                self.initial_learning_rate,
            ]
            lr_end = [self.max_lr, self.initial_learning_rate, min_lr]
        else:
            milestones = [
                self.pct_start * self.total_steps - 1,
                self.total_steps - 1,
            ]
            lr_start = [self.initial_learning_rate, self.max_lr]
            lr_end = [self.max_lr, min_lr]

        if self.anneal_strategy == "cos":
            anneal_func = self._annealing_cos
        else:
            anneal_func = self._annealing_linear

        start_step = 0
        pct = torch.div(
            torch.sub(global_step, start_step), (milestones[0] - start_step),
        )
        lr = anneal_func(lr_start[0], lr_end[0], pct)
        start_step = milestones[0]
        for idx, milestone in enumerate(milestones[1:]):
            pct = torch.div(
                torch.sub(global_step, start_step), (milestone - start_step),
            )
            lr = torch.where(
                global_step > milestones[idx],
                anneal_func(lr_start[idx + 1], lr_end[idx + 1], pct),
                lr,
            )
            start_step = milestone
        return lr
