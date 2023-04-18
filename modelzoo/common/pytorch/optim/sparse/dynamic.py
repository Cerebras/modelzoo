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

from abc import ABC, abstractmethod

import torch
from torch.optim.optimizer import required

from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch.optim.sparse.base import BaseSparsityOptimizer
from modelzoo.common.pytorch.optim.sparse.utils import (
    make_mask_topk_sparsity,
    tiebreak_for_topk,
)


class DynamicSparsityOptimizer(BaseSparsityOptimizer, ABC):
    r"""Abstract base class for a dynamic sparsity optimizer.
        Subclasses must implement the update_sparsity function.

    Args:
        params (iterable): iterable of parameters to sparsify or dicts defining
            parameter groups to sparsify
        optimizers_and_state_names (tuple, list(tuple)): a tuple or list of
            tuple where the where the first element of the tuple is the
            optimizer and the second is a list of a optimizer state names to
            sparsify.
        sparsity_schedule (list): Ordered list of (step, sparsity) tuples

    Example:
        >>> optimizer = torch.optim.SGD(
                model.parameters(), lr=0.1, momentum=0.9
            )
        >>> sparsity_opt = DynamicSparsityOptimizer(
                [p for n,p in model.named_parameters() if should_sparsify(n,p)],
                sparsity_schedule=[(0, 0.0) (5, 0.5)],
            )
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
        >>> sparsity_opt.step()
    """

    def __init__(
        self,
        params,
        optimizers_and_state_names=[],
        sparsity_schedule=required,
        **kwargs,
    ):
        defaults = {"sparsity_schedule": sparsity_schedule, **kwargs}

        # When using CS, we execute the initial step 0 schedule and initialize
        # the masks on CPU, though during training it all happens on device:

        # |      Training Device | GPU | CS  |
        # | Operation            |           |
        # | ---------------------------------|
        # | step 0 schedule      | CPU | CPU |
        # | initial mask         | GPU | CPU |
        # | training schedule    | GPU | CS  |
        # | training mask update | GPU | CS  |

        self._step = torch.tensor(0, dtype=torch.int64)

        super().__init__(
            params=params,
            optimizers_and_state_names=optimizers_and_state_names,
            defaults=defaults,
        )
        self._step_to_param_device()

    def _step_to_param_device(self):
        """
        Transfers the step (and thus schedule computation) to the same device
        the first parameter is on.
        """
        for group in self.param_groups:
            for p in group["params"]:
                self._step = self._step.to(p.device)
                break
            break

    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        # Go verify the schedule of the just-added param_group.
        self._validate_schedule(self.param_groups[-1]["sparsity_schedule"])

    def _validate_schedule(self, schedule):
        if not schedule:
            raise ValueError(
                "sparsity_schedule must contain at least one entry"
            )
        for item in schedule:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(
                    f"sparsity_schedule be must pairs of (step, sparsity). "
                    f"Found `{item}`"
                )
        # Ensure the schedule is sorted by steps
        if schedule != sorted(schedule):
            raise ValueError("sparsity_schedule must be in sorted order.")

    def init_sparsity(self):
        # Called from __init__ via BaseSparsityOptimizer.__init__

        # Computes the initial value for all param's sparsities. If any
        # group's schedule indicates it should update for step 0, we'll use the
        # normal interface to compute the sparsity, and the p.grad will be None
        # Otherwise, the initial masks will be all ones for full-dense.
        for group in self.param_groups:
            # step==0 during init_sparsity, so use the regular schedule
            # processing to detect if we need to compute an initial sparsity
            # pattern.
            starts_sparse, sparsity = self.process_schedule(group)
            if starts_sparse:
                for p in group["params"]:
                    if cm.use_cs():
                        # On CS, we must do this intiailization on CPU
                        cpu_view = cm.to_cpu(p)
                    else:
                        # On GPU, the parameter and mask stay on device, so
                        # move sparsity to it.
                        cpu_view = p
                        sparsity = sparsity.to(p.device)

                    # Initialize sparsity pattern using the normal update API
                    self.state[p]['mask'] = self.update_mask(
                        cpu_view, None, sparsity, group
                    ).to(p.device)
            else:
                # Then we start dense
                for p in group["params"]:
                    self.state[p]['mask'] = torch.ones_like(p, dtype=torch.bool)

    def state_dict(self):
        state_dict = super(DynamicSparsityOptimizer, self).state_dict()
        state_dict["step"] = self._step
        return state_dict

    def load_state_dict(self, state_dict):
        super(DynamicSparsityOptimizer, self).load_state_dict(state_dict)

        self._step = state_dict['step']
        self._step_to_param_device()

    @abstractmethod
    @torch.no_grad()
    def update_mask(self, p, mask, sparsity, group):
        """
        Compute an updated sparsity pattern.

        Args:
            p (torch.Tensor): the parameter to sparsify
            mask (torch.tensor(dtype=torch.bool)): the current mask
                of param p
            sparsity (torch.tensor(dtype=torch.float32)): the desired
                sparsity level
            group (dict): The param group dict with any additional options
        Returns:
            The updated sparsity pattern on parameter p
        """

    def process_schedule(self, group):
        """
        Given a parameter group, determine whether we want to update sparsity
        on this step, as well as the sparsity level at this step

        Args:
            group (dict): a parameter group to sparsify. Contains schedule
                information at key "sparsity_schedule"
        Returns:
            A pair (torch.tensor(dtype=torch.bool), torch.tensor(dtype=torch.float32))
                corresponding to whether to sparsify on the current step, and the
                current sparsity level
        """
        schedule = group["sparsity_schedule"]

        # [(step, sparsity), (step, sparsity)] -->
        # torch.tensor(steps...), torch.tensor(sparsities...)
        steps, sparsities = zip(*schedule)
        steps = torch.tensor(steps, dtype=torch.int64)
        # Add an extra entry at the end equal to the last target sparsity to
        # prevent index out of bounds after the final sparsity update.
        sparsities = list(sparsities) + [sparsities[-1]]
        sparsities = torch.tensor(sparsities, dtype=torch.float32)

        if cm.use_cs() and self._step.device == cm.device():
            # WSE during training (i.e. not step 0, which takes the CPU/GPU
            # path below to evaluate schedule during initialization):
            # Ensure the schedule tensors are constant in the compute graph and
            # not passed in as runtime buffers.
            steps = cm.make_constant(steps)
            sparsities = cm.make_constant(sparsities)
        else:
            # CPU, GPU, or WSE on step 0
            # Ensure these constants are on the same device as _step otherwise:
            # RuntimeError: Expected all tensors to be on the same device, but
            # found at least two devices, cuda:0 and cpu!
            steps = steps.to(self._step.device)
            sparsities = sparsities.to(self._step.device)

        is_update_step = steps == self._step
        should_update = torch.any(is_update_step)

        # Sum the number of steps we've passed.
        next_index = torch.sum(
            (self._step > steps).int(), 0, keepdim=True, dtype=torch.int32,
        )
        # next sparsity. If should_update, the target sparsity.
        next_sparsity = sparsities.index_select(0, next_index)

        return should_update, next_sparsity

    @torch.no_grad()
    def step(self, closure=None):

        # The weights and optimizer state were just updated. In case we
        # _decrease_ sparsity here instead of increasing it, apply the current
        # sparsity pattern.
        self.apply_sparsity()

        # By convention, `step` counts number of fwd/bwd/gradient evaluations of
        # the model (`step==0` is model initialization time). If
        # `sparsity_optimizer.step()` is called after weights have been updated
        # (which is recommended), we are effectively setting up the sparsity
        # pattern for the next step. Thus, increment step here so
        # self.process_schedule can indicate if this is a step to update.
        self._step.add_(1)

        for group in self.param_groups:
            should_update, sparsity = self.process_schedule(group)
            for p in group["params"]:
                # In case there are multiple devices, ensure the sparsity is
                # on the parameter's device; it comes from the device we
                # evaluated the schedule on, usually the device of step.
                sparsity = sparsity.to(p.device)

                mask = self.state[p]['mask']

                updated_mask = self.update_mask(p, mask, sparsity, group)
                # Rewrite into the existing mask tensor for state tracking
                mask.data = torch.where(should_update, updated_mask, mask)

        self.apply_sparsity()


class GMPSparsityOptimizer(DynamicSparsityOptimizer):
    r"""Implements Gradual Magnitude Pruning https://arxiv.org/abs/1506.02626.
        Sparsity increases monotonically based on weight magnitude.

    Args:
        params (iterable): iterable of parameters to sparsify or dicts defining
            parameter groups to sparsify
        optimizers (tuple, list(tuple)): a tuple or list of tuple where the
            where the first element of the tuple is the optimzer and the second
            is a list of a optimizer state names to sparsify.
        sparsity_schedule (list): List of (step, sparsity) tuples

    Example:
        >>> optimizer = torch.optim.SGD(
                model.parameters(), lr=0.1, momentum=0.9
            )
        >>> sparsity_opt = GMPSparsityOptimizer(
                [p for n,p in model.named_parameters() if should_sparsify(n,p)],
                sparsity_schedule=[(0, 0.0) (5, 0.5)],
            )
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
        >>> sparsity_opt.step()

    """

    def __init__(
        self,
        params,
        optimizers_and_state_names=[],
        sparsity_schedule=required,
        **kwargs,
    ):
        # Set default tiebreak mechanism (don't break ties)
        if "tiebreak" not in kwargs:
            kwargs["tiebreak"] = "none"
        if "tiebreak_eps" not in kwargs:
            kwargs["tiebreak_eps"] = 1e-8
        super().__init__(
            params, optimizers_and_state_names, sparsity_schedule, **kwargs
        )

    @torch.no_grad()
    def update_mask(self, p, mask, sparsity, group):
        score = p.abs()
        tiebreak = group["tiebreak"]
        tiebreak_eps = group["tiebreak_eps"]
        if tiebreak != "none":
            score = tiebreak_for_topk(score, tiebreak, tiebreak_eps)
        return make_mask_topk_sparsity(score, sparsity)
