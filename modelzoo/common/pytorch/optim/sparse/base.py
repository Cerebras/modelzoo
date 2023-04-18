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

import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager

import torch
from torch.optim.optimizer import Optimizer


class BaseSparsityOptimizer(Optimizer, ABC):
    r"""Abstract base class for a dynamic sparsity optimizer.
        Subclasses must implement the init_sparsity and step function.

    Args:
        params (iterable): iterable of parameters to sparsify or dicts defining
            parameter groups to sparsify
        optimizers_and_state_names (tuple, list(tuple)): a tuple or list of
            tuple where the where the first element of the tuple is the
            optimizer and the second is a list of a optimizer state names to
            sparsify.
        defaults (dict): Defaults for param_groups

    """

    def __init__(
        self, params, optimizers_and_state_names=[], defaults={}, **kwargs
    ):
        super().__init__(params, defaults)

        if optimizers_and_state_names:
            if not isinstance(optimizers_and_state_names, list):
                optimizers_and_state_names = [optimizers_and_state_names]
            for opt in optimizers_and_state_names:
                assert isinstance(
                    opt, tuple
                ), f'Expected type: tuple or list of tuples'
                o, s = opt
                assert isinstance(o, Optimizer)
                for _s in s:
                    assert isinstance(_s, str)
            self.optimizers_and_state_names = optimizers_and_state_names
        else:
            warnings.warn(f'sparsity will not be applied to optimizer state')
            self.optimizers_and_state_names = []

        self.init_sparsity()

        # Verify sparsity was initialized for every parameter and move the mask
        # to the parameter device just in case.
        for group in self.param_groups:
            for p in group["params"]:
                assert (
                    'mask' in self.state[p]
                ), "sparsity was not initialized for all parameters"
                mask = self.state[p]['mask']
                if mask.device != p.device:
                    self.state[p]['mask'] = mask.to(p.device)

    @abstractmethod
    def init_sparsity(self):
        """
        Compute the initial sparsity pattern for each of the parameters.
        """

    def state_dict(self):
        # Adapted from torch.optim.Optimizer, but we use param_names

        # param_names used in place of params
        param_groups = []

        # map parameter -> name
        name_map = {}
        for group in self.param_groups:
            name_map.update(dict(zip(group["params"], group["param_names"])))
            group = group.copy()
            del group["params"]
            param_groups.append(group)

        state = {name_map[p]: v for p, v in self.state.items()}

        return {"state": state, "param_groups": param_groups}

    def load_state_dict(self, state_dict):
        # Adapted from torch.optim.Optimizer, but we use param_names

        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError(
                "loaded state dict has a different number of "
                "parameter groups"
            )

        # map name -> parameter
        name_map = {}
        for group in self.param_groups:
            name_map.update(dict(zip(group["param_names"], group["params"])))

        for group, saved_group in zip(groups, saved_groups):
            if group["param_names"] != saved_group["param_names"]:
                raise ValueError(
                    "loaded state dict contains different parameters than "
                    "the current optimizer"
                )

        def to_device(param, value, key=None):
            r"""Make a deep copy of value, transferring all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                return value.to(param.device)
            elif isinstance(value, dict):
                return {k: to_device(param, v, key=k) for k, v in value.items()}
            elif isinstance(value, abcs.Iterable):
                return type(value)(to_device(param, v) for v in value)
            else:
                return value

        # Copy state associated with params (moving tensors to param device).
        state = defaultdict(dict)
        for param_name, v in state_dict['state'].items():
            param = name_map[param_name]
            state[param] = to_device(param, v)

        # Update parameter groups, resetting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group

        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)
        ]
        self.__setstate__({'state': state, 'param_groups': param_groups})

    def add_param_group(self, param_group):
        # SparsityOptimizer accepts named_params tuples instead
        named_params = param_group["params"]
        if isinstance(named_params, list):
            # list of tuples
            names, params = zip(*named_params)
        elif isinstance(named_params, tuple):
            # single tuple
            names, params = named_params
            params = [params]
            names = [names]

        param_group["params"] = params
        param_group["param_names"] = names
        super().add_param_group(param_group)

    @torch.no_grad()
    def apply_param_sparsity(self):
        """
        Apply the sparsity pattern to the parameters.
        """
        self._apply_masks_to_params()

    @torch.no_grad()
    def apply_grad_sparsity(self):
        """
        Apply the sparsity pattern to the gradients of the parameters.
        """
        self._apply_masks_to_grads()

    @torch.no_grad()
    def apply_sparsity(self):
        """
        Apply the sparsity pattern to the parameters and optimizer states.
        """
        self._apply_masks_to_params()
        self._apply_masks_to_opt_state()

    @torch.no_grad()
    @contextmanager
    def sparse_params(self):
        """
        Context manager applying sparsity to params upon entry and sparsity to
        gradients upon exit.
        """
        self.apply_param_sparsity()
        yield
        self.apply_grad_sparsity()

    @torch.no_grad()
    def _apply_masks_to_params(self):
        for group in self.param_groups:
            for p in group['params']:
                p.mul_(self.state[p]['mask'])

    @torch.no_grad()
    def _apply_masks_to_grads(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    mask = self.state[p]['mask']
                    # In the case there any NaNs in the unused gradients that
                    # correspond to zero'd out weights, we use a selection to
                    # replace these NaNs with zeros. (multiplying with the mask
                    # would preserve them).
                    # DLS will skip a weight update if there is a NaN in the
                    # gradient, but we only want this to happen if there is a
                    # NaN in gradients corresponding to non-zero weights. This
                    # is the behavior of the CS2 which doesn't even compute the
                    # full gradients on most steps.
                    zero = torch.zeros_like(p.grad)
                    p.grad.data = torch.where(mask, p.grad, zero)

    @torch.no_grad()
    def _apply_masks_to_opt_state(self):
        for opt, opt_states_to_mask in self.optimizers_and_state_names:
            for p in self.state:
                if p in opt.state:
                    for s_name in opt_states_to_mask:
                        if s_name in opt.state[p]:
                            opt.state[p][s_name].mul_(self.state[p]['mask'])
