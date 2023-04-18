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
Abstract base class for Cerebras Optimizers.
"""
from abc import ABC, abstractmethod

import torch
from torch.optim import Optimizer


class CSOptimizer(Optimizer, ABC):
    """
    Cerebras Base Optimizer class
    """

    def __init__(self, params, defaults, enable_global_step=False):
        """
        Cerebras Base Optimizer class handles preinitialization of
        optimizer states for non-CS runs, making the implementation
        of the optimizer compatible with both CS and non-CS runs.
        It also preinitializes global steps tensor and provides a method
        to retrieve the global steps.
        """
        super(CSOptimizer, self).__init__(params, defaults)

        self.preinitialize()

        if enable_global_step:
            for group in self.param_groups:
                for p in group['params']:
                    self.state[p]["step"] = torch.tensor(
                        0.0, device="cpu", dtype=torch.float32
                    ).to(p.device)

        self.post_load_state_dict()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.post_load_state_dict()

    def post_load_state_dict(self):
        """
        Actions to perform after initializing state and loading the state dict
        """

        def tensor_cast(value):
            if isinstance(value, int):
                value = torch.tensor(value, dtype=torch.int32)
            elif isinstance(value, float):
                value = torch.tensor(value, dtype=torch.float32)
            elif isinstance(value, (list, tuple)):
                value = type(value)(map(tensor_cast, value))
            return value

        # Convert all python scalars in the param groups to 32 bit torch tensors
        for param_group in self.param_groups:
            keys = list(param_group.keys())
            for key in keys:
                if key == "params":
                    for p in param_group["params"]:
                        state_names = list(self.state[p].keys())
                        for name in state_names:
                            value = self.state[p].pop(name)
                            self.state[p][name] = tensor_cast(value)
                else:
                    value = param_group.pop(key)
                    param_group[key] = tensor_cast(value)

    def _get_global_step(self, p):
        """
        Increases the global steps by 1 and returns the current
        value of global step tensor in torch.float32 format.
        """
        self.state[p]["step"] += 1.0
        global_step = self.state[p]["step"]
        return global_step

    @abstractmethod
    def state_names_to_sparsify(self):
        """
        Return the names of of per-parameter states that need to be sparsified
        when applying sparsity to the underlying parameters.
        """

    @abstractmethod
    def preinitialize(self):
        """
        Allocates tensors for the optimizer state to allow direct compilation
        of the model before the first step.
        """
        raise NotImplementedError(
            "preinitialize must be implemented in a child class!"
        )

    @abstractmethod
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        raise NotImplementedError("step must be implemented in a child class!")

    def to(self, device=None):
        """Moves optimizer state onto specified device or onto corresponding
           parameter's device if no device is specified.

        Args:
            device (optional): Device to move state tensors to. If not specified,
            the corresponding parameter's device will be used.

        Returns:
            self
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                to_device = device if device is not None else p.device
                for key in state:
                    state[key] = state[key].to(to_device)
        return self
