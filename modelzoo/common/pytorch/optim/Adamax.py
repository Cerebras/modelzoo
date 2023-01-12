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

from typing import Iterable, Tuple

import torch
from torch import nn

from modelzoo.common.pytorch.optim.CSOptimizer import CSOptimizer


class Adamax(CSOptimizer):
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        maximize: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
        )
        super().__init__(params, defaults)

    def preinitialize(self):
        """
        Allocates tensors for the optimizer state to allow direct compilation
        of the model before the first step.
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]

                # State initialization

                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(p, device="cpu").to(
                    p.device
                )
                # Exponential moving average of infinity norm
                state["exp_inf"] = torch.zeros_like(p, device="cpu").to(
                    p.device
                )

                beta1, _ = group["betas"]

                # beta1 ^ step, initialized for used on step 1
                state["beta1_power"] = torch.tensor(beta1).to(p.device)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError(
                        'Adamax does not support sparse gradients'
                    )

                maximize = group["maximize"]

                grad = p.grad.data
                grad = grad if not maximize else -grad

                if group["weight_decay"] > 0.0:
                    grad = grad.add(p.data, alpha=group["weight_decay"])

                state = self.state[p]

                exp_avg, exp_inf = state["exp_avg"], state["exp_inf"]
                beta1, beta2 = group["betas"]

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time.
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                torch.maximum(
                    exp_inf.mul(beta2),
                    grad.abs().add_(group["eps"]),
                    out=exp_inf,
                )
                update = exp_avg / exp_inf

                bias_correction = 1.0 - state["beta1_power"]
                update.div_(bias_correction)
                # Update `beta1^step` for the next step.
                state["beta1_power"] *= beta1

                # Scale the update by the learning rate.
                update *= group["lr"]

                # Finally, update the weight data.
                p.data.sub_(update)

        return loss
