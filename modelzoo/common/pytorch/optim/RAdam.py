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
from modelzoo.common.pytorch.utils import to_tensor


class RAdam(CSOptimizer):
    r"""RAdam optimizer implemented to conform to execution within the
    constraints of the Cerebras WSE.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0]"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0]"
            )
        if not 0.0 <= weight_decay:
            raise ValueError(
                f"Invalid weight_decay value: {weight_decay} - should be >= 0.0"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,)
        super().__init__(params, defaults, enable_global_step=True)

    def state_names_to_sparsify(self):
        return ["exp_avg", "exp_avg_sq"]

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
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(p, device="cpu").to(
                    p.device
                )

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            beta1, beta2 = group['betas']
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group['params']:
                if p.grad is not None:

                    grad = p.grad
                    state = self.state[p]
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    global_step = self._get_global_step(p)

                    if group["weight_decay"] > 0.0:
                        grad = grad.add(p, alpha=group["weight_decay"])

                    beta1t = torch.pow(
                        to_tensor(beta1).to(p.device), global_step
                    )
                    beta2t = torch.pow(
                        to_tensor(beta2).to(p.device), global_step
                    )

                    bias_correction1 = 1 - beta1t
                    bias_correction2 = 1 - beta2t

                    # Decay the first and second moment running average coefficient
                    # In-place operations to update the averages at the same time.
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(
                        grad, grad, value=1.0 - beta2
                    )

                    # correcting bias for the first moving moment
                    update = exp_avg / bias_correction1

                    # maximum length of the approximated SMA
                    rho_inf = 2 / (1 - beta2) - 1
                    # compute the length of the approximated SMA
                    rho_t = (
                        rho_inf - 2 * global_step * beta2t / bias_correction2
                    )

                    one = to_tensor(1.0).to(p.device)
                    five = to_tensor(5.0).to(p.device)

                    # Compute the variance rectification term and update parameters accordingly
                    rect = torch.where(
                        torch.gt(rho_t, five),
                        torch.sqrt(
                            (rho_t - 4.0)
                            * (rho_t - 2.0)
                            * rho_inf
                            / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)
                        ),
                        one,
                    )
                    adaptive_lr = torch.where(
                        torch.gt(rho_t, five),
                        torch.sqrt(bias_correction2)
                        / exp_avg_sq.sqrt().add_(eps),
                        one,
                    )

                    update *= rect
                    update *= adaptive_lr

                    update *= group["lr"]

                    p.sub_(update)

        return loss
