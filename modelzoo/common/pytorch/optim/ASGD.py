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
Cerebras implementation of ASGD optimizer. Adapted from the `torch.optim.ASGD`
implementation.
"""

import torch

from modelzoo.common.pytorch.optim.CSOptimizer import CSOptimizer


class ASGD(CSOptimizer):
    """
    ASGD optimizer implemented to conform to execution within the constraints
    of the Cerebras WSE, including pre-initializing optimizer state.

    For more details, see https://dl.acm.org/citation.cfm?id=131098
    """

    def __init__(
        self,
        params,
        lr=1e-2,
        lambd=1e-4,
        alpha=0.75,
        t0=1e6,
        weight_decay=0,
        maximize: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )

        defaults = dict(
            lr=lr,
            lambd=lambd,
            alpha=alpha,
            t0=t0,
            weight_decay=weight_decay,
            maximize=maximize,
        )
        super(ASGD, self).__init__(params, defaults, enable_global_step=True)

    def preinitialize(self):
        """
        Allocates tensors for the optimizer state to allow direct compilation
        of the model before the first step.
        """
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["eta"] = torch.tensor(group["lr"]).to(p.device)
                self.state[p]["mu"] = torch.tensor(1.0).to(p.device)
                self.state[p]["ax"] = torch.zeros_like(p, device="cpu").to(
                    p.device
                )

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
            Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lambd = group["lambd"]
            lr = group["lr"]
            t0 = group["t0"]

            for p in group["params"]:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            "ASGD does not support sparse gradients"
                        )
                    alpha = torch.tensor(
                        group["alpha"], dtype=torch.float32, device=p.device
                    )
                    state = self.state[p]
                    grad = p.grad
                    grad = grad if not group["maximize"] else -grad
                    mu = state["mu"]
                    ax = state["ax"]
                    eta = state["eta"]
                    step = self._get_global_step(p)

                    if group["weight_decay"] != 0:
                        grad = grad.add(p, alpha=group["weight_decay"])

                    # decay term
                    p.mul_(1 - lambd * eta)

                    # update parameter
                    p.add_(grad * eta.neg())

                    # averaging
                    new_ax = torch.where(mu == 1, p, ax.add(p.sub(ax).mul(mu)))
                    ax.copy_(new_ax)

                    new_eta = lr / torch.pow(1 + lambd * lr * step, alpha)
                    eta.copy_(new_eta)

                    new_mu = 1 / torch.maximum(
                        torch.ones(size=[], dtype=mu.dtype),
                        torch.tensor(step - t0, dtype=mu.dtype),
                    )
                    mu.copy_(new_mu)

        return loss
