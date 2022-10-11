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

import torch

from modelzoo.common.pytorch.optim.CSOptimizer import CSOptimizer


class Adadelta(CSOptimizer):
    """
    Adadelta optimizer implemented to perform the required
    pre-initialization of the optimizer state.
    """

    def __init__(
        self,
        params,
        lr=1.0,
        rho=0.9,
        eps=1e-6,
        weight_decay=0,
        maximize: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= rho <= 1.0:
            raise ValueError("Invalid rho value: {}".format(rho))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )

        defaults = dict(
            lr=lr,
            rho=rho,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
        )
        super(Adadelta, self).__init__(params, defaults)

    def preinitialize(self):
        """
        Allocates tensors for the optimizer state to allow direct compilation
        of the model before the first step.
        """
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]["square_avg"] = torch.zeros_like(
                    p, device="cpu"
                ).to(p.device)
                self.state[p]["acc_delta"] = torch.zeros_like(
                    p, device="cpu"
                ).to(p.device)

    def step(self, closure=None):
        """Performs a single optimization step.

            Args:
                closure (callable, optional): A closure that reevaluates the model
                    and returns the loss.
            """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            rho = group['rho']
            eps = group["eps"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError(
                        "Adadelta does not support sparse gradients."
                    )

                state = self.state[p]
                square_avg = state["square_avg"]
                acc_delta = state["acc_delta"]

                p_data = p.data

                grad = grad if not maximize else -grad

                if weight_decay != 0:
                    grad = grad.add(p_data, alpha=weight_decay)

                square_avg.mul_(rho).addcmul_(grad, grad, value=1 - rho)
                std = square_avg.add(eps).sqrt_()
                delta = acc_delta.add(eps).sqrt_().div_(std).mul_(grad)
                acc_delta.mul_(rho).addcmul_(delta, delta, value=1 - rho)
                p_data.add_(-lr * delta)

        return loss
