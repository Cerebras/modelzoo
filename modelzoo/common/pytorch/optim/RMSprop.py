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


class RMSprop(CSOptimizer):
    """
    RMSprop optimizer implemented to perform the required
    pre-initialization of the optimizer state.
    """

    def __init__(
        self,
        params,
        lr=1e-2,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0,
        momentum=0,
        centered=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            alpha=alpha,
            eps=eps,
            centered=centered,
            weight_decay=weight_decay,
        )

        super(RMSprop, self).__init__(params, defaults)

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
                if group['momentum'] > 0:
                    self.state[p]["momentum_buffer"] = torch.zeros_like(
                        p, device="cpu"
                    ).to(p.device)
                if group['centered']:
                    self.state[p]["grad_avg"] = torch.zeros_like(
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
            alpha = group["alpha"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eps = group["eps"]
            centered = group["centered"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError(
                        "RMSprop does not support sparse gradients."
                    )

                p_data = p.data
                state = self.state[p]
                square_avg = state["square_avg"]

                if weight_decay != 0:
                    grad = grad.add(p_data, alpha=weight_decay)

                square_avg.mul_(alpha).addcmul_(grad, grad, value=1.0 - alpha)

                if centered:
                    grad_avg = state["grad_avg"]
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    avg = (
                        square_avg.addcmul(grad_avg, grad_avg, value=-1.0)
                        .sqrt_()
                        .add_(eps)
                    )
                else:
                    avg = square_avg.sqrt().add_(eps)

                if momentum > 0.0:
                    momentum_buffer = state["momentum_buffer"]
                    momentum_buffer.mul_(momentum).addcdiv_(grad, avg)
                    p_data.add_(-lr * momentum_buffer)
                else:
                    p_data.addcdiv_(-lr * grad, avg)

        return loss
