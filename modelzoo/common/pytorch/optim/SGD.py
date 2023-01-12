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


class SGD(CSOptimizer):
    """
    SGD optimizer implemented to conform to execution within the constraints
    of the Cerebras WSE, including pre-initializing optimizer state
    """

    def __init__(
        self,
        params,
        lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        maximize=False,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                f"Nesterov momentum requires a `momentum` and zero `dampening`. "
                f"`momentum` was {momentum} and `dampening` was {dampening}."
            )

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
        )

        super(SGD, self).__init__(params, defaults)

    def preinitialize(self):
        """
        Allocates tensors for the optimizer state to allow direct compilation
        of the model before the first step.
        """
        for group in self.param_groups:
            for p in group['params']:
                if group['momentum'] != 0:
                    self.state[p]["momentum_buffer"] = torch.zeros_like(
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
            momentum = group['momentum']
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError("SGD does not support sparse gradients.")

                p_data = p.data

                grad = grad if not maximize else -grad

                if weight_decay != 0:
                    grad = grad.add(p_data, alpha=weight_decay)

                if momentum != 0:
                    buf = self.state[p]["momentum_buffer"]

                    buf.mul_(momentum).add_(grad, alpha=1.0 - dampening)

                    if nesterov:
                        grad.add_(buf, alpha=momentum)
                    else:
                        grad = buf

                p_data.add_(-lr * grad)

        return loss
