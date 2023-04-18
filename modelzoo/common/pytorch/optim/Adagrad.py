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


class Adagrad(CSOptimizer):
    r"""Adagrad optimizer implemented to conform to execution within the
    constraints of the Cerebras WSE.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
        maximize (bool, optional): maximize the params based on the objective,
            instead of minimizing (default: False)

    Adaptive Subgradient Methods for Online Learning and Stochastic
    Optimization: http://jmlr.org/papers/v12/duchi11a.html

    """

    def __init__(
        self,
        params,
        lr=1e-2,
        lr_decay=0,
        weight_decay=0,
        initial_accumulator_value=0,
        eps=1e-6,
        maximize: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay)
            )
        if not 0.0 <= initial_accumulator_value:
            raise ValueError(
                "Invalid initial_accumulator_value value: {}".format(
                    initial_accumulator_value
                )
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(
            lr=lr,
            lr_decay=lr_decay,
            eps=eps,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            maximize=maximize,
        )
        super(Adagrad, self).__init__(params, defaults, enable_global_step=True)

    def state_names_to_sparsify(self):
        return ["sum"]

    def preinitialize(self):
        """
        Allocates tensors for the optimizer state to allow direct compilation
        of the model before the first step.
        """
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]["sum"] = torch.full_like(
                    p,
                    group["initial_accumulator_value"],
                    memory_format=torch.preserve_format,
                    device="cpu",
                ).to(p.device)

    @torch.no_grad()
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
            lr_decay = group["lr_decay"]
            eps = group["eps"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError(
                        "Adagrad does not support sparse gradients."
                    )

                state = self.state[p]
                state_sum = state["sum"]

                global_step = self._get_global_step(p)

                grad = grad if not maximize else -grad

                if group["weight_decay"] > 0:
                    grad = grad.add(p, alpha=weight_decay)

                state_sum.addcmul_(grad, grad, value=1.0)
                std = state_sum.sqrt().add_(eps)

                grad.div_(1.0 + (global_step - 1.0) * lr_decay)
                p.addcdiv_(-lr * grad, std)

        return loss
