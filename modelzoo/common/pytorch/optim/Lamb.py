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
    Copyright cybertonai and Cerebras, see LICENSE_LambOptimizer
"""

import torch

from modelzoo.common.pytorch.optim.CSOptimizer import CSOptimizer


class Lamb(CSOptimizer):
    r"""Implements Lamb algorithm.
    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.
    
    .. _Large Batch Optimization for Deep Learning\: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0,
        adam=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1])
            )
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, adam=adam
        )

        super(Lamb, self).__init__(params, defaults)

    def state_names_to_sparsify(self):
        return ["exp_avg", "exp_avg_sq"]

    def preinitialize(self):
        """
        Allocates tensors for the optimizer state to allow direct compilation
        of the model before the first step.
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p, device="cpu").to(
                    p.device
                )
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p, device="cpu").to(
                    p.device
                )

    @torch.no_grad()
    def step(self, closure=None):
        r"""Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        'Lamb does not support sparse gradients, consider SparseAdam instad.'
                    )

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Paper v3 does not use debiasing.
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                # Apply bias to lr to avoid broadcast.
                step_size = group[
                    'lr'
                ]  # * math.sqrt(bias_correction2) / bias_correction1

                weight_norm = p.pow(2).sum().sqrt().clamp(0, 10).to(torch.float)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(p, alpha=group['weight_decay'])

                adam_norm = adam_step.pow(2).sum().sqrt().to(torch.float)
                # pytorch version for future reference (we don't support weight_norm == 0 or adam_norm == 0)
                # if weight_norm == 0 or adam_norm == 0:
                #     trust_ratio = 1
                # else:
                #     trust_ratio = weight_norm / adam_norm
                zero = torch.tensor(
                    0.0, dtype=torch.float32, device=weight_norm.device
                )
                trust_ratio = torch.where(
                    torch.gt(weight_norm, zero),
                    torch.where(
                        torch.gt(adam_norm, zero),
                        weight_norm / adam_norm,
                        torch.tensor(
                            1.0, dtype=torch.float32, device=weight_norm.device
                        ),
                    ),
                    torch.tensor(
                        1.0, dtype=torch.float32, device=weight_norm.device
                    ),
                )
                if group['adam']:
                    trust_ratio = 1

                update_step = adam_step.mul(trust_ratio)
                p.sub_(update_step * step_size)

        return loss
