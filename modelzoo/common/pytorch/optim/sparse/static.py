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

import torch
from torch.optim.optimizer import required

from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch.optim.sparse.base import BaseSparsityOptimizer
from modelzoo.common.pytorch.optim.sparse.utils import make_mask_topk_sparsity


class StaticSparsityOptimizer(BaseSparsityOptimizer):
    r"""Implements a static sparsity optimizer.

    Args:
        params (iterable): iterable of parameters to sparsify or dicts defining
            parameter groups to sparsify
        optimizers_and_state_names (tuple, list(tuple)): a tuple or list of
            tuple where the where the first element of the tuple is the
            optimizer and the second is a list of a optimizer state names to
            sparsify.
        sparsity (float): target sparsity
        init_method (str): method by which masks are initialized

    Example:
        >>> optimizer = torch.optim.SGD(
                model.parameters(), lr=0.1, momentum=0.9
            )
        >>> sparsity_opt = StaticSparsityOptimizer(
                [p for n,p in model.named_parameters() if should_sparsify(n,p)],
                sparsity=0.5,
            )
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
        >>> sparsity_opt.step()

    """

    def __init__(
        self,
        params,
        optimizers_and_state_names=[],
        sparsity=required,
        init_method='topk',
        **kwargs,
    ):
        if sparsity is not required and (
            type(sparsity) == float and not (0.0 < sparsity < 1.0)
        ):
            raise ValueError('Invalid sparsity level: {}'.format(sparsity))

        if kwargs:
            warnings.warn(f"Unused arguments: {kwargs}")

        defaults = {
            'sparsity': sparsity,
            'init_method': init_method,
        }

        super().__init__(params, optimizers_and_state_names, defaults)

    def _init_mask(self, p, sparsity, init_method):
        if init_method == 'topk':
            return make_mask_topk_sparsity(p.abs(), sparsity)

        elif init_method == 'random':
            randn = p.new_empty(p.shape).uniform_()
            return make_mask_topk_sparsity(randn, sparsity)

        elif init_method == 'from_weights':
            return p != 0

        raise NotImplementedError(
            f'Mask init method not implemented or valid: {init_method}'
        )

    def init_sparsity(self):
        for group in self.param_groups:
            for p in group['params']:
                if cm.use_cs():
                    # On CS, we must do this intiailization on CPU
                    cpu_view = cm.to_cpu(p)
                else:
                    # On GPU, the parameter and mask stay on device
                    cpu_view = p

                self.state[p]['mask'] = self._init_mask(
                    cpu_view, group['sparsity'], group['init_method']
                ).to(p.device)

    @torch.no_grad()
    def step(self, closure=None):

        # Merely apply the mask to maintain initial sparsity pattern.
        self.apply_sparsity()
