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

# https://pytorch.org/docs/stable/_modules/torch/nn/modules/normalization.html#LayerNorm
import numbers
from typing import Tuple

import torch
from torch import Tensor, nn


class BiaslessLayerNorm(nn.Module):
    r"""Applies Layer Normalization without a bias (beta) like in PaLM. Note
    that this is not the same as RMSNorm which also doesn't shift the
    distribution by considering the mean.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`\text{normalized\_shape}`.
            The values are initialized to 1.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    """

    def __init__(
        self,
        normalized_shape: Tuple[int, ...],
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

        self.weight = nn.Parameter(
            torch.empty(self.normalized_shape, **factory_kwargs)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight.data.fill_(1.0)

    def forward(self, input: Tensor) -> Tensor:
        return nn.functional.layer_norm(
            input, self.normalized_shape, self.weight, None, self.eps
        )
