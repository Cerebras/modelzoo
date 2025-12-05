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

from typing import Literal, Optional

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from cerebras.modelzoo.common.utils.model.recompute_api import recompute_region
from cerebras.modelzoo.config import ModelConfig
from cerebras.modelzoo.layers.init import (
    InitializerConfig,
    TruncatedNormalInitializer,
    ZerosInitializer,
)


class DinoHeadConfig(ModelConfig):
    name: Literal["DinoHead"]
    "Name of the model. Must be set to `DinoHead`."

    input_size: int = ...
    """
    Size of the input to the MLP (multi-layer perceptron). 
    Please note: MLP is being used interchangeably with FeedForward network.
    """

    output_size: int = ...
    "Final size of output sample after the Head is applied."

    num_layers: int = 3
    """
    Number of layers in the MLP.
    Please note: MLP is being used interchangeably with FeedForward network.
    """

    hidden_size: int = 2048
    "Intermediate size of input."

    bottleneck_size: int = 256
    "Size of the activations after the MLP has been applied."

    use_bias_in_mlp: bool = True
    "If set to True, a bias term is applied inside the Linear layers of the MLP."

    norm_last_layer: bool = False
    "When true, apply layer normalization to the last layer of the MLP."

    initializer: Optional[InitializerConfig] = None
    """
    Weight Initialization to use for the Head, defaults to truncated_normal if set to None.
    See ../../../../layers/init.py for more weight initializers. 
    """

    initializer_range: float = 0.02
    """
    The standard deviation of the truncated_normal_initializer as the
    default initializer.
    """

    @property
    def __model_cls__(self):
        return DinoHead


# copied from Dino repo: https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/dino_head.py
class DinoHead(nn.Module):
    def __init__(self, config: DinoHeadConfig):
        if isinstance(config, dict):
            config = DinoHeadConfig(**config)

        super().__init__()

        num_layers = max(config.num_layers, 1)
        self.mlp = self._build_mlp(
            num_layers,
            config.input_size,
            config.bottleneck_size,
            hidden_size=config.hidden_size,
            bias=config.use_bias_in_mlp,
        )
        self.last_layer = weight_norm(
            nn.Linear(config.bottleneck_size, config.output_size, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if config.norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

        self.initializer = config.initializer
        self.initializer_range = config.initializer_range
        if self.initializer is None:
            self.initializer = TruncatedNormalInitializer(
                std=self.initializer_range,
                mean=0.0,
                a=self.initializer_range * -2.0,
                b=self.initializer_range * 2.0,
            )
        self.__reset_parameters()

    def reset_parameters(self):
        self.__reset_parameters()

    def __reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                if hasattr(layer, "weight") and hasattr(layer.weight, "data"):
                    self.initializer(layer.weight.data)
                if hasattr(layer, "bias") and hasattr(layer.bias, "data"):
                    ZerosInitializer()(layer.bias.data)

        self.last_layer.weight_g.data.fill_(1.0)

    def _build_mlp(
        self,
        num_layers,
        input_size,
        bottleneck_size,
        hidden_size=None,
        bias=True,
    ):
        if num_layers == 1:
            return nn.Sequential(
                *[nn.Linear(input_size, bottleneck_size, bias=bias)]
            )
        else:
            layers = [nn.Linear(input_size, hidden_size, bias=bias)]
            layers.append(nn.GELU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_size, hidden_size, bias=bias))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_size, bottleneck_size, bias=bias))
            return nn.Sequential(*layers)

    def forward(self, data):
        bsz, n_imgs, other_dims = (
            data.shape[0],
            data.shape[1],
            data.shape[2:],
        )  # data shape(bsz, num_images, H)

        data_dim = data.ndim
        if data_dim == 3:  # for cls token only
            output_shape = (bsz, n_imgs, -1)
        elif data_dim == 4:
            output_shape = (bsz, n_imgs, other_dims[0], -1)
        data = torch.reshape(data, (bsz * n_imgs, *other_dims))
        data = self.mlp(data)
        eps = 1e-6 if data.dtype == torch.float16 else 1e-12
        data = nn.functional.normalize(data, dim=-1, p=2, eps=eps)

        # annotate last_layer for specialized recompute schedule
        @recompute_region("ibot_patch_opt")
        def recompute_last_layer(data):
            res = self.last_layer(data)
            res = torch.reshape(res, output_shape)
            return res

        # only annotate head that has sequence dimension
        if data_dim == 4:
            return recompute_last_layer(data)
        else:
            data = self.last_layer(data)
        data = torch.reshape(data, output_shape)
        return data
