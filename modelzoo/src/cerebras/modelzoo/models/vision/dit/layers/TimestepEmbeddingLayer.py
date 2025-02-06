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

import math

import torch
import torch.nn as nn

from cerebras.modelzoo.layers.FeedForwardNetwork import (
    FeedForwardNetwork,
    FeedForwardNetworkConfig,
)
from cerebras.modelzoo.models.vision.dit.layers.GaussianDiffusion import index


class TimestepEmbeddingLayer(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self,
        num_diffusion_steps,
        hidden_size,
        frequency_embedding_size=256,
        nonlinearity="silu",
        kernel_initializer: str = "xavier_uniform",
        bias_initializer: str = "zeros",
    ):
        super().__init__()
        self.timestep_embedding = self.create_timestep_embedding(
            seq_len=num_diffusion_steps, dim=frequency_embedding_size
        )
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.ffn = FeedForwardNetwork(
            FeedForwardNetworkConfig(
                input_unit=frequency_embedding_size,
                layers_units=[hidden_size, hidden_size],
                layers_activation=[nonlinearity, None],
                use_bias=True,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
            )
        )

        # Initialize weights and bias
        self.__reset_parameters()

    def reset_parameters(self):
        self.__reset_parameters()

    def __reset_parameters(self):
        self.ffn.reset_parameters()

    @staticmethod
    def create_timestep_embedding(seq_len, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        Slightly different than `EmbeddingLayer.create_fix_pos_embedding`.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        position = torch.arange(seq_len, dtype=torch.float32)
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        )
        args = position[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return torch.nn.Parameter(embedding, requires_grad=False)

    def forward(self, t):
        t_freq = index(self.timestep_embedding, t)
        t_emb = self.ffn(t_freq)
        return t_emb
