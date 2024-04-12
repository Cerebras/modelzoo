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
import torch.nn as nn

from cerebras.modelzoo.layers.AttentionHelper import get_attention_module
from cerebras.modelzoo.models.vision.dit.layers.vae.ResNetBlock2D import (
    ResnetBlock2D,
)


class UNetMidBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        attention_type="aiayn_attention",
        extra_attn_params=None,
    ):
        super().__init__()
        resnet_groups = (
            resnet_groups
            if resnet_groups is not None
            else min(in_channels // 4, 32)
        )
        self.add_attention = add_attention
        extra_attn_params = (
            {} if extra_attn_params is None else extra_attn_params
        )
        AttentionModule = get_attention_module(
            attention_type, extra_attn_params
        )
        self.output_scale_factor = output_scale_factor

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=self.output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []
        norms = []

        for _ in range(num_layers):
            if self.add_attention:
                group_norm = nn.GroupNorm(
                    num_channels=in_channels,
                    num_groups=resnet_groups,
                    eps=resnet_eps,
                    affine=True,
                )
                if attn_num_head_channels is not None:
                    num_heads = in_channels // attn_num_head_channels
                else:
                    num_heads = 1
                attention_layer = AttentionModule(
                    embed_dim=in_channels,
                    num_heads=num_heads,
                    inner_dim=None,
                    dropout=0.0,
                    batch_first=True,
                    attention_type="scaled_dot_product",
                    softmax_dtype_fp32=True,
                    use_projection_bias=True,
                    use_ffn_bias=True,
                )
                norms.append(group_norm)
                attentions.append(attention_layer)
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=self.output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.norms = nn.ModuleList(norms)
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None):
        hidden_states = self.resnets[0](hidden_states, temb)
        for norm, attn, resnet in zip(
            self.norms, self.attentions, self.resnets[1:]
        ):
            if attn is not None:
                residual = hidden_states
                batch, channel, height, width = hidden_states.shape

                # norm
                hidden_states = norm(hidden_states)

                # attn
                hidden_states = hidden_states.view(
                    batch, channel, height * width
                ).transpose(1, 2)

                attn_mask = torch.ones(
                    hidden_states.shape[1],
                    hidden_states.shape[1],
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                hidden_states = attn(
                    hidden_states,
                    hidden_states,
                    hidden_states,
                    attn_mask=attn_mask,
                )
                hidden_states = hidden_states.transpose(-1, -2).reshape(
                    batch, channel, height, width
                )

                # residual connection
                hidden_states = (
                    hidden_states + residual
                ) / self.output_scale_factor

            hidden_states = resnet(hidden_states, temb)

        return hidden_states
