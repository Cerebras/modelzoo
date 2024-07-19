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
from typing import Callable, List, Optional, Union

import torch
from torch import Tensor, nn

import cerebras.pytorch as cstorch
import cerebras.pytorch.nn.functional as F
from cerebras.modelzoo.common.half_dtype import maybe_to_half_dtype
from cerebras.modelzoo.layers.activations import (
    get_activation,
    is_glu_activation,
)
from cerebras.modelzoo.layers.create_initializer import create_initializer
from cerebras.modelzoo.layers.FeedForwardNetwork import (
    FeedForwardNetwork,
    FeedForwardNetworkConfig,
)


class TopKExpertsSparseLinear(nn.Module):
    """Fused linear layer of top-k sparse experts.

    Args:
        num_experts (int): Number of experts.
        in_features (int): Size of each input sample.
        out_features (int): Size of each input sample.
        bias (bool): If set to `False`, the layer will not learn an additive bias.
            Default: `True`.
        device (torch.device): Optional device. Default: `None`.
        dtype (torch.dtype): Optional dtype. Default: `None`.

    Shapes:
        input: (batch_size, seq_len, top_k, hidden_in)
        topk_probs: (batch_size, seq_len, top_k)
        output: (batch_size, seq_len, top_k, hidden_out)
    """

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.expert_weights = nn.Parameter(
            torch.empty(
                (out_features, num_experts, in_features), **factory_kwargs
            )
        )
        if bias:
            self.expert_biases = nn.ParameterList(
                [
                    torch.empty(out_features, **factory_kwargs)
                    for _ in range(num_experts)
                ]
            )
        else:
            self.register_parameter('expert_biases', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.expert_weights, -stdv, stdv)
        if self.expert_biases is not None:
            for bias in self.expert_biases:
                torch.nn.init.uniform_(bias, -stdv, stdv)

    def forward(
        self, input: torch.Tensor, topk_indices: torch.Tensor
    ) -> torch.Tensor:
        input = maybe_to_half_dtype(input)
        expert_weights = maybe_to_half_dtype(self.expert_weights)

        output = F.sparse_matmul(input, topk_indices, expert_weights)

        B, S, K, H = output.shape
        ones = torch.ones_like(topk_indices.float())
        zeros = torch.zeros_like(topk_indices.float())
        if self.expert_biases:
            expert_biases = [
                maybe_to_half_dtype(val) for val in self.expert_biases
            ]
            for i, bias in enumerate(expert_biases):
                bias = bias[None, None, :].expand(B, S, -1)
                bias = bias[..., None, :].expand(-1, -1, K, -1)
                indices_offset = (
                    topk_indices.float()
                    - torch.tensor(i, device=output.device).float()
                )
                mask = torch.where(indices_offset == 0, ones, zeros)
                mask = mask.to(output.dtype)
                mask = mask[..., None].broadcast_to(output.shape)
                output += bias * mask

        return output


class TopKExpertsSparseSingleFeedForwardLayer(nn.Module):
    """Fused single feed forward layer of top-k sparse experts.

    Args:
        num_experts (int): Number of experts.
        in_features (int): Size of each input sample.
        out_features (int): Size of each input sample.
        activation (str/Callable): Optional activation. Default: `None`.
        dropout (float): Optional dropout probability. Default: `None`.
        use_bias (bool): If set to `False`, the layer will not learn an additive bias.
            Default: `True`.
        device (torch.device): Optional device. Default: `None`.

    Shapes:
        input: (batch_size, seq_len, top_k, hidden_in)
        topk_indices: (batch_size, seq_len, top_k)
        output: (batch_size, seq_len, top_k, hidden_out)
    """

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        activation: Optional[Union[str, Callable[[Tensor], Tensor]]] = None,
        dropout: Optional[float] = None,
        use_bias: Optional[bool] = False,
        device=None,
    ):
        super().__init__()
        self.linear_layer = TopKExpertsSparseLinear(
            num_experts, in_features, out_features, use_bias, device
        )

        self.is_glu_activation = is_glu_activation(activation)
        if self.is_glu_activation:
            self.linear_layer_for_glu = TopKExpertsSparseLinear(
                num_experts, in_features, out_features, use_bias, device
            )

        self.act_layer = None
        if activation:
            self.act_layer = get_activation(activation)

        self.dropout_layer = None
        if dropout and dropout > 0.0:
            self.dropout_layer = nn.Dropout(p=dropout)

    def forward(
        self, input: torch.Tensor, topk_indices: torch.Tensor
    ) -> torch.Tensor:
        if self.is_glu_activation:
            glu_component_1 = self.linear_layer(input, topk_indices)
            glu_component_2 = self.linear_layer_for_glu(input, topk_indices)
            output = self.act_layer(glu_component_1, glu_component_2)
        else:
            output = self.linear_layer(input, topk_indices)
            if self.act_layer:
                output = self.act_layer(output)

        if self.dropout_layer:
            output = self.dropout_layer(output)

        return output


class TopKExpertsSparseFeedForwardNetwork(nn.Module):
    """Fused feed forward network of top-k sparse experts.

    Args:
        config (FeedForwardNetworkConfig): Feed forward network config.

    Shapes:
        input: (batch_size, seq_len, hidden_in)
        topk_probs: (batch_size, seq_len, top_k)
        topk_indices: (batch_size, seq_len, top_k)
        output: (batch_size, seq_len, hidden_out)
    """

    def __init__(self, config: FeedForwardNetworkConfig) -> None:
        super().__init__()
        self.config = config
        self.fused_ffns = nn.ModuleList(
            [
                TopKExpertsSparseSingleFeedForwardLayer(
                    num_experts=self.config.num_experts,
                    in_features=in_features,
                    out_features=out_features,
                    activation=activation,
                    dropout=dropout,
                    use_bias=self.config.use_bias,
                    device=self.config.device,
                )
                for in_features, out_features, activation, dropout in zip(
                    self.config.input_units,
                    self.config.layers_units,
                    self.config.layers_activation,
                    self.config.layers_dropout_rates,
                )
            ]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer_num, linear_layer_module in enumerate(self.fused_ffns):
            weight_initializer = create_initializer(
                self.config.kernel_initializer
            )
            if layer_num == self.config.num_dense_layers - 1:
                weight_initializer = create_initializer(
                    self.config.output_layer_initializer
                )
            # Initialize linear layer weights associated with the
            # 'GLU' type activation function with the kernel_initializer
            if hasattr(linear_layer_module, 'linear_layer_for_glu'):
                weight_initializer(
                    linear_layer_module.linear_layer_for_glu.expert_weights.data
                )
            weight_initializer(
                linear_layer_module.linear_layer.expert_weights.data
            )
            if self.config.use_bias:
                bias_initializer = create_initializer(
                    self.config.bias_initializer
                )
                for bias in linear_layer_module.linear_layer.expert_biases:
                    bias_initializer(bias.data)

    def forward(
        self,
        input: torch.Tensor,
        topk_probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        input = maybe_to_half_dtype(input)
        topk_probs = maybe_to_half_dtype(topk_probs)

        output = input[:, :, None, :].expand(-1, -1, self.config.top_k, -1)
        for ffn_layer in self.fused_ffns:
            output = ffn_layer(output, topk_indices)

        output = output * topk_probs[..., None]
        output = output.sum(-2, dtype=output.dtype)
        return output


class ExpertsFeedForwardNetwork(nn.Module):
    """Functional dense implementation of n-Experts feed forward network.

    Args:
        config (FeedForwardNetworkConfig): Feed forward network config.

    Shapes:
        input: (batch_size, seq_len, hidden_in)
        selected_probs: top_k * (batch_size, seq_len, 1)
        selected_experts: top_k * (batch_size, seq_len, 1)
        output: (batch_size, seq_len, hidden_out)
    """

    def __init__(self, config: FeedForwardNetworkConfig) -> None:
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [FeedForwardNetwork(config) for _ in range(self.config.num_experts)]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for expert in self.experts:
            expert.reset_parameters()

    def forward(
        self,
        input: torch.Tensor,
        selected_probs: List[torch.Tensor],
        selected_experts: List[torch.Tensor],
    ) -> torch.Tensor:
        input = maybe_to_half_dtype(input)
        selected_probs = [maybe_to_half_dtype(val) for val in selected_probs]
        selected_experts = [val.to(torch.int16) for val in selected_experts]

        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(input))

        output = torch.zeros_like(input)
        for i in range(self.config.num_experts):
            curr_expert = torch.tensor(
                i, device=input.device, dtype=torch.int16
            )
            for j in range(self.config.top_k):
                output += (
                    selected_probs[j]
                    * expert_outputs[i]
                    * (selected_experts[j] == curr_expert)
                )

        return output


class SparseMoEBlock(nn.Module):
    """
    Sparsely-gated Mixture-of-Expert block, that can be used as drop-in
    for FeedForwardNetwork.

    Args:
        config (FeedForwardNetworkConfig): Feed forward network config.
    """

    def __init__(self, config: FeedForwardNetworkConfig):
        super(SparseMoEBlock, self).__init__()
        self.config = config
        assert (
            self.config.num_experts > 1
        ), "expected num_experts > 1, but got {self.config.num_experts=}"

        self.gate = nn.Linear(
            self.config.input_unit, self.config.num_experts, bias=False
        )
        if self.config.moe_optimized_impl():
            self.experts = TopKExpertsSparseFeedForwardNetwork(self.config)
        else:
            self.experts = ExpertsFeedForwardNetwork(self.config)

        self.reset_parameters()

    def reset_parameters(self):
        weight_initializer = create_initializer(self.config.gate_initializer)
        weight_initializer(self.gate.weight.data)
        self.experts.reset_parameters()

    def forward(self, x, **extra_args):
        x = maybe_to_half_dtype(x)
        maybe_half_dtype = x.dtype

        if self.config.routing_algorithm == "hash":
            assert extra_args.get("expert_hash_idx", None) is not None
            expert_hash_idx = extra_args["expert_hash_idx"]
            routing_weights = cstorch.nn.functional.one_hot(
                expert_hash_idx.to(torch.int64),
                num_classes=self.config.num_experts,
            ).to(x.dtype)
            routing_weights_fp32 = routing_weights.float()
        elif self.config.routing_algorithm == "learned":
            router_logits = self.gate(x)
            if self.config.router_fp32:
                router_logits = router_logits.float()
            routing_weights_fp32 = nn.functional.softmax(router_logits, dim=-1)
            routing_weights = routing_weights_fp32.to(maybe_half_dtype)
        else:
            raise ValueError(
                f'Unknown MoE routing algorithm: {self.config.routing_algorithm}'
            )

        if self.config.moe_optimized_impl():
            topk_probs, topk_indices = routing_weights.topk(self.config.top_k)
            denom = torch.sum(topk_probs, dim=-1).to(x.dtype)
            topk_probs /= denom[..., None]
            output = self.experts(x, topk_probs, topk_indices)

            expert_mask_shape = *topk_probs.shape[:-1], self.config.num_experts
            expert_mask = torch.zeros(
                expert_mask_shape, device=x.device, dtype=x.dtype
            )
            expert_mask.scatter_(2, topk_indices, 1)

            return output, routing_weights_fp32, expert_mask.float()

        selected_experts = []
        selected_probs = []
        routing_weights_top_k = torch.clone(routing_weights)
        expert_mask = torch.zeros_like(routing_weights)
        for k in range(self.config.top_k):
            selected_prob, selected_expert = torch.max(
                routing_weights_top_k, dim=-1, keepdim=True
            )
            selected_experts.append(selected_expert)
            selected_probs.append(selected_prob)
            # Mask the current top value to find the next top value
            expert_mask += cstorch.nn.functional.one_hot(
                torch.argmax(routing_weights_top_k, dim=-1),
                num_classes=self.config.num_experts,
            ).to(maybe_half_dtype)
            routing_weights_top_k = torch.where(
                expert_mask == 0, routing_weights_top_k, 0
            )

        # Normalize Top-K weights
        denom = selected_probs[0]
        for k in range(1, self.config.top_k):
            denom = denom + selected_probs[k]

        for k in range(self.config.top_k):
            selected_probs[k] = selected_probs[k] / denom

        output = self.experts(x, selected_probs, selected_experts)

        return output, routing_weights_fp32, expert_mask.float()
