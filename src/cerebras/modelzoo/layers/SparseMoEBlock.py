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

import copy
import math
from typing import Callable, List, Optional, Union

import torch
from torch import Tensor, nn

import cerebras.pytorch as cstorch
import cerebras.pytorch.nn.functional as F
from cerebras.modelzoo.common.half_dtype import maybe_to_half_dtype
from cerebras.modelzoo.common.utils.model.mixture_of_experts_api import (
    expert_annotation,
)
from cerebras.modelzoo.layers.activations import (
    get_activation,
    is_glu_activation,
)
from cerebras.modelzoo.layers.create_initializer import create_initializer
from cerebras.modelzoo.layers.FeedForwardNetwork import (
    FeedForwardNetwork,
    FeedForwardNetworkConfig,
)

INVALID_INDEX = -1


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
            self.expert_biases = nn.Parameter(
                torch.empty((out_features, num_experts, 1), **factory_kwargs)
            )
        else:
            self.register_parameter('expert_biases', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.expert_weights, -stdv, stdv)
        if self.expert_biases is not None:
            torch.nn.init.uniform_(self.expert_biases, -stdv, stdv)

    def forward(
        self, input: torch.Tensor, topk_indices: torch.Tensor
    ) -> torch.Tensor:
        input = maybe_to_half_dtype(input)
        expert_weights = maybe_to_half_dtype(self.expert_weights)

        output = F.sparse_matmul(
            input, topk_indices.to(torch.int64), expert_weights
        )

        B, S, K, _ = output.shape
        ones = torch.ones(B, S, K, 1, device=input.device, dtype=input.dtype)
        if self.expert_biases is not None:
            expert_biases = maybe_to_half_dtype(self.expert_biases)
            output += F.sparse_matmul(
                ones, topk_indices.to(torch.int64), expert_biases
            )

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
        # For MoE layers, shared experts are also added as experts
        # We dont use this in routing logic, but they process as regular experts
        num_shared_experts = (
            self.config.moe_params.num_shared_experts
            if self.config.moe_params.num_shared_experts is not None
            else 0
        )
        self.num_experts_all = (
            self.config.moe_params.num_experts + num_shared_experts
        )
        self.top_k_all = self.config.moe_params.top_k + num_shared_experts
        self.fused_ffns = nn.ModuleList(
            [
                TopKExpertsSparseSingleFeedForwardLayer(
                    num_experts=self.num_experts_all,
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
                bias_initializer(
                    linear_layer_module.linear_layer.expert_biases.data
                )

    def forward(
        self,
        input: torch.Tensor,
        topk_probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        input = maybe_to_half_dtype(input)
        topk_probs = maybe_to_half_dtype(topk_probs)

        output = input[:, :, None, :].expand(-1, -1, self.top_k_all, -1)
        for ffn_layer in self.fused_ffns:
            output = ffn_layer(output, topk_indices)

        output = output * topk_probs[..., None]
        output = output.sum(-2, dtype=output.dtype)
        return output


class SparseActLinear(nn.Module):
    """Token-level sparse activation layer.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each input sample.
        bias (bool): If set to `False`, the layer will not learn an additive bias.
            Default: `True`.
        device (torch.device): Optional device. Default: `None`.
        dtype (torch.dtype): Optional dtype. Default: `None`.

    Shapes:
        input: (batch_size, seq_len, hidden_in)
        act_mask: (batch_size, seq_len)
        output: (batch_size, seq_len, hidden_out)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, **factory_kwargs)
            )
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.weight, -stdv, stdv)
        if self.bias is not None:
            torch.nn.init.uniform_(self.bias, -stdv, stdv)

    def forward(
        self, input: torch.Tensor, act_mask: torch.Tensor
    ) -> torch.Tensor:
        input = maybe_to_half_dtype(input)
        weight = maybe_to_half_dtype(self.weight)

        output = F.sparse_act_matmul(input, act_mask.to(torch.int16), weight)

        B, S, H = output.shape
        if self.bias is not None:
            bias = maybe_to_half_dtype(self.bias)
            bias = bias[None, None, :].expand(B, S, -1)
            mask = act_mask.to(output.dtype)
            mask = mask[..., None].broadcast_to(output.shape)
            output += bias * mask

        return output


class SparseActSingleFeedForwardLayer(nn.Module):
    """Token-level sparse activation single feed forward layer.

    Args:
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
        in_features: int,
        out_features: int,
        activation: Optional[Union[str, Callable[[Tensor], Tensor]]] = None,
        dropout: Optional[float] = None,
        use_bias: Optional[bool] = False,
        device=None,
    ):
        super().__init__()
        self.linear_layer = SparseActLinear(
            in_features, out_features, use_bias, device
        )

        self.is_glu_activation = is_glu_activation(activation)
        if self.is_glu_activation:
            self.linear_layer_for_glu = SparseActLinear(
                in_features, out_features, use_bias, device
            )

        self.act_layer = None
        if activation:
            self.act_layer = get_activation(activation)

        self.dropout_layer = None
        if dropout and dropout > 0.0:
            self.dropout_layer = nn.Dropout(p=dropout)

    def forward(
        self, input: torch.Tensor, act_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.is_glu_activation:
            glu_component_1 = self.linear_layer(input, act_mask)
            glu_component_2 = self.linear_layer_for_glu(input, act_mask)
            output = self.act_layer(glu_component_1, glu_component_2)
        else:
            output = self.linear_layer(input, act_mask)
            if self.act_layer:
                output = self.act_layer(output)

        if self.dropout_layer:
            output = self.dropout_layer(output)

        return output


class SparseActFeedForwardNetwork(nn.Module):
    """Feed forward network of top-k sparse experts. Composed of primitives.

    Args:
        config (FeedForwardNetworkConfig): Feed forward network config.

    Shapes:
        input: (batch_size, seq_len, hidden_in)
        topk_probs: (batch_size, seq_len, top_k)
        topk_indices: (batch_size, seq_len, top_k)
        output: (batch_size, seq_len, hidden_out)

    Conditions:
        Each token can only be routed to one expert once.
    """

    def __init__(self, config: FeedForwardNetworkConfig) -> None:
        super().__init__()
        self.config = config

        # Routed experts
        self.num_experts = self.config.moe_params.num_experts

        def create_ffn(config):
            return nn.ModuleList(
                [
                    SparseActSingleFeedForwardLayer(
                        in_features=in_features,
                        out_features=out_features,
                        activation=activation,
                        dropout=dropout,
                        use_bias=self.config.use_bias,
                        device=self.config.device,
                    )
                    for in_features, out_features, activation, dropout in zip(
                        config.input_units,
                        config.layers_units,
                        config.layers_activation,
                        config.layers_dropout_rates,
                    )
                ]
            )

        self.experts = nn.ModuleList(
            [create_ffn(self.config) for _ in range(self.num_experts)]
        )

        self.top_k = self.config.moe_params.top_k

        # Shared experts
        self.num_shared_experts = (
            self.config.moe_params.num_shared_experts
            if self.config.moe_params.num_shared_experts is not None
            else 0
        )
        self.shared_experts = nn.ModuleList(
            [FeedForwardNetwork(config) for _ in range(self.num_shared_experts)]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for expert in self.shared_experts:
            expert.reset_parameters()

        for expert in self.experts:
            for layer_num, linear_layer_module in enumerate(expert):
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
                        linear_layer_module.linear_layer_for_glu.weight.data
                    )
                weight_initializer(linear_layer_module.linear_layer.weight.data)
                if self.config.use_bias:
                    bias_initializer = create_initializer(
                        self.config.bias_initializer
                    )
                    bias_initializer(linear_layer_module.linear_layer.bias.data)

    def forward(
        self,
        input: torch.Tensor,
        topk_probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        assert (
            topk_probs.shape == topk_indices.shape
        ), "Expected topk_probs and topk_indices to have the same shapes. {topk_probs.shape} != {topk_indices.shape}"
        input = maybe_to_half_dtype(input)
        output = torch.zeros_like(input)

        # Shared experts
        print(f"len(self.shared_experts) = {len(self.shared_experts)}")
        for expert in self.shared_experts:
            output += expert(input)

        # Routed Experts
        topk_probs = maybe_to_half_dtype(topk_probs)

        ones = torch.ones_like(topk_probs)
        zeros = torch.zeros_like(topk_probs)

        for i, expert in enumerate(self.experts):

            @expert_annotation(
                num_experts=self.num_experts,
                top_k=self.top_k,
                expert_block_id=i,
            )
            def add_routed_expert(input, topk_indices, output):
                # Compute token mask for expert i
                indices_mask_k = torch.where(
                    (topk_indices - i).to(topk_probs.dtype) == 0, ones, zeros
                )
                indices_mask, _ = indices_mask_k.max(
                    dim=-1
                )  # returns values, indices. Only care about values

                # Compute probability mask for expert i
                probs_mask = topk_probs * indices_mask_k
                probs_mask, _ = probs_mask.max(
                    dim=-1, keepdim=True
                )  # returns values, indices. Only care about values

                exp_in = input
                if self.config.moe_params.probability_before_ffn:
                    exp_in = exp_in * probs_mask.broadcast_to(exp_in.shape)

                exp_out = exp_in
                for ffn in expert:
                    exp_out = ffn(exp_out, indices_mask)

                if not self.config.moe_params.probability_before_ffn:
                    exp_out = exp_out * probs_mask.broadcast_to(exp_out.shape)

                return output + exp_out

            output = add_routed_expert(input, topk_indices, output)

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
        # For MoE layers, shared experts are also added as experts
        # We dont use this in routing logic, but they process as regular experts
        num_shared_experts = (
            self.config.moe_params.num_shared_experts
            if self.config.moe_params.num_shared_experts is not None
            else 0
        )
        self.num_experts_all = (
            self.config.moe_params.num_experts + num_shared_experts
        )
        self.top_k_all = self.config.moe_params.top_k + num_shared_experts
        self.experts = nn.ModuleList(
            [FeedForwardNetwork(config) for _ in range(self.num_experts_all)]
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
        for i in range(self.num_experts_all):
            curr_expert = torch.tensor(
                i, device=input.device, dtype=torch.int16
            )
            for j in range(self.top_k_all):
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
        # Get a copy of config as we mutate it for shared experts
        self.config = copy.deepcopy(config)
        total_experts = self.config.moe_params.num_experts

        # Shared routing support added to our learned routing method
        if self.config.moe_params.num_shared_experts is not None:
            assert (
                self.config.moe_params.routing_algorithm == "learned"
            ), "Shared experts work with learned routing"
            assert (
                self.config.moe_params.num_shared_experts > 0
            ), "Invalid value of shared experts for MoE"
            total_experts += self.config.moe_params.num_shared_experts

        assert (
            total_experts > 1
        ), "expected num total experts > 1, but got {self.config.moe_params.num_experts} + {self.config.moe_params.num_shared_experts}="

        self.gate = nn.Linear(
            self.config.input_unit,
            self.config.moe_params.num_experts,
            bias=False,
        )
        if self.config.moe_optimized_impl():
            self.experts = TopKExpertsSparseFeedForwardNetwork(self.config)
        elif self.config.moe_experimental_impl():
            self.experts = SparseActFeedForwardNetwork(self.config)
        else:
            self.experts = ExpertsFeedForwardNetwork(self.config)

        self.reset_parameters()

    def reset_parameters(self):
        weight_initializer = create_initializer(
            self.config.moe_params.gate_initializer
        )
        weight_initializer(self.gate.weight.data)
        self.experts.reset_parameters()

    def forward(self, x, **extra_args):
        sinkhorn_error = None

        def sinkhorn(r, n_iters=1):
            r_old = r
            error = 0
            for _ in range(n_iters):
                r = r - torch.logsumexp(r, dim=2, keepdim=True)
                r = r - torch.logsumexp(r, dim=1, keepdim=True)
                error = torch.mean(torch.abs(r_old - r))
                r_old = r
            return torch.exp(r), error

        x = maybe_to_half_dtype(x)
        maybe_half_dtype = x.dtype

        if self.config.moe_params.routing_algorithm == "hash":
            assert extra_args.get("expert_hash_idx", None) is not None
            expert_hash_idx = extra_args["expert_hash_idx"]
            routing_weights = cstorch.nn.functional.one_hot(
                expert_hash_idx.to(torch.int64),
                num_classes=self.config.moe_params.num_experts,
            ).to(x.dtype)
            routing_weights_fp32 = routing_weights.float()
        elif self.config.moe_params.routing_algorithm == "learned":
            router_logits = self.gate(x)
            if self.config.moe_params.router_fp32:
                router_logits = router_logits.float()
            if (
                self.config.moe_params.router_selection_nonlinearity
                == "softmax"
            ):

                routing_weights_fp32 = nn.functional.softmax(
                    router_logits, dim=-1
                )
            elif (
                self.config.moe_params.router_selection_nonlinearity
                == "sinkhorn"
            ):
                with torch.no_grad():
                    routing_weights, sinkhorn_error = sinkhorn(
                        router_logits.detach(),
                        self.config.moe_params.sinkhorn_n_iters,
                    )
                    routing_weights_fp32 = routing_weights.float()
            elif (
                self.config.moe_params.router_selection_nonlinearity
                == "sigmoid"
            ):
                routing_weights_fp32 = nn.functional.sigmoid(router_logits)
            else:
                raise ValueError(
                    f'Unknown MoE routing nonlinearity: {self.config.moe_params.router_selection_nonlinearity}'
                )

            routing_weights = routing_weights_fp32.to(maybe_half_dtype)

            if (
                self.config.moe_params.expert_weighting_nonlinearity
                != self.config.moe_params.router_selection_nonlinearity
            ):
                if (
                    self.config.moe_params.expert_weighting_nonlinearity
                    == "softmax"
                ):
                    expert_weights_fp32 = nn.functional.softmax(
                        router_logits, dim=-1
                    )
                elif (
                    self.config.moe_params.expert_weighting_nonlinearity
                    == "sinkhorn"
                ):
                    with torch.no_grad():
                        routing_weights, sinkhorn_error = sinkhorn(
                            router_logits.detach(),
                            self.config.moe_params.sinkhorn_n_iters,
                        )
                        expert_weights_fp32 = routing_weights.float()
                elif (
                    self.config.moe_params.expert_weighting_nonlinearity
                    == "sigmoid"
                ):
                    expert_weights_fp32 = nn.functional.sigmoid(router_logits)
                else:
                    raise ValueError(
                        f'Unknown MoE Expert weighting nonlinearity: {self.config.moe_params.expert_weighting_nonlinearity}'
                    )

                expert_weights = expert_weights_fp32.to(maybe_half_dtype)

        else:
            raise ValueError(
                f'Unknown MoE routing algorithm: {self.config.moe_params.routing_algorithm}'
            )

        if (
            self.config.moe_optimized_impl()
            or self.config.moe_experimental_impl()
        ):
            # Compute TopK
            topk_probs, topk_indices = routing_weights.topk(
                self.config.moe_params.top_k
            )

            # Cast to i16 first since i64 is not supported on-wafer. Only cast
            # back to i64 for torch operations that require i64 input (e.g.
            # scatter, gather, etc.)
            topk_indices = topk_indices.to(torch.int16)
            if (
                self.config.moe_params.num_experts
                > torch.iinfo(torch.int16).max + 1
            ):
                # Ensure number of experts is within range to cast to i16
                raise ValueError(
                    f'Number of experts need to be <= 2^16, but got {self.config.moe_params.num_experts}'
                )

            # If a different non-linearity is used for expert weighting, gather the expert weights
            # according to the selected expert index
            if (
                self.config.moe_params.expert_weighting_nonlinearity
                != self.config.moe_params.router_selection_nonlinearity
            ):
                topk_probs = torch.gather(
                    expert_weights, -1, topk_indices.to(torch.int64)
                )

            else:
                num_experts = routing_weights.shape[-1]
                expert_mask = (
                    cstorch.nn.functional.one_hot(
                        topk_indices.to(torch.int64), num_classes=num_experts
                    )
                    .to(x.dtype)
                    .sum(dim=2)
                )

            if self.config.moe_params.expert_weighting_normalization:
                denom = torch.sum(topk_probs, dim=-1).to(x.dtype)
                # Add the probability from the null expert
                if (
                    self.config.moe_params.null_expert_bias is not None
                    and self.config.moe_params.null_expert_bias > 0
                ):
                    null_prob = torch.full_like(
                        topk_probs[..., 0],
                        self.config.moe_params.null_expert_bias,
                        dtype=x.dtype,
                    )
                    denom = denom + null_prob
                topk_probs /= denom[..., None]

            # moe_optimized_impl only:
            # If we have shared experts, add the probability and index to selected experts
            if (
                self.config.moe_params.num_shared_experts is not None
                and self.config.moe_optimized_impl()
            ):
                # Let the last experts be the shared ones
                shared_expert_index = self.config.moe_params.num_experts
                for _ in range(self.config.moe_params.num_shared_experts):
                    topk_prob = torch.full_like(
                        topk_probs[..., 0], 1.0, dtype=topk_probs.dtype
                    )
                    topk_prob = topk_prob.unsqueeze(-1)
                    topk_probs = torch.cat((topk_probs, topk_prob), dim=-1)
                    topk_index = torch.full_like(
                        topk_indices[..., 0],
                        shared_expert_index,
                        dtype=topk_indices.dtype,
                    )
                    shared_expert_index += 1
                    topk_index = topk_index.unsqueeze(-1)
                    topk_indices = torch.cat((topk_indices, topk_index), dim=-1)

            output = self.experts(x, topk_probs, topk_indices)

            return output, routing_weights_fp32, expert_mask.float()

        selected_experts = []
        selected_probs = []
        routing_weights_top_k = torch.clone(routing_weights)
        expert_mask = torch.zeros_like(routing_weights)
        for k in range(self.config.moe_params.top_k):
            selected_prob, selected_expert = torch.max(
                routing_weights_top_k, dim=-1, keepdim=True
            )

            # If a different non-linearity is used for expert weighting, gather the expert weights
            # according to the selected expert index
            if (
                self.config.moe_params.expert_weighting_nonlinearity
                != self.config.moe_params.router_selection_nonlinearity
            ):
                selected_prob = torch.gather(
                    expert_weights, -1, selected_expert
                )

            selected_experts.append(selected_expert)
            selected_probs.append(selected_prob)
            # Mask the current top value to find the next top value
            expert_mask += cstorch.nn.functional.one_hot(
                torch.argmax(routing_weights_top_k, dim=-1),
                num_classes=self.config.moe_params.num_experts,
            ).to(maybe_half_dtype)
            routing_weights_top_k = torch.where(
                expert_mask == 0, routing_weights_top_k, 0
            )

        if self.config.moe_params.expert_weighting_normalization:
            # Normalize Top-K weights
            # Add the optional null router probability for Top k=0 scenario
            if (
                self.config.moe_params.null_expert_bias is not None
                and self.config.moe_params.null_expert_bias > 0
            ):
                denom = torch.full_like(
                    selected_probs[0], self.config.moe_params.null_expert_bias
                )
                for k in range(0, self.config.moe_params.top_k):
                    denom = denom + selected_probs[k]
            else:
                denom = selected_probs[0]
                for k in range(1, self.config.moe_params.top_k):
                    denom = denom + selected_probs[k]

            for k in range(self.config.moe_params.top_k):
                selected_probs[k] = selected_probs[k] / denom

        # If we have shared experts, add the probability and index to selected experts
        if self.config.moe_params.num_shared_experts is not None:
            # Let the last experts be the shared ones
            shared_expert_index = self.config.moe_params.num_experts
            for _ in range(self.config.moe_params.num_shared_experts):
                selected_prob = torch.full_like(selected_probs[0], 1.0)
                selected_probs.append(selected_prob)
                selected_expert = torch.full_like(
                    selected_experts[0], shared_expert_index
                )
                shared_expert_index += 1
                selected_experts.append(selected_expert)

        output = self.experts(x, selected_probs, selected_experts)

        return output, routing_weights_fp32, expert_mask.float()
