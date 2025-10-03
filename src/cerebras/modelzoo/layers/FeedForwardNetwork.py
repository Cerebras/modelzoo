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

import enum
from typing import Callable, ClassVar, List, Literal, Optional, Union
from warnings import warn

import torch.nn as nn
from annotated_types import Len
from pydantic import field_validator, model_validator
from torch import Tensor

# Use typing once we move to Python 3.11
from typing_extensions import Annotated

from cerebras.modelzoo.config import BaseConfig, ModelConfig
from cerebras.modelzoo.layers.activations import (
    get_activation,
    is_glu_activation,
)
from cerebras.modelzoo.layers.init import InitializerConfig


class MoEConfig(BaseConfig):
    num_experts: int = 0
    "Number of experts used for MoE, 0 means MoE is disabled"
    num_shared_experts: Optional[int] = None
    "Number of shared experts used by MoE"
    top_k: int = 1
    "K value for the number of top experts to be selected from all experts"
    load_balancing_loss_coef: float = 0.0
    "Weight for the load balancing loss"
    null_expert_bias: Optional[float] = 0.0
    "Optional bias to add null expert prob to the routing"
    moe_implementation: Literal["functional", "optimized", "experimental"] = (
        "experimental"
    )
    "Whether to use the functional or Optimized implementation of MoE"
    router_fp32: bool = True
    "Selects the precision of routing weights to be float"
    routing_algorithm: Literal["hash", "learned"] = "learned"
    "Routing algorithm to use for selection of experts"
    router_selection_nonlinearity: Optional[
        Literal["sigmoid", "sinkhorn", "softmax"]
    ] = "softmax"
    "Non linearity used for routing algorithm expert selection, to be used with 'learned' routing"
    expert_weighting_nonlinearity: Optional[
        Literal["sigmoid", "sinkhorn", "softmax"]
    ] = "softmax"
    "Non linearity used for expert probability weightings, to be used with 'learned' routing"
    expert_weighting_normalization: Optional[bool] = True
    "Normalize expert weights or not before weighting the experts"
    sinkhorn_n_iters: Optional[int] = 1
    "Number of iterations for sinkhorn nonlinearity"
    gate_initializer: Optional[InitializerConfig] = None
    "Initializer used for router gating network"
    probability_before_ffn: bool = True
    "Compute routing probabilities before the FFN layer instead of after"

    def post_init(self, context):
        super().post_init(context)

    @model_validator(mode="after")
    def validate_moe(self):
        if self.routing_algorithm == "hash":
            if self.load_balancing_loss_coef > 0.0:
                raise ValueError(
                    "Load Balancing Loss not supported with hash routing"
                )
            if self.router_selection_nonlinearity is not None:
                warn(
                    f"Routing non-linearity {self.router_selection_nonlinearity} is ignored with hash routing"
                )
            if self.expert_weighting_nonlinearity is not None:
                warn(
                    f"Expert weighting non-linearity {self.expert_weighting_nonlinearity} is ignored with hash routing"
                )

        elif self.routing_algorithm == "learned":
            if (
                self.null_expert_bias is not None
                and self.null_expert_bias != 0.0
                and not self.expert_weighting_normalization
            ):
                raise ValueError(
                    "expert_weighting_normalization should be set to True when using null_expert_bias"
                )

        if self.router_selection_nonlinearity == "sinkhorn":
            if self.load_balancing_loss_coef != 0.0:
                raise ValueError(
                    "Load Balancing Loss not supported with sinkhorn routing nonlinearity"
                )
            if self.sinkhorn_n_iters is None:
                raise ValueError(
                    "sinkhorn_n_iters cannot be None for sinkhorn nonlinearity"
                )
        total_experts = self.num_experts
        if self.num_shared_experts is not None:
            total_experts += self.num_shared_experts
        if total_experts > 1:
            if self.load_balancing_loss_coef < 0.0:
                raise ValueError(
                    f"load_balancing_loss_coef cannot be less than 0, got {self.load_balancing_loss_coef}"
                )

            if not (self.top_k >= 1 and self.top_k <= self.num_experts):
                raise ValueError(
                    f"{self.top_k=} should be [1, {self.num_experts=}]"
                )

            if self.routing_algorithm == "hash" and self.top_k != 1:
                raise ValueError(
                    f"{self.top_k=} but should be 1 for hash routing"
                )
        return self


class StaticDualExpertLinear(nn.Module):
    """
    Description of the linear op where tokens are sent to two different experts based on 'token_modality_idx'.
    """

    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype=None
    ):
        super().__init__()
        self.linear_img = nn.Linear(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.linear_text = nn.Linear(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )

    def forward(self, input, token_modality_idx):
        """
        We can model this part as a MOE structure, which we have one expert
        for each text and image portion. The text and image portion is masked
        by `token_modality_idx`.
        TODO: we can leverage the MOE optimization by using the MOE Sparse linear layer.

        Args:
            token_modality_idx: tensor to mask different modality.
                value '1' is for image token and value '0' is for text token.
        """
        assert (
            token_modality_idx != None
        ), "Expect 'token_modality_idx' when 'static_dual_expert' is True."
        img_out = self.linear_img(input)
        text_out = self.linear_text(input)

        token_modality_idx = token_modality_idx.to(img_out.dtype)[
            :, :, None
        ].broadcast_to(*img_out.shape)
        x = text_out * (1 - token_modality_idx) + img_out * token_modality_idx
        return x


class SingleFeedForwardLayer(nn.Module):
    """
    Initialize Single FFN layer instance.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = False,
        activation: Optional[Union[str, Callable[[Tensor], Tensor]]] = None,
        dropout: Optional[float] = None,
        device=None,
        static_dual_expert: bool = False,
    ):
        super(SingleFeedForwardLayer, self).__init__()

        self.static_dual_expert = static_dual_expert
        if static_dual_expert == True:
            linearOp = StaticDualExpertLinear
        else:
            linearOp = nn.Linear
        self.linear_layer = linearOp(
            in_features,
            out_features,
            bias=use_bias,
            device=device,
        )

        self.is_glu_activation = is_glu_activation(activation)
        if self.is_glu_activation:
            self.linear_layer_for_glu = linearOp(
                in_features,
                out_features,
                bias=use_bias,
                device=device,
            )

        if activation:
            self.act_layer = get_activation(activation)
        else:
            self.act_layer = None

        if dropout and dropout > 0.0:
            self.dropout_layer = nn.Dropout(p=dropout)
        else:
            self.dropout_layer = None

    def forward(self, inputs, **extra_args):
        if self.static_dual_expert:
            linear_inputs = [inputs, extra_args.get("token_modality_idx", None)]
        else:
            linear_inputs = [inputs]
        if self.is_glu_activation:
            glu_component_1 = self.linear_layer(*linear_inputs)
            glu_component_2 = self.linear_layer_for_glu(*linear_inputs)
            outputs = self.act_layer(glu_component_1, glu_component_2)
        else:
            outputs = self.linear_layer(*linear_inputs)
            if self.act_layer:
                outputs = self.act_layer(outputs)
        if self.dropout_layer:
            outputs = self.dropout_layer(outputs)
        return outputs


class FeedForwardNetworkConfig(ModelConfig):
    """Feed forward network config.

    Args:
        input_unit (int): integer for number of in_features of input.
        layers_units (list[int]): List of units for each layer.
        layers_activation (list[str]): List of activation types (str) for each layer.
        layers_dropout_rates (list[float]): List of dropout rates (float) for each
            layer.
        use_bias (bool): If `True`, use bias throughout all layers.
        kernel_initializer: Kernel initializer. Defaults to
            `"xavier_uniform"`.
        bias_initializer: Bias initializer. Defaults to `"zeros"`.
        output_layer_initializer: If not None, initialize the last projection
            layer with this initializer. Defaults to None.
        moe_implementation (str): "functional", "optimized", or "experimental" implementation. Defaults to "optimized".
        device (optional): Device to create the model parameters on, can be a cuda device or CS device.
        moe_params (optional [MoEConfig]): config params for setting up MoE
    """

    name: Literal["FeedForwardNetwork", "MLP"]

    input_unit: int = ...
    layers_units: Annotated[List[int], Len(min_length=1)] = ...
    layers_activation: Optional[
        List[Union[str, Callable[[Tensor], Tensor], None]]
    ] = None
    layers_dropout_rates: Optional[List[Union[float, None]]] = None
    use_bias: bool = False
    kernel_initializer: InitializerConfig = "xavier_uniform"
    bias_initializer: InitializerConfig = "zeros"
    output_layer_initializer: Optional[InitializerConfig] = None
    device: Optional[str] = None
    static_dual_expert: bool = False
    # Class variables.
    MoEImpl: ClassVar[enum.Enum] = enum.Enum(
        "MoEImpl", ["functional", "optimized", "experimental"]
    )
    moe_params: Optional[MoEConfig] = None

    @field_validator("name", mode="after")
    def validate_name(cls, name):
        if name == "MLP":
            warn(
                "Passing 'MLP' as the model name is deprecated. "
                "Please use 'FeedForwardNetwork' instead.",
                category=FutureWarning,
            )
            return "FeedForwardNetwork"
        return name

    @property
    def num_dense_layers(self):
        return len(self.layers_units)

    @property
    def input_units(self):
        return [self.input_unit] + self.layers_units[:-1]

    @model_validator(mode="after")
    def validate_layers_activation(self):
        if (
            self.layers_activation
            and len(self.layers_activation) != self.num_dense_layers
        ):
            raise ValueError(
                "len(layers_activation) should equal the number"
                " of dense layers."
            )

        return self

    @model_validator(mode="after")
    def validate_layers_dropout_rates(self):
        if (
            self.layers_dropout_rates
            and len(self.layers_dropout_rates) != self.num_dense_layers
        ):
            raise ValueError(
                "len(layers_dropout_rates) should equal the number "
                "of dense layers."
            )

        return self

    def moe_experimental_impl(self) -> bool:
        """Return True if experimental implementation is used."""
        return (
            self.MoEImpl[self.moe_params.moe_implementation]
            == self.MoEImpl.experimental
        )

    def moe_optimized_impl(self) -> bool:
        """Return True if optimized implementation is used."""
        return (
            self.MoEImpl[self.moe_params.moe_implementation]
            == self.MoEImpl.optimized
        )

    def moe_functional_impl(self) -> bool:
        """Return True if functional implementation is used."""
        return (
            self.MoEImpl[self.moe_params.moe_implementation]
            == self.MoEImpl.functional
        )

    def post_init(self, context):
        if self.output_layer_initializer is None:
            self.output_layer_initializer = self.kernel_initializer

        if not self.layers_activation:
            self.layers_activation = [None] * self.num_dense_layers
        if not self.layers_dropout_rates:
            self.layers_dropout_rates = [None] * self.num_dense_layers

    @property
    def __model_cls__(self):
        return FeedForwardNetwork


class FeedForwardNetwork(nn.Module):
    """
    A feed forward network that consists of a stack of fully connected\
    layers arranged as [LinearLayer -> Activation -> Dropout] block
    repeated `len(layers_units)` times.

    Args:
        config (FeedForwardNetworkConfig): Feed forward network config.
    """

    def __init__(self, config: FeedForwardNetworkConfig):
        """
        Initialize the FFN object instance.
        """
        super(FeedForwardNetwork, self).__init__()
        self.config = config
        self.static_dual_expert = config.static_dual_expert

        # This sets the namespace of the layer.
        # Using `nn.ModuleList` to have clear namespace such as
        # `ffn.{layer_num}.weight` and `ffn.{layer_num}.bias`
        # Class attributes cannot have `.` in their names when
        # inheriting from `nn.Module` and therefore cannot generate
        # attribute names on the fly and hence the need to use ModuleList.
        self.ffn = nn.ModuleList(
            [
                SingleFeedForwardLayer(
                    in_features,
                    out_features,
                    use_bias=self.config.use_bias,
                    activation=activation,
                    dropout=dropout,
                    device=config.device,
                    static_dual_expert=config.static_dual_expert,
                )
                for in_features, out_features, activation, dropout in zip(
                    self.config.input_units,
                    self.config.layers_units,
                    self.config.layers_activation,
                    self.config.layers_dropout_rates,
                )
            ]
        )

        # Initialize weights in Linear layers.
        self.__reset_parameters()

    def reset_parameters(self):
        self.__reset_parameters()

    def __reset_parameters(self):
        # Initialize weights for all Linear layers
        for layer_num, linear_layer_module in enumerate(self.ffn):
            if layer_num == self.config.num_dense_layers - 1:
                weight_initializer = self.config.output_layer_initializer
            else:
                weight_initializer = self.config.kernel_initializer

            # Initialize linear layer weights associated with the
            # 'GLU' type activation function with the kernel_initializer
            for m in linear_layer_module.modules():
                if type(m) == nn.Linear:
                    weight_initializer(m.weight.data)

                    if m.bias is not None:
                        self.config.bias_initializer(m.bias.data)

    def forward(self, inputs, **extra_args):
        outputs = inputs
        for ffn_layer in self.ffn:
            outputs = ffn_layer(outputs, **extra_args)

        return outputs
