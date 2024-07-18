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
from dataclasses import dataclass
from typing import Callable, ClassVar, List, Optional, Union

import torch.nn as nn
from torch import Tensor

from cerebras.modelzoo.layers.activations import (
    get_activation,
    is_glu_activation,
)
from cerebras.modelzoo.layers.create_initializer import create_initializer


class StaticDualExpertLinear(nn.Module):
    """
    Description of the linear op where tokens are sent to two different experts based on 'token_modality_idx'
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
                value '1' is for image tocken and value '0' is for text token.
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


@dataclass
class FeedForwardNetworkConfig:
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
        num_experts (int): The number of experts in the MoE block. Defaults to 1.
        top_k (int): The number of experts to be used on each token. Defaults to None.
        gate_initializer: Router (gate) initializer. Defaults to `"xavier_uniform"`.
        load_balancing_loss_coef (float): The float coefficient to scale the load balancing loss.
            Defaults to `0.01`.
        router_fp32 (bool): If `True`, the router operate in FP32 dtype, Defaults to `True`.
        routing_algorithm (str): The routing algorithm used in Mixture-of-Experts
            models. Choose from: "learned", "hash". Defaults to "learned".
        moe_implementation (str): "functional" or "optimized" implementation. Defaults to "optimized".
        device (optional): Device to create the model parameters on, can be a cuda device or CS device.
    """

    input_unit: int
    layers_units: List[int]
    layers_activation: Optional[
        List[Union[str, Callable[[Tensor], Tensor]]]
    ] = None
    layers_dropout_rates: Optional[List[float]] = None
    use_bias: bool = False
    kernel_initializer: str = "xavier_uniform"
    bias_initializer: str = "zeros"
    output_layer_initializer: Optional[str] = None
    num_experts: int = 1
    top_k: Optional[int] = None
    gate_initializer: Optional[str] = "xavier_uniform"
    load_balancing_loss_coef: Optional[float] = 0.01
    router_fp32: bool = True
    routing_algorithm: str = "learned"
    moe_implementation: str = "optimized"
    device: str = None
    static_dual_expert: bool = False

    # Class variables.
    MoEImpl: ClassVar[enum.Enum] = enum.Enum(
        "MoEImpl", ["functional", "optimized"]
    )

    def __post_init__(self):
        self.num_dense_layers = len(self.layers_units)
        self.input_units = [self.input_unit] + self.layers_units[:-1]
        if self.output_layer_initializer is None:
            self.output_layer_initializer = self.kernel_initializer

        assert (
            self.num_dense_layers > 0
        ), "Number of dense layers should be at least 1."

        if self.layers_activation:
            assert len(self.layers_activation) == self.num_dense_layers, (
                "len(layers_activation) should equal the number"
                " of dense layers."
            )
        else:
            self.layers_activation = [None] * self.num_dense_layers

        if self.layers_dropout_rates:
            assert len(self.layers_dropout_rates) == self.num_dense_layers, (
                "len(layers_dropout) should equal the number" "of dense layers."
            )
        else:
            self.layers_dropout_rates = [None] * self.num_dense_layers

        if self.num_experts > 1:
            assert (
                self.load_balancing_loss_coef >= 0.0
            ), f"load_balancing_loss_coef cannot be less than 0, got {self.load_balancing_loss_coef}"
            assert (
                self.top_k >= 1 and self.top_k <= self.num_experts
            ), f"{self.top_k=} should be [1, {self.num_experts=}]"
            if self.routing_algorithm == "hash":
                assert (
                    self.top_k == 1
                ), f"{self.top_k=} but should be 1 for hash routing"

    def moe_optimized_impl(self) -> bool:
        """Return True if optimized implementation is used."""
        return self.MoEImpl[self.moe_implementation] == self.MoEImpl.optimized

    def moe_functional_impl(self) -> bool:
        """Return True if functional implementation is used."""
        return self.MoEImpl[self.moe_implementation] == self.MoEImpl.functional


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
            weight_initializer = create_initializer(
                self.config.kernel_initializer
            )
            if layer_num == self.config.num_dense_layers - 1:
                weight_initializer = create_initializer(
                    self.config.output_layer_initializer
                )
            # Initialize linear layer weights associated with the
            # 'GLU' type activation function with the kernel_initializer
            for m in linear_layer_module.modules():
                if type(m) == nn.Linear:
                    weight_initializer(m.weight.data)

                    if m.bias is not None:
                        create_initializer(self.config.bias_initializer)(
                            m.bias.data
                        )

    def forward(self, inputs, **extra_args):
        outputs = inputs
        for ffn_layer in self.ffn:
            outputs = ffn_layer(outputs, **extra_args)

        return outputs
