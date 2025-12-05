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

from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from annotated_types import Ge, Le
from typing_extensions import Annotated

import cerebras.pytorch as cstorch
from cerebras.modelzoo.config import ModelConfig
from cerebras.pytorch.metrics import AccuracyMetric


class MNISTConfig(ModelConfig):
    name: Literal["fc_mnist"]
    """Name of the model."""

    use_bias: bool = True
    """Whether to use bias in the linear layers."""

    hidden_sizes: Optional[List[int]] = None
    """Size of hidden layers. This option takes precedence over `depth` and `hidden_size`."""

    depth: Optional[int] = None
    """Number of hidden layers. This option is used iff `hidden_sizes` is not set."""

    hidden_size: Optional[int] = None
    """Size of each hidden layer. This option is used iff `hidden_sizes` is not set."""

    activation_fn: Optional[Literal["relu"]] = None
    """Non-linearity to apply to the hidden layers."""

    dropout: Annotated[float, Ge(0), Le(1)] = 0
    """Whether to use dropout to randomly zero out elements of hidden layer inputs."""

    def post_init(self, context):
        if self.hidden_sizes is not None:
            self.depth = len(self.hidden_sizes)
        else:
            if self.hidden_size is None or self.depth is None:
                raise ValueError(
                    "hidden_size and depth, or hidden_sizes must be provided"
                )

            self.hidden_sizes = [self.hidden_size] * self.depth

    @property
    def __model_cls__(self):
        return MNIST


class MNIST(nn.Module):
    def __init__(self, config: MNISTConfig):
        super().__init__()
        self.fc_layers = []
        use_bias = config.use_bias
        input_size = 784

        for hidden_size in config.hidden_sizes:
            fc_layer = nn.Linear(input_size, hidden_size, bias=use_bias)
            self.fc_layers.append(fc_layer)
            input_size = hidden_size
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.last_layer = nn.Linear(input_size, 10, bias=use_bias)

        self.nonlin = nn.ReLU() if config.activation_fn == "relu" else None

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs):
        x = torch.flatten(inputs, 1)
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            if self.nonlin:
                x = self.nonlin(x)
            x = self.dropout(x)

        pred_logits = self.last_layer(x)
        return pred_logits


class MNISTModelConfig(MNISTConfig):
    name: Literal["fc_mnist"]
    """Name of the model."""

    disable_softmax: bool = False
    """Disable softmax on output logits."""

    compute_eval_metrics: bool = False
    """Compute evaluation metrics (currently only accuracy)."""

    to_float16: bool = False
    """Whether to convert the model to 16-bit floating precision."""


class MNISTModel(nn.Module):
    def __init__(self, config: MNISTModelConfig):
        if isinstance(config, dict):
            if "model" in config:
                config = config["model"]
            config = MNISTModelConfig(**config)

        super().__init__()

        self.model = self.build_model(config)
        self.loss_fn = nn.NLLLoss()
        self.disable_softmax = config.disable_softmax

        self.accuracy_metric = None
        if config.compute_eval_metrics:
            self.accuracy_metric = AccuracyMetric(name="accuracy")

    def build_model(self, config: MNISTModelConfig):
        model = MNIST(config)
        if config.to_float16:
            return model.to(cstorch.amp.get_half_dtype())
        return model

    def forward(self, data):
        inputs, labels = data
        pred_logits = self.model(inputs)
        if not self.model.training and self.accuracy_metric:
            labels = labels.clone()
            predictions = pred_logits.argmax(-1).int()
            self.accuracy_metric(labels=labels, predictions=predictions)
        if not self.disable_softmax:
            pred_logits = F.log_softmax(pred_logits, dim=1)
        loss = self.loss_fn(pred_logits, labels)
        return loss
