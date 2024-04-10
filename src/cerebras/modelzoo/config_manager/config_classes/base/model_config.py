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

"""
Config classes of Model Configs

"""
from dataclasses import dataclass
from typing import List, Literal, Optional, Union

from cerebras.modelzoo.config_manager.config_classes.base.base_config import (
    BaseConfig,
)


@dataclass
class InitializerConfig(BaseConfig):
    name: str = Literal[
        "constant",
        "ones",
        "zeros",
        "eye" "uniform",
        "normal",
        "xavier_normal",
        "glorot_normal",  # alias for `xavier_normal`
        "xavier_uniform",
        "glorot_uniform",  # alias for `xavier_uniform`
        "truncated_normal",
        "variance_scaling",
        "lecun_normal",
        "lecun_uniform",
        "kaiming_normal",
        "kaiming_uniform",
    ]
    mean: Optional[float] = None
    std: Optional[float] = None
    a: Optional[float] = None
    b: Optional[float] = None
    nonlinearity: Optional[
        Literal[
            "linear",
            "conv1d",
            "conv2d",
            "conv3d",
            "conv_transpose1d",
            "conv_transpose2d",
            "conv_transpose3d",
            "sigmoid",
            "tanh",
            "relu",
            "leaky_relu",
        ]
    ] = None
    mode: Optional[str] = None
    scale: Optional[float] = None
    distribution: Optional[str] = None


@dataclass
class NormKWArgsConfig(BaseConfig):
    pass


@dataclass
class LoraConfig:
    r: int = 0
    "Rank of LoRA matrix projections"
    alpha: int = 1
    "Scaling factor (see paper for additional details)"
    dropout: float = 0.0
    "Dropout to apply to LoRA updates"
    fan_in_fan_out: bool = False
    merge_weights: bool = True
    """Determines whether lora weights should be merged/folded
    into underlying layers"""
    target_modules: Optional[list] = None
    """A list of module names that must all exist in layers
    that will be converted to LoRA. For example, setting target_modules
    to ["TransformerDecoderLayer", "Linear"] would mean that all linear
    layers that were children of a TransformerDecoderLayer would be
    converted to LoRA."""


@dataclass()
class ModelConfig(BaseConfig):
    mixed_precision: bool = False
    "Enable to run the model in mixed precision mode"

    fp16_type: Optional[
        Literal["bfloat16", "float16", "cbfloat16"]
    ] = "bfloat16"
    "Type of 16bit precision used"

    boundary_casting: Optional[bool] = False
    lora_params: Optional[Union[LoraConfig, List[LoraConfig]]] = None
