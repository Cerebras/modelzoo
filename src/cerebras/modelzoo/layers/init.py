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
from abc import ABC, abstractmethod
from typing import Any, Literal, Union, get_args
from warnings import warn

import torch
from pydantic import Field, field_validator, model_validator
from pydantic.functional_validators import BeforeValidator
from torch.nn.init import _calculate_fan_in_and_fan_out, trunc_normal_

# Use typing once we move to Python 3.11
from typing_extensions import Annotated

from cerebras.modelzoo.config import NamedConfig


class Initializer(NamedConfig, ABC):
    @abstractmethod
    def __call__(self, tensor):
        pass

    @model_validator(mode="before")
    @classmethod
    def remove_unused(cls, data):
        used = {
            f: data[f]
            for f in cls.model_fields
            if f in data and data[f] is not None
        }
        unused = set(data) - set(used)
        if unused:
            name = get_args(cls.model_fields["name"].annotation)[0].title()
            warn(
                f"{name} Initializer got the following unused inputs:\n"
                f"{sorted(unused)}\n"
                f"This is deprecated behaviour and will be disallowed in the future."
            )
        return used


class ConstantInitializer(Initializer):
    name: Literal["constant"]
    val: float = ...

    def __call__(self, tensor):
        return torch.nn.init.constant_(tensor, self.val)


class OnesInitializer(Initializer):
    name: Literal["ones"]

    def __call__(self, tensor):
        return torch.nn.init.ones_(tensor)


class ZerosInitializer(Initializer):
    name: Literal["zeros"]

    def __call__(self, tensor):
        return torch.nn.init.zeros_(tensor)


class EyeInitializer(Initializer):
    name: Literal["eye"]

    def __call__(self, tensor):
        return torch.nn.init.eye_(tensor)


class UniformInitializer(Initializer):
    name: Literal["uniform"]
    a: float = -0.05
    b: float = 0.05

    def __call__(self, tensor):
        return torch.nn.init.uniform_(tensor, a=self.a, b=self.b)


class NormalInitializer(Initializer):
    name: Literal["normal"]
    mean: float = 0.0
    std: float = 0.05

    def __call__(self, tensor):
        return torch.nn.init.normal_(tensor, mean=self.mean, std=self.std)


class XavierNormalInitializer(Initializer):
    name: Literal["xavier_normal", "glorot_normal"]
    gain: float = 1.0

    def __call__(self, tensor):
        return torch.nn.init.xavier_normal_(tensor, gain=self.gain)


class XavierUniformInitializer(Initializer):
    name: Literal["xavier_uniform", "glorot_uniform"]
    gain: float = 1.0

    def __call__(self, tensor):
        return torch.nn.init.xavier_uniform_(tensor, gain=self.gain)


class KaimingNormalInitializer(Initializer):
    name: Literal["kaiming_normal"]
    a: float = 0.0
    mode: Literal["fan_in", "fan_out"] = "fan_in"
    nonlinearity: Literal["leaky_relu", "relu"] = "leaky_relu"

    def __call__(self, tensor):
        return torch.nn.init.kaiming_normal_(
            tensor, a=self.a, mode=self.mode, nonlinearity=self.nonlinearity
        )


class KaimingUniformInitializer(Initializer):
    name: Literal["kaiming_uniform"]
    a: float = 0.0
    mode: Literal["fan_in", "fan_out"] = "fan_in"
    nonlinearity: Literal["leaky_relu", "relu"] = "leaky_relu"

    def __call__(self, tensor):
        return torch.nn.init.kaiming_uniform_(
            tensor, a=self.a, mode=self.mode, nonlinearity=self.nonlinearity
        )


class TruncatedNormalInitializer(Initializer):
    name: Literal["truncated_normal"]
    mean: float = 0.0
    std: float = 0.05
    a: float = -0.1
    b: float = 0.1

    @field_validator("a", mode="before")
    @classmethod
    def validate_a(cls, a):
        if a is None:
            return -0.1
        return a

    @field_validator("b", mode="before")
    @classmethod
    def validate_b(cls, b):
        if b is None:
            return 0.1
        return b

    def post_init(self, context):
        if "a" not in self.model_fields_set:
            self.a = -2 * self.std
        if "b" not in self.model_fields_set:
            self.b = 2 * self.std

    def __call__(self, tensor):
        return torch.nn.init.trunc_normal_(
            tensor, mean=self.mean, std=self.std, a=self.a, b=self.b
        )


class VarianceScalingInitializer(Initializer):
    name: Literal["variance_scaling"]
    scale: float = 1.0
    mode: Literal["fan_in", "fan_out", "fan_avg"] = "fan_in"
    distribution: Literal["uniform", "normal", "truncated_normal"] = (
        "truncated_normal"
    )

    @field_validator("mode", mode="before")
    @classmethod
    def validate_mode(cls, mode):
        if mode is None:
            return "fan_in"
        return mode

    @field_validator("distribution", mode="before")
    @classmethod
    def validate_distribution(cls, distribution):
        if distribution is None:
            return "truncated_normal"
        return distribution

    def __call__(self, tensor):
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        if self.mode == 'fan_in':
            denom = max(1.0, fan_in)
        elif self.mode == 'fan_out':
            denom = max(1.0, fan_out)
        elif self.mode == 'fan_avg':
            denom = (fan_in + fan_out) / 2
            denom = max(1.0, denom)

        variance = self.scale / denom

        if self.distribution == "truncated_normal":
            # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            trunc_normal_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
        elif self.distribution == "normal":
            tensor.normal_(std=math.sqrt(variance))
        elif self.distribution == "uniform":
            bound = math.sqrt(3 * variance)
            tensor.uniform_(-bound, bound)
        else:
            raise ValueError(f"invalid distribution {distribution}")

        return tensor


class LecunNormalInitializer(VarianceScalingInitializer):
    name: Literal["lecun_normal"]


class LecunUniformInitializer(VarianceScalingInitializer):
    name: Literal["lecun_uniform"]

    distribution: Literal["uniform", "normal", "truncated_normal"] = "uniform"


def get_discriminator(v: Any) -> Any:
    """Converts a string to a dictionary with a `name` key.

    This is used to handle the case where the user provides the name of the
    initializer as a string instead of a dictionary.
    """
    if isinstance(v, str):
        return {"name": v}
    return v


InitializerConfig = Annotated[
    Union[
        ConstantInitializer,
        OnesInitializer,
        ZerosInitializer,
        EyeInitializer,
        UniformInitializer,
        NormalInitializer,
        XavierNormalInitializer,
        XavierUniformInitializer,
        KaimingNormalInitializer,
        KaimingUniformInitializer,
        TruncatedNormalInitializer,
        VarianceScalingInitializer,
        LecunNormalInitializer,
        LecunUniformInitializer,
    ],
    Field(discriminator="name"),
    BeforeValidator(get_discriminator),
]
