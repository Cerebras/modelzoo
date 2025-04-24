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

"""Named matmul annotation support."""

from dataclasses import dataclass
from typing import Optional

from cerebras.pytorch.core.annotation import AnnotationMode, create_annotation


@dataclass
class NamedConfig(AnnotationMode.Config):
    name: Optional[str]


def get_attribute(config, is_backward: bool):
    """Returns named_matmul attribute"""
    postfix = "bwd" if is_backward else "fwd"
    name = f"{config.name}.{postfix}"
    return AnnotationMode.Attribute('named_matmul', name)


def named_matmul(
    name: Optional[str] = None,
    enable_fwd: bool = True,
    enable_bwd: bool = True,
):
    return create_annotation(
        NamedConfig,
        get_attribute,
        name=name,
        enable_fwd=enable_fwd,
        enable_bwd=enable_bwd,
    )
