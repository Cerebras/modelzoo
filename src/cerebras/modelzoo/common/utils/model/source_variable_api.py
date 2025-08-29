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
Pytorch-level API for source variable annotation

This API aims to give user ability to annotate operations as breakpoints.
Operations annotated will have `cs.internal = {source_vars = []}` attribute,
and this attribute can be used as a mapping between PyTorch operations and
their tensor dumps, for numeric validation.
"""

from dataclasses import dataclass
from typing import Optional

from cerebras.pytorch.core.annotation import AnnotationMode, create_annotation


@dataclass
class SourceVariableNamedConfig(AnnotationMode.Config):
    name: Optional[str]


def get_source_variable_attribute(config, is_backward: bool):
    postfix = "bwd" if is_backward else "fwd"
    name = f"{config.name}.{postfix}"
    return AnnotationMode.Attribute('source_vars', [name])


def source_variable(
    name: Optional[str] = None,
    enable_fwd: bool = True,
    enable_bwd: bool = False,
):
    return create_annotation(
        SourceVariableNamedConfig,
        get_source_variable_attribute,
        name=name,
        enable_fwd=enable_fwd,
        enable_bwd=enable_bwd,
    )
