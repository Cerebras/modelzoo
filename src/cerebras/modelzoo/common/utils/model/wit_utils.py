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
This module provides a PyTorch-level API for Wafer Instruction Tuning (WIT).

WIT allows users to control the behavior of the ML compiler by annotating function closures. 
Through this API, users can specify whether a function closure should be optimized by the 
scheduling algorithm or through common subexpression elimination (CSE). 

The primary components of this module include:
- `WitConfig`: A configuration class for enabling or disabling schedule optimization 
  and CSE within annotated closures.
- `wafer_instruction_tuning`: A Python decorator for annotating checkpoint operations for the recompute algorithm. 
  This ensures that the annotated operation includes the "wit" attribute in the compiler's 
  internal representation.

This API is designed to provide fine-grained control over compiler optimizations, enabling 
users to tailor the behavior of the ML compiler to their specific needs.
"""

from dataclasses import asdict, dataclass, fields

from cerebras.pytorch.core.annotation import AnnotationMode, create_annotation


@dataclass
class WitConfig(AnnotationMode.Config):
    """
    Wafer instruction tuning config.
    This is an annotation mode to annotate functions in PyTorch.
    Within the annotated closure, all the lowered annotated IR will have this annotation.
    Currently supports enabling schedule optimization and enabling common subexpression elimination.
    """

    schedule_opt: bool = True
    cse_opt: bool = True
    force_recompute_opt: bool = False


def get_wit_attribute(config: WitConfig, is_backward: bool):
    """Returns recompute_region attribute"""
    config_dict = asdict(config)
    for field in fields(AnnotationMode.Config):
        del config_dict[field.name]
    return AnnotationMode.Attribute('wit', config_dict)


def wafer_instruction_tuning(
    schedule_opt=True, cse_opt=True, force_recompute_opt=False
):
    """
    Python decorator for annotating checkpoint operation for the recompute algorithm.
    The annotated operation will have the "wit" attribute in "cs.internal"
    DictionaryAttr.
    """
    return create_annotation(
        WitConfig,
        get_wit_attribute,
        schedule_opt=schedule_opt,
        cse_opt=cse_opt,
        force_recompute_opt=force_recompute_opt,
        enable_fwd=True,
        enable_bwd=True,
    )
