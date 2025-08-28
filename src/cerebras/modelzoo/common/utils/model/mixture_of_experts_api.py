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
Pytorch-level API for Mixture of Experts

This API allows the user to annotate Mixture-of-Expert ops metadata that is used to calculate the
expected sparsity.
"""

from dataclasses import dataclass

from cerebras.pytorch.backend import use_cs
from cerebras.pytorch.core.annotation import AnnotationMode, create_annotation


@dataclass
class ExpertsConfig(AnnotationMode.Config):
    num_experts: int = 0
    top_k: int = 0
    expert_block_id: int = 0

    def __post_init__(self):
        if not isinstance(self.num_experts, (int, type(None))):
            raise TypeError(
                f"Expected `num_experts` to be {int}, got {type(self.num_experts)}."
            )
        if not isinstance(self.top_k, (int, type(None))):
            raise TypeError(
                f"Expected `top_k` to be {int}, got {type(self.top_k)}."
            )
        if not isinstance(self.expert_block_id, (int, type(None))):
            raise TypeError(
                f"Expected `expert_block_id` to be {int}, got {type(self.expert_block_id)}."
            )


def get_attribute(config: ExpertsConfig, is_backward: bool):
    """Returns expert_block attribute"""
    return AnnotationMode.Attribute(
        'expert_block_info',
        {
            "num_experts": config.num_experts,
            "top_k": config.top_k,
            "expert_block_id": config.expert_block_id,
            "is_backward": is_backward,
        },
    )


# Function decorator to annotate MoE metadata
# - number of total experts in layer
# - top_k
# - current expert (int id) that this op belongs to
def expert_annotation(num_experts: int, top_k: int, expert_block_id: int):
    """
    Returns an annotating function which wraps the given function.
    """
    if not use_cs():
        return lambda fn: fn

    return create_annotation(
        ExpertsConfig,
        get_attribute,
        num_experts=num_experts,
        top_k=top_k,
        expert_block_id=expert_block_id,
        enable_fwd=True,
        enable_bwd=True,
    )


# Annotate expected sparsity
# Valid range: [0.0, 1.0)
# - 0.0 indicates no sparsity (no values are zero)
# - 0.9999.. indicates 99.99..% sparsity (almost all values are 0)
#
# ST-122: Currently the backend uses expert_annotation
# Eventually this will be deprecated and replaced with a more generic backend
# once stack is updated.
def average_sparsity_annotation(average_sparsity: float):
    assert (
        average_sparsity >= 0.0 and average_sparsity < 1.0
    ), f"Expected 0.0 <= average_sparsity < 1.0. Instead got average_sparsity = {average_sparsity}"

    density = 1.0 - average_sparsity
    num_experts = int(2**31 - 1)
    top_k = int(density * float(num_experts))

    return expert_annotation(
        num_experts=num_experts, top_k=top_k, expert_block_id=0
    )
