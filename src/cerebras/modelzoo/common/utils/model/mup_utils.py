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

import logging
from dataclasses import asdict, is_dataclass
from typing import List, Optional, Union

import torch

from cerebras.modelzoo.config_manager.config_classes.base.model_config import (
    InitializerConfig,
)
from cerebras.pytorch.utils.utils import FilterCallable, make_param_filter

SUPPORTED_MUP_INITIALIZERS = [
    "normal",
    "truncated_normal",
]


class LRAdjustmentGroup:
    """
    Stores data for a group of params that share a learning rate scalar.
    Stores a callable that returns True if a given model param corresponds
    to the group. Additionally, it stores the scale that should be applied to
    the LR of the model params that correspond to the group.
    """

    def __init__(
        self,
        param_filter: Union[str, List[str]],
        scale: Optional[float] = 1.0,
    ):
        """
        param_filter: A string or a list of strings that contains glob expressions
        used to match whether a given model param name belongs to the group.

        scale: The scale that should be applied to the LR of this group
        """
        # Convert the strings into a callable that returns True if a given param
        # name corresponds to the LR group
        self.param_filter = make_param_filter(param_filter)
        self.scale = scale

    def set_scale(self, scale):
        self.scale = scale


def scale_initializers_by_dimension(
    initializers: Union[InitializerConfig, List[InitializerConfig]],
    width_scale: Optional[float] = None,
    depth_scale: Optional[float] = None,
):
    """
    Scales the std of an initializer or list of initializers by the specified
    width and depth scalars. Unsupported initializers are ignored and a warning
    is printed to the user.
    """
    if not width_scale:
        width_scale = 1.0
    if not depth_scale:
        depth_scale = 1.0
    mup_scalar = width_scale * depth_scale

    if not isinstance(initializers, list):
        initializers = [initializers]

    for initializer in initializers:
        if type(initializer) == str:
            initializer = {"name": initializer}
        if "name" not in initializer:
            raise ValueError("Initializer name must be provided")
        initializer_name = initializer["name"].lower()
        if initializer_name not in SUPPORTED_MUP_INITIALIZERS:
            raise RuntimeError(
                f"Initializer {initializer} does not support mup scaling. "
                f"Please use a supported initializer from the following: "
                f"{SUPPORTED_MUP_INITIALIZERS}"
            )
            continue

        if initializer_name == "normal":
            initializer["std"] = initializer.get("std", 1.0) * mup_scalar
        elif initializer_name == "truncated_normal":
            std = initializer.get("std", 1.0)
            initializer["std"] = std * mup_scalar
            initializer["a"] = initializer.get("a", -2 * std) * mup_scalar
            initializer["b"] = initializer.get("b", 2 * std) * mup_scalar
            std = None


def is_mup(model: Union[dict, torch.nn.Module]):
    if is_dataclass(model):
        model = asdict(model)
        return any(
            name.startswith('mup_base_') and model[name] is not None
            for name in model
        )
    if not isinstance(model, dict):
        model = {k: getattr(model, k) for k in dir(model)}
    return any(
        name.startswith('mup_base_') and model[name] is not None
        for name in model
    )


def process_lr_adjustment_params(
    model_lr_adjustment_groups, params_lr_adjustment_groups
):
    """
    Parses the model's supported lr adjustment groups and optionally overrides
    any user set scales
    Args:
        model_lr_adjustment_groups (dict): Keys are the
        LR group name and the values are LRAdjustmentGroup instances
        params_lr_adjustment_groups (dict): Keys are the
        LR group name and the values are the scale override value

    Returns:
        Tuple: A tuple consisting of a list of the adjustment scales with a
        corresponding list of parameter filter callables used to identify params
        that belong to the scales
    """
    lr_adjustment_scales: List[float] = []
    lr_adjustment_filters: List[List[FilterCallable]] = []
    for lr_group_name, lr_group in model_lr_adjustment_groups.items():
        if lr_group_name in params_lr_adjustment_groups:
            param_scale = params_lr_adjustment_groups.get(lr_group_name, 1.0)
            if param_scale != lr_group.scale:
                if lr_group.scale != 1.0:
                    logging.warning(
                        f"Overriding the scale for adjust_learning_rate group {lr_group_name} "
                        f"with the provided value of {param_scale}"
                    )
                lr_group.set_scale(param_scale)

        if lr_group.scale != 1.0:
            # Prior to 2.3 release, we had a single LR adjustment group which matched
            # a number of parameters. In 2.3, however, the adjustment groups were broken
            # up to allow finer-grained control over scaling different parameters.
            # To keep backwards compatibility with older checkpoints, we need to ensure
            # ordering of param groups remains the same even if old configs/checkpoints
            # are provided. As such, we merge params that have the same scale here which
            # ensures same ordering as before. If we treated them as separate groups and
            # merged them later on, the ordering would have been different.
            for idx in range(len(lr_adjustment_scales)):
                if lr_group.scale == lr_adjustment_scales[idx]:
                    lr_adjustment_filters[idx].append(lr_group.param_filter)
                    break
            else:
                lr_adjustment_scales.append(lr_group.scale)
                lr_adjustment_filters.append([lr_group.param_filter])

    return (lr_adjustment_scales, lr_adjustment_filters)
