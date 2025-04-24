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

from cerebras.modelzoo.config import BaseConfig
from cerebras.modelzoo.layers.init import (
    Initializer,
    NormalInitializer,
    TruncatedNormalInitializer,
)
from cerebras.pytorch.utils.utils import FilterCallable, make_param_filter

SUPPORTED_MUP_INITIALIZERS = [
    "normal",
    "truncated_normal",
]


class LRAdjustmentGroup(BaseConfig):
    model_config = dict(frozen=False)

    """
    Stores data for a group of params that share a learning rate scalar.
    Stores a callable that returns True if a given model param corresponds
    to the group. Additionally, it stores the scale that should be applied to
    the LR of the model params that correspond to the group.
    """

    param_filter_patterns: Union[str, List[str]] = ...
    """
    A string or a list of strings that contains glob expressions
    used to match whether a given model param name belongs to the group.
    """

    scale: Optional[float] = 1.0
    """The scale that should be applied to the LR of this group"""

    def __init__(self, param_filter_patterns, **kwargs):
        # This init is required to accept param_filter_patterns as a positional argument
        super().__init__(param_filter_patterns=param_filter_patterns, **kwargs)

    def set_scale(self, scale):
        self.scale = scale

    @property
    def param_filter(self):
        # Convert the strings into a callable that returns True if a given param
        # name corresponds to the LR group
        return make_param_filter(self.param_filter_patterns)


def scale_initializers_by_dimension(
    initializers: Union[Initializer, List[Initializer]],
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

    scaled_initializers = []
    for initializer in initializers:
        if isinstance(initializer, NormalInitializer):
            scaled_initializers.append(
                initializer.copy(update=dict(std=initializer.std * mup_scalar))
            )
        elif isinstance(initializer, TruncatedNormalInitializer):
            scaled_initializers.append(
                initializer.copy(
                    update=dict(
                        std=initializer.std * mup_scalar,
                        a=initializer.a * mup_scalar,
                        b=initializer.b * mup_scalar,
                    )
                )
            )
        elif isinstance(initializer, (str, dict)):
            #
            # TODO(SW-137621): This codepath is deprecated and will be removed in a future release
            #
            if isinstance(initializer, str):
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

            from cerebras.modelzoo.layers.create_initializer import (
                _get_spec_value,
            )

            if initializer_name == "normal":
                initializer["std"] = (
                    _get_spec_value(initializer, "std", 0.05) * mup_scalar
                )
            elif initializer_name == "truncated_normal":
                std = _get_spec_value(initializer, "std", 0.05)
                initializer["std"] = std * mup_scalar
                initializer["a"] = (
                    _get_spec_value(initializer, "a", -2 * std) * mup_scalar
                )
                initializer["b"] = (
                    _get_spec_value(initializer, "b", 2 * std) * mup_scalar
                )
                std = None

            scaled_initializers.append(initializer)
        else:
            raise TypeError(
                f"Only normal and truncated normal initializers are supported for muP. "
                f"Got: {initializer}"
            )

    return scaled_initializers


def is_mup(model: Union[dict, torch.nn.Module], print_log: bool = True):
    if isinstance(model, BaseConfig):
        model = model.dict()
    elif is_dataclass(model):
        model = asdict(model)
    elif not isinstance(model, dict):
        model = {
            k: getattr(model, k) for k in dir(model) if not k.startswith("__")
        }

    mup_params_found = any(
        name.startswith('mup_base_') and model[name] is not None
        for name in model
    )
    if mup_params_found and print_log:
        logging.info("This is a muP configured run")

    return mup_params_found


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
        LR group name and the values are the scale override value.

    Returns:
        Tuple: A tuple consisting of a list of the adjustment scales with a
        corresponding list of parameter filter callables used to identify params
        that belong to the scales
    """
    if any(
        key not in model_lr_adjustment_groups
        for key in params_lr_adjustment_groups
    ):
        missing_lr_adjust_keys = [
            key
            for key in params_lr_adjustment_groups
            if key not in model_lr_adjustment_groups
        ]
        raise ValueError(
            f"Learning rate adjustment key(s) from config parameters do not "
            f"match any model LR adjustment argument:\n{missing_lr_adjust_keys} "
            f"are not in the model's set: {sorted(model_lr_adjustment_groups)}"
        )

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
