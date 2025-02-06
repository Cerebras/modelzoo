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

"""Helper functions related to configuration of the Optimizer."""

from collections import defaultdict

import torch

from cerebras.modelzoo.common.utils.model.mup_utils import (
    LRAdjustmentGroup,
    process_lr_adjustment_params,
)
from cerebras.pytorch.utils.utils import convert_glob_to_regex


def _named_parameters_requiring_grad(model):
    """
    Returns the named paramters that should be passed to the optimizer
    i.e. are trainable because they require gradients.
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            yield name, param


def _get_adaptive_lr_layers(model, lr_adjustment_layer_filters):
    """
    Args:
        model: Pytorch model
        lr_adjustment_layer_filter: List of callables that return True if a given
        param name belongs to the adaptive_lr_layer group.

    Returns:
        list: list of layer names for the given lr_adjustment_layer_type
    """
    return [
        n
        for n, p in model.named_parameters()
        if any(f(n, p) for f in lr_adjustment_layer_filters)
    ]


def _should_apply_weight_decay(model, param_name):
    """

    Args:
        model: Pytorch model
        param_name (torch.nn.Parameter): model param name

    Returns:
        bool: whether to apply weight decay for the give param_name
    """
    norm_modules = (
        torch.nn.LayerNorm,
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.GroupNorm,
        torch.nn.SyncBatchNorm,
    )
    if 'bias' in param_name:
        return False
    for name, module in model.named_modules():
        if name in param_name:
            if isinstance(module, norm_modules):
                return False
    return True


def _partition_params_groups_with_weight_decay(
    model, param_groups, weight_decay
):
    """
    Args:
        model : Pytorch model
        param_groups (list): optimizer param_groups.
        weight_decay (float): value of weight decay rate

    Returns:
        list: param_groups as list of dicts, split based on the weight_decay rate
    """
    refined_params_groups = []
    for _ in range(2 * len(param_groups)):
        refined_params_groups.append({"params": []})
    for idx, param_group_ in enumerate(param_groups):
        # Set group's weight decay params
        refined_params_groups[2 * idx]["weight_decay"] = weight_decay
        refined_params_groups[2 * idx + 1]["weight_decay"] = 0.0
        for name, param in param_group_["params"]:
            if _should_apply_weight_decay(model, name):
                refined_params_groups[2 * idx]["params"].append((name, param))
            else:
                refined_params_groups[2 * idx + 1]["params"].append(
                    (name, param)
                )
        # Propogate tags to new param groups
        # all param groups are being split in half here
        # need to propagate the tags to both halves
        tags = param_group_.get("tags", None)
        if tags:
            refined_params_groups[2 * idx]["tags"] = tags
            refined_params_groups[2 * idx + 1]["tags"] = tags

    return refined_params_groups


def _construct_adjusted_lr_param_group(param_group, adjusted_lr):
    new_param_group = {
        "params": [],
        "weight_decay": param_group["weight_decay"],
        "adjust_learning_rate": adjusted_lr,
    }

    # Propogate tags to new param groups
    tags = param_group.get("tags", None)
    if tags:
        new_param_group["tags"] = tags

    return new_param_group


def _partition_params_groups_with_adjusted_lr(
    model,
    param_optimizer_grouped,
    lr_adjustment_groups,
):
    """
    Generates param_groups based on the lr_adjustment_layers
    Each lr adjustment layer_type will have a group asociated with it.

    Args:
        model : Pytorch model
        param_optimizer_grouped (list): param_groups before the split based on lr_adjustment_layers
        lr_adjustment_scalars (list): lr adjustment scalars
        lr_adjustment_filters (list): callables for each scalar that return
        True if a given param name corresponds to a param that should be scaled
        by the scalar value

    Returns:
        list: list of dicts of param groups
    """
    lr_adjustment_scalars, lr_adjustment_filters = lr_adjustment_groups

    if lr_adjustment_scalars:
        param_groups_with_lr_adjustment = []
        for param_group in param_optimizer_grouped:
            refined_param_groups = [
                _construct_adjusted_lr_param_group(
                    param_group,
                    (
                        lr_adjustment_scalars[idx]
                        if idx < len(lr_adjustment_scalars)
                        else 1.0
                    ),
                )
                for idx in range(len(lr_adjustment_scalars) + 1)
            ]
            # collect all the params whose layer_type is not in lr_adjustment_layers
            # in the last param group
            adaptive_lr_layers = [
                _get_adaptive_lr_layers(model, lr_adjust_layer_filters_)
                for lr_adjust_layer_filters_ in lr_adjustment_filters
            ]
            for name, param in param_group["params"]:
                param_in_adjust_lr_groups = False
                for idx, _ in enumerate(lr_adjustment_scalars):
                    # check if param belongs to one of the adaptive lr layer types
                    if any(
                        adaptive_lr_layer_ in name
                        for adaptive_lr_layer_ in adaptive_lr_layers[idx]
                    ):
                        refined_param_groups[idx]["params"].append(
                            (name, param)
                        )
                        param_in_adjust_lr_groups = True
                # if param doesn't belongs to one of the adaptive lr layer types,
                # put it in the last refined_param_group
                if not param_in_adjust_lr_groups:
                    refined_param_groups[-1]["params"].append((name, param))

            # remove empty param groups
            refined_param_groups = [
                param_group_
                for param_group_ in refined_param_groups
                if param_group_["params"]
            ]

            if "tags" not in param_group:
                # remove duplicate groups
                unique_scales = []
                merged_refined_param_groups = []
                for param_group in refined_param_groups:
                    scale = param_group['adjust_learning_rate']
                    if scale not in unique_scales:
                        unique_scales.append(scale)

                for scale in unique_scales:
                    merged_refined_param_groups.append(
                        {
                            "params": [],
                            "weight_decay": param_group["weight_decay"],
                            "adjust_learning_rate": scale,
                        }
                    )
                    for param_group in refined_param_groups:
                        if param_group['adjust_learning_rate'] == scale:
                            merged_refined_param_groups[-1]['params'].extend(
                                param_group['params']
                            )

                param_groups_with_lr_adjustment.append(
                    merged_refined_param_groups
                )
            else:
                param_groups_with_lr_adjustment.append(refined_param_groups)
    else:
        param_groups_with_lr_adjustment = param_optimizer_grouped

    # flatten the param group list if nested
    param_groups_with_lr_adjustment_flattened = []
    for groups in param_groups_with_lr_adjustment:
        if isinstance(groups, list):
            for group_ in groups:
                param_groups_with_lr_adjustment_flattened.append(group_)
        else:
            param_groups_with_lr_adjustment_flattened.append(groups)

    return param_groups_with_lr_adjustment_flattened


def partition_params_group_with_tags(param_group: dict, optimizer_params: dict):
    """
    Splits a single param group into multiple param groups based on glob
    patterns found in optimizer_params. Patterns are expected to be located
    under the key ``params`` which contains a list of patterns and tags following
    this format:

    .. code:: yaml

        params:
            - params: "glob-pattern-1"
              tag: "tag1"
            - params: "glob-pattern-2"
              tag: "tag2"

    Args:
        param_group (dict): a single param group containing a list of name-param
            pairs under the ``params`` key.
        optimizer_params (dict): optimizer parameters.
    """

    filters = optimizer_params.get("params", [])
    if isinstance(filters, dict):
        filters = [filters]

    # Construct a map where each parameter stores the list of tags.
    # At the end each list of test will refere to a specific group
    # of parameters.
    param_to_tags = {(name, param): [] for name, param in param_group["params"]}
    for param_filter in filters:
        tag = param_filter["tag"]
        assert isinstance(param_filter, dict)
        patterns = param_filter["params"]
        if not isinstance(patterns, list):
            patterns = [patterns]
        patterns = list(map(convert_glob_to_regex, patterns))
        matches = [
            (name, param)
            for name, param in param_group["params"]
            if any(pattern.search(name) for pattern in patterns)
        ]
        if matches:
            for pair in matches:
                param_to_tags[pair].append(tag)
        else:
            raise ValueError(
                f"Param filter {tag} with patterns {patterns} "
                f"did not match any params."
            )

    # Create a map from tags to list of parameters.
    partitioned_param_groups = defaultdict(list)
    for param_name_pair, tags in param_to_tags.items():
        partitioned_param_groups[frozenset(tags)].append(param_name_pair)

    return [
        (
            {
                "params": params,
                "tags": set(tags),
            }
            if tags
            else {
                "params": params,
            }
        )
        for tags, params in partitioned_param_groups.items()
    ]


def configure_param_groups(model: dict, optimizer_params: dict):
    """
    Groups the optimizer parameters into non-overlapping groups.
    The groups are formed along the two axis:
    i) if weight_decay is >0 or not for the param
    ii) unique adjust_learning_rate (learning_rate_scaling) for the param

    Args:
        model (dict): Pytorch model
        optimizer_params (dict): optimizer paramters

    Returns:
        _type_: _description_
    """
    # muP backwards compatibility
    default_lr_adjustment_groups = {
        "embedding": LRAdjustmentGroup("*embedding*weight"),
        "decoder_kernel": LRAdjustmentGroup(
            ["*decoder*dense*weight", "*decoder*linear*weight"]
        ),
    }
    lr_adjustment_groups = process_lr_adjustment_params(
        getattr(model, "lr_adjustment_groups", default_lr_adjustment_groups),
        optimizer_params.get("adjust_learning_rate", {}),
    )
    param_optimizer = list(_named_parameters_requiring_grad(model))
    # default: assemble all params in 1 group
    param_optimizer_grouped = [{"params": list(param_optimizer)}]

    param_optimizer_grouped = partition_params_group_with_tags(
        param_optimizer_grouped[0], optimizer_params
    )

    # Parse weight decay
    weight_decay = optimizer_params.get("weight_decay", 0.0)

    # split param_groups in 2 groups: with and without weight decay
    param_optimizer_grouped = _partition_params_groups_with_weight_decay(
        model,
        param_optimizer_grouped,
        weight_decay,
    )

    # remove empty param groups if tags are being used
    if any("tags" in group for group in param_optimizer_grouped):
        param_optimizer_grouped = [
            param_group_
            for param_group_ in param_optimizer_grouped
            if param_group_["params"]
        ]

    # create additional param groups for each layer type with lr adjustment scalar
    param_optimizer_grouped = _partition_params_groups_with_adjusted_lr(
        model,
        param_optimizer_grouped,
        lr_adjustment_groups,
    )
    # remove param name from the (name, param) tuple as the name was only used for referencing
    # while grouping params
    for group_idx in range(len(param_optimizer_grouped)):
        param_list = []
        for _, param in param_optimizer_grouped[group_idx]["params"]:
            param_list.append(param)
        param_optimizer_grouped[group_idx].pop("params")
        param_optimizer_grouped[group_idx]["params"] = param_list

    return param_optimizer_grouped


def flatten_optimizer_params(kwargs):
    """
    Config classes package optimizer related params in a sub dict.
    ALthough, if we use native yaml config, they come unrolled.
    This utility unwraps the optimizer related params(if present)
    into an unroller optimizer param dict for consistency.
    Args:
        kwargs : Input args dict
    Returns:
        flattened_args: Flattened dict
    """
    additional_dict = kwargs.get('optim_params', {})
    flattened_dict = kwargs.copy()

    for key, value in additional_dict.items():
        new_key = f"{key}"
        flattened_dict[new_key] = value

    if 'optim_params' in flattened_dict:
        # Remove the 'optim_params' key from the flattened dictionary
        del flattened_dict['optim_params']

    return flattened_dict
