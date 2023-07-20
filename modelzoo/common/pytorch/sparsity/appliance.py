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
Preview release of static sparsity work.

This module contains a function that runs in the appliance_client to modify
groups of tensors while in-flight to the appliance according to sparsification
settings. It also contains a function to help build those tensor groups and
configure the sparsifier.

This only works in appliance mode.
"""

import functools
import logging
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np

from cerebras_appliance.appliance_manager import (
    TensorGroup,
    TensorGrouper,
    TensorSendPayload,
)
from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch.PyTorchBaseModel import PyTorchBaseModel


def compute_mask(
    params: dict, weight: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a sparsity mask for the given weight according to params

    Args:
        params: configuration of the sparsity to compute
        weight: The initial weight values.

    Returns:
        mask with np.dtype bool of same shape as weight indicating sparsity
        pattern: True: keep weight. False: prune weight.

        regrow with np.dtype bool indicating positions which _were_ pruned that
        should instead be regrown as zeros
    """
    # Get the numpy sparsifier params.
    init_method = params["init_method"]
    sparsity = params["sparsity"]

    # Positions of all NaNs.
    pruned = ~np.isfinite(weight)
    num_pruned = np.count_nonzero(pruned)

    rng = np.random.default_rng(params["seed"])

    if init_method == "random":
        # Score is randomly distributed values in range (0,1)
        score = rng.random(weight.shape, dtype=weight.dtype)

        # Existing pruned positions should be low magnitude to be chosen only
        # if we are _decreasing_ sparsity, so negate their score.
        if num_pruned:
            score[pruned] *= -1
    elif init_method == "topk":
        # score is weight magnitude
        score = np.abs(weight)

        # The logical magnitude of pruned weights is zero, but their literal
        # magnitude is NaN. In order to prevent NaNs from disrupting tokp and
        # to possibly regrow deterministically, use a seeded RNG to put a
        # random negative score into all pruned positions. Such a negative
        # score will only be selected by topk if we are _decreasing_ sparsity.
        if num_pruned:
            score[pruned] = -rng.random(num_pruned, dtype=weight.dtype)
    elif init_method == "from_zeros":
        # Result mask is actually independent of requested sparsity level.
        # In order to be "idempotent", either a pre-existing prune _or_ a zero
        # count as pruned. However, never regrow.
        return ~(weight == 0 | pruned), np.zeros_like(pruned)

    out_groups = params.get("out_groups")
    in_groups = params.get("in_groups")

    score_shape = score.shape
    if out_groups:
        # [N*O, I] -> [N, O*I]
        score = score.reshape(out_groups, -1)
    elif in_groups:
        # [O, N*I] -> [N*I, O] -> [N, I*O]
        score = score.transpose(1, 0).reshape(in_groups, -1)
    else:
        score = score.reshape(1, -1)

    # Compute the number of elements to keep, rounding toward sparsity
    numel = score.shape[-1]
    keep = numel - int(np.round(sparsity * numel))
    # Compute the indices of the score to keep (within each group)
    keep_index = np.argpartition(score, -keep, axis=-1)[:, -keep:]

    # Put True into the kept positions (within each group)
    mask = np.zeros(score.shape, dtype=np.bool)
    np.put_along_axis(mask, keep_index, values=True, axis=-1)

    # Ungroup mask back to shape of score
    if in_groups:
        O, I = score_shape
        # [N, I*O] -> [N*I, O] -> [O, I*N]
        mask = mask.reshape(I, O).transpose(1, 0)
    else:
        mask = mask.reshape(score_shape)

    # If a weight is kept now but it was pruned before, we're regrowing.
    regrow = mask & pruned
    return mask, regrow


def appliance_sparsify(
    params: dict, weight_fw_name: str, tensors: List[TensorSendPayload]
) -> None:
    """
    This function is used as a closure on the TensorGroup for applying
    sparsity. params and weight_fw_name are added as a functools.partial
    closure and must be pickleable to send over to the multiprocessing.Pool.

    Args:
        params: numpy sparsifier params for this single weight.
        weight_fw_name: The fw_name of the weight in tensors.
        tensors: tensors in the group, with their tensor member set.
    """

    # Compute the mask for these sparsity params and the main tensor
    for tensor in tensors:
        if tensor.fw_name == weight_fw_name:
            mask, regrow = compute_mask(params, tensor.tensor)
            break

    if params.get("dump_masks"):
        np.save(weight_fw_name, mask)

    # Modify `tensors` in-place, poking nans into their pruned positions.
    fw_names = []
    for tensor in tensors:
        fw_names.append(tensor.fw_name)
        # Poke NaN into pruned positions.
        tensor.tensor[~mask] = float('nan')
        # Regrow unpruned into zero magnitude for both weight and opt state.
        tensor.tensor[regrow] = 0
    logging.debug(f"Applied sparsity {params} to {fw_names}")


def sparsify_grouper(
    sparse_tensor_groups: Dict[str, Tuple[dict, List[str]]],
    tensors: Iterable[TensorSendPayload],
) -> Iterable[TensorGroup]:
    """
    Constuct a grouping of tensors from the given lazy tensors. The group
    will consist of weights needing sparsification and their associated
    optimizer state, or single-tensor "groups" for tensors not needing
    sparsification.

    Args:
        sparse_tensor_groups: FW names of weights needing sparsity applied
            mapping to the and the sparsity params and optimizer state fw
            names.
        tensors: all tensors to actually send

    Yields:
        TensorGroup objects with either single tensors or group of tensors
            with a closure to apply sparsity
    """

    # sparse_tensor_groups is based on fw_name, so build that lookup here.
    all_tensors = {tensor.fw_name: tensor for tensor in tensors}

    # First, build all groups for parameters that need sparsity.
    for name, (params, opt_names) in sparse_tensor_groups.items():
        if name not in all_tensors:
            # Possibly duplicate or unused weight.
            logging.warning(
                f"Couldn't find {name} for sparsification. This parameter "
                f"may be unused or a duplicate."
            )
            continue

        # Remove main weight tensors.
        group = [all_tensors.pop(name)]

        # Remove associated optimizer state tensors.
        for opt_name in opt_names:
            group.append(all_tensors.pop(opt_name))

        # Build the closure which needs to be pickle-able.
        closure = functools.partial(appliance_sparsify, params, name)

        # Process these tensors as a group, running the sparsify logic into
        # the weight sending process pool.
        yield TensorGroup(tensors=group, closure=closure)

    # Return a single item group for all remaining tensors.
    for tensor in all_tensors.values():
        yield TensorGroup(tensors=[tensor])


def validate_sparsity_params(params: dict):
    """
    Validates the sparsity block of the model configuration. A ValueError will
    be raised if there are any invalid or unsupported settings.
    """

    # validate init_method, sparsity, and seed
    def validate_group(block, context):
        init_methods = ["random", "topk", "from_zeros"]
        init_method = block.get("init_method")
        if init_method is None:
            raise ValueError(
                f"{context} is missing required `init_method`. Valid "
                f"options are: {init_methods}."
            )
        elif init_method not in init_methods:
            raise ValueError(
                f"{context} has invalid `init_method`: \"{init_method}\". "
                f"Valid options are: {init_methods}."
            )

        sparsity = block.get("sparsity")
        if sparsity is None:
            raise ValueError(
                f"{context} is missing required `sparsity` which must be a "
                f"fraction in the range [0.0, 1.0)."
            )
        elif not isinstance(sparsity, float) or sparsity < 0 or sparsity >= 1:
            raise ValueError(
                f"{context} has invalid `sparsity`: {sparsity}, which must be "
                f"a fraction in the range [0.0, 1.0)."
            )

        seed = block.get("seed")
        if seed is not None and not isinstance(seed, int):
            raise ValueError(
                f"{context} has invalid `seed`: {seed}, which must be "
                f"an integer used to seed np.random.seed."
            )

        if "out_groups" in block and "in_groups" in block:
            raise ValueError(
                f"{context} has specified both `out_groups` and `in_groups` "
                f"which are mutually exclusive."
            )

        g = block.get("out_groups")
        if g is not None and (not isinstance(g, int) or g <= 0):
            raise ValueError(
                f"{context} has invalid `out_groups`: {g}, which must be a "
                f"positive integer."
            )
        g = block.get("in_groups")
        if g is not None and (not isinstance(g, int) or g <= 0):
            raise ValueError(
                f"{context} has invalid `in_groups`: {g}, which must be a "
                f"positive integer."
            )

    validate_top = True
    param_name_patterns = params.get("param_name_patterns")
    if param_name_patterns is None:
        # valid
        pass
    elif isinstance(param_name_patterns, str):
        try:
            re.compile(param_name_patterns)
        except Exception as e:
            raise ValueError(
                f"Invalid `param_name_patterns` \"{param_name_patterns}\". "
                f"When `param_name_patterns` is a string, it must be a regex "
                f"to match against parameter names."
            ) from e
    elif isinstance(param_name_patterns, list):
        # A list of several patterns, all of which get the default setting.
        for pattern in param_name_patterns:
            try:
                re.compile(pattern)
            except Exception as e:
                raise ValueError(
                    f"Invalid `param_name_patterns` entry \"{pattern}\". "
                    f"When `param_name_patterns` is a list, each entry must "
                    f"be a regex to match against parameter names."
                ) from e
    elif isinstance(param_name_patterns, dict):
        # An entire params group per pattern. Use the global sparsify_params as
        # defaults, filling in each group's customization.
        for pattern, block in param_name_patterns.items():
            try:
                re.compile(pattern)
            except Exception as e:
                raise ValueError(
                    f"Invalid `param_name_patterns` entry \"{pattern}\". "
                    f"When `param_name_patterns` is a dict, each key must "
                    f"be a regex to match against parameter names."
                ) from e
            if not isinstance(block, dict):
                raise ValueError(
                    f"Invalid setting for `param_name_patterns` entry "
                    f"\"{pattern}\". When `param_name_patterns` is a dict, "
                    f"each value must itself be a dictionary containing more "
                    f"sparsity settings specific to each group."
                )
            validate_top = False
            # build group starting from default settings overriding with block
            group = {**params, **block}
            validate_group(group, context=f"`{pattern}`")
    else:
        raise ValueError(
            f"Invalid `param_name_patterns`: \"{param_name_patterns}\". "
            f"This can be a regex, a list of regex, or a dict mapping regex "
            f"to a dictionary of further sparsity settings."
        )

    if validate_top:
        validate_group(params, context="`sparsity`")


def build_sparsify_grouper(
    params: dict, model: PyTorchBaseModel
) -> TensorGrouper:
    """
    Construct a function building the tensor groups according to parameters
    needing sparsification and their associated optimizer state.

    Args:
        params: top-level "sparsity" params from the yaml
        model: Model to pull parameters and associated optimzer state from.

    Returns:
        Function to be used as the appliance send_weights_grouper
    """

    # Determine which paramters need sparsity, construct the sparsifier params
    # from the yaml, and group those with their optimizer state tensor names.

    state_dict = model.get_state()

    # Map fw_weight_name:(params, [opt_names])
    # This dict is the only communication between build_sparsity_groups and the
    # enclosed grouper function which is returned.
    sparse_tensor_groups = {}

    sparsify_params = params.copy()

    # Set up the base seed if no param group overrides it.
    seed = sparsify_params.get("seed")
    if seed is None:
        # Choose a random 32bit number
        sparsify_params["seed"] = np.random.randint(1 << 32)

    def should_sparsify(name, param):
        # By default, sparsify params that are > 1D and not embedding or norm.
        name = name.lower()
        if (
            len(param.shape) <= 1
            or "embedding" in name
            or "norm" in name
            or "lm_head" in name
        ):
            return False
        return True

    def get_sparsify_params(name, param):
        if should_sparsify(name, param):
            # Return defaults.
            return sparsify_params
        return None

    # Check if there is a yaml specified param name pattern
    param_name_patterns = sparsify_params.pop("param_name_patterns", None)
    if isinstance(param_name_patterns, str):
        # Just a single config changing which params the defaults apply to.
        pattern = re.compile(param_name_patterns)

        # pylint: disable=function-redefined
        def should_sparsify(name, param):
            return pattern.search(name)

    elif isinstance(param_name_patterns, list):
        # A list of several patterns, all of which get the default setting.
        patterns = list(map(re.compile, param_name_patterns))

        # pylint: disable=function-redefined
        def should_sparsify(name, param):
            return any(map(lambda patt: patt.search(name), patterns))

    elif isinstance(param_name_patterns, dict):
        # An entire params group per pattern. Use the global sparsify_params as
        # defaults, filling in each group's customization.
        patterns = [
            (re.compile(pattern), {**sparsify_params, **p})
            for pattern, p in param_name_patterns.items()
        ]

        # pylint: disable=function-redefined
        def get_sparsify_params(name, param):
            for pattern, params in patterns:
                if pattern.search(name):
                    # Return the params for this group.
                    return params
            # No match means no sparsity.
            return None

    optimizer = model.get_optimizer()
    optimizer_state = state_dict.get("optimizer", {}).get("state", {})

    # For printing a single info level summary log statement.
    summary_info = defaultdict(int)

    for i, (name, param) in enumerate(model.model.named_parameters()):
        weight_sparsify_params = get_sparsify_params(name, param)
        if not weight_sparsify_params:
            # None result means no sparsity at all.
            continue
        cm.set_attribute(param, "sparse", True)
        sparsity = weight_sparsify_params["sparsity"]
        cm.set_attribute(param, "sparsity", sparsity)

        # All names the appliance deals with are flattened with "."
        name = "model." + name

        # Go find related optimizer state tensors.
        # Then, find the state_dict key for that tensor.
        opt_state_names = []
        if optimizer:
            opt_state = optimizer.state.get(param, {})
        else:
            opt_state = {}
        for state_name, state_tensor in opt_state.items():
            if state_tensor.shape != param.shape:
                # Only consider optimizer state of same shape as parameter.
                continue
            id_state = id(state_tensor)
            for param_id, state in optimizer_state.items():
                if state_name in state and id(state[state_name]) == id_state:
                    # Found the state_dict key for this related tensor.
                    opt_state_name = f"optimizer.state.{param_id}.{state_name}"
                    opt_state_names.append(opt_state_name)
                    cm.set_attribute(state_tensor, "sparse", True)
                    cm.set_attribute(state_tensor, "sparsity", sparsity)
                    break

        # Aggregate summary before setting per weight unique seed.
        summary_info[tuple(weight_sparsify_params.items())] += 1

        # Copy the per-weight params since we modify it to set a unique seed.
        weight_sparsify_params = weight_sparsify_params.copy()
        # Every weight needs a unique seed in case random pattern is used.
        # Sparsity is applied to tensors independently in parallel
        # processes and the rng state between weights is not shared.
        weight_sparsify_params["seed"] += i

        logging.debug(
            f"Sparsity for \"{name}\" + {opt_state_names}: "
            f"{weight_sparsify_params}"
        )

        sparse_tensor_groups[name] = (weight_sparsify_params, opt_state_names)

    for tuple_params, count in summary_info.items():
        logging.info(
            f"Will apply sparsity to {count} weights with {dict(tuple_params)}"
        )
    if not summary_info:
        logging.warning("No sparsity applied.")
        if param_name_patterns:
            logging.warning(
                f"Sparsity was configured with custom `param_name_patterns`: "
                f"{param_name_patterns}, but no parameter names matched "
                f"those patterns. Check the model.named_parameters()."
            )
        else:
            logging.warning(
                "Sparsity was configured with the default heuristic of "
                "applying to multidimensional parameters excluding bias, "
                "embedding, and normalization layers. But no parameters met "
                "those conditions. Check the model.named_parameters() and "
                "specify `param_name_patterns` in the sparsity config block "
                "to opt-in to sparsify some parameters."
            )

    return functools.partial(sparsify_grouper, sparse_tensor_groups)
