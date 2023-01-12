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

from cerebras_appliance.appliance_manager import (
    TensorGroup,
    TensorGrouper,
    TensorSendPayload,
)
from modelzoo.common.model_utils.sparsity.sparsifiers import SPARSIFIER_MAP
from modelzoo.common.model_utils.sparsity.utils import extract_mask_from_weight
from modelzoo.common.pytorch.PyTorchBaseModel import PyTorchBaseModel


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
    # Get the numpy sparsifier params.
    # Translate from 1.8 style yaml to numpy sparsifier setting.
    mask_type = params["init_method"]
    if mask_type == "random":
        mask_type = "constant"
    sparsity_level = params["sparsity"]
    # Every weight needs a unique seed in case random pattern is used.
    # numpy.random.seed must be 32bit int
    seed = (params["seed"] + hash(weight_fw_name)) % (1 << 32)

    # Construct numpy sparsifier.
    sparsifier = SPARSIFIER_MAP[mask_type](
        n_iter=0,
        sparsity_level=sparsity_level,
        sparsity_distribution="uniform",  # Its a single weight anyway.
        seed=seed,
    )
    sparse_val = float("nan")  # hardcoded

    # Build the dense weight dict with only the single weight.
    # We've already determined we'll sparsify this at a higher level, so just
    # use "weight" as the name to force it through.
    WEIGHT = "weight"
    dense_weights_dict = {}
    for tensor in tensors:
        if tensor.fw_name == weight_fw_name:
            dense_weights_dict[WEIGHT] = tensor.tensor

    # Go compute new sparsity pattern and add NaNs.
    sparse_weight = sparsifier.get_masked_weights(
        0, dense_weights_dict, sparse_val
    )[WEIGHT]

    mask = extract_mask_from_weight(sparse_weight, sparse_val)

    # Modify `tensors` in-place.
    fw_names = []
    for tensor in tensors:
        fw_names.append(tensor.fw_name)
        if tensor.fw_name == weight_fw_name:
            tensor.tensor = sparse_weight
        else:
            tensor.tensor *= mask
    logging.debug(f"Applied {sparsity_level} sparsity to {fw_names}")


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

    # validate type
    valid_types = (
        "For 1.7, the only valid setting is `sideband`, but 1.8 will support "
        "`static`, `gmp`, `set`, or `rigl`."
    )
    sparsity_type = params.get("type")
    if sparsity_type is None:
        raise ValueError(
            f"To use sparsity, a `type` field must be present. {valid_types}"
        )
    elif sparsity_type != "sideband":
        raise ValueError(
            f"Invalid sparsity `type` \"{sparsity_type}\". {valid_types}"
        )

    # validate init_method, sparsity, and seed
    def validate_group(block, context):
        init_methods = ["random", "topk", "balanced-topk", "checkerboard"]
        init_method = block.get("init_method")
        if init_method not in init_methods:
            if init_method is None:
                raise ValueError(
                    f"{context} is missing required `init_method`. Valid "
                    f"options are: {init_methods}."
                )
            else:
                raise ValueError(
                    f"{context} has invalid `init_method`: \"{init_method}\". "
                    f"Valid options are: {init_methods}."
                )
        sparsity = block.get("sparsity")
        if sparsity is None:
            raise ValueError(
                f"{context} is missing required `sparsity` which must be a "
                f"fraction between 0.0 and 1.0 (inclusive)."
            )
        elif not isinstance(sparsity, float) or sparsity < 0 or sparsity > 1:
            raise ValueError(
                f"{context} has invalid `sparsity`: {sparsity}, which must be "
                f"a fraction between 0.0 and 1.0 (inclusive)."
            )

        seed = block.get("seed")
        if seed is not None and not isinstance(seed, int):
            raise ValueError(
                f"{context} has invalid `seed`: {seed}, which must be "
                f"an integer used to seed np.random.seed."
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

    sparsify_params = {
        "sparsity": params.get("sparsity", 0.0),
        "init_method": params.get("init_method", "topk"),
        "seed": params.get("seed", 0),
    }

    def should_sparsify(name, param):
        # By default, sparsify params that are > 1D and not embedding or norm.
        name = name.lower()
        if len(param.shape) <= 1 or "embedding" in name or "norm" in name:
            return False
        return True

    def get_sparsify_params(name, param):
        if should_sparsify(name, param):
            # Return defaults.
            return sparsify_params
        return None

    # Check if there is a yaml specified param name pattern
    param_name_patterns = params.get("param_name_patterns", None)
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
    optimizer_state = state_dict["optimizer"]["state"]

    # For printing a single info level summary log statement.
    summary_info = defaultdict(int)

    for name, param in model.model.named_parameters():
        weight_sparsify_params = get_sparsify_params(name, param)
        if not weight_sparsify_params:
            # None result means no sparsity at all.
            continue

        # All names the appliance deals with are flattened with "."
        name = "model." + name

        # Go find related optimizer state tensors.
        # Then, find the state_dict key for that tensor.
        opt_state_names = []
        for state_name, state_tensor in optimizer.state.get(param, {}).items():
            if state_tensor.shape != param.shape:
                # Only consider optimizer state of same shape as parameter.
                continue
            id_state = id(state_tensor)
            for param_id, state in optimizer_state.items():
                if state_name in state and id(state[state_name]) == id_state:
                    # Found the state_dict key for this related tensor.
                    opt_state_name = f"optimizer.state.{param_id}.{state_name}"
                    opt_state_names.append(opt_state_name)
                    break

        logging.debug(
            f"Sparsity for \"{name}\" + {opt_state_names}: "
            f"{weight_sparsify_params}"
        )
        summary_info[tuple(weight_sparsify_params.items())] += 1

        sparse_tensor_groups[name] = (weight_sparsify_params, opt_state_names)

    for tuple_params, count in summary_info.items():
        logging.info(
            f"Will apply sparsity to {count} weights with {dict(tuple_params)}"
        )

    return functools.partial(sparsify_grouper, sparse_tensor_groups)
