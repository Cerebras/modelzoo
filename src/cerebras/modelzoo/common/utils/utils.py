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

import os
import shutil
from typing import List, Optional, Union

import yaml

from cerebras.pytorch.experimental import Compression
from cerebras.pytorch.nn import SelectiveGrad
from cerebras.pytorch.sparse.configure import default_sparse_param_filter


def check_and_create_output_dirs(output_dir, filetype):
    contains_filetype = False
    if os.path.isdir(output_dir):
        for fname in os.listdir(output_dir):
            if filetype in fname:
                contains_filetype = True
                break

    if contains_filetype:
        _in = input(
            f"Output directory already contains {filetype} file(s)."
            + " Do you want to delete the folder to write"
            + " new records in the same output folder name? (yes/no): "
        )
        if _in.lower() in ["y", "yes"]:
            shutil.rmtree(output_dir)
        elif _in.lower() in ["n", "no"]:
            raise IsADirectoryError(
                "Create a new folder for the files you want to write!!"
            )
        else:
            raise ValueError(f"Inputs can be yes, no, y or n. Received {_in}!!")

    os.makedirs(output_dir, exist_ok=True)


def save_params(params, model_dir, fname="params.yaml"):
    """
    Writes and saves a dictionary to a file in the model_dir.

    :param dict params: dict we want to write to a file in model_dir
    :param string model_dir: Directory we want to write to
    :param string fname: Name of file in model_dir we want to save to.
    """
    if not model_dir:
        raise ValueError(
            "model_dir is not provided. For saving params, user-defined"
            + " model_dir must be passed either through the command line"
            + " or from the yaml"
        )

    params_fname = os.path.join(
        model_dir,
        fname,
    )
    try:
        os.makedirs(os.path.dirname(params_fname), exist_ok=True)
    except OSError as error:
        raise ValueError(
            f"Invalid path {model_dir} provided. Check the model_dir path!"
        )

    with open(params_fname, "w+") as _fout:
        yaml.dump(params, _fout, default_flow_style=False, sort_keys=False)


def update_debug_args_with_mem_limits(
    debug_args: 'DebugArgs', runconfig: Optional[dict] = None
):
    """Update debug args with runconfig memory limits"""
    if not runconfig:
        return

    prop_to_pb_attr = {
        "compile_crd_memory_gi": debug_args.debug_usr.compile_coord_resource,
        "execute_crd_memory_gi": debug_args.debug_usr.execute_coord_resource,
        "wrk_memory_gi": debug_args.debug_usr.worker_resource,
        "act_memory_gi": debug_args.debug_usr.activation_resource,
        "cmd_memory_gi": debug_args.debug_usr.command_resource,
        "wgt_memory_gi": debug_args.debug_usr.weight_resource,
    }

    for prop, pb_attr in prop_to_pb_attr.items():
        if prop in runconfig and runconfig[prop] is not None:
            pb_attr.memory_bytes = runconfig[prop] << 30  # gi to bytes


def format_rate(rate):
    return f"{rate:.2g}" if rate < 1.0 else f"{rate:.2f}"


def configure_compression(config: Union[dict, List[dict]]) -> List[Compression]:
    """
    Takes in a dictionary of configs and returns the output as a list of compressions
    """

    def get_compression_from_dict(single_config) -> Compression:
        if not isinstance(single_config, dict):
            raise ValueError(
                "Improper compression format due to configuration not being a dictionary"
            )
        if "format" not in single_config:
            raise ValueError(
                "Improper compression format due to configuration not having \"format\" as a field"
            )
        if "param_filter" not in single_config:
            raise ValueError(
                "Improper compression format due to configuration not having \"param_filter\" as a field"
            )

        return Compression(
            single_config["format"], single_config["param_filter"]
        )

    if isinstance(config, dict):
        # then turn this single dictionary value to a compression
        return [get_compression_from_dict(config)]
    elif isinstance(config, list):
        return list(map(get_compression_from_dict, config))

    raise ValueError(
        "Improper compression format due to configuration not being a dictionary or a list of configs"
    )


def configure_selective_gradient(
    config: Union[dict, List[dict]]
) -> List[SelectiveGrad]:
    """
    Takes in a dictionary of selective grad configs and returns the output as a list of
    SelectiveGrad
    """

    def get_selective_grad_from_dict(single_config) -> SelectiveGrad:
        # use the sparsity filter as well as a default filter
        param_filter = single_config.get("param_filter", None)
        if param_filter is None:
            param_filter = default_sparse_param_filter

        # make init_method an optional field
        if "init_method" in single_config:
            return SelectiveGrad(param_filter, single_config["init_method"])

        return SelectiveGrad(param_filter)

    if isinstance(config, dict):
        return [get_selective_grad_from_dict(config)]
    elif isinstance(config, list):
        return list(map(get_selective_grad_from_dict, config))
    raise ValueError(
        "Improper compression format due to configuration not being a dictionary or a list of configs"
    )


def merge_recursively(d1: dict, d2: dict, delval=None):
    """Merge new dict into orig dict recursively"""
    if isinstance(d1, dict) and isinstance(d2, dict):
        merged = {}
        for key, v1 in d1.items():
            if key not in d2:
                merged[key] = v1
            elif (v2 := d2[key]) is not delval and (
                value := merge_recursively(v1, v2, delval)
            ) is not delval:
                merged[key] = value

        for key, v2 in d2.items():
            if key not in d1:
                merged[key] = v2

        return merged

    elif isinstance(d1, (list, tuple)) and isinstance(d2, (list, tuple)):

        def get_all():
            yield from d1
            yield from d2

        return type(d1)(get_all())

    if d2 is not delval:
        return d2

    return d1
