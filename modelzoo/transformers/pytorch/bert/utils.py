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
import os

from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch import modes


def set_defaults(params):
    """
    Update any missing parameters in the params dictionary with default values

    Args:
        params: The dictionary containing the params
    """
    for section in ["train_input", "eval_input"]:
        for key in ["vocab_file"]:
            if params.get(section, {}).get(key):
                params[section][key] = os.path.abspath(params[section][key])

    model_params = params["model"]
    params["model"]["disable_nsp"] = model_params.get("disable_nsp", False)
    params["model"]["enable_vts"] = model_params.get("enable_vts", False)
    # Pass settings into data loader.
    for model_key in (
        "disable_nsp",
        "vocab_size",
        "enable_vts",
        "mixed_precision",
    ):
        for input_key in ("train_input", "eval_input"):
            params[input_key][model_key] = model_params.get(model_key)

    params["model"]["max_position_embeddings"] = model_params.get(
        "max_position_embeddings", params["train_input"]["max_sequence_length"],
    )
    params["model"]["to_float16"] = model_params.get("to_float16", False)
    params["model"]["use_bfloat16"] = model_params.get("use_bfloat16", False)
    params["optimizer"]["disable_lr_steps_reset"] = params["optimizer"].get(
        "disable_lr_steps_reset", True
    )


def set_custom_stack_params(params):
    if cm.use_cs():
        from modelzoo.common.pytorch import cbtorch

        state = cbtorch.state()
        state.full_config.placement.optimize_buses.deltat_relative_margin = 0.5
        if params["train_input"]["max_sequence_length"] <= 512:
            state.full_config.matching.kernel.no_dcache_spill_splits = True

        runconfig_params = params["runconfig"]
        if runconfig_params["mode"] == modes.EVAL:
            state.full_config.matching.add_pack_and_unpack.max_egress_per_pack = (
                1
            )
            state.full_config.placement.prep_recolor_kernels.wrap_pack_kernel = (
                True
            )

        if runconfig_params.get("multireplica", False):
            reshape_filter = (
                "reshape*,core/in_splits[-1]:!1,core/out_splits[-1]:!1;"
            )
            state.full_config.placement.place.custom_split_filter = (
                reshape_filter
            )


def check_unused_model_params(model_params):
    """
    While setting up the model, we pop used settings from model_params.
    This function sends a warning about any unused parameters.
    """
    model_params.pop("to_float16", None)
    model_params.pop("mixed_precision", None)
    if len(model_params) > 0:
        logging.warning(
            "The following model params are unused: "
            + ", ".join(model_params.keys())
        )
    logging.root.setLevel(logging.INFO)
