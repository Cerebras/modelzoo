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
import warnings
import tempfile

from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch.utils import get_debug_args


def get_default_inis():
    return {
        "ws_opt_refinement_opt": "true",
        "ws_opt_act_buffer_cols": 8,
        "ws_opt_target_max_actv_wios": 16,
        "ws_opt_target_wgt_port_group_size": 12,
        "ws_rt_enable_worker_flow": "true",
        "ws_run_memoize_actv_mem_mapping": "true",
        "ws_rt_mem_limit_high": "0.5",
        "ws_opt_lower_transform_in_waf_to_rt": "true",
        "ws_opt_duplicate_weight_source": "false",
        "ol_csp_allow_transpose": "true",
    }


def set_defaults(params):
    """
    Update any missing parameters in the params dictionary with default values

    Args:
        params: The dictionary containing the params
    """

    def use_cs():
        k8s_cs_ip = os.environ.get("K8S_CS_IP", None)
        if k8s_cs_ip:
            cs_ip = k8s_cs_ip
        else:
            cs_ip = params["runconfig"]["cs_ip"]

        validate_only = params["runconfig"]["validate_only"]
        compile_only = params["runconfig"]["compile_only"] or validate_only
        appliance = params["runconfig"].get("appliance", False)
        use_cs = bool(cs_ip or compile_only or appliance)
        return use_cs

    if use_cs():
        from cerebras_appliance.run_utils import write_debug_args

        if not params["runconfig"].get("debug_args_path"):
            cwd = os.getcwd()

            with tempfile.TemporaryDirectory(dir=cwd) as ini_dir:
                # Create and populate debug.ini
                with open(os.path.join(ini_dir, "debug.ini"), "w") as fp:
                    for key, val in get_default_inis().items():
                        fp.write(f"{key}: {val}\n")

                # Uses debug.ini from current path if it exists
                debug_args = get_debug_args(None, ini_dir)

            # in the case debug_args_path is not set
            # create a default debug_args file in the debug_args_dir
            debug_args_path = os.path.join(
                params["runconfig"]["model_dir"], ".debug_args.proto"
            )
            write_debug_args(debug_args, debug_args_path)
            params["runconfig"]["debug_args_path"] = debug_args_path

    # Model params:
    if "input_channels" not in params["model"].keys():
        params["model"]["input_channels"] = params["train_input"].get(
            "image_shape"
        )[-1]

    params["model"]["num_classes"] = params["train_input"]["num_classes"]
    params["model"]["skip_connect"] = params["model"].get("skip_connect", True)
    params["model"]["downscale_method"] = params["model"].get(
        "downscale_method", "max_pool"
    )
    params["model"]["downscale_first_conv"] = params["model"].get(
        "downscale_first_conv", False,
    )

    params["model"]["residual_blocks"] = params["model"].get(
        "residual_blocks", False
    )
    params["model"]["use_conv3d"] = params["model"].get("use_conv3d", False)
    params["model"]["downscale_encoder_blocks"] = params["model"].get(
        "downscale_encoder_blocks",
        False if (params["model"]["downscale_method"] == "max_pool") else True,
    )
    params["model"]["downscale_bottleneck"] = params["model"].get(
        "downscale_bottleneck", False
    )
    if (params["model"]["downscale_method"] == "max_pool") and (
        params["model"]["downscale_encoder_blocks"]
    ):
        warnings.warn(
            "Setting downscale_encoder_blocks has no effect when using max_pool"
        )
    if (params["model"]["downscale_method"] == "max_pool") and (
        params["model"]["downscale_bottleneck"]
    ):
        warnings.warn(
            "Setting downscale_bottleneck has no effect when using max_pool"
        )

    # Param defaults for metrics
    params["model"]["eval_ignore_classes"] = params["model"].get(
        "eval_ignore_classes", None
    )
    params["model"]["compute_eval_metrics"] = params["model"].get(
        "compute_eval_metrics", True
    )
    params["model"]["eval_metrics"] = params["model"].get(
        "eval_metrics", ["mIOU", "DSC", "Acc"]
    )
    params["model"]["use_bfloat16"] = params["model"].get("use_bfloat16", False)
    downscale_method = params["model"]["downscale_method"]
    convs_per_block = params["model"]["convs_per_block"]
    skip_connect = params["model"]["skip_connect"]
    if (
        skip_connect
        and downscale_method == "strided_conv"
        and len(convs_per_block) == 1
    ):
        raise ValueError(
            f"skip_connect cannot be True when "
            f"downscale_method = {downscale_method} "
            f"and len(convs_per_block) = {len(convs_per_block)}. "
            f"Either set `skip_connect` = `False` (or) "
            f"change `downscale_method` = `max_pool`."
        )

    # Pass settings into data loader.
    for model_key in ("mixed_precision", "loss"):
        for input_key in ("train_input", "eval_input"):
            params[input_key][model_key] = params["model"].get(model_key)


def set_custom_stack_params(params):
    # TODO: Add custom stack params for UNet if any.
    # Fcn used in `model.py` __init__
    if cm.use_cs():
        from modelzoo.common.pytorch import cbtorch

        state = cbtorch.state()
        runconfig_params = params["runconfig"]
