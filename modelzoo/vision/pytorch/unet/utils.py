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

from modelzoo.common.pytorch import cb_model as cm
from modelzoo.vision.pytorch.utils.run_utils import (
    get_default_inis,
    update_runconfig_debug_args_path,
)


def set_defaults(params):
    """
    Update any missing parameters in the params dictionary with default values

    Args:
        params: The dictionary containing the params
    """

    default_inis_dict = get_default_inis()
    update_runconfig_debug_args_path(params, default_inis_dict)

    # For performance
    if params["runconfig"]["log_steps"] > 1:
        params["runconfig"]["skip_train_recv_activations"] = True

    if params["runconfig"]["checkpoint_steps"] == 0:
        logging.warning(
            f"Setting `runconfig.checkpoint_steps` to max_steps. "
            f"Setting to 0 only saves initial checkpoint"
        )
        params["runconfig"]["checkpoint_steps"] = params["runconfig"][
            "max_steps"
        ]

    # Data params:
    params["train_input"]["normalize_data_method"] = params["train_input"].get(
        "normalize_data_method", None
    )

    # Model params:
    if "input_channels" not in params["model"].keys():
        params["model"]["input_channels"] = params["train_input"].get(
            "image_shape"
        )[-1]

    input_mode = params["runconfig"]["mode"]
    shape_key = (
        "image_shape"
        if params["train_input"].get("image_shape")
        else "input_shape"
    )
    if input_mode == "eval_all":
        input_mode = "eval"
    params["model"]["image_shape"] = params[f"{input_mode}_input"].get(
        shape_key
    )
    params["model"]["batch_size"] = params[f"{input_mode}_input"]["batch_size"]
    convert_to_onehot = params["model"]["loss"] == "multilabel_bce"
    params['train_input']['convert_to_onehot'] = convert_to_onehot
    params['train_input']["use_worker_cache"] = params['train_input'].get(
        "use_worker_cache", False
    )
    params['eval_input']['convert_to_onehot'] = convert_to_onehot
    params['eval_input']["use_worker_cache"] = params['eval_input'].get(
        "use_worker_cache", False
    )
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
        logging.warning(
            "Setting downscale_encoder_blocks has no effect when using max_pool"
        )
    if (params["model"]["downscale_method"] == "max_pool") and (
        params["model"]["downscale_bottleneck"]
    ):
        logging.warning(
            "Setting downscale_bottleneck has no effect when using max_pool"
        )

    # ignore_background_class only used by dice + cross entropy loss
    params["model"]["ignore_background_class"] = params["model"].get(
        "ignore_background_class", True
    )

    # Param defaults for metrics
    params["model"]["eval_ignore_classes"] = params["model"].get(
        "eval_ignore_classes", []
    )
    params["model"]["compute_eval_metrics"] = params["model"].get(
        "compute_eval_metrics", True
    )
    params["model"]["eval_metrics"] = params["model"].get(
        "eval_metrics", ["mIOU", "DSC", "Acc"]
    )
    params["model"]["use_bfloat16"] = params["model"].get("use_bfloat16", False)

    if params["model"]["use_bfloat16"]:
        params["optimizer"]["loss_scaling_factor"] = 1.0

    params["model"]["shuffle_seed"] = params["train_input"].get("shuffle_seed")

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
    for model_key in ("mixed_precision", "loss", "use_bfloat16"):
        for input_key in ("train_input", "eval_input"):
            params[input_key][model_key] = params["model"].get(model_key)


def set_custom_stack_params(params):
    # TODO: Add custom stack params for UNet if any.
    # Fcn used in `model.py` __init__
    if cm.use_cs():
        from modelzoo.common.pytorch import cbtorch

        state = cbtorch.state()
        runconfig_params = params["runconfig"]
