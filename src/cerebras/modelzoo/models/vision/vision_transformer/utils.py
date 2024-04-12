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


def set_defaults(params):
    params["model"]["fp16_type"] = params["model"].get("fp16_type", "bfloat16")
    params["optimizer"]["loss_scaling_factor"] = params["optimizer"].get(
        "loss_scaling_factor", 1.0
    )

    params["train_input"]["mixed_precision"] = params["model"][
        "mixed_precision"
    ]
    params["train_input"]["fp16_type"] = params["model"]["fp16_type"]
    params["eval_input"]["mixed_precision"] = params["model"]["mixed_precision"]
    params["eval_input"]["fp16_type"] = params["model"]["fp16_type"]

    return params


def set_vit_model_defaults(model_params):
    if "default_initializer" not in model_params:
        model_params["default_initializer"] = {
            "name": "truncated_normal",
            "std": model_params["initializer_range"],
            "mean": 0.0,
            "a": model_params["initializer_range"] * -2.0,
            "b": model_params["initializer_range"] * 2.0,
        }

    if "image_size" not in model_params:
        default_image_size = [224, 224]
        model_params["image_size"] == default_image_size
        logging.warning(
            f"image_size is not set, defaulted to be {default_image_size}"
        )

    if "num_channels" not in model_params:
        default_num_channels = 3
        model_params["num_channels"] == default_num_channels
        logging.warning(
            f"num_channels is not set, defaulted to be {default_num_channels}"
        )

    if "patch_size" not in model_params:
        default_patch_size = 3
        model_params["patch_size"] == default_patch_size
        logging.warning(
            f"patch_size is not set, defaulted to be {default_patch_size}"
        )

    if "prepend_cls_token" not in model_params:
        default_prepend_cls_token = 3
        model_params["prepend_cls_token"] == default_prepend_cls_token
        logging.warning(
            f"prepend_cls_token is not set, defaulted to be {default_prepend_cls_token}"
        )

    return model_params
