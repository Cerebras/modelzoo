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

import math

from cerebras.modelzoo.models.multimodal.multimodal_utils import (
    set_model_defaults_for_components,
)


def set_input_defaults(params):
    tparams = params["train_input"]
    eparams = params["eval_input"]
    mparams = params["model"]

    # Copy params from model section to
    # train_input and eval_input section
    _copy_params_across(params)

    tparams["use_worker_cache"] = tparams.get("use_worker_cache", False)
    eparams["use_worker_cache"] = eparams.get("use_worker_cache", False)

    tparams["noaugment"] = tparams.get("noaugment", False)
    eparams["noaugment"] = eparams.get("noaugment", False)

    # TODO: be consistent with naming `image_size` vs `image_data_size`
    tparams["image_data_size"] = [
        mparams["image_model"]["num_channels"],
        *mparams["image_model"]["image_size"],
    ]
    eparams["image_data_size"] = tparams["image_data_size"]

    if len(tparams["image_data_size"]) == 3:
        tparams["num_patches"] = (
            tparams["image_data_size"][-1]
            // mparams["image_model"]["patch_size"][0]
        ) * (
            tparams["image_data_size"][-2]
            // mparams["image_model"]["patch_size"][1]
        )
    else:
        tparams["num_patches"] = tparams["image_data_size"][0]
        # when len(tparams["image_data_size"]) == 2, we assume image has already been
        # embedded and the ViT model is not included in the LLaVA model
    eparams["num_patches"] = tparams["num_patches"]

    mparams["image_feature_select_mode"] = mparams.get(
        "image_feature_select_mode", "patch"
    )
    _valid_vals = ["cls_patch", "patch"]
    _msg_feature = f"Incorrect value for `image_feature_select_mode`, valid are one of {_valid_vals}"
    assert mparams["image_feature_select_mode"] in _valid_vals, _msg_feature


def set_model_defaults(params):
    mparams = params["model"]
    mpparams = params["model"].get("projector", None)
    if (
        mpparams is None
        and mparams["image_model"]["hidden_size"]
        != mparams["text_model"]["hidden_size"]
    ):
        raise ValueError(
            f"The model should have a projector when the image model "
            f"and text model do not have the same `hidden_size`"
        )

    # Loss related
    mparams["loss_scaling"] = mparams.get("loss_scaling", "num_tokens").lower()
    mparams["loss_weight"] = mparams.get("loss_weight", 1.0)
    mparams["label_smoothing"] = mparams.get("label_smoothing", 0.0)
    mparams["z_loss_eps"] = mparams.get("z_loss_eps", 0.0)

    # set defaults for model.image_model, model.text_model,
    # model.projector.image_model, model.projector.text_model
    set_model_defaults_for_components(
        params, model_keys=("image_model", "text_model")
    )

    if mparams["projector"]:
        for comp in mparams["projector"].values():
            if comp["name"] == "FeedForwardNetwork":
                # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
                # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
                # https://github.com/pytorch/pytorch/issues/57109
                comp["kernel_initializer"] = {
                    "name": "kaiming_uniform",
                    "a": math.sqrt(5),
                }
                # Note: Using Kaiming_uniform directly on bias tensor
                # results in PyTorch error:`ValueError: Fan in and fan out
                # can not be computed for tensor with fewer than 2 dimensions`
                # While this mismatches the src code, since we load from
                # HF -> CS converted checkpoint, this is initialized in the
                # checkpoint correctly
                comp["bias_initializer"] = {
                    "name": "zeros",
                }

    mparams["image_feature_select_layer_idx"] = mparams.get(
        "image_feature_select_layer_idx", -1
    )  # Last encoder layer as default
    if mparams["image_feature_select_layer_idx"]:
        # Convert negative index and positive index representing layer_id of encoder to positive index. All indices are zero-based.
        mparams["image_feature_select_layer_idx"] = (
            mparams["image_feature_select_layer_idx"]
            % mparams["image_model"]["num_hidden_layers"]
        )


def set_defaults(params):
    set_model_defaults(params)
    set_input_defaults(params)

    return params


def _copy_params_across(params):
    # Pass model settings into data loader.
    _model_to_input_map = [
        "mixed_precision",
        "fp16_type",
        ("image_model.patch_size", "patch_size"),
        ("image_model.prepend_cls_token", "prepend_cls_token"),
        ("text_model.vocab_size", "vocab_size"),
    ]
    for _key_map in _model_to_input_map:
        if isinstance(_key_map, tuple):
            assert len(_key_map) == 2, f"Tuple {_key_map} does not have len=2"
            model_key, input_key = _key_map
        else:
            model_key = input_key = _key_map

        for section in ["train_input", "eval_input"]:
            if "." not in model_key:
                params[section][input_key] = params["model"][model_key]
            else:
                # Split `image_model.patch_size`
                # into ['image_model', 'patch_size'] and traverse
                _key = model_key.split(".")
                val = params["model"]
                for _k in _key:
                    val = val[_k]
                params[section][_key[-1]] = val
