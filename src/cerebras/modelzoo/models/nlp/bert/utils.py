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


def set_defaults(params):
    """
    Update any missing parameters in the params object with default values.

    Args:
        params: An dict/object containing the parameters
    """
    if "trainer" not in params:
        _set_defaults(params)
    else:
        _set_defaults_v2(params)


def _set_defaults(params):
    for section in ["train_input", "eval_input"]:
        for key in ["vocab_file"]:
            if params.get(section, {}).get(key):
                params[section][key] = os.path.abspath(params[section][key])

    model_params = params["model"]
    params["model"]["disable_nsp"] = model_params.get("disable_nsp", False)
    # Pass settings into data loader.
    for model_key in (
        "disable_nsp",
        "vocab_size",
    ):
        for input_key in ("train_input", "eval_input"):
            params[input_key][model_key] = model_params.get(model_key)

    params["model"]["max_position_embeddings"] = model_params.get(
        "max_position_embeddings",
        params["train_input"]["max_sequence_length"],
    )
    params["model"]["fp16_type"] = model_params.get("fp16_type", "float16")
    params["optimizer"]["log_summaries"] = params["optimizer"].get(
        "log_summaries", False
    )

    # Attention softmax is fp32 by default.
    params["model"]["attention_softmax_fp32"] = True

    # Attention softmax is bf16 for precision_opt_level: 2
    if params["runconfig"].get("precision_opt_level", 1) == 2:
        params["model"]["attention_softmax_fp32"] = False

    if (
        params["model"].get("fp16_type", "bfloat16") == "cbfloat16"
        and params["runconfig"].get("precision_opt_level", 1) == 1
    ):
        params["model"]["attention_softmax_fp32"] = False


def _set_defaults_v2(params):
    if "base" in params:
        params = params["base"]
    fit = params.get("trainer", {}).get("fit", {})
    for section in ["train_dataloader", "val_dataloader"]:
        for key in ["vocab_file"]:
            if fit.get(section, {}).get(key):
                fit[section][key] = os.path.abspath(fit[section][key])

    model_params = params.get("trainer", {}).get("init", {}).get("model", {})
    model_params.setdefault("disable_nsp", False)
    # Pass settings into data loader.
    for model_key in (
        "disable_nsp",
        "vocab_size",
    ):
        for input_key in ("train_dataloader", "val_dataloader"):
            fit[input_key][model_key] = model_params.get(model_key)

    model_params.setdefault(
        "max_position_embeddings",
        fit["train_dataloader"]["max_sequence_length"],
    )
    precision_params = (
        params.get("trainer", {}).get("init", {}).get("precision", {})
    )
    precision_params.setdefault("fp16_type", "float16")
    precision_params.setdefault("log_loss_scale", False)

    # Attention softmax is fp32 by default.
    model_params["attention_softmax_fp32"] = True
    # Attention softmax is bf16 for precision_opt_level: 2
    if precision_params.get("precision_opt_level", 1) == 2:
        model_params["attention_softmax_fp32"] = False

    if (
        precision_params.get("fp16_type", "bfloat16") == "cbfloat16"
        and precision_params.get("precision_opt_level", 1) == 1
    ):
        model_params["attention_softmax_fp32"] = False
