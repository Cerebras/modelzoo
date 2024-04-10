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


def set_defaults(params):
    """
    Update any missing parameters in the params dictionary with default values

    Args:
        params: The dictionary containing the params
    """
    params["model"]["src_max_position_embeddings"] = params["model"].get(
        "src_max_position_embeddings",
        params["train_input"].get("src_max_sequence_length"),
    )
    params["model"]["tgt_max_position_embeddings"] = params["model"].get(
        "tgt_max_position_embeddings",
        params["train_input"].get("tgt_max_sequence_length"),
    )
    # Enable bf16 by default and set loss scaling factor to 1.0 when sw-82646 is resolved.
    params["model"]["fp16_type"] = params["model"].get("fp16_type", "float16")
    params["optimizer"]["loss_scaling_factor"] = params["optimizer"].get(
        "loss_scaling_factor", "dynamic"
    )
    params["optimizer"]["log_summaries"] = params["optimizer"].get(
        "log_summaries", False
    )

    params["train_input"]["dynamic_loss_weight"] = (
        params["model"].get("mlm_loss_scaling", "batch_size")
        == "precomputed_num_masked"
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
