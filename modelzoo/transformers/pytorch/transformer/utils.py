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

from modelzoo.common.pytorch import cb_model as cm


def set_defaults(params):
    """
    Update any missing parameters in the params dictionary with default values

    Args:
        params: The dictionary containing the params
    """
    params["model"]["src_max_position_embeddings"] = params["model"].get(
        "src_max_position_embeddings",
        params["train_input"]["src_max_sequence_length"],
    )
    params["model"]["tgt_max_position_embeddings"] = params["model"].get(
        "tgt_max_position_embeddings",
        params["train_input"]["tgt_max_sequence_length"],
    )
    params["model"]["use_bfloat16"] = params["model"].get("use_bfloat16", False)
    params["optimizer"]["loss_scaling_factor"] = params["optimizer"].get(
        "loss_scaling_factor", "dynamic"
    )
    params["train_input"]["dynamic_loss_weight"] = (
        params["model"].get("mlm_loss_scaling", "batch_size")
        == "precomputed_num_masked"
    )


def set_custom_stack_params(params):
    if cm.use_cs():
        from modelzoo.common.pytorch import cbtorch

        state = cbtorch.state()
        state.full_config.matching.kernel.enable_pipelined_mlm_loss = True
        state.full_config.matching.kernel.inc_pwt_estimate = True
