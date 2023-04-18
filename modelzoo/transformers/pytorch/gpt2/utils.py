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


def set_custom_stack_params():
    if cm.use_cs():
        from modelzoo.common.pytorch import AP_DISABLED, cbtorch

        state = cbtorch.state()
        state.full_config.matching.kernel.inc_pwt_estimate = True
        state.full_config.matching.kernel.enable_pipelined_mlm_loss = True
        state.full_config.matching.autogen_policy = AP_DISABLED


def set_attention_kernel(params):
    '''
    Set attention kernel related params
    :param params: model_params
    :return:
    '''

    # POL0 : fused-attention
    # POL1/2 : opitmized-attention
    attention_kernel_type = params["model"].get(
        "attention_kernel", "optimized_beta"
    )
    if params["model"]["precision_opt_level"] == 0:
        attention_kernel_type = params["model"].get(
            "attention_kernel", "default"
        )
    params["model"]["attention_kernel"] = attention_kernel_type

    # Attention softmax is fp32 by default.
    params["model"]["attention_softmax_fp32"] = True

    if (
        params["model"]["precision_opt_level"] == 1
        and params["model"]["attention_kernel"] == "default"
    ) or params["model"].get("precision_opt_level", 1) == 2:
        params["model"]["attention_softmax_fp32"] = False


def set_defaults(params):
    """
    Update any missing parameters in the params dictionary with default values

    Args:
        params: The dictionary containing the params
    """
    params["model"]["use_bfloat16"] = params["model"].get("use_bfloat16", True)
    params["optimizer"]["loss_scaling_factor"] = params["optimizer"].get(
        "loss_scaling_factor", 1.0
    )
    params["optimizer"]["log_summaries"] = params["optimizer"].get(
        "log_summaries", False
    )
    params["model"]["precision_opt_level"] = params["model"].get(
        "precision_opt_level", 1
    )
    set_attention_kernel(params)
