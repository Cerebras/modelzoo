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


def set_custom_stack_params(params):
    if cm.use_cs():
        from modelzoo.common.pytorch import AP_DISABLED, cbtorch

        state = cbtorch.state()
        state.full_config.matching.kernel.inc_pwt_estimate = True
        state.full_config.matching.kernel.enable_pipelined_mlm_loss = True
        state.full_config.matching.autogen_policy = AP_DISABLED


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
