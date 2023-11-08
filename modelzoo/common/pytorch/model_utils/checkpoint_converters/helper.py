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

from modelzoo.common.pytorch.model_utils.checkpoint_converters.base_converter import (
    ConfigConversionError,
)


def convert_use_rms_layer_norm_helper(
    self,
    old_key,
    new_key,
    old_state_dict,
    new_state_dict,
    from_index,
    action_fn_args,
):
    if from_index == 0:
        new_state_dict[new_key] = (
            "rmsnorm" if old_state_dict[old_key] else "layernorm"
        )
    else:
        if old_state_dict[old_key] == "rmsnorm":
            new_state_dict[new_key] = True
        elif old_state_dict[old_key] == "layernorm":
            new_state_dict[new_key] = False
        else:
            raise ConfigConversionError(
                "{} did not support {}".format(
                    self.formats()[0], old_state_dict[old_key]
                )
            )


def convert_use_biasless_layer_norm_helper(
    self,
    old_key,
    new_key,
    old_state_dict,
    new_state_dict,
    from_index,
    action_fn_args,
):
    if from_index == 0:
        new_state_dict[new_key] = (
            "biasless-layernorm" if old_state_dict[old_key] else "layernorm"
        )
    else:
        if old_state_dict[old_key] == "biasless-layernorm":
            new_state_dict[new_key] = True
        elif old_state_dict[old_key] == "layernorm":
            new_state_dict[new_key] = False
        else:
            raise ConfigConversionError(
                "{} did not support {}".format(
                    self.formats()[0], old_state_dict[old_key]
                )
            )
