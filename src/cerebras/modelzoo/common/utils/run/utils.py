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

"""Contains common run related utilies for cv and nlp models"""

import os


def get_default_inis():
    return {
        "ws_cv_auto_inis": True,
        "ws_opt_target_max_actv_wios": 16,
        "ws_opt_max_wgt_port_group_size": 8,
        "ws_run_memoize_actv_mem_mapping": True,
        "ws_opt_bidir_act": False,
        "ws_variable_lanes": True,
    }


def update_runconfig_debug_args_path(params, default_inis_dict):
    from cerebras.appliance.utils.debug_args import (
        DebugArgs,
        get_debug_args,
        write_debug_args,
    )
    from cerebras.appliance.utils.ini import set_default_ini

    if not params["runconfig"].get("debug_args_path"):
        debug_args = DebugArgs()
        # in the case debug_args_path is not set
        # create a default debug_args file in the debug_args_dir
        debug_args_path = os.path.join(
            params["runconfig"]["model_dir"], ".debug_args.proto"
        )

    else:
        debug_args_path = params["runconfig"]["debug_args_path"]
        debug_args = get_debug_args(debug_args_path)

    set_default_ini(debug_args, **default_inis_dict)
    write_debug_args(debug_args, debug_args_path)
    params["runconfig"]["debug_args_path"] = debug_args_path


class DeviceType:
    """Supported Devices for Running Modelzoo Scripts"""

    CSX = "CSX"
    CPU = "CPU"
    GPU = "GPU"
    # to be used to reference when device type does not matter
    ANY = "ANY"

    @classmethod
    def devices(cls):
        """Valid strategies"""
        return [cls.CSX, cls.CPU, cls.GPU]


def update_dataclass_from_dict(obj, data_dict):
    """Adds new members to dataclass from dict"""
    for key, value in data_dict.items():
        setattr(obj, key, value)
