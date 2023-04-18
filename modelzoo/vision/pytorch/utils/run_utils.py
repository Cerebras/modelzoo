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
import tempfile

from modelzoo.common.pytorch.utils import get_debug_args
from modelzoo.common.run_utils.utils import DeviceType


def get_default_inis():
    return {
        "ws_cv_auto_inis": True,
        "ws_opt_target_max_actv_wios": 16,
        "ws_opt_max_wgt_port_group_size": 8,
        "ws_run_memoize_actv_mem_mapping": True,
        "ws_opt_bidir_act": False,
        "ws_variable_lanes": True,
    }


def use_cs(params):
    use_cs = params["runconfig"]["target_device"] == DeviceType.CSX
    return use_cs


def update_runconfig_debug_args_path(params, default_inis_dict):
    if use_cs(params):
        from cerebras_appliance.run_utils import write_debug_args

        if not params["runconfig"].get("debug_args_path"):
            cwd = os.getcwd()

            with tempfile.TemporaryDirectory(dir=cwd) as ini_dir:
                # Create and populate debug.ini
                with open(os.path.join(ini_dir, "debug.ini"), "w") as fp:
                    for key, val in default_inis_dict.items():
                        fp.write(f"{key}: {val}\n")

                # Uses debug.ini from current path if it exists
                debug_args = get_debug_args(None, ini_dir)

            # in the case debug_args_path is not set
            # create a default debug_args file in the debug_args_dir
            debug_args_path = os.path.join(
                params["runconfig"]["model_dir"], ".debug_args.proto"
            )
            write_debug_args(debug_args, debug_args_path)
            params["runconfig"]["debug_args_path"] = debug_args_path
