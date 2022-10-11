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

"""
Utils for params parsing for FC model
"""
import os

import yaml

_curdir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YAML_PATH = os.path.join(_curdir, "configs/params.yaml")


def get_params(params_file=DEFAULT_YAML_PATH):
    with open(params_file, "r") as stream:
        params = yaml.safe_load(stream)
    return params


def get_custom_stack_params(params):
    stack_params = {}
    if params["runconfig"]["multireplica"]:
        from cerebras.pb.stack.full_pb2 import FullConfig

        config = FullConfig()
        config.target_num_replicas = -1
        stack_params["config"] = config
        os.environ["CEREBRAS_CUSTOM_MONITORED_SESSION"] = "True"
    return stack_params


def set_defaults(params):
    # set default weight_servers
    params["runconfig"]["num_wgt_servers"] = params["runconfig"].get(
        "num_wgt_servers", 12
    )
