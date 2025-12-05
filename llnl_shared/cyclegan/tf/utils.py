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

import yaml

_curdir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PARAMS_FILE = os.path.join(_curdir, "params.yaml")


def get_params(subgraph_type, param_path=DEFAULT_PARAMS_FILE, config="base"):
    """ Get params from yaml file and put into dict """
    params = {}
    with open(param_path) as fid:
        params_all = yaml.safe_load(fid)
    if config:
        params_all = params_all[config]
    params['runconfig'] = params_all['runconfig']
    params['input'] = params_all['train_input']
    params['model'] = params_all['model'][subgraph_type]
    params['optimizer'] = params_all['optimizer']

    params['input'].update(
        {
            'max_steps': params['runconfig']['max_steps'],
            'input_names': params['model']['input_names'],
            'output_names': params['model']['output_names'],
            'mixed_precision': params_all['model']['mixed_precision'],
        }
    )
    params['model'].update(
        {'mixed_precision': params_all['model']['mixed_precision'],}
    )
    return params
