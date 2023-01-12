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

import argparse
import os

import torchvision.datasets as dset
import yaml

_curdir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YAML_PATH = os.path.join(_curdir, "configs", "params.yaml")

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--params",
    default=DEFAULT_YAML_PATH,
    help="Path to .yaml file with model parameters",
)


def get_params(params_file=DEFAULT_YAML_PATH, config=None):
    with open(params_file, "r") as stream:
        params = yaml.safe_load(stream)
    return params


args = parser.parse_args()
params = get_params(args.params)
root = params["train_input"]["data_dir"]

if not os.path.exists(root):
    os.mkdir(root)

# if not exist, download mnist dataset
train_set = dset.MNIST(root=root, train=True, download=True)
test_set = dset.MNIST(root=root, train=False, download=True)
