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
Script to download FC MNIST data.
"""

import argparse
import os
import sys

import tensorflow_datasets as tfds

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from modelzoo.fc_mnist.tf.utils import DEFAULT_YAML_PATH, get_params

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--params",
    default=DEFAULT_YAML_PATH,
    help="Path to .yaml file with model parameters",
)

args = parser.parse_args()
params = get_params(args.params)

data_dir = params["train_input"]["data_dir"]
data = tfds.load("mnist", data_dir=data_dir)
