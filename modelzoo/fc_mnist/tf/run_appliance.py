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

#!/usr/bin/env python3

"""
Run script for running on cerebras appliance cluster
"""

import os
import sys

import tensorflow as tf

# Disable eager execution
tf.compat.v1.disable_eager_execution()

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from modelzoo.common.tf.appliance_utils import ExecutionStrategy, run_appliance
from modelzoo.fc_mnist.tf.data import eval_input_fn, train_input_fn
from modelzoo.fc_mnist.tf.model import model_fn
from modelzoo.fc_mnist.tf.utils import get_custom_stack_params, set_defaults


def main():
    run_appliance(
        model_fn,
        train_input_fn,
        eval_input_fn,
        supported_strategies=[
            ExecutionStrategy.weight_streaming,
            ExecutionStrategy.pipeline,
        ],
        default_params_fn=set_defaults,
        stack_params_fn=get_custom_stack_params,
        enable_cs_summaries=True,
    )


if __name__ == '__main__':
    main()
