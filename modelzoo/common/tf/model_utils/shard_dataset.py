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

import json
import os


def shard_dataset(dataset, use_multiple_workers, input_context=None):
    """
    Shard a dataset based on whether we are using a multi-gpu setting
    or using a Cerebras System with multiple-workers. For single worker scenario on the Cerebras System,
    it's best to pass use_multiple_workers as False

    :param tf.data.Dataset dataset: TF dataset to shard
    :param bool use_multiple_workers: Specifies whether using multiple_workers
        with the Cerebras System or not
    :param dict input_context: Given by distributed strategy for training

    :returns dataset: Sharded if either input_context or use_multiple_workers
    is passed, else just returns the dataset
    """

    # Add multi-gpu input context.
    if not use_multiple_workers and input_context:
        num_workers = input_context.num_input_pipelines
        worker_id = input_context.input_pipeline_id
        dataset = dataset.shard(num_workers, worker_id)
    # Add a multi-worker context for the Cerebras System data loading.
    # In this case, no input context should be generated.
    elif use_multiple_workers and "TF_CONFIG" in os.environ:
        config = json.loads(os.environ["TF_CONFIG"])
        num_workers = len(config["cluster"]["worker"])
        worker_id = config["task"]["index"]
        dataset = dataset.shard(num_workers, worker_id)

    return dataset
