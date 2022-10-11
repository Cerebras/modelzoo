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

import tensorflow as tf

from modelzoo.common.tf.input.TfRecordsProcessor import TfRecordsProcessor


class GptTfRecordsProcessor(TfRecordsProcessor):
    """
    Creates dataset from pre-compiled TF records
    :param dict params: dict containing training
        input parameters for creating dataset.
    Expects the following fields:

    - "data_dir" (str): Path to directory containing the TF records.
    - "max_sequence_length" (int): Maximum length of the sequence. Should
      match the sequence length in the TF records.
    - "batch_size" (int): Batch size.
    - "shuffle" (bool): Flag to enable data shuffling.
    - "shuffle_seed" (int): Shuffle seed.
    - "shuffle_buffer" (int): Shuffle buffer size.
    - "repeat" (bool): Flag to enable data repeat.
    - "use_multiple_workers" (bool): If `True`, dataset will be sharded.
    - "n_parallel_reads" (int): Used for call to `tf.data.Dataset.interleave`.
    - "skip_steps" (int): Number of steps to skip the dataset after batching.
    - "mixed_precision" (bool): Casts input mask to `float16` if set to `True`.
      Otherwise, the generated mask is `float32`.
    """

    def __init__(self, params):

        super(GptTfRecordsProcessor, self).__init__(
            params["data_dir"],
            params["batch_size"],
            shuffle=params.get("shuffle", True),
            shuffle_seed=params.get("shuffle_seed", None),
            shuffle_buffer=params.get("shuffle_buffer", None),
            repeat=params.get("repeat", True),
            use_multiple_workers=params.get("use_multiple_workers", False),
            n_parallel_reads=params.get("n_parallel_reads", 4),
            skip_steps=params.get("skip_steps", 0),
        )

        self.max_sequence_length = params["max_sequence_length"]
        self.mp_type = (
            tf.float16 if params.get("mixed_precision") else tf.float32
        )

    def map_fn(self, raw_record):
        """
        Parses a serialized protobuf example into a dictionary
        of input features and labels for BERT pretraining.
        """
        feature_map = {
            "input_ids": tf.io.FixedLenFeature(
                [self.max_sequence_length], tf.int64
            ),
            "input_mask": tf.io.FixedLenFeature(
                [self.max_sequence_length], tf.int64
            ),
            "labels": tf.io.FixedLenFeature(
                [self.max_sequence_length], tf.int64
            ),
        }
        example = tf.io.parse_example(
            serialized=raw_record, features=feature_map,
        )

        example["input_mask"] = tf.equal(example["input_mask"], 0)

        # Given the example record, organize features and labels dicts
        feature = {
            "input_ids": tf.cast(example["input_ids"], tf.int32),
            "input_mask": tf.cast(example["input_mask"], self.mp_type),
        }

        labels = tf.cast(example["labels"], tf.int32)

        return feature, labels
