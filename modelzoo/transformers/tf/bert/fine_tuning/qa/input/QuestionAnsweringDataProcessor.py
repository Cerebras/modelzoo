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
Processor for SQuAD v1.1
"""

import os

import tensorflow as tf

from modelzoo.common.tf.input.utils import transform_dataset


class QuestionAnsweringDataProcessor:
    """
    Creates dataset from pre-compiled TF records
    :param <dict> params: dict containing training input parameters for creating dataset

    - "data_file" (str): Path to the data file.
    - "batch_size" (int): Batch size.
    - "max_sequence_length" (int): Maximum sequence length.
    - "shuffle" (bool): Flag to enable data shuffling.
    - "shuffle_buffer" (int): Shuffle buffer size.
    - "shuffle_seed" (int): Shuffle seed.
    - "repeat" (bool): Repeat data (for WSE must be True).
    """

    def __init__(self, params):
        super(QuestionAnsweringDataProcessor, self).__init__()
        self.data_file = params["data_file"]
        self.batch_size = params["batch_size"]
        self.max_sequence_length = params["max_sequence_length"]
        self.shuffle = params.get("shuffle", True)
        self.shuffle_seed = params.get("shuffle_seed", None)
        self.shuffle_buffer = params.get("shuffle_buffer", None)
        self.repeat = params.get("repeat", True)

        assert self.batch_size > 0, "Batch size should be positive."
        assert os.path.exists(
            self.data_file
        ), f"data_file does not exist: {self.data_file}"

    def map_fn(self, raw_record, mode):
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
            "segment_ids": tf.io.FixedLenFeature(
                [self.max_sequence_length], tf.int64
            ),
        }
        if mode != tf.estimator.ModeKeys.PREDICT:
            feature_map["start_positions"] = tf.io.FixedLenFeature(
                [1], tf.int64
            )
            feature_map["end_positions"] = tf.io.FixedLenFeature([1], tf.int64)
        else:
            feature_map["unique_ids"] = tf.io.FixedLenFeature([1], tf.int64)

        example = tf.io.parse_example(
            serialized=raw_record, features=feature_map,
        )

        for name in list(example.keys()):
            feature = example[name]
            if feature.dtype == tf.int64:
                feature = tf.cast(feature, tf.int32)
                example[name] = feature
            if name == 'input_mask':
                example[name] = tf.cast(tf.equal(feature, 0), tf.int32)

        # Given the example record, organize features and labels dicts
        feature = {
            "input_ids": example["input_ids"],
            "input_mask": example["input_mask"],
            "segment_ids": example["segment_ids"],
        }

        if mode != tf.estimator.ModeKeys.PREDICT:
            label = tf.concat(
                [example["start_positions"], example["end_positions"]], 1
            )
            return feature, label
        else:
            # During prediction, these ids are used to match windows
            # with specific examples.
            feature["unique_ids"] = example["unique_ids"]
            return feature

    def create_tf_dataset(self, mode, input_context=None):
        """
        Create tf dataset.

        :param tf.estimator.ModeKeys mode: mode input pipe will be used for
        :param dict input_context: Given by distributed strategy for training
        :returns: tf dataset
        """
        assert mode in [
            tf.estimator.ModeKeys.TRAIN,
            tf.estimator.ModeKeys.EVAL,
            tf.estimator.ModeKeys.PREDICT,
        ], f"Mode must be TRAIN, EVAL, or PREDICT."

        dataset = tf.data.TFRecordDataset(self.data_file)

        return transform_dataset(
            dataset,
            lambda raw_record: self.map_fn(raw_record, mode),
            self.batch_size,
            is_training=mode == tf.estimator.ModeKeys.TRAIN,
            shuffle=self.shuffle,
            shuffle_buffer=self.shuffle_buffer,
            repeat=self.repeat,
            seed=self.shuffle_seed,
        )
