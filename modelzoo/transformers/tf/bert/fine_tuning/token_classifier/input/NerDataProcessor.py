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

import tensorflow as tf

from modelzoo.common.tf.input.utils import transform_dataset
from modelzoo.common.tf.model_utils.shard_dataset import shard_dataset


class NerDataProcessor:
    def __init__(self, params):
        """
        Dataset generator.
        :param dict params: List of training input parameters for creating dataset
        Expects the following fields:
            - "vocab_file" (str): Path to the vocab file.
            - "label_vocab_file" (str): Path to json file with class name to class index.
            - "data_dir" (str): Path to directory containing the TF Records.
            - "batch_size" (int): Batch size.
            - "max_sequence_length" (int): Maximum length of the sequence.
            - "repeat" (bool): Flag to enable data repeat.
            - "shuffle" (bool): Flag to enable data shuffling.
            - "shuffle_seed" (int): Shuffle seed.
            - "shuffle_buffer" (int): Shuffle buffer size.
            - "use_multiple_workers" (bool): Specifies whether using multiple_workers
                with the Cerebras System or not.
        """

        self.data_dir = params["data_dir"]

        self.vocab_file = params["vocab_file"]
        self.label_vocab_file = params["label_vocab_file"]
        if not os.path.exists(self.label_vocab_file):
            raise FileNotFoundError(f"{self.label_vocab_file} not found.")

        self.batch_size = params["batch_size"]
        self.max_sequence_length = params["max_sequence_length"]
        self.shuffle = params.get("shuffle", True)
        self.shuffle_seed = params.get("shuffle_seed", None)
        self.shuffle_buffer = params.get("shuffle_buffer", None)
        self.repeat = params.get("repeat", True)
        self.use_multiple_workers = params.get("use_multiple_workers", False)

    def map_fn(self, raw_record):
        """
        Parses a serialized protobuf example into a dictionary
        of input features and labels for Token classifier finetuning.
        """

        feature_map = {
            'input_ids': tf.io.FixedLenFeature(
                [self.max_sequence_length], tf.int64
            ),
            'input_mask': tf.io.FixedLenFeature(
                [self.max_sequence_length], tf.int64
            ),
            'segment_ids': tf.io.FixedLenFeature(
                [self.max_sequence_length], tf.int64
            ),
            'label_ids': tf.io.FixedLenFeature(
                [self.max_sequence_length], tf.int64
            ),
        }

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
            "labels": example["label_ids"],
        }

        labels = example["label_ids"]

        return (feature, labels)

    def create_tf_dataset(self, is_training=True, input_context=None):
        """
        Create tf dataset.

        :param bool is_training: Specifies whether the data is for training
        :param dict input_context: Given by distributed strategy for training
        :returns: tf dataset
        """

        if not isinstance(self.data_dir, list):
            self.data_dir = [self.data_dir]

        file_pattern = []

        for file in self.data_dir:
            file_pattern.append(os.path.join(file, '*.tfrecord*'))

        filelist = tf.data.Dataset.list_files(
            file_pattern, shuffle=(self.shuffle and is_training)
        )

        if is_training:
            filelist = shard_dataset(
                filelist, self.use_multiple_workers, input_context
            )

            n_parallel_reads = 4
            dataset = filelist.interleave(
                map_func=tf.data.TFRecordDataset,
                cycle_length=n_parallel_reads,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                # only allow nondeterminism when shuffling unseeded
                deterministic=not (self.shuffle and self.shuffle_seed is None),
            )

        else:
            dataset = filelist.flat_map(map_func=tf.data.TFRecordDataset)

        return transform_dataset(
            dataset,
            self.map_fn,
            self.batch_size,
            is_training,
            shuffle=self.shuffle,
            shuffle_buffer=self.shuffle_buffer,
            repeat=self.repeat,
            seed=self.shuffle_seed,
        )
