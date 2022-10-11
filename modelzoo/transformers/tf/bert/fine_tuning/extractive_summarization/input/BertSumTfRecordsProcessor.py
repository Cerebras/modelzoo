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
Processor for handling TF records data for BERT extractive summarization.
"""

import tensorflow as tf

from modelzoo.common.tf.input.TfRecordsProcessor import TfRecordsProcessor


class BertSumTfRecordsProcessor(TfRecordsProcessor):
    """
    Creates dataset from pre-compiled TF records.
    :param <dict> params: dict containing training input parameters for creating dataset.
    """

    def __init__(self, params):
        super(BertSumTfRecordsProcessor, self).__init__(
            params["data_dir"],
            params["batch_size"],
            shuffle=params.get("shuffle", True),
            shuffle_seed=params.get("shuffle_seed", None),
            shuffle_buffer=params.get("shuffle_buffer", None),
            repeat=params.get("repeat", True),
            use_multiple_workers=params.get("use_multiple_workers", False),
            n_parallel_reads=params.get("n_parallel_reads", 4),
        )
        self.max_sequence_length = params["max_sequence_length"]
        self.max_cls_tokens = params["max_cls_tokens"]

    def map_fn(self, raw_record):
        """
        Parses a serialized protobuf example into a dictionary
        of input features and labels for BERT extractive summarization.
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
            "cls_indices": tf.io.FixedLenFeature(
                [self.max_cls_tokens], tf.int64
            ),
            "labels": tf.io.FixedLenFeature([self.max_cls_tokens], tf.int64),
            "cls_weights": tf.io.FixedLenFeature(
                [self.max_cls_tokens], tf.float32
            ),
        }
        example = tf.io.parse_example(
            serialized=raw_record, features=feature_map
        )

        for name in list(example.keys()):
            feature = example[name]
            if feature.dtype == tf.int64:
                feature = tf.cast(feature, tf.int32)
                example[name] = feature

        # Given the example record, organize features and labels dicts.
        feature = {
            "input_ids": example["input_ids"],
            "input_mask": example["input_mask"],
            "segment_ids": example["segment_ids"],
            "cls_indices": example["cls_indices"],
            "labels": example["labels"],
            "cls_weights": example["cls_weights"],
        }
        label = example["labels"]
        return feature, label
