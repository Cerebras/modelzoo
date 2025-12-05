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
Processor for the SecondaryStructure dataset.
"""
import numpy as np
import tensorflow as tf
from genentech_shared.bert.tf.utils import get_pfam_vocab

from modelzoo.common.tf.input.TfRecordsProcessor import TfRecordsProcessor


class SecondaryStructureDataProcessor(TfRecordsProcessor):
    """
    SecondaryStructureDataProcessor dataset processor for BERT token classification
        fine-tuning task.
    Reads precompiled tf records for sequences.
    :param dict params: List of training input parameters for creating dataset.
    """

    def __init__(self, params):
        super(SecondaryStructureDataProcessor, self).__init__(
            params["data_dir"],
            params["batch_size"],
            shuffle=params.get("shuffle", True),
            shuffle_seed=params.get("shuffle_seed", None),
            shuffle_buffer=params.get("shuffle_buffer", None),
            repeat=params.get("repeat", True),
            use_multiple_workers=params.get("use_multiple_workers", False),
            n_parallel_reads=params.get("n_parallel_reads", 4),
            map_before_batch=params.get("map_before_batch", True),
        )
        self.max_sequence_length = params["max_sequence_length"]
        self.num_classes = params["num_classes"]
        self.vocab, self.min_aa_id, self.max_aa_id = get_pfam_vocab()

    def _create_input_features(
        self, raw_input_ids, raw_label, number_of_real_ids
    ):
        input_ids = self._pad_sequence(raw_input_ids, number_of_real_ids)
        label = self._pad_sequence(raw_label, number_of_real_ids)
        input_mask = np.ones_like(input_ids, np.int32)
        input_mask[:number_of_real_ids] = 0
        return input_ids, label, input_mask

    def _pad_sequence(self, raw_input_sequence, number_of_real_ids):
        input_sequence = raw_input_sequence[: self.max_sequence_length].tolist()
        num_pad_positions = self.max_sequence_length - number_of_real_ids

        input_sequence = np.array(
            input_sequence + [self.vocab["[PAD]"]] * num_pad_positions,
            np.int32,
        )
        return input_sequence

    def map_fn(self, raw_record):
        """
        Parses a serialized protobuf example into a dictionary
        of input features and labels for BERT token classification
            fine-tuning task.
        """
        feature_map = {
            "primary": tf.io.FixedLenSequenceFeature([], tf.int64),
            "evolutionary": tf.io.FixedLenSequenceFeature([30], tf.float32),
            "ss3": tf.io.FixedLenSequenceFeature([], tf.int64),
            "ss8": tf.io.FixedLenSequenceFeature([], tf.int64),
            "disorder": tf.io.FixedLenSequenceFeature([], tf.int64),
            "interface": tf.io.FixedLenSequenceFeature([], tf.int64),
        }

        context_map = {
            "id": tf.io.FixedLenFeature([], tf.string),
            "protein_length": tf.io.FixedLenFeature([], tf.int64),
        }

        raw_context, raw_features, _ = tf.io.parse_sequence_example(
            serialized=raw_record,
            context_features=context_map,
            sequence_features=feature_map,
        )

        primary = tf.cast(raw_features["primary"], tf.int32)
        protein_length = tf.cast(raw_context["protein_length"], tf.int32)
        ss = tf.cast(
            raw_features["ss3" if self.num_classes == 3 else "ss8"], tf.int32
        )

        input_ids, label, input_mask = tf.numpy_function(
            self._create_input_features,
            [primary, ss, protein_length],
            [tf.int32, tf.int32, tf.int32],
        )

        input_ids.set_shape(self.max_sequence_length)
        input_mask.set_shape(self.max_sequence_length)
        label.set_shape(self.max_sequence_length)

        # Given the example record, organize features and labels dicts.
        features = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": tf.zeros(self.max_sequence_length, dtype=tf.int32),
            "labels": label,
        }
        return features, label
