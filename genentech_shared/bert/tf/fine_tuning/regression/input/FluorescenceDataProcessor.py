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
Processor for the Fluorescence dataset.
"""
import numpy as np
import tensorflow as tf
from genentech_shared.bert.tf.utils import get_pfam_vocab

from modelzoo.common.tf.input.TfRecordsProcessor import TfRecordsProcessor


class FluorescenceDataProcessor(TfRecordsProcessor):
    """
    FluorescenceDataProcessor dataset processor for BERT regression fine-tuning task.
    Reads precompiled tf records for sequences.

    :param dict params: List of training input parameters for creating dataset.
    """

    def __init__(self, params):
        super(FluorescenceDataProcessor, self).__init__(
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
        self.vocab, self.min_aa_id, self.max_aa_id = get_pfam_vocab()

    def _create_input_features(self, raw_input_ids, number_of_real_ids):
        input_ids = raw_input_ids[: self.max_sequence_length - 1].tolist()
        num_pad_positions = self.max_sequence_length - number_of_real_ids - 1

        input_ids = np.array(
            [self.vocab["[CLS]"]]
            + input_ids
            + [self.vocab["[PAD]"]] * num_pad_positions,
            np.int32,
        )

        input_mask = np.ones_like(input_ids, np.int32)
        input_mask[:number_of_real_ids] = 0
        return input_ids, input_mask

    def map_fn(self, raw_record):
        """
        Parses a serialized protobuf example into a dictionary
        of input features and labels for BERT regression fine-tuning task.
        """
        feature_map = {
            "primary": tf.io.FixedLenSequenceFeature([1], tf.int64),
        }

        context_map = {
            "protein_length": tf.io.FixedLenFeature([1], tf.int64),
            "log_fluorescence": tf.io.FixedLenFeature([1], tf.float32),
            "num_mutations": tf.io.FixedLenFeature([1], tf.int64),
        }

        raw_context, raw_features, _ = tf.io.parse_sequence_example(
            raw_record,
            context_features=context_map,
            sequence_features=feature_map,
        )

        input_ids, input_mask = tf.numpy_function(
            self._create_input_features,
            [
                tf.cast(raw_features["primary"][:, 0], tf.int32),
                tf.cast(raw_context["protein_length"][0], tf.int32),
            ],
            [tf.int32, tf.int32],
        )

        input_ids.set_shape(self.max_sequence_length)
        input_mask.set_shape(self.max_sequence_length)

        # Given the example record, organize features and labels dicts.
        features = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": tf.zeros(self.max_sequence_length, dtype=tf.int32),
            "labels": tf.cast(
                tf.expand_dims(raw_context["log_fluorescence"][0], -1),
                tf.float32,
            ),
        }
        label = features["labels"]
        return features, label
