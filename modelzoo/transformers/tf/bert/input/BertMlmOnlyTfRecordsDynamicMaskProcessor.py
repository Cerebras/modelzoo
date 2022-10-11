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
Processor for performing dynamic masking on MLM only data for BERT
"""
import random

import tensorflow as tf

from modelzoo.common.tf.input.TfRecordsProcessor import TfRecordsProcessor
from modelzoo.common.tf.input.utils import bucketed_batch
from modelzoo.transformers.data_processing.mlm_only_processor import (
    create_masked_lm_features,
)
from modelzoo.transformers.data_processing.Tokenization import FullTokenizer
from modelzoo.transformers.data_processing.utils import get_output_type_shapes


class BertMlmOnlyTfRecordsDynamicMaskProcessor(TfRecordsProcessor):
    """
    Reads TF records containing sequences of tokens, adds MLM features
    on the fly. Resulting dataset is for MLM task only.
    These TFRecords do not contain `segment_ids` and `next_sentence_labels`
    Should be used in conjunction with
        `disable_nsp: True`
        `use_segment_embedding: False`
        settings in the model section of the config yaml file during pre-training
    :param <dict> params: dict containing training input parameters for creating dataset
    
    Expects the following fields:
    
    - "data_dir" (string): path to the data files to use.
    - "batch_size" (int): Batch size.
    - "shuffle" (bool): Flag to enable data shuffling.
    - "shuffle_seed" (int): Shuffle seed.
    - "shuffle_buffer" (int): Shuffle buffer size.
    - "repeat" (bool): Flag to enable data repeat.
    - "use_multiple_workers" (bool): Specifies whether using multiple workers with the Cerebras System for training
    - "mask_whole_word" (bool): Specifies whether to mask whole words
    - "max_sequence_length" (int): the maximum sequence length
    - "max_predictions_per_seq" (int): maximum number of masked LM predictions per sequence
    - "masked_lm_prob" (float): Probability of generating a masked token
    - "mixed_precision" (bool): Specifies whether to generate float16 instead of float32 tensors for mixed-precision training
    """

    def __init__(self, params):
        super(BertMlmOnlyTfRecordsDynamicMaskProcessor, self).__init__(
            params["data_dir"],
            params["batch_size"],
            shuffle=params.get("shuffle", True),
            shuffle_seed=params.get("shuffle_seed", None),
            shuffle_buffer=params.get("shuffle_buffer", None),
            repeat=params.get("repeat", True),
            use_multiple_workers=params.get("use_multiple_workers", False),
            n_parallel_reads=params.get("n_parallel_reads", 4),
            map_before_batch=True,
        )
        self.tokenizer = FullTokenizer(params["vocab_file"], params["do_lower"])
        self.vocab_words = self.tokenizer.get_vocab_words()
        self.max_sequence_length = params["max_sequence_length"]
        self.mask_whole_word = params.get("mask_whole_word", False)
        self.max_predictions_per_seq = params["max_predictions_per_seq"]
        self.masked_lm_prob = params.get("masked_lm_prob", 0.15)
        self.document_separator_token = params.get(
            "document_separator_token", "[SEP]"
        )
        self.input_pad_id = params.get("input_pad_id", None)
        self.mlm_pad_id = params.get("mlm_pad_id", None)
        self.rng = random.Random(params.get("shuffle_seed", None))
        self.output_type_shapes = get_output_type_shapes(
            self.max_sequence_length,
            self.max_predictions_per_seq,
            mlm_only=True,
        )
        self.mp_type = (
            tf.float16 if params.get("mixed_precision") else tf.float32
        )
        self.scale_mlm_weights = params.get("scale_mlm_weights", False)

        # buckets must be either an integer or a list of boundaries of length
        # `num_buckets` - 1. If it is a list, it should exclude `0` and
        # `max_sequence_length`. Buckets are inclusive on the bottom and
        # exclusive on the top.
        self.buckets = params.get("buckets", 1)
        if isinstance(self.buckets, int):
            self.buckets = [
                int((i + 1) * self.max_sequence_length / self.buckets)
                for i in range(self.buckets - 1)
            ]

    def _parse_raw_tfrecord(self, raw_record):
        feature_map = {"tokens": tf.io.FixedLenSequenceFeature((), tf.string)}
        _, raw_features = tf.io.parse_single_sequence_example(
            raw_record, sequence_features=feature_map
        )
        return raw_features["tokens"]

    def _create_masked_lm_features(self, tokens):
        tokens = [token.decode() for token in tokens]
        features, label = create_masked_lm_features(
            tokens,
            self.vocab_words,
            self.max_sequence_length,
            self.mask_whole_word,
            self.max_predictions_per_seq,
            self.masked_lm_prob,
            self.document_separator_token,
            self.rng,
            self.tokenizer,
            self.output_type_shapes,
            inverted_mask=True,
        )
        return (
            features["input_ids"],
            features["input_mask"],
            features["masked_lm_ids"],
            features["masked_lm_positions"],
            features["masked_lm_weights"],
            label,
        )

    def batch_fn(self, dataset):
        if self.buckets:
            return bucketed_batch(
                dataset,
                element_length_func=lambda f, _: (
                    self.max_sequence_length - tf.reduce_sum(f["input_mask"])
                ),
                bucket_boundaries=self.buckets,
                batch_size=self.batch_size,
                no_padding=True,
                drop_remainder=True,
            )
        else:
            return dataset.batch(self.batch_size, drop_remainder=True)

    def map_fn(self, raw_record):
        """
        Parses a serialized protobuf example into a list of tokens,
        then creates masked lm features and labels for BERT pretraining.
        """

        tokens = self._parse_raw_tfrecord(raw_record)
        (
            input_ids,
            input_mask,
            masked_lm_ids,
            masked_lm_positions,
            masked_lm_weights,
            label,
        ) = tf.numpy_function(
            self._create_masked_lm_features,
            [tokens],
            [tf.int32, tf.int32, tf.int32, tf.int32, tf.float32, tf.int32],
        )
        input_ids.set_shape(self.max_sequence_length)
        input_mask.set_shape(self.max_sequence_length)
        masked_lm_ids.set_shape(self.max_predictions_per_seq)
        masked_lm_positions.set_shape(self.max_predictions_per_seq)
        masked_lm_weights.set_shape(self.max_predictions_per_seq)
        label.set_shape(())
        features = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "masked_lm_ids": masked_lm_ids,
            "masked_lm_positions": masked_lm_positions,
            "masked_lm_weights": tf.cast(masked_lm_weights, self.mp_type),
        }

        if self.input_pad_id is not None:
            features["input_ids"] = tf.where(
                tf.cast(features["input_mask"], tf.bool),
                self.input_pad_id,
                features["input_ids"],
            )
        if self.mlm_pad_id is not None:
            features["masked_lm_ids"] = tf.where(
                tf.cast(features["masked_lm_weights"], tf.bool),
                features["masked_lm_ids"],
                self.mlm_pad_id,
            )

        return features, label

    def post_batch_map_fn(self, features, label):
        """
        When appropriate, scale mlm weights by `batch_size / num_valid_tokens`.
        This is used to compute the correct scaling factor on the loss without
        running into precision issues. Intended for use in situations when the
        loss will be divided by `batch_size` at the time of computation.
        """
        if self.scale_mlm_weights:
            mlm_weights = features["masked_lm_weights"]
            scale = self.batch_size / tf.reduce_sum(mlm_weights)
            features["masked_lm_weights"] = tf.cast(
                mlm_weights * scale, self.mp_type
            )
        return features, label
