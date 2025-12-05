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
Processor for the Pfam dataset
"""
import os
import random

import numpy as np
import tensorflow as tf
from genentech_shared.bert.tf.utils import get_pfam_vocab

from modelzoo.common.tf.input.utils import bucketed_batch, transform_dataset
from modelzoo.common.tf.model_utils.shard_dataset import shard_dataset


class PfamDataProcessor:
    """
    Pfam dataset processor for BERT pre-training.
    Reads precompiled tf records for sequences, masks dynamically.

    :param dict params: List of training input parameters for creating dataset.
    """

    def __init__(self, params):
        self.data_dir = params["data_dir"]
        self.batch_size = params["batch_size"]
        self.max_sequence_length = params["max_sequence_length"]
        self.max_predictions_per_seq = params["max_predictions_per_seq"]
        self.masked_lm_prob = params.get("masked_lm_prob", 0.15)
        self.mp_type = (
            tf.float16 if params.get("mixed_precision") else tf.float32
        )
        self.shuffle = params.get("shuffle", True)
        self.shuffle_buffer = params.get("shuffle_buffer", 10 * self.batch_size)
        self.shuffle_seed = params.get("shuffle_seed", None)
        self.repeat = params.get("repeat", True)

        # Buckets must be either an integer or a list of boundaries of length
        # `num_buckets - 1`. If it is a list, it should exclude `0` and
        # `max_sequence_length`. Buckets are inclusive on the bottom
        # and exclusive on the top.
        self.buckets = params.get("buckets", 1)
        if isinstance(self.buckets, int):
            self.buckets = [
                int((i + 1) * self.max_sequence_length / self.buckets)
                for i in range(self.buckets - 1)
            ]

        # For sharding on the Cerebras System, we need to explicitly retrieve `TF_CONFIG`.
        self.use_multiple_workers = params.get("use_multiple_workers", False)

        # padding tokens for vsl
        self.input_pad_id = params.get("input_pad_id", None)
        self.segment_pad_id = params.get("segment_pad_id", None)
        self.mlm_pad_id = params.get("mlm_pad_id", None)
        self.scale_mlm_weights = params.get("scale_mlm_weights", False)

        assert self.batch_size > 0, "Batch size should be positive."
        assert (
            self.max_sequence_length > 0
        ), "Max sequence length should be positive."
        assert (
            self.max_predictions_per_seq > 0
        ), "Max predictions per seq should be positive."

        self.vocab, self.min_aa_id, self.max_aa_id = get_pfam_vocab()

        # No seed if not deterministic.
        self.deterministic = False
        self.rng = random.Random()

    def _create_mlm_input_features(self, raw_input_ids):
        """
        Truncates, masks, and pads input ids.

        When an id is masked, it is:
            - replaced with [MASK] 80% of the time.
            - replaced with a random amino acid 10% of the time.
            - left the same 10% of the time.

        :param array raw_input_ids: sequence from a TFRecord.

        :returns: input_ids, masked_lm_positions, masked_lm_ids.
        """
        input_ids = raw_input_ids[: self.max_sequence_length - 1].tolist()
        num_ids = len(input_ids)
        num_pad_pos = self.max_sequence_length - num_ids - 1
        input_ids = (
            [self.vocab["[CLS]"]]
            + input_ids
            + [self.vocab["[PAD]"]] * num_pad_pos
        )
        num_to_predict = min(
            self.max_predictions_per_seq,
            max(1, int(round(num_ids * self.masked_lm_prob))),
        )
        masked_lm_positions = sorted(
            self.rng.sample(range(1, num_ids + 1), num_to_predict)
        )
        masked_lm_ids = [input_ids[pos] for pos in masked_lm_positions]
        for pos in masked_lm_positions:
            random_val = self.rng.random()
            if random_val < 0.8:
                input_ids[pos] = self.vocab["[MASK]"]
            elif random_val < 0.9:
                input_ids[pos] = self.rng.randint(
                    self.min_aa_id, self.max_aa_id
                )

        masked_lm_padding = [0] * (
            self.max_predictions_per_seq - num_to_predict
        )
        masked_lm_positions += masked_lm_padding
        masked_lm_ids += masked_lm_padding

        return (
            np.int32(input_ids),
            np.int32(masked_lm_positions),
            np.int32(masked_lm_ids),
        )

    def _map_fn(self, raw_record):
        """
        Parses a serialized protobuf example,
        creates input features for pretraining.
        """

        feature_map = {
            "primary": tf.io.FixedLenSequenceFeature([], tf.int64),
        }
        _, raw_features = tf.io.parse_single_sequence_example(
            serialized=raw_record, sequence_features=feature_map,
        )
        raw_input_ids = tf.cast(raw_features["primary"], tf.int32)
        input_ids, masked_lm_positions, masked_lm_ids = tf.numpy_function(
            self._create_mlm_input_features,
            [raw_input_ids],
            [tf.int32, tf.int32, tf.int32],
        )
        input_ids.set_shape(self.max_sequence_length)
        masked_lm_positions.set_shape(self.max_predictions_per_seq)
        masked_lm_ids.set_shape(self.max_predictions_per_seq)
        input_mask = tf.cast(tf.equal(input_ids, self.vocab["[PAD]"]), tf.int32)
        masked_lm_weights = tf.cast(
            tf.not_equal(masked_lm_positions, 0), self.mp_type
        )
        segment_ids = tf.zeros(self.max_sequence_length, dtype=tf.int32)

        if self.input_pad_id is not None:
            input_ids = tf.where(
                tf.cast(input_mask, tf.bool), self.input_pad_id, input_ids,
            )
        if self.segment_pad_id is not None:
            segment_ids = tf.where(
                tf.cast(input_mask, tf.bool), self.segment_pad_id, segment_ids,
            )
        if self.mlm_pad_id is not None:
            masked_lm_ids = tf.where(
                tf.cast(masked_lm_weights, tf.bool),
                masked_lm_ids,
                self.mlm_pad_id,
            )

        features = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "masked_lm_ids": masked_lm_ids,
            "masked_lm_positions": masked_lm_positions,
            "masked_lm_weights": masked_lm_weights,
            "segment_ids": segment_ids,
        }
        label = tf.zeros(())  # Pretraining is MLM only

        return features, label

    def post_batch_weight_scale_fn(self, feature, label):
        """
        When appropriate, scale mlm weights by `batch_size` / `num_valid_tokens`.
        This is used to compute the correct scaling factor on the loss without
        running into precision issues. Intended for use in situations when the
        loss will be divided by `batch_size` at the time of computation.
        """
        if self.scale_mlm_weights:
            mlm_weights = feature["masked_lm_weights"]
            scale = self.batch_size / tf.reduce_sum(mlm_weights)
            feature["masked_lm_weights"] = tf.cast(
                mlm_weights * scale, self.mp_type
            )
        return feature, label

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

    def create_tf_dataset(
        self, mode=tf.estimator.ModeKeys.TRAIN, input_context=None
    ):
        """
        Create tf dataset.

        :param mode: tf.estimator.ModeKeys.TRAIN (default) or
            tf.estimator.ModeKeys.EVAL or tf.estimator.ModeKeys.PREDICT.
        :param dict input_context: Given by distributed strategy for training.
        :returns: tf dataset.
        """

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        # Only allow nondeterminism when training and shuffling unseeded.
        self.deterministic = (not is_training) or not (
            self.shuffle and self.shuffle_seed is None
        )
        if self.deterministic:
            self.rng = random.Random(self.shuffle_seed)

        file_pattern = os.path.join(self.data_dir, "*.tfrecord")
        n_parallel_reads = 4
        filelist = tf.data.Dataset.list_files(
            file_pattern, shuffle=self.shuffle, seed=self.shuffle_seed
        )
        filelist = shard_dataset(
            filelist, self.use_multiple_workers, input_context
        )

        dataset = filelist.interleave(
            map_func=tf.data.TFRecordDataset,
            cycle_length=n_parallel_reads,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=self.deterministic,
        )

        return transform_dataset(
            dataset,
            self._map_fn,
            self.batch_size,
            is_training,
            shuffle=self.shuffle,
            shuffle_buffer=self.shuffle_buffer,
            repeat=self.repeat,
            seed=self.shuffle_seed,
            map_before_batch=True,
            batch_fn=self.batch_fn,
            post_batch_map_fn=self.post_batch_weight_scale_fn,
            # We specify number of parallel calls to `1` when `deterministic` is set to `True`.
            # This allows us to have fixed results regardless how threads are dealing with randomness
            # produced under `map_fn` function.
            num_parallel_calls=1
            if self.deterministic
            else tf.data.experimental.AUTOTUNE,
        )
