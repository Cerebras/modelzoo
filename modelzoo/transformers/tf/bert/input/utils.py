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

import numpy as np
import tensorflow as tf

from modelzoo.common.tf.input.utils import (
    create_bytes_feature,
    create_float_feature,
    create_int_feature,
)


def create_tf_train_example(features, labels):
    """
    Create tf.train.Example intances for features commonly used in BERT
    """
    features_dict = {}
    features_dict["input_ids"] = create_int_feature(features["input_ids"])
    features_dict["input_mask"] = create_int_feature(features["input_mask"])
    features_dict["masked_lm_positions"] = create_int_feature(
        features["masked_lm_positions"]
    )
    features_dict["masked_lm_ids"] = create_int_feature(
        features["masked_lm_ids"]
    )
    features_dict["masked_lm_weights"] = create_float_feature(
        features["masked_lm_weights"]
    )

    if "segment_ids" in features:
        # MLM + NSP dataset
        features_dict["segment_ids"] = create_int_feature(
            features["segment_ids"]
        )
        features_dict["next_sentence_labels"] = create_int_feature([labels])

    tf_example = tf.train.Example(
        features=tf.train.Features(feature=features_dict)
    )

    return tf_example


def create_unmasked_tokens_example(tokens):
    """
    Create tf.train.Example containing variable length sequence of tokens.
    """
    array = [create_bytes_feature(token.encode()) for token in tokens]
    features = {"tokens": tf.train.FeatureList(feature=array)}
    features = tf.train.FeatureLists(feature_list=features)
    tf_example = tf.train.SequenceExample(feature_lists=features)
    return tf_example


def parse_raw_tfrecord(
    raw_record,
    max_sequence_length,
    max_predictions_per_seq,
    mp_type,
    mlm_only=False,
):

    feature_map = {
        'input_ids': tf.io.FixedLenFeature([max_sequence_length], tf.int64),
        'input_mask': tf.io.FixedLenFeature([max_sequence_length], tf.int64),
        'masked_lm_positions': tf.io.FixedLenFeature(
            [max_predictions_per_seq], tf.int64
        ),
        'masked_lm_ids': tf.io.FixedLenFeature(
            [max_predictions_per_seq], tf.int64
        ),
        'masked_lm_weights': tf.io.FixedLenFeature(
            [max_predictions_per_seq], tf.float32
        ),
    }

    if not mlm_only:
        feature_map["segment_ids"] = tf.io.FixedLenFeature(
            [max_sequence_length], tf.int64
        )
        feature_map["next_sentence_labels"] = tf.io.FixedLenFeature(
            [1], tf.int64
        )

    example = tf.io.parse_example(serialized=raw_record, features=feature_map,)

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
        "masked_lm_ids": example["masked_lm_ids"],
        "masked_lm_positions": example["masked_lm_positions"],
        "masked_lm_weights": tf.cast(example["masked_lm_weights"], mp_type),
    }

    if "segment_ids" in example:
        feature["segment_ids"] = example["segment_ids"]

    if "next_sentence_labels" in example:
        # example["next_sentence_labels"] is of shape (bsz, 1).
        # Converting it to (bsz, )
        label = tf.reshape(example["next_sentence_labels"], shape=[-1])
    else:
        # Currently label=None is not supported.
        label = tf.convert_to_tensor(
            np.empty(feature["input_ids"].shape[0]), dtype=tf.int32
        )

    return (feature, label)
