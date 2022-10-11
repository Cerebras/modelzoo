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

""" Util functions for data pipeline. """

import tensorflow as tf

from modelzoo.common.tf.input.utils import bucketed_batch


def get_special_tokens():
    """
    Function to get generic special tokens used in Transformer model.
    """
    return {
        "PAD": "<pad>",
        "UNK": "<unk>",
        "SOS": "<s>",
        "EOS": "</s>",
    }


def get_special_tokens_index(
    special_tokens, src_tokenizer=None, tgt_tokenizer=None,
):
    if src_tokenizer and tgt_tokenizer:
        special_tokens_index = {}
        for key, word in special_tokens.items():
            src_id = src_tokenizer.get_id(word)
            tgt_id = tgt_tokenizer.get_id(word)

            if src_id != tgt_id:
                raise ValueError(
                    f"{key} - {word} has different ids in src(src_id: {src_id}) and tgt(tgt_id: {tgt_id}) vocab files"
                )
            else:
                special_tokens_index[key] = src_id
    else:
        special_tokens_index = {
            "PAD": 0,
            "UNK": 1,
            "SOS": 2,
            "EOS": 3,
        }

    return special_tokens_index


def check_special_tokens_index(
    special_tokens, src_tokenizer=None, tgt_tokenizer=None,
):
    if src_tokenizer and tgt_tokenizer:
        for key, word in special_tokens.items():
            src_tokenizer.check_word_and_get_id(word)
            tgt_tokenizer.check_word_and_get_id(word)


def create_filter_fn(src_max_sequence_length, tgt_max_sequence_length):
    """
    Filter map function for a tf.data.Dataset.
    Discards the inputs where the sentences are less 
        than zero or greater than source and target sequence lengths.
    :param int src_max_sequence_length Max sequence length for encoder input.
    :param int tgt_max_sequence_length Max sequence length for decoder input.

    returns a function handle to pass to tf.data.Dataset.filter
    """

    def _filter_fn(feature, label):
        src_len = feature["encoder_input_length"]
        tgt_len = feature["decoder_input_length"]
        src_cond = tf.logical_and(
            src_len > 0, src_len <= src_max_sequence_length
        )
        tgt_cond = tf.logical_and(
            tgt_len > 0, tgt_len <= tgt_max_sequence_length
        )
        return tf.logical_and(src_cond, tgt_cond)

    return _filter_fn

    def key_func(feature, label):
        src_len = feature["encoder_input_length"]
        tgt_len = feature["decoder_input_length"]
        # Calculate bucket_width by maximum source sequence length.
        # Pairs with length [0, bucket_width) go to bucket 0, length
        # [bucket_width, 2 * bucket_width) go to bucket 1, etc.
        # Pairs with length over ((num_bucket-1) * bucket_width) words
        # all go into the last bucket.
        bucket_width = (
            src_max_sequence_length + num_buckets - 1
        ) // num_buckets

        # Bucket sentence pairs by the length of their source sentence
        # and target sentence.
        bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
        # key_func must return int64 per the group_by_window function's
        # requirement.
        return tf.cast(tf.minimum(num_buckets, bucket_id), tf.int64)

    def reduce_func(unused_key, windowed_data):
        return batching_fn(windowed_data)

    if num_buckets > 1:
        batched_dataset = dataset.apply(
            tf.data.experimental.group_by_window(
                key_func=key_func,
                reduce_func=reduce_func,
                window_size=batch_size,
            )
        )
    else:
        batched_dataset = dataset.apply(batching_fn)

    return batched_dataset


def create_masks_map_fn(pad_id, pad_id_label=None):
    def mask_map_fn(feature, label):
        # Add two additional tensors (masks to indicate where there is padding).
        # Attention layer expects 1's in padded positions and 0's in non-padded positions.

        # encoder_mask -> 1's in padded positions and 0's in non-padded positions.
        # decoder_mask -> 1's in padded positions and 0's in non-padded positions.
        encoder_mask = tf.cast(
            tf.equal(feature["encoder_input_ids"], pad_id), tf.int32,
        )
        decoder_mask = tf.cast(tf.equal(label, pad_id), tf.int32)

        if pad_id_label is not None:
            label = tf.where(
                tf.cast(decoder_mask, tf.bool), pad_id_label, label
            )

        return (
            {
                **feature,
                **{"encoder_mask": encoder_mask, "decoder_mask": decoder_mask,},
            },
            label,
        )

    return mask_map_fn


def create_batched_dataset(
    dataset,
    batch_size,
    buckets,
    src_max_sequence_length,
    tgt_max_sequence_length,
    pad_id,
):
    """
    Takes a dataset of sequences and returns a padded, batched dataset
        with examples bucketed by source sequence length.

    :param (tf.data.Dataset) dataset :
        Each example must be a tuple (x, y) with form:
        ({
            "encoder_input_ids": [tf.int32, ...],
            "encoder_input_length": tf.int32,
            "decoder_input_ids": [tf.int32, ...],
            "decoder_input_length": tf.int32,
        },
        [tf.int32, ...] # decoder_target_output_ids)
    :param int batch_size: size of each batch.
    :param int num_buckets: number of buckets for grouping sequences.
    :param int src_max_sequence_length: max length of encoder_input_ids.
    :param int tgt_max_sequence_length: max length of decoder_input_ids.
    :param int pad_id: padding token id for src and tgt sequences.

    :returns: tf dataset
        dataset (tf.Dataset): contain ``feature, label`` with the following
        keys and tensors.
        feature:
                "encoder_input_ids": [tf.int32, ...]
                "encoder_input_length": tf.int32
                "decoder_input_ids": [tf.int32, ...]
                "decoder_input_length": tf.int32
        label: [tf.int32, ...]  # decoder_target_output_ids
    """
    padded_shapes = (
        {
            "encoder_input_ids": [src_max_sequence_length],
            "decoder_input_ids": [tgt_max_sequence_length],
            "encoder_input_length": [],
            "decoder_input_length": [],
        },
        [tgt_max_sequence_length],
    )
    padding_values = (
        {
            "encoder_input_ids": pad_id,
            "decoder_input_ids": pad_id,
            "encoder_input_length": 0,
            "decoder_input_length": 0,
        },
        pad_id,
    )

    if buckets:
        return bucketed_batch(
            dataset,
            element_length_func=lambda f, l: tf.maximum(
                f["encoder_input_length"], f["decoder_input_length"]
            ),
            bucket_boundaries=buckets,
            batch_size=batch_size,
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            drop_remainder=True,
        )
    else:
        return dataset.padded_batch(
            batch_size,
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            drop_remainder=True,
        )


def scale_loss(
    features, labels, batch_size, tgt_max_sequence_length, compute_dtype
):
    """
    Create `loss_scale` input that can be used to scale the loss by
    the number of target tokens in the batch. Scaled by `batch_size`
    in the input function for numerical reasons.
    """
    loss_scale = batch_size / tf.reduce_sum(features["decoder_input_length"])
    loss_scale = tf.cast(loss_scale, compute_dtype)
    loss_scale = tf.broadcast_to(
        loss_scale, (batch_size, tgt_max_sequence_length)
    )
    features["loss_scale"] = loss_scale

    return features, labels
