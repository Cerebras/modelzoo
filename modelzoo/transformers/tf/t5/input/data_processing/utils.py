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

from modelzoo.transformers.tf.transformer.input.data_processing.utils import (  # noqa
    create_batched_dataset,
    create_masks_map_fn,
    scale_loss,
)


def get_special_tokens():
    """
    Function to get generic special tokens used in T5 model.
    :return dict with special tokens for T5 model to use.
    """
    return {
        "PAD": "<pad>",
        "UNK": "<unk>",
        # The huggingface reference implementation maps
        # SOS token to PAD token, and it won't work when
        # setting `use_vsl: True`, instead we map it to
        # <s> token.
        "SOS": "<s>",
        "EOS": "</s>",
    }


def get_additional_special_tokens(extra_ids):
    """
    Function to get generic special tokens used in T5 model.
    :param int extra_ids: number of additional special tokens
        for T5 model to use.
    :return list of extra tokens each of format
        <extra_id_{num}> where `num` is a number between
         `0` and `extra_ids` - 1.
    """
    return [f"<extra_id_{i}>" for i in range(extra_ids)]


def convert_text_to_token_ids(text, tokenizer):
    """
    Helper function to convert a string of text to integer ids.
    :param text tf.string sentence to split.
    :param tokenizer T5BaseTokenizer instance of `BaseTokenizer` to convert
        text to indices based on vocab file.
    :return tensor containing tf.int32 indices.
    """
    text = tf.compat.v1.string_split([text]).values
    return tokenizer.convert_tokens_tensor_to_ids(text)


def noise_token_span_to_unique_sentinel(tokens, noise_mask, vocab_size):
    """
    Replace each run of consecutive noise tokens with a different sentinel.
    The idea here is to be able to align the dropped spans in the inputs
    with the markers in the targets.

    We want to generate training examples like
    "We hold <X> to be <Y> that" -> "<X> these truths <Y> self evident <Z>"

    Sentinels assigned in decreasing order within the sequence starting at
    `vocab_size` - 1.  That is, we appropriate the last tokens in the
    vocabulary for additional use as sentinels.

    :param Tensor tokens: 1d integer tokens tensor.
    :param Tensor noise_mask: A 1d boolean tensor with mask to apply noise.
    :param int vocab_size: Size of the vocabulary with tokens.
    :return: Tensor with sentinels of the same type and shape as
        `tokens`.

    Based on `~google-research/text-to-text-transformer/t5/data/preprocess.py:noise_token_span_to_unique_sentinel`.
    """
    previous_token_is_noise = tf.pad(noise_mask[:-1], [[1, 0]])

    first_noise_tokens = tf.logical_and(
        noise_mask, tf.logical_not(previous_token_is_noise)
    )
    subsequent_noise_tokens = tf.logical_and(
        noise_mask, previous_token_is_noise
    )

    sentinel = (
        _sentinel_id(vocab_size)
        + 1
        - tf.cumsum(tf.cast(first_noise_tokens, tokens.dtype))
    )

    tokens = tf.where(first_noise_tokens, sentinel, tokens)
    return tf.boolean_mask(tokens, tf.logical_not(subsequent_noise_tokens))


def _sentinel_id(vocab_size):
    """
    Token ID to use as a sentinel.
    By default, we use the last token in the vocabulary.
    :param int vocab_size: Size of the vocabulary with tokens.
    :return: int id of the sentinel.
    """
    return vocab_size - 1


def _random_segmentation(num_items, num_segments, seed):
    """
    Partition a sequence of items randomly into non-empty segments.
    :param int num_items: An integer scalar > 0.
    :param int num_segments: An integer scalar in `[1, num_items]`.
    :param int seed: an integer seed.

    :return: A Tensor with shape `[num_segments]` containing positive
        integers that add up to `num_items`.

    Based on `~google-research/text-to-text-transformer/t5/data/preprocess.py:random_segmentation`.
    """
    first_in_segment = tf.random.shuffle(
        tf.cast(tf.range(num_items - 1) < num_segments - 1, tf.int32), seed
    )
    first_in_segment_padded = tf.pad(first_in_segment, [[1, 0]])
    segment_id = tf.cumsum(first_in_segment_padded)
    segment_length = tf.math.segment_sum(tf.ones_like(segment_id), segment_id)
    return segment_length


def random_spans_noise_mask(
    length, seeds=[0, 1], noise_density=0.15, mean_noise_span_length=3.0
):
    """
    Noise mask consisting of random spans of noise tokens.

    The number of noise tokens and the number of noise spans and non-noise spans
    are determined deterministically as follows:

          `num_noise_tokens` = round(`length` * `noise_density`)
          `num_nonnoise_spans` = `num_noise_spans` = round(
          `num_noise_tokens` / `mean_noise_span_length`)

    Spans alternate between non-noise and noise, beginning with non-noise.
    Subject to the above restrictions, all masks are equally likely.

    :param Tensor length: An int32 scalar (length of the incoming token sequence).
    :param Tensor seeds: An int32 Tensor, shaped (2, 2).
    :param float noise_density: A float - approximate density of output mask.
    :param float mean_noise_span_length: A number used in the noise mask calculation.

    :return: A boolean tensor with shape `[length]`.

    Based on `~google-research/text-to-text-transformer/t5/data/preprocess.py:random_spans_noise_mask`.
    """
    # Increase the length to avoid degeneracy.
    original_length = length
    length = tf.maximum(length, 2)

    # Calculate number of noised tokens.
    num_noise_tokens = tf.cast(
        tf.round(tf.cast(length, tf.float32) * noise_density), tf.int32
    )

    # Avoid degeneracy by ensuring positive numbers of noise and non-noise tokens.
    num_noise_tokens = tf.minimum(tf.maximum(num_noise_tokens, 1), length - 1)

    num_noise_spans = tf.cast(
        tf.round(
            tf.cast(num_noise_tokens, tf.float32) / mean_noise_span_length
        ),
        tf.int32,
    )
    # Avoid degeneracy by ensuring positive number of noise spans.
    num_noise_spans = tf.maximum(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    # Pick the lengths of the noise spans and the non-noise spans.
    noise_span_lengths = _random_segmentation(
        num_noise_tokens, num_noise_spans, seeds[0]
    )
    nonnoise_span_lengths = _random_segmentation(
        num_nonnoise_tokens, num_noise_spans, seeds[1]
    )

    # Stack both lengths into one tensor.
    interleaved_span_lengths = tf.reshape(
        tf.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
        [num_noise_spans * 2],
    )

    span_starts = tf.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = tf.math.unsorted_segment_sum(
        tf.ones_like(span_starts), span_starts, length
    )
    span_num = tf.cumsum(span_start_indicator)
    is_noise = tf.equal(span_num % 2, 1)
    return is_noise[:original_length]


def get_raw_sequence_lengths(
    max_sequence_length, corruption_prob=0.15, mean_span_len=3
):
    """
    T5 span corruption takes a sequence `raw_sequence` and corrupts spans to
    generate sequences `masked_input` and `target`. This function computes
    the maximum possible length of `raw_sequence` such that `masked_input`
    has length no greater than `max_sequence_length`. It outputs this length
    along with the maximum length of `targets` for this length of
    `raw_sequences`.

    :param int max_sequence_length: The maximum length of the encoder inputs
        after masking.
    :param float corruption_prob: The fraction of tokens that are corrupted
        for the denoising objective.
    :param int mean_span_len: The average length of a corrupted span.
    :returns: an integer such that if a sequence is clipped to this length
        before masking then it will have length at most max_sequence_length
        after masking; an integer that is the maximum possible length of a
        decoder sequence.
    """

    def get_post_masking_lens(unmasked_len):
        noise_tokens = int(round(unmasked_len * corruption_prob))
        nonnoise_tokens = unmasked_len - noise_tokens
        noise_spans = int(round(noise_tokens / mean_span_len))
        masked_src_len = nonnoise_tokens + noise_spans + 1
        masked_tgt_len = noise_tokens + noise_spans + 1
        return masked_src_len, masked_tgt_len

    raw_input_len = max_sequence_length
    while get_post_masking_lens(raw_input_len)[0] < max_sequence_length:
        raw_input_len += 1
    _, targets_len = get_post_masking_lens(raw_input_len)
    return raw_input_len, targets_len


def select_random_chunk(tokens, max_length=65536, seed=None):
    """
    Select a random chunk of a sample. This is used to prevnet bias towards
    very long passages in the corpus.

    :param tf.Tensor tokens: an int32 tensor of token indices.
    :param int max_length: the maximum allowed length of a sample before
        splitting.
    :returns: a tf.Tensor that is a random chunk of `tokens` if
        `len(tokens) > max_length` or `tokens` otherwise.
    """
    num_tokens = tf.size(tokens)
    num_segments = tf.math.ceil(
        tf.cast(num_tokens, tf.float32) / tf.cast(max_length, tf.float32)
    )
    num_segments = tf.cast(num_segments, tf.int32)
    start_index = max_length * tf.random.uniform(
        [], maxval=num_segments, dtype=tf.int32, seed=seed
    )
    end_index = tf.minimum(start_index + max_length, num_tokens)
    return tokens[start_index:end_index]


def concatenate_documents(
    dataset, num_to_concatenate=128, pad_id=0, **map_args
):
    """
    Concatenate unrelated documents together to reduce the need for padding.

    Adapted from https://github.com/google-research/text-to-text-transfer-transformer/blob/b7d861c6b383c97eda6de7dfe5d2867a3c6afd0a/t5/data/preprocessors.py#L2167

    :param tf.data.Dataset dataset: The input dataset.
    :param int num_to_concatenate: How many documents to concatanate together.
    :params int pad_id: The vocab id reserved for padding values. Must not occur
        anywhere in the dataset.
    :param **map_args: keyword arguments to be passed to `tf.data.Dataset.map`.
    :returns: a `tf.data.Dataset` with concatenated samples.
    """
    dataset = dataset.padded_batch(
        batch_size=num_to_concatenate,
        padded_shapes=[-1],
        padding_values=pad_id,
    )

    def flatten_batch(x):
        x = tf.reshape(x, [-1])
        return tf.boolean_mask(x, tf.not_equal(x, pad_id))

    return dataset.map(flatten_batch, **map_args)


def split_sequences(dataset, length, **map_args):
    """
    Split a long sequence into shorter sequences of the specified length.

    :param tf.data.Dataset dataset: the dataset to split sequences for.
    :param int length: the maximum allowed length of a sample in the resulting
        dataset.
    :param **map_args: keyword arguments to be passed to `tf.data.Dataset.map`.
    :returns: a `tf.data.Dataset` containing exactly the same samples as before
        split into seperate samples such that no element of the dataset has
        length longer than specified.
    """

    def split_fn(tokens):
        n_tokens = tf.size(tokens)
        num_segments = tf.cast(
            tf.math.ceil(
                tf.cast(n_tokens, tf.float32) / tf.cast(length, tf.float32)
            ),
            tf.int32,
        )
        padding = num_segments * length - n_tokens
        padded = tf.pad(tokens, [[0, padding]])
        original_lengths = tf.concat(
            [tf.repeat(length, num_segments - 1), [length - padding]], axis=0
        )
        return tf.reshape(padded, [-1, length]), original_lengths

    def strip_padding(tokens, original_length):
        return tokens[:original_length]

    dataset = dataset.map(split_fn, **map_args)
    dataset = dataset.unbatch()
    dataset = dataset.map(strip_padding, **map_args)
    return dataset


def construct_denoising_objective(tokens, vocab_size, sos_token, eos_token):
    """
    Formats a raw sequence into a corrupted sequence and corresponding denoising
    targets.

    :param tf.Tensor tokens: an int32 tensor of uncorrupted token indices.
    :param int vocab_size: the size of the vocabulary.
    :param int sos_token: the index of the `SOS` token in the vocabulary.
    :param int eos_token: the index of the `EOS` token in the vocabulary.
    :returns: a tuple `(feature_dict, label)` of denoising source and target
        tensors.
    """
    noise_mask = random_spans_noise_mask(tf.shape(tokens)[0])
    encoder_input_ids = noise_token_span_to_unique_sentinel(
        tokens, noise_mask, vocab_size
    )
    decoder_input_ids = noise_token_span_to_unique_sentinel(
        tokens, tf.logical_not(noise_mask), vocab_size
    )

    # Add special tokens.
    eos_token = tf.constant([eos_token], dtype=tf.int32)
    decoder_output_ids = tf.concat((decoder_input_ids, eos_token), 0)

    sos_token = tf.constant([sos_token], dtype=tf.int32)
    decoder_input_ids = tf.concat((sos_token, decoder_input_ids), 0)

    features = {
        "encoder_input_ids": encoder_input_ids,
        "encoder_input_length": tf.size(encoder_input_ids, tf.int32),
        "decoder_input_ids": decoder_input_ids,
        "decoder_input_length": tf.size(decoder_input_ids, tf.int32),
    }
    label = decoder_output_ids
    return features, label
