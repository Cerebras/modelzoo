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

import math

import numpy as np


def shuffle(dataset, shuffle_buffer, rng):
    """
    Perform a buffered shuffle on an iterator.

    :param iterable dataset: The iterable to shuffle.
    :param int shuffle_buffer: The number of elements to buffer for the shuffle.
    :param random.Random rng: The random generator used as a source of
        randomness for shuffling.
    :yields: Elements of the original iterable in a different order.
    """
    data_buffer = []
    for x in dataset:
        if len(data_buffer) < shuffle_buffer:
            data_buffer.append(x)
        else:
            index = rng.randrange(shuffle_buffer)
            yield data_buffer[index]
            data_buffer[index] = x

    rng.shuffle(data_buffer)
    for x in data_buffer:
        yield x


def flat_map(fn, dataset):
    """
    Map a function over an iterator and flatten the result.

    :param callable fn: The function to apply to each element.
    :param iterator dataset: A stream of elements to map `fn` over.
    :yields: A flattened version of `fn` applied to each element of `dataset`.
    """
    for x in dataset:
        for output in fn(x):
            yield output


def select_random_chunk(tokens, max_length=65536, rng=None):
    """
    Select a random chunk of a sample. This is used to prevent bias towards
    very long passages in the corpus.

    :param list tokens: A list of token indices.
    :param int max_length: the maximum allowed length of a sample before
        splitting.
    :param np.random.Generator rng: The numpy random generator to be used as
        the source of randomness for this function.
    :returns: A list that is a random chunk of `tokens` if
        `len(tokens) > max_length` or `tokens` otherwise.
    """
    num_tokens = len(tokens)
    num_segments = math.ceil(num_tokens / max_length)
    if not num_segments:
        return tokens
    # np.random.uniform sometimes includes the upper bound, so we subtract 1e-5
    # to make the effective range [0, num_segments)
    start_index = max_length * int(rng.uniform(0, num_segments - 1e-5))
    end_index = min(start_index + max_length, num_tokens)
    return tokens[start_index:end_index]


def concatenate_documents(dataset, num_to_concatenate=128, pad_id=0):
    """
    Concatenate unrelated documents together to reduce the need for padding.

    :param iterable dataset: The input dataset.
    :param int num_to_concatenate: How many documents to concatanate together.
    :params int pad_id: The vocab id reserved for padding values. Must not occur
        anywhere in the dataset.
    :yields: new samples made from concatenating samples in `dataset`.
    """
    new_sample = []
    count = 0
    for x in dataset:
        new_sample.extend(x)
        count += 1
        if count == num_to_concatenate:
            yield new_sample
            new_sample = []
            count = 0


def _random_segmentation(num_items, num_segments, rng):
    """
    Partition a sequence of items randomly into non-empty segments.
    :param int num_items: An integer scalar > 0.
    :param int num_segments: An integer scalar in `[1, num_items]`.
    :param int seed: an integer seed.
    :param np.random.Generator rng: The numpy random generator to be used as
        the source of randomness for this function.
    :return: A numpy array with shape `[num_segments]` containing positive
        integers that add up to `num_items`.
    """
    first_in_segment = np.arange(num_items - 1) < num_segments - 1
    rng.shuffle(first_in_segment)
    first_in_segment_padded = np.pad(first_in_segment, [[1, 0]])
    segment_id = np.cumsum(first_in_segment_padded)

    segment_length = np.zeros(num_segments, dtype=np.int32)
    data = np.ones_like(segment_id)
    for i in range(len(segment_length)):
        segment_length[i] = sum(
            data[j] for j in range(len(segment_id)) if segment_id[j] == i
        )
    return segment_length


def random_spans_noise_mask(
    length, noise_density=0.15, mean_noise_span_length=3.0, rng=None
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
    :param int length: Length of the incoming token sequence.
    :param float noise_density: A float - approximate density of output mask.
    :param float mean_noise_span_length: A number used in the noise mask calculation.
    :param np.random.Generator rng: The numpy random generator to be used as
        the source of randomness for this function.
    :return: A boolean np.array with shape `[length]`.
    """
    assert rng is not None, "You must specify a random number generator"
    assert isinstance(
        rng, np.random.Generator
    ), f"rng must be a `np.random.Generator` object, got {type(rng)}"

    # Increase the length to avoid degeneracy.
    original_length = length
    length = max(length, 2)

    # Calculate number of noised tokens.
    num_noise_tokens = round(length * noise_density)

    # Avoid degeneracy by ensuring positive numbers of noise and non-noise tokens.
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)

    num_noise_spans = round(num_noise_tokens / mean_noise_span_length)

    # Avoid degeneracy by ensuring positive number of noise spans.
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    # Pick the lengths of the noise spans and the non-noise spans.
    noise_span_lengths = _random_segmentation(
        num_noise_tokens, num_noise_spans, rng=rng
    )
    nonnoise_span_lengths = _random_segmentation(
        num_nonnoise_tokens, num_noise_spans, rng=rng
    )

    # Stack both lengths into one tensor.
    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
        [num_noise_spans * 2],
    )

    span_starts = np.cumsum(interleaved_span_lengths)[:-1]

    span_start_indicator = np.zeros(length, dtype=np.int32)
    data = np.ones_like(span_starts)
    for i in range(len(span_start_indicator)):
        span_start_indicator[i] = sum(
            data[j] for j in range(len(span_starts)) if span_starts[j] == i
        )

    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)
    return is_noise[:original_length]


def _sentinel_id(vocab_size):
    """
    Token ID to use as a sentinel.
    By default, we use the last token in the vocabulary.
    :param int vocab_size: Size of the vocabulary with tokens.
    :return: int id of the sentinel.
    """
    return vocab_size - 1


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
    :param list tokens: A list of uncorrupted token indices.
    :param np.array noise_mask: A 1d boolean tensor with mask to apply noise.
    :param int vocab_size: Size of the vocabulary with tokens.
    :return: np.array with sentinels of the same type and shape as
        `tokens`.
    """
    previous_token_is_noise = np.pad(noise_mask[:-1], [[1, 0]])

    first_noise_tokens = np.logical_and(
        noise_mask, np.logical_not(previous_token_is_noise)
    )
    subsequent_noise_tokens = np.logical_and(
        noise_mask, previous_token_is_noise
    )

    sentinel = _sentinel_id(vocab_size) + 1 - np.cumsum(first_noise_tokens)

    tokens = np.where(first_noise_tokens, sentinel, tokens)
    return np.extract(np.logical_not(subsequent_noise_tokens), tokens)


def construct_denoising_objective(
    tokens, vocab_size, sos_token, eos_token, rng
):
    """
    Formats a raw sequence into a corrupted sequence and corresponding denoising
    targets.
    :param list tokens: A list of uncorrupted token indices.
    :param int vocab_size: The size of the vocabulary.
    :param int sos_token: The index of the `SOS` token in the vocabulary.
    :param int eos_token: The index of the `EOS` token in the vocabulary.
    :param np.random.Generator rng: The numpy random generator to be used as
        the source of randomness for this function.
    :returns: a tuple `(feature_dict, label)` of denoising source and target
        numpy arrays.
    """
    assert rng is not None, "You must specify a random number generator"
    assert isinstance(
        rng, np.random.Generator
    ), f"rng must be a `np.random.Generator` object, got {type(rng)}"

    noise_mask = random_spans_noise_mask(len(tokens), rng=rng)
    encoder_input_ids = noise_token_span_to_unique_sentinel(
        tokens, noise_mask, vocab_size
    )
    decoder_input_ids = noise_token_span_to_unique_sentinel(
        tokens, np.logical_not(noise_mask), vocab_size
    )

    # Add special tokens.
    decoder_output_ids = np.concatenate(
        (decoder_input_ids, [eos_token]), axis=0
    )
    decoder_input_ids = np.concatenate(([sos_token], decoder_input_ids), axis=0)

    labels = decoder_output_ids
    return {
        "input_ids": encoder_input_ids,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels,
    }


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
    :returns: An integer such that if a sequence is clipped to this length
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


def split_sequences(tokens, length):
    """
    Split a long sequence into shorter sequences of the specified length.
    :param list tokens: A list of token indices.
    :param int length: The maximum allowed length of a sample.

    :returns: A list of sequences containing exactly the same samples as before
        split into seperate samples such that no element of the dataset has
        length longer than specified.
    """
    n_tokens = len(tokens)
    num_segments = math.ceil(n_tokens / length)
    padding = num_segments * length - n_tokens
    padded = np.pad(tokens, [[0, padding]])
    original_lengths = np.concatenate(
        [np.repeat(length, num_segments - 1), [length - padding]], axis=0
    )
    sequences = list(np.reshape(padded, [-1, length]))
    for i, (sequence, original_length) in enumerate(
        zip(sequences, original_lengths)
    ):
        sequences[i] = sequence[:original_length]
    return sequences


def create_transformer_input_features(
    src_tokens,
    tgt_tokens,
    src_max_sequence_length,
    tgt_max_sequence_length,
    input_pad_id,
    attn_mask_pad_id,
    labels_pad_id,
    tokenize,
    sos_token="<s>",
    eos_token="</s>",
):
    """
    Creates features for Transformer model input.

    :param list src_tokens: Input tokens to process.
    :param list tgt_tokens: Target tokens to process.
    :param int src_max_sequence_length: Maximum sequence length of the encoder input.
    :param int tgt_max_sequence_length: Maximum sequence length of the decoder input.
    :param int input_pad_id: Input sequence padding id.
    :param int attn_mask_pad_id: Attention mask padding id.
    :param int labels_pad_id: Labels padding id.
    :param callable tokenize: Method to tokenize the input sequence.
    :param str sos_token: the index of the `SOS` token in the vocabulary.
    :param str eos_token: the index of the `EOS` token in the vocabulary.

    :returns: A dict with includes:
            * np.array[int.32] input_ids: Numpy array with encoder input token indices.
                Shape: (`src_max_sequence_length`).
            * np.array[int.32] decoder_input_ids: Numpy array with decoder input token indices.
                Shape: (`tgt_max_sequence_length`).
            * np.array[int.32] attention_mask: Numpy array with attention mask for encoder.
               Shape: (`src_max_sequence_length`).
            * np.array[int.32] decoder_attention_mask: Numpy array with attention mask for decoder.
               Shape: (`tgt_max_sequence_length`).
               `1` indicates the non masked token, and `0` indicates the masked token.
    """
    # Convert tokens to integer ids.
    src_token_ids = tokenize(src_tokens)
    tgt_token_ids = tokenize(tgt_tokens)

    # Filter out examples based on the sequence length.
    if not (0 < len(src_token_ids) < src_max_sequence_length) or not (
        0 < len(tgt_token_ids) < tgt_max_sequence_length
    ):
        return {}

    # Add special tokens.
    labels = np.concatenate((tgt_token_ids, [eos_token]), axis=0)
    decoder_input_ids = np.concatenate(([sos_token], tgt_token_ids), axis=0)

    # Pad input sequences.
    features = pad_t5_input_features(
        src_max_sequence_length,
        tgt_max_sequence_length,
        input_pad_id,
        attn_mask_pad_id,
        labels_pad_id,
        {
            "input_ids": src_token_ids,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        },
    )

    return features


def pad_t5_input_features(
    src_max_sequence_length,
    tgt_max_sequence_length,
    input_pad_id,
    attn_mask_pad_id,
    labels_pad_id,
    features,
):
    """
    Provides padding for T5 input features.

    :param src_max_sequence_length:
    :param tgt_max_sequence_length:
    :param input_pad_id:
    :param attn_mask_pad_id:
    :param labels_pad_id:
    :param features:
    :return: dict with padded features.
    """
    input_ids = (
        np.ones((src_max_sequence_length,), dtype=np.int32) * input_pad_id
    )
    decoder_input_ids = (
        np.ones((tgt_max_sequence_length,), dtype=np.int32) * input_pad_id
    )
    attention_mask = (
        np.ones((src_max_sequence_length,), dtype=np.int32) * attn_mask_pad_id
    )
    decoder_attention_mask = (
        np.ones((tgt_max_sequence_length,), dtype=np.int32) * attn_mask_pad_id
    )
    labels = np.ones((tgt_max_sequence_length,), dtype=np.int32) * labels_pad_id

    input_ids[: len(features["input_ids"])] = features["input_ids"]
    decoder_input_ids[: len(features["decoder_input_ids"])] = features[
        "decoder_input_ids"
    ]
    labels[: len(features["labels"])] = features["labels"]
    attention_mask[: len(features["input_ids"])] = 1
    decoder_attention_mask[: len(features["decoder_input_ids"])] = 1

    return {
        "input_ids": input_ids,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_attention_mask": decoder_attention_mask,
        "labels": labels,
        "decoder_input_length": np.array([len(features["decoder_input_ids"])]),
    }


def parse_text(text, do_lower):
    """
    Postprocessing of the CSV file.
    :param: str text: String with an input text.

    :return: List of parsed tokens.
    """
    tokens = text.split()
    return (
        list(map(lambda token: token.lower(), tokens)) if do_lower else tokens
    )
