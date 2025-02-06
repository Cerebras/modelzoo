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

import random

import numpy as np

from cerebras.modelzoo.data_preparation.nlp.tokenizers.BPETokenizer import (
    BPETokenizer,
)


def training_data_generator(
    input_files,
    vocab_file,
    encoder_file,
    max_sequence_length,
    buffer_size=1e6,
    overlap_size=None,
    short_seq_prob=0,
    inverted_mask=False,
    add_special_tokens=True,
    eos_token="<|endoftext|>",
    pad_token="<|endoftext|>",
    input_ids_dtype="int32",
    input_mask_dtype="int32",
    labels_dtype="int32",
):
    """
    Generator function used to create input dataset
    for GPT2Model.

    :param list[str] input_files: List of input files.
    :param str vocab_file: Vocabulary file, to build tokenization from
    :param str encoder_file: Encoder file, map from word-pieces to
         token IDs for tokenization
    :param int max_sequence_length: Maximum length of the sequence to generate
    :param int short_seq_prob: Probability of a short sequence. Defaults to 0.
        Sometimes we want to use shorter sequences to minimize the mismatch
        between pre-training and fine-tuning.
    :param int buffer_size: Read buffer size. Defaults to 1MB.
    :param int overlap_size: Size of overlap when forming sequences from
      buffered token ids in a sliding window fashion.
      Defaults to None, which sets the overlap of max_sequence_length/4.
    :param bool inverted_mask: If set to False, has 0's on padded positions and
        1's elsewhere. Otherwise, "inverts" the mask, so that 1's are on padded
        positions and 0's elsewhere.
    :param str eos_token: End of sequence token. Defaults to "<|endoftext|>".
    :param str pad_token: Pad token. Defaults to "<|endoftext|>".
    :param str input_ids_dtype: Type of input ids. Defaults to "int32".
    :param str input_mask_dtype: Type of mask. Defaults to "int32".
    :param str labels_dtype: Type of labels. Defaults to "int32".

    :returns: yields training examples (feature, label)
    """
    assert (
        eos_token == "<|endoftext|>" and pad_token == "<|endoftext|>"
    ), "EOS and PAD tokens are given by '<|endoftext|>' for now."

    num_input_files = len(input_files)

    rng = random.Random()

    tokenizer = BPETokenizer(vocab_file, encoder_file)
    # id("<|endoftext|>") = 50256
    eos_id = 50256
    pad_id = 50256

    def _generate_train_example(token_ids):
        return _create_features_labels(
            token_ids,
            max_sequence_length,
            short_seq_prob,
            inverted_mask,
            pad_id,
            input_ids_dtype,
            input_mask_dtype,
            labels_dtype,
            rng,
        )

    if overlap_size is None:
        overlap_size = int(max_sequence_length / 4)
    assert overlap_size >= 0, "overlap_size must be non-negative."

    buffer_token_ids = [eos_id] if add_special_tokens else []

    for _file_num, _file in enumerate(input_files):
        with open(_file, 'r') as _fin:
            document_text = _fin.read()

        token_ids = tokenizer.encode(document_text)

        # skip empty documents
        if len(token_ids) == 0:
            continue

        if add_special_tokens:
            token_ids.append(eos_id)
        buffer_token_ids.extend(token_ids)

        # if buffer_size hasn't been reached
        # and current document is not the last one,
        # then continue adding elements into buffer
        if (
            len(buffer_token_ids) < buffer_size
            and _file_num < num_input_files - 1
        ):
            continue

        # generate sequences from buffer
        # in a sliding window fashion
        start_idx = 0
        end_idx = max_sequence_length + 1

        while end_idx <= len(buffer_token_ids):
            # need n+1 tokens to generate
            # (feature, label) of length n
            yield _generate_train_example(
                buffer_token_ids[start_idx:end_idx]
            ), _file_num
            start_idx = end_idx - overlap_size - 1
            end_idx = start_idx + max_sequence_length + 1

        # generate last example from  buffer
        if start_idx < len(buffer_token_ids) - 1:
            yield _generate_train_example(
                buffer_token_ids[-max_sequence_length - 1 :],
            ), _file_num

        buffer_token_ids = [eos_id] if add_special_tokens else []


def _create_features_labels(
    token_ids,
    max_sequence_length,
    short_seq_prob=0,
    inverted_mask=False,
    pad_id=0,
    input_ids_dtype="int32",
    input_mask_dtype="int32",
    labels_dtype="int32",
    rng=random.Random(),
):
    """
    Given a list of token_ids, generate input sequence
    and labels.
    """

    assert len(token_ids) >= 2, "token_ids must have at least 2 elements."

    if rng.random() < short_seq_prob:
        token_ids = token_ids[0 : rng.randint(2, max_sequence_length - 1)]

    input_ids = token_ids[:-1]
    labels = token_ids[1:]
    input_mask = [1] * len(input_ids)

    # padding
    num_pad = max_sequence_length - len(input_ids)
    padding = [pad_id] * num_pad

    input_ids.extend(padding)
    labels.extend(padding)
    input_mask.extend([0] * num_pad)

    # assertions to ensure correct output shapes
    assert (
        len(input_ids) == max_sequence_length
        and len(labels) == max_sequence_length
        and len(input_mask) == max_sequence_length
    ), "Wrong sequence length"

    # create feature dict
    features = dict()
    features["input_ids"] = getattr(np, input_ids_dtype)(input_ids)
    features["attention_mask"] = getattr(np, input_mask_dtype)(input_mask)

    if inverted_mask:
        features['attention_mask'] = np.equal(
            features['attention_mask'], 0
        ).astype(features['attention_mask'].dtype)
    labels = getattr(np, labels_dtype)(labels)
    features['labels'] = labels

    return features
