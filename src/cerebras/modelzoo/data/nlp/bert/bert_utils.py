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

import os
import random

import numpy as np
import torch


class Vocab(dict):
    """
    Class to store vocab related attributes.
    """

    def __init__(self, inp_list, oov_id=-1):
        super(Vocab, self).__init__(inp_list)
        self.oov_id = oov_id

    def update(self, input_val):
        super(Vocab, self).update(input_val)
        self.inv_dict = {idx: token_str for token_str, idx in self.items()}

    def __missing__(self, key):
        return self.oov_id

    def forward(self, tokens):
        return [self[token] for token in tokens]

    def backward(self, ids):
        return [self.inv_dict[token_id] for token_id in ids]


def get_meta_data(data_dir):
    """
    Read data from meta files.
    :param str data_dir: Path to the input directory.
    :return: Processed meta data.
    """
    if not isinstance(data_dir, list):
        data_dir = [data_dir]

    meta_data = {}
    for file_name in data_dir:
        meta_file = os.path.join(file_name, "meta.dat")
        assert os.path.exists(
            meta_file
        ), f"Meta file is missing in the input directory: {data_dir}."

        with open(meta_file, "r") as fin:
            for line in fin.readlines():
                line = line.strip().split()
                meta_data[os.path.join(file_name, line[0])] = int(line[1])
    return meta_data


def parse_text(text, do_lower):
    """
    Postprocessing of the CSV file.
    This code should parse commas that are part of the token strings.
    :param: str text: String with an input text.

    :return: List of parsed tokens.
    """
    tokens = eval(text)
    return (
        list(map(lambda token: token.lower(), tokens)) if do_lower else tokens
    )


def shard_and_shuffle_data(files_per_task, shuffle, shuffle_seed):
    """
    Shard the data across the processes.

    :param: list files_per_task: List of files with input data.
    :param bool shuffle: Whether to shuffle data or not.
    :param bool shuffle_seed: Seed to use for shuffling.
    :return: A tuple with:
        * int processed_buffers: Counter for how many buffers of data processed so far.
        * list files_per_worker: Files to process for the input worker.
        * int shuffle_seed: Updated shuffle seed.
        * random.Random rng: Object with shuffle function.
    """

    worker_info = torch.utils.data.get_worker_info()
    files_per_task_per_worker = []

    if worker_info is not None:
        worker_id = worker_info.id
        num_workers = worker_info.num_workers

        assert num_workers <= len(files_per_task), (
            f"Number of processes should be less than number of files, "
            f"Got `num_workers` equal to {num_workers} and `num_files` equal to {len(files_per_task)}."
        )

        # Gather files for the input worker based in the file index and
        # number of workers.
        for file_index, file_len_start_id in enumerate(files_per_task):
            if file_index % num_workers == worker_id:
                files_per_task_per_worker.append(
                    file_len_start_id
                )  # Tuple of csv_filepath, num_examples_to consider, start_idx

        # Use a unique seed for each worker.
        if shuffle_seed is not None:
            shuffle_seed += worker_id + 1
    else:
        # num_worker = 0 case
        files_per_task_per_worker = files_per_task

    rng = random.Random(shuffle_seed)

    processed_buffers = 0

    if shuffle:
        rng.shuffle(files_per_task_per_worker)

    return processed_buffers, files_per_task_per_worker, shuffle_seed, rng


def create_masked_lm_predictions(
    tokens,
    max_sequence_length,
    mask_token_id,
    max_predictions_per_seq,
    input_pad_id,
    attn_mask_pad_id,
    labels_pad_id,
    tokenize,
    vocab_size,
    masked_lm_prob,
    rng,
    exclude_from_masking,
    mask_whole_word,
    replacement_pool=None,
):
    """
    Creates the predictions for the masked LM objective.

    :param list tokens: Tokens to process.
    :param int max_sequence_length: Maximum sequence length.
    :param int mask_token_id: Id of the masked token.
    :param int max_predictions_per_seq: Maximum number of masked LM predictions per sequence
    :param int input_pad_id: Input sequence padding id.
    :param int attn_mask_pad_id: Attention mask padding id.
    :param int labels_pad_id: Labels padding id.
    :param callable tokenize: Method to tokenize the input sequence.
    :param str vocab_size: Size of the vocabulary file.
    :param float masked_lm_prob: Masked LM probability.
    :param random.Random rng: Object with shuffle function.
    :param list exclude_from_masking: List of tokens to exclude from masking.
    :param bool mask_whole_word: Whether to mask the whole words or not.
    :param list replacement_pool: List of ids which should be included
        when replacing tokens with random words from vocab. Default is None
        and means that we can take any token from the vocab.

    :returns: tuple which includes:
            * np.array[int.32] input_ids: Numpy array with input token indices.
                Shape: (`max_sequence_length`).
            * np.array[int.32] labels: Numpy array with labels.
               Shape: (`max_sequence_length`).
            * np.array[int.32] attention_mask
               Shape: (`max_sequence_length`).
            * np.array[int.32] masked_lm_mask: Numpy array with a mask of
               predicted tokens.
               Shape: (`max_predictions`)
               `0` indicates the non masked token, and `1` indicates the masked token.
    """
    input_ids = np.ones((max_sequence_length,), dtype=np.int32) * input_pad_id
    attention_mask = (
        np.ones((max_sequence_length,), dtype=np.int32) * attn_mask_pad_id
    )
    labels = np.ones((max_sequence_length,), dtype=np.int32) * labels_pad_id
    masked_lm_mask = np.zeros((max_sequence_length,), dtype=np.int32)

    # Convert tokens to integer ids.
    token_ids = tokenize(tokens)
    num_tokens = len(token_ids)
    input_ids[:num_tokens] = token_ids
    attention_mask[:num_tokens] = 1

    # Form predictions for the MLM task.
    num_to_predict = min(
        max_predictions_per_seq,
        max(1, int(round(num_tokens * masked_lm_prob))),
    )

    # Track which tokens have been used.
    num_current_predictions = 0
    token_indices = list(range(num_tokens))
    rng.shuffle(token_indices)

    masked_token_indices = []
    while num_current_predictions < num_to_predict:
        # Stop adding masked token indices, if we reach the limit.
        if (
            len(token_indices) == 0
            or len(masked_token_indices) >= max_predictions_per_seq
        ):
            break

        current_token_index = token_indices[0]
        current_token = tokens[current_token_index]

        if current_token in exclude_from_masking:
            token_indices.pop(0)
            continue

        if mask_whole_word:
            # Get span of the word for whole word masking.
            span = get_whole_word_span(tokens, current_token_index)
        else:
            span = [current_token_index, current_token_index + 1]

        # Calculate the number of predicted tokens at the current iteration.
        span_len = span[1] - span[0]

        for index in range(span[0], span[1]):
            token_indices.remove(index)

        if len(masked_token_indices) + span_len > max_predictions_per_seq:
            # Only add the word if it does not overflow the maximum
            # predictions this should only happen with whole word masking.
            continue

        masked_token_indices.extend(range(span[0], span[1]))

        # Add these tokens to the labels.
        labels[span[0] : span[1]] = input_ids[span[0] : span[1]]

        num_current_predictions += span_len

    for masked_token_index in masked_token_indices:
        masked_lm_mask[masked_token_index] = 1
        rnd = rng.random()
        if rnd < 0.8:
            # Mask it `80%` of the time
            input_ids[masked_token_index] = mask_token_id
        elif rnd < 0.9:
            # `10%` of the time replace with random token
            # `random.randint` is inclusive `[0, len(vocab) - 1]`.
            if replacement_pool is not None:
                random_token_id = rng.choice(replacement_pool)
            else:
                random_token_id = rng.randint(0, vocab_size - 1)
            input_ids[masked_token_index] = random_token_id
        else:
            # `10%` of the time leave input as is.
            pass

    return input_ids, labels, attention_mask, masked_lm_mask


def get_whole_word_span(tokens, start_index):
    """
    Returns the whole word start and end
    indices.

    :param: list tokens: Tokens to process.
    :param: int start_index: Start index.

    :returns: tuple with start and end index of the word
              from the token list.
    """
    end_index = len(tokens)
    if start_index < len(tokens) - 1:
        end_index = min(start_index + 1, len(tokens) - 1)
        while tokens[end_index].startswith("##"):
            end_index += 1
            if end_index > len(tokens) - 1:
                break

    while tokens[start_index].startswith("##") and start_index > 0:
        start_index -= 1

    return start_index, end_index


def build_vocab(vocab_file, do_lower, oov_token):
    """

    Load up the vocab file.
    :param: str vocab_file: Path to the vocab file.
    :param: bool do_lower: Whether the tokens should be
        converted to lower case.
    :param str oov_token: Token reserved for the out of vocabulary tokens.

    :returns: A tuple with:
            * dict vocab: Contains the words from the vocab as keys
                and indices as values.
            * int vocab_size: Size of the resulted vocab.
    """
    assert os.path.exists(vocab_file), f"Vocab file not found {vocab_file}."

    with open(vocab_file, "r") as fin:
        vocab = fin.readlines()

    vocab_words = [convert_to_unicode(word).strip() for word in vocab]
    vocab_size = len(vocab)

    if do_lower:
        vocab_words = list(map(lambda word: word.lower(), vocab_words))

    vocab = Vocab({}, vocab_words.index(oov_token))
    vocab.update({word: id for id, word in enumerate(vocab_words)})
    return vocab, vocab_size


def convert_to_unicode(text):
    """

    Converts `text` to unicode, assuming utf-8 input.
    Returns text encoded in a way suitable for print or `tf.compat.v1.logging`.
    """
    if isinstance(text, str):
        return text
    return text.decode("utf-8", "ignore")
