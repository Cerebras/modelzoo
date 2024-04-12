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
LMDataTokenGenerator Module

This module provides the LMDataTokenGenerator class which is designed to process
text data and create features suitable for language modeling tasks.

Usage:
    tokenizer = LMDataTokenGenerator(dataset_params,max_sequence_length,tokenizer)
    tokenized_features = tokenizer.encode("Sample text for processing.")
"""

import random
from typing import Any, Dict, List, Tuple

import ftfy
import numpy as np

from cerebras.modelzoo.data_preparation.nlp.hdf5_preprocessing.utils import (
    split_text_and_tokenize,
    validate_tokens,
    wikitext_detokenizer,
)


def create_features_auto_lm(
    token_ids,
    max_sequence_length,
    short_seq_prob=0,
    inverted_mask=False,
    pad_id=0,
    min_len=10,
    input_ids_dtype="int32",
    input_mask_dtype="int32",
    labels_dtype="int32",
    rng=None,
):
    """Given a list of token_ids, generate input sequence and labels.

    Args:
        token_ids (sequence): List containing token ids for creating features,
            labels and input mask from.
        max_sequence_length (int): Maximum sequence length for data writes.
        short_seq_prob (float): Probability of generating short sequences from
            data. Defaults to `0`.
        inverted_mask (bool): Invert mask if specified for runtime execution.
            Defaults to `False`.
        min_len (int): Minimum length of token_ids to be considered a valid
            sequence.
        pad_id (int): Id for pad token. Defaults to `0`.
        input_ids_dtype (str): Dtype as string for input ids.
            Defaults to `int32`.
        input_mask_dtype (str): Dtype as string for input mask.
            Defaults to `int32`.
        labels_dtype (str): Dtype as string for labels. Defaults to `int32`.
        rng (random.Random obj): Instance of random object, with states set.
            Defaults to `None`.

    Returns:
        Tuple containing features, labels and number of pad tokens
    """
    if not validate_tokens(token_ids, min_len=min_len):
        return []

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
    features["input_mask"] = getattr(np, input_mask_dtype)(input_mask)

    if inverted_mask:
        features["input_mask"] = np.equal(features["input_mask"], 0).astype(
            features["input_mask"].dtype
        )
    labels = getattr(np, labels_dtype)(labels)

    return np.stack([features["input_ids"], features["input_mask"], labels])


class LMDataTokenGenerator:
    def __init__(self, params, tokenizer, eos_id, pad_id):
        """
        Initialize the LMDataTokenGenerator class.

        Args:
            vocab_file (str): Path to the vocabulary file.
            encoder_file (str): Path to the encoder file.
            max_seq_length (int, optional): Maximum sequence length. Defaults to 2048.
        """
        dataset_params = params["dataset"]
        processing_params = params["processing"]
        self.tokenizer = tokenizer
        self.use_ftfy = dataset_params.pop("use_ftfy", False)
        self.ftfy_normalizer = dataset_params.pop("ftfy_normalizer", "NFC")
        self.wikitext_detokenize = dataset_params.pop(
            "wikitext_detokenize", False
        )
        self.pack_sequences = dataset_params.pop("pack_sequences", True)
        self.min_sequence_len = dataset_params.pop("min_sequence_len", 10)
        self.input_ids_dtype = dataset_params.pop("input_ids_dtype", "int32")
        self.input_mask_dtype = dataset_params.pop("input_mask_dtype", "int32")
        self.inverted_mask = dataset_params.pop("inverted_mask", False)
        self.inverted_mask = dataset_params.pop("inverted_mask", False)

        self.max_seq_length = processing_params.pop("max_seq_length", 2048)
        self.short_seq_prob = processing_params.pop("short_seq_prob", 0.0)
        self.seed = processing_params.pop("seed", 0)
        self.files_per_record = processing_params.pop(
            "files_per_record", 50000
        )  ## redundant param

        self.split_text_to_tokenize = processing_params.pop(
            "split_text_to_tokenize", False
        )
        if self.split_text_to_tokenize:
            self.chunk_len_to_split = processing_params.pop(
                "chunk_len_to_split", 2000
            )
            self.remove_bos_in_chunks = processing_params.pop(
                "remove_bos_in_chunks", False
            )
        self.eos_id = eos_id
        self.pad_id = pad_id

        self.rng = random.Random()
        self.rng.seed(self.seed)
        self.prefix = []

    def tokenize_text(self, text: str) -> List[int]:
        """
        Tokenize the provided text.

        Args:
            text (str): Text to tokenize.

        Returns:
            List[int]: List of token IDs.
        """
        return self.tokenizer.encode(text)

    def process_chunks(
        self, tokenized_text_chunks: List[List[int]]
    ) -> Tuple[List[Any], Dict]:
        """
        Processes chunks of tokenized text and returns processed features along with the total padding added.

        Args:
        tokenized_text_chunks (List[List[int]]): A list of tokenized text chunks, where each chunk is represented as a list of integers.

        Returns:
        Tuple[List[Any], Dict]: A tuple containing a list of processed results and dataset stats.
        """

        results = []  # List to store the processed results
        stats = {
            "loss_valid_tokens": 0,
            "num_tokens": 0,
            "num_pad_tokens": 0,
            "non_pad_tokens": 0,
            "num_masked_tokens": 0,
            "empty_chunks": 0,
        }
        # Iterate over each chunk in the tokenized text chunks
        for chunk in tokenized_text_chunks:
            # Process the chunk and get the processed result and number of padding tokens added
            processed = create_features_auto_lm(
                chunk,
                self.max_seq_length,
                short_seq_prob=self.short_seq_prob,
                inverted_mask=self.inverted_mask,
                pad_id=self.pad_id,
                min_len=self.min_sequence_len,
                input_ids_dtype=self.input_ids_dtype,
                input_mask_dtype=self.input_mask_dtype,
                labels_dtype=self.input_ids_dtype,
                rng=self.rng,
            )
            # If the processed chunk is not empty, add the results to the list and update the total padding
            if processed != []:
                loss_valid_tokens = int(processed[1, :].sum())
                num_pad = int((processed[0, :] == self.pad_id).sum())
                stats["num_pad_tokens"] += num_pad
                stats["non_pad_tokens"] += self.max_seq_length - num_pad
                stats["num_masked_tokens"] += (
                    self.max_seq_length - loss_valid_tokens
                )
                stats["loss_valid_tokens"] += loss_valid_tokens
                stats["num_tokens"] += len(processed[0])
                results.append(processed)
            else:
                stats["empty_chunks"] += 1

        # Return the list of processed results and data stats
        return results, stats

    def tokenize_text_auto_lm(self, text: str) -> Tuple[List[np.ndarray], Dict]:
        """
        Tokenize the text and create features for auto-regressive language modeling.

        Args:
            text (str): Text to tokenize.

        Returns:
            Tuple[List[np.ndarray], Dict]: Tuple of encoded features for auto-regressive language modeling and dataset stats.
        """
        discarded_files = 0
        num_false_discards = 0
        # tokenize text
        if self.split_text_to_tokenize:
            # TODO: implement a better fix for this by updating the tokenizer
            # normalization rules. This is a temporary fix and it may
            # cause issues with the spacing tokens being repeated.
            tokenized_text = split_text_and_tokenize(
                text,
                self.tokenizer,
                max_tok_len=self.chunk_len_to_split,
                remove_bos_in_chunks=self.remove_bos_in_chunks,
            )
        else:
            tokenized_text = self.tokenize_text(text)
        if self.eos_id is not None:
            tokenized_text += [self.eos_id]
        all_text = self.prefix + tokenized_text
        tokenized_text_chunks = [
            all_text[i : i + self.max_seq_length + 1]
            for i in range(0, len(all_text), self.max_seq_length)
        ]

        # reset prefix
        self.prefix = []

        # update prefix if last chunk is < max_seq_length
        num_tokens_last_chunk = len(tokenized_text_chunks[-1])

        if self.pack_sequences:
            if num_tokens_last_chunk < self.max_seq_length + 1:
                last_chunk = tokenized_text_chunks.pop(-1)
                self.prefix.extend(last_chunk)
            if tokenized_text_chunks == []:
                num_false_discards = 1
        elif num_tokens_last_chunk < 2:
            _ = tokenized_text_chunks.pop(-1)
            discarded_files += 1
        results, stats = self.process_chunks(tokenized_text_chunks)
        stats["discarded_files"] = discarded_files - num_false_discards
        return results, stats

    def encode(self, data: str) -> Tuple[List[np.ndarray], Dict]:
        """
        Tokenize and encode the data for auto-regressive language modeling.

        Args:
            data (str): Text data to encode.

        Returns:
            Tuple[List[np.ndarray], Dict]: Tuple of encoded features for auto-regressive language modeling and dataset stats.
        """

        raw_chars_count = len(data)
        raw_bytes_count = len(data.encode("utf-8"))
        files_processed = 1
        discarded_files = 0
        normalized_chars_count = raw_chars_count
        normalized_bytes_count = raw_bytes_count

        if self.use_ftfy:
            data = ftfy.fix_text(data, normalization=self.ftfy_normalizer)
            normalized_chars_count = len(data)
            normalized_bytes_count = len(data.encode("utf-8"))
        if self.wikitext_detokenize:
            data = wikitext_detokenizer(data)
            normalized_chars_count = len(data)
            normalized_bytes_count = len(data.encode("utf-8"))

        sample, stats = self.tokenize_text_auto_lm(data)
        discarded_files = stats["discarded_files"]
        if sample == []:
            discarded_files += 1
        data_stats = {
            "discarded": discarded_files,
            "processed": files_processed,
            "successful": files_processed - discarded_files,
            "raw_chars_count": raw_chars_count,
            "raw_bytes_count": raw_bytes_count,
            "num_pad_tokens": stats["num_pad_tokens"],
            "non_pad_tokens": stats["non_pad_tokens"],
            "num_masked_tokens": stats["num_masked_tokens"],
            "loss_valid_tokens": stats["loss_valid_tokens"],
            "num_tokens": stats["num_tokens"],
            "normalized_chars_count": normalized_chars_count,
            "normalized_bytes_count": normalized_bytes_count,
        }
        return sample, data_stats

    def encode_leftover_prefix(
        self, prefix: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Processes the leftover prefix which is a list of ndarray tokens into chunks based
        on max sequence length.

        The last chunk is handled specifically if it's shorter than the max sequence
        length. If the last chunk has less than two tokens, it's discarded.

        Args:
            prefix (List[np.ndarray]): The prefix list of token arrays to process.

        Returns:
            Tuple[List[np.ndarray], Dict]: A tuple containing the processed token chunks as
            a list of ndarrays and the dataset stats.
        """
        tokenized_text_chunks = [
            prefix[i : i + self.max_seq_length + 1]
            for i in range(0, len(prefix), self.max_seq_length)
        ]

        # Handle last chunk if shorter than max_seq_length
        num_tokens_last_chunk = len(tokenized_text_chunks[-1])
        if num_tokens_last_chunk < self.max_seq_length + 1:
            _ = tokenized_text_chunks.pop(-1)
        elif num_tokens_last_chunk < 2:
            _ = tokenized_text_chunks.pop(-1)

        # Assuming process_chunks is a method that processes each tokenized chunk
        results, stats = self.process_chunks(tokenized_text_chunks)

        return results, stats

    def get_token_id(self, token: str) -> int:
        """
        Get the token ID for the given token.

        Args:
            token (str): Token for which the ID is needed.

        Returns:
            int: Token ID.
        """
        return self.tokenizer.get_token_id(token)
