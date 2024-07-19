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
PretrainingTokenGenerator Module

This module provides the PretrainingTokenGenerator class which is designed to process
text data and create features suitable for language modeling tasks.

Usage:
    tokenizer = PretrainingTokenGenerator(dataset_params, max_sequence_length, tokenizer)
    tokenized_features = tokenizer.encode("Sample text for processing.")
"""

import logging
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import ftfy
import numpy as np

from cerebras.modelzoo.data_preparation.data_preprocessing.utils import (
    split_text_and_tokenize,
    validate_tokens,
    wikitext_detokenizer,
)

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def create_features_auto_lm(
    token_ids: List[int],
    max_sequence_length: int,
    short_seq_prob: float = 0,
    inverted_mask: bool = False,
    pad_id: int = 0,
    min_len: int = 10,
    input_ids_dtype: str = "int32",
    input_mask_dtype: str = "int32",
    labels_dtype: str = "int32",
    rng: random.Random = None,
    eos_id: int = 0,
) -> np.ndarray:
    """Given a list of token_ids, generate input sequence and labels.

    Args:
        token_ids (List[int]): List containing token ids for creating features,
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
        rng (random.Random): Instance of random object, with states set.
            Defaults to `None`.

    Returns:
        np.ndarray: Array containing features, labels, and input mask.
    """
    if not validate_tokens(token_ids, min_len=min_len):
        return []

    if rng.random() < short_seq_prob:
        token_ids = token_ids[0 : rng.randint(2, max_sequence_length - 1)]

    input_ids = token_ids[:-1]
    labels = token_ids[1:]
    input_mask = [1] * len(labels)

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


class PretrainingTokenGenerator:
    def __init__(
        self, params: Dict[str, Any], tokenizer: Any, eos_id: int, pad_id: int
    ):
        """
        Initialize the PretrainingTokenGenerator class.

        Args:
            params (Dict[str, Any]): Parameters for the dataset and processing.
            tokenizer (Any): Tokenizer to use for tokenization.
            eos_id (int): End-of-sequence token ID.
            pad_id (int): Padding token ID.
        """
        dataset_params = params["dataset"]
        processing_params = params["processing"]
        self.tokenizer = tokenizer
        self.use_ftfy = dataset_params.pop("use_ftfy", False)
        self.ftfy_normalizer = dataset_params.pop("ftfy_normalizer", "NFC")
        self.training_objective = dataset_params.pop("training_objective", None)
        self.mlm = (
            (self.training_objective == 'mlm')
            if self.training_objective is not None
            else False
        )
        self.wikitext_detokenize = dataset_params.pop(
            "wikitext_detokenize", False
        )
        self.pack_sequences = dataset_params.pop("pack_sequences", True)
        self.min_sequence_len = dataset_params.pop("min_sequence_len", 10)
        self.input_ids_dtype = dataset_params.pop("input_ids_dtype", "int32")
        self.input_mask_dtype = dataset_params.pop("input_mask_dtype", "int32")
        self.inverted_mask = dataset_params.pop("inverted_mask", False)
        self.seed = processing_params.pop("seed", 0)
        self.max_seq_length = processing_params.pop("max_seq_length", 2048)
        self.short_seq_prob = processing_params.pop("short_seq_prob", 0.0)
        self.semantic_drop_mask = processing_params.pop(
            "semantic_drop_mask", {}
        )

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
        self.sample_features = ["input_ids", "attention_mask", "labels"]

        ##MLM fields
        if self.mlm:
            import math

            self.mlm_fraction = dataset_params.pop("mlm_fraction", 0.15)
            self.max_predictions = math.ceil(
                self.mlm_fraction * self.max_seq_length
            )
            self.mlm_with_gather = dataset_params.pop("mlm_with_gather", False)
            self.ignore_index = dataset_params.pop(
                "ignore_index", -100
            )  # default value for torch.nn.CrossEntropyLoss
            self.excluded_tokens = dataset_params.pop(
                "excluded_tokens",
                ['<cls>', '<pad>', '<eos>', '<unk>', '<null_1>', '<mask>'],
            )
            self.allowable_token_ids = self.get_allowable_token_ids()
            self.special_tokens_ids = {
                self.tokenizer.cls_token_id,
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.unk_token_id,
            }

    def get_data_stats(
        self,
        sample: np.ndarray,
        lvt: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Get data statistics from the sample.

        Args:
            sample (np.ndarray): Tokenized sample.

        Returns:
            Dict[str, int]: Data statistics.
        """
        stats = {
            "num_pad_tokens": 0,
            "non_pad_tokens": 0,
            "num_masked_tokens": 0,
            "loss_valid_tokens": 0,
            "num_tokens": 0,
        }
        if sample == []:
            return stats
        stats["num_pad_tokens"] = int((sample[0, :] == self.pad_id).sum())
        stats["non_pad_tokens"] = int(
            np.logical_and(
                sample[0, :] != self.eos_id, sample[0, :] != self.pad_id
            ).sum()
        )
        stats["num_tokens"] = int(sample[0, :].shape[0])

        if self.mlm:
            stats["loss_valid_tokens"] = lvt
        else:
            stats["loss_valid_tokens"] = int(sample[1, :].sum())
        stats["num_masked_tokens"] = (
            self.max_seq_length - stats["loss_valid_tokens"]
        )

        return stats

    def get_allowable_token_ids(self) -> List[int]:
        """Generate a list of token IDs that can be masked."""
        excluded_token_ids = {
            self.tokenizer.convert_tokens_to_ids(tok)
            for tok in self.excluded_tokens
            if tok in self.tokenizer.get_vocab()
        }
        allowable_token_ids = [
            tok_id
            for tok, tok_id in self.tokenizer.get_vocab().items()
            if tok_id not in excluded_token_ids
        ]
        return list(allowable_token_ids)

    def mask_single_sequence(
        self, input_ids: List[int]
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Masks tokens in a single sequence according to the MLM strategy.
        When self.mlm_with_gather is False, the returning len(labels) == len(input_ids)
        When self.mlm_with_gather is True, the returning len(labels) == self.max_predictions

        Args:
            input_ids (List[int]): Original sequence of token IDs.

        Returns:
            Tuple[List[int], List[int], List[int], List[int]]:
                - input_ids: Modified sequence with masked tokens.
                - masked_lm_positions: Positions of the masked tokens, empty if not self.mlm_with_gather.
                - masked_lm_mask: Binary indicators (1s) for positions that were masked, empty if not self.mlm_with_gather.
                - labels: Original token IDs of the masked tokens for label purposes.
        """
        sequence = np.array(input_ids.copy())
        masked_lm_positions = []
        masked_lm_mask = []
        labels = (
            [] if self.mlm_with_gather else [self.ignore_index] * len(input_ids)
        )
        indices_can_be_masked = [
            i
            for i, token_id in enumerate(input_ids)
            if token_id not in self.special_tokens_ids
        ]

        # Calculate the number of tokens to mask
        num_tokens_to_mask = min(
            int(self.mlm_fraction * len(indices_can_be_masked)),
            self.max_predictions,
        )
        if num_tokens_to_mask > 0:
            # Randomly select tokens to mask
            indices_to_mask = sorted(
                self.rng.sample(indices_can_be_masked, k=num_tokens_to_mask)
            )

            for pos in indices_to_mask:
                original_token_id = sequence[pos].copy()
                prob = self.rng.random()
                if prob < 0.8:  # 80% of the time, replace with [MASK]
                    sequence[pos] = self.tokenizer.mask_token_id
                elif prob < 0.9:  # 10% of the time, replace with a random token
                    # Ensure selected token is not a special token
                    masked_token_id = np.random.choice(self.allowable_token_ids)
                    sequence[pos] = masked_token_id
                elif prob <= 1.0:
                    pass  # 10% of the time, keep the original token

                # Store the original token ID as the label
                if self.mlm_with_gather:
                    masked_lm_positions.append(pos)
                    masked_lm_mask.append(1)
                    labels.append(original_token_id)
                else:
                    labels[pos] = original_token_id

        if self.mlm_with_gather:
            # Pad the lists to reach max_predictions length
            num_paddings = self.max_predictions - len(masked_lm_positions)
            masked_lm_positions = masked_lm_positions + [0] * num_paddings
            masked_lm_mask = masked_lm_mask + [0] * num_paddings
            labels = labels + [self.ignore_index] * num_paddings

        return list(sequence), masked_lm_positions, masked_lm_mask, labels

    def process_chunks(
        self, tokenized_text_chunks: List[List[int]]
    ) -> Tuple[List[np.ndarray], Dict[str, int]]:
        """
        Processes chunks of tokenized text and returns processed features along with the total padding added.

        Args:
            tokenized_text_chunks (List[List[int]]): A list of tokenized text chunks, where each chunk is represented as a list of integers.

        Returns:
            Tuple[List[np.ndarray], Dict[str, int]]: A tuple containing a list of processed results and dataset stats.
        """
        results = {"data": []}  # List to store the processed results
        stats = defaultdict(int)
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
                eos_id=self.eos_id,
            )

            # If the processed chunk is not empty, add the results to the list and update the total padding
            if len(processed) != 0:
                processed_stats = self.get_data_stats(processed)
                for key in processed_stats:
                    stats[key] += processed_stats[key]
                results["data"].append(processed)

        # Return the list of processed results and data stats
        return results, stats

    def process_chunks_mlm(
        self, tokenized_text_chunks: List[List[int]]
    ) -> Tuple[List[Any], Dict]:
        """
        Processes chunks of tokenized text and returns processed features along with the total padding added.

        Args:
        tokenized_text_chunks (List[List[int]]): A list of tokenized text chunks, where each chunk is represented as a list of integers.

        Returns:
        Tuple[List[Any], Dict]: A tuple containing a list of processed results and dataset stats.
        """

        results = {
            'data': [],
            'labels': [],
        }  # List to store the processed result

        stats = defaultdict(int)

        masked_lm_positions_list = []
        masked_lm_mask_list = []
        input_id_list = []
        labels_list = []
        attention_mask_list = []
        # Iterate over each chunk in the tokenized text chunks
        for chunk in tokenized_text_chunks:
            input_ids, masked_lm_positions, masked_lm_mask, labels = (
                self.mask_single_sequence(chunk)
            )
            num_pad = self.max_seq_length - len(input_ids)
            attention_mask = [1] * len(input_ids) + [0] * num_pad
            input_ids = input_ids + [self.pad_id] * num_pad

            input_id_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)
            masked_lm_positions_list.append(masked_lm_positions)
            masked_lm_weights_list.append(masked_lm_weights)

            lvt = len(labels) - labels.count(self.ignore_index)
            processed_stats = self.get_data_stats(
                np.expand_dims(np.array(input_ids), 0), lvt
            )
            for key in processed_stats:
                stats[key] += processed_stats[key]

        if len(tokenized_text_chunks) > 0:
            results['data'] = np.stack(
                [np.array(input_id_list), np.array(attention_mask_list)], axis=1
            )
            if self.mlm_with_gather:
                results['labels'] = np.stack(
                    [
                        np.array(labels_list),
                        np.array(masked_lm_positions_list),
                        np.array(masked_lm_weights_list),
                    ],
                    axis=1,
                )
            else:
                results['labels'] = np.stack(
                    [np.array(labels_list)],
                    axis=1,
                )

        # Return the list of processed results and data stats
        return results, stats

    def clean_text(self, data: str) -> str:
        """
        Clean the provided text.

        Args:
            data (str): Text to clean.

        Returns:
            str: Cleaned text.
        """
        if self.use_ftfy:
            data = ftfy.fix_text(data, normalization=self.ftfy_normalizer)
        if self.wikitext_detokenize:
            data = wikitext_detokenizer(data)

        return data

    def tokenize_data(
        self, semantic_data_array: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Tuple[List[np.ndarray], Dict[str, int]]:
        """
        Tokenize the text and create features for auto-regressive language modeling.

        Args:
            semantic_data_dict (Union[Dict[str, Any], List[Dict[str, Any]]]): Data to tokenize.

        Returns:
            Tuple[List[np.ndarray], Dict[str, int]]: Tuple of encoded features for auto-regressive language modeling and dataset stats.
        """
        text, raw_data_stats = self.parse_semantic_data_array(
            semantic_data_array
        )
        if text == "":
            return {"data": []}, raw_data_stats
        discarded_files = 0

        if self.mlm:
            tokenized_data = self.tokenizer(
                text,
                max_length=self.max_seq_length,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
            )
            input_ids, attention_mask = (
                tokenized_data['input_ids'],
                tokenized_data['attention_mask'],
            )
            tokenized_data_stats = dict()
            results = dict()

            tokenized_data_stats["processed"] = 1
            tokenized_data_stats["successful"] = 0
            if input_ids == []:
                tokenized_data_stats["discarded"] = 1
                return {"data": [], "labels": []}, tokenized_data_stats

            tokenized_data_stats["successful"] = 1

            input_ids, masked_lm_positions, masked_lm_mask, labels = (
                self.mask_single_sequence(input_ids)
            )

            results['data'] = np.stack(
                [np.array(input_ids), np.array(attention_mask)], axis=0
            ).reshape(1, 2, self.max_seq_length)

            if self.mlm_with_gather:
                results['labels'] = np.stack(
                    [
                        np.array(labels),
                        np.array(masked_lm_positions),
                        np.array(masked_lm_mask),
                    ],
                    axis=0,
                ).reshape(1, 3, self.max_predictions)
            else:
                results['labels'] = np.stack(
                    [
                        np.array(labels),
                    ],
                    axis=0,
                ).reshape(1, 1, self.max_seq_length)

            tokenized_data_stats["non_pad_tokens"] = sum(
                1 for id in input_ids if id != self.pad_id
            )
            tokenized_data_stats["num_pad_tokens"] = (
                self.max_seq_length - tokenized_data_stats["non_pad_tokens"]
            )
            tokenized_data_stats["num_tokens"] = self.max_seq_length
            tokenized_data_stats["num_masked_tokens"] = input_ids.count(
                self.tokenizer.mask_token_id
            )
            tokenized_data_stats["loss_valid_tokens"] = len(
                labels
            ) - labels.count(self.ignore_index)
            tokenized_data_stats.update(raw_data_stats)

            return results, tokenized_data_stats

        # tokenize text
        if self.split_text_to_tokenize:
            tokenized_text = split_text_and_tokenize(
                text,
                self.tokenizer,
                max_tok_len=self.chunk_len_to_split,
                remove_bos_in_chunks=self.remove_bos_in_chunks,
            )
        else:
            tokenized_text = self.tokenizer.encode(text)

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
        elif num_tokens_last_chunk < 2:
            _ = tokenized_text_chunks.pop(-1)
            discarded_files += 1
        results, tokenized_data_stats = (
            self.process_chunks_mlm(tokenized_text_chunks)
            if self.mlm
            else self.process_chunks(tokenized_text_chunks)
        )
        tokenized_data_stats["discarded"] = discarded_files
        tokenized_data_stats["processed"] = 1
        tokenized_data_stats["successful"] = (
            tokenized_data_stats["processed"]
            - tokenized_data_stats["discarded"]
        )
        tokenized_data_stats.update(raw_data_stats)
        return results, tokenized_data_stats

    def parse_semantic_data_array(
        self, semantic_data_array: List[Dict[str, Any]]
    ) -> Tuple[str, Dict[str, int]]:
        """
        Parse semantic data dictionary.

        Args:
            entry (Union[Dict[str, Any], List[Dict[str, Any]]]): Data entry.

        Returns:
            Tuple[str, Dict[str, int]]: Parsed text and raw data statistics.
        """
        if not semantic_data_array:
            return "", {}
        text = ""
        raw_data_stats = {
            "raw_chars_count": 0,
            "raw_bytes_count": 0,
            "normalized_chars_count": 0,
            "normalized_bytes_count": 0,
            "total_raw_docs": 1,
            "raw_docs_skipped": 0,
        }
        for entry in semantic_data_array:
            content = entry["content"]
            drop_mask = entry.get("semantic_drop_mask")
            for region in content:
                include_tags = region.get("include_tags", False)
                region_key = list(region.keys())[0]
                region_val = region.get(region_key)
                if not region_val or region_val == "":
                    logger.warning(f"Empty doc. Skipping this example ")
                    raw_data_stats["raw_docs_skipped"] = 1
                    return "", raw_data_stats
                drop_region = (drop_mask and drop_mask.get(region_key)) or (
                    self.semantic_drop_mask
                    and self.semantic_drop_mask.get(region_key)
                )
                if drop_region:
                    region_val = ""
                text += region_val

        raw_data_stats["raw_chars_count"] = len(text)
        raw_data_stats["raw_bytes_count"] = len(text.encode("utf-8"))
        text = self.clean_text(text)
        raw_data_stats["normalized_chars_count"] = len(text)
        raw_data_stats["normalized_bytes_count"] = len(text.encode("utf-8"))

        return text, raw_data_stats

    def encode(
        self, semantic_data_array: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """
        Tokenize and encode the data for auto-regressive language modeling.

        Args:
            semantic_data_array (Union[Dict[str, Any], List[Dict[str, Any]]]): Data to encode.

        Returns:
            Tuple[Dict[str, Any], Dict[str, int]]: Tuple of encoded features for auto-regressive language modeling and dataset stats.
        """
        tokenized_data, data_stats = self.tokenize_data(semantic_data_array)
        if tokenized_data["data"] == []:
            return {}, data_stats
        else:
            data = tokenized_data

        return data, data_stats

    def encode_leftover_prefix(
        self, prefix: List[np.ndarray]
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """
        Processes the leftover prefix which is a list of ndarray tokens into chunks based
        on max sequence length.

        The last chunk is handled specifically if it's shorter than the max sequence
        length. If the last chunk has less than two tokens, it's discarded.

        Args:
            prefix (List[np.ndarray]): The prefix list of token arrays to process.

        Returns:
            Tuple[Dict[str, Any], Dict[str, int]]: A tuple containing the processed token chunks as
            a list of ndarrays and the dataset stats.
        """
        tokenized_text_chunks = (
            [
                prefix[i : i + self.max_seq_length]
                for i in range(0, len(prefix), self.max_seq_length)
            ]
            if self.mlm
            else [
                prefix[i : i + self.max_seq_length + 1]
                for i in range(0, len(prefix), self.max_seq_length)
            ]
        )

        # Handle last chunk if shorter than max_seq_length
        num_tokens_last_chunk = len(tokenized_text_chunks[-1])
        if num_tokens_last_chunk < self.max_seq_length + 1:
            _ = tokenized_text_chunks.pop(-1)
        elif num_tokens_last_chunk < 2:
            _ = tokenized_text_chunks.pop(-1)

        results, stats = self.process_chunks(tokenized_text_chunks)
        if results["data"] == []:
            return {}, stats
        data = results

        return data, stats
