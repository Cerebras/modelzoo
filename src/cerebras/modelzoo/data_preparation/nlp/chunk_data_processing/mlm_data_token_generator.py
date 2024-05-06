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
MLMTokenGenerator Module

This module provides the MLMTokenGenerator class which is designed to prepare tokenized
data for training with the Masked Language Modeling (MLM) objective, commonly used in
pre-training transformers like BERT. The class uses a tokenizer object (compatible with
Hugging Face's transformers library) to tokenize text data and creates masked tokens
features which are essential for MLM training. It supports dynamic masking of tokens
where a configurable percentage of tokens are masked randomly, facilitating the training
of deep learning models on tasks that require understanding of context and word relationships.

The MLMTokenGenerator handles tokenization, applies the MLM masking strategy, and prepares
appropriate outputs for model training, including the indices of masked tokens and their
original values, which are used as labels in MLM.

Usage:
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    mlm_generator = MLMTokenGenerator(
        params={'dataset':{}, 'processing':{'max_seq_length':512, 'mlm_fraction':0.15}},
        tokenizer=tokenizer,
        eos_id=tokenizer.eos_token_id,
        pad_id=tokenizer.pad_token_id
    )
    data = ["This is a sample input sentence", "This is another example"]
    tokenized_data = tokenizer(data, padding=True, return_tensors='pt', max_length=512)
    input_ids = tokenized_data['input_ids'].tolist()
    masked_input_ids, masked_positions, masked_labels, original_labels = mlm_generator.mask_single_sequence(input_ids[0])

Attributes:
    tokenizer (PreTrainedTokenizer): A tokenizer object, instance of transformers.PreTrainedTokenizer.
    eos_id (int): Token ID used to signify the end of a sentence.
    pad_id (int): Token ID used to signify padding.
    max_seq_length (int): Maximum length of tokens in a sequence, sequences longer than this are truncated.
    mlm_fraction (float): Fraction of tokens in each sequence to be masked.
    max_predictions (int): Maximum number of tokens to mask in a sequence.
    seed (int): Random seed for reproducibility.
    rng (random.Random): Random number generator for masking logic.
    excluded_tokens (List[str]): Tokens that should not be masked.
    allowable_token_ids (List[int]): Token IDs that can be masked.
    special_tokens_ids (Set[int]): Token IDs of special tokens that should not be masked.
"""

import math
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np


class MLMTokenGenerator:
    def __init__(
        self,
        params: Dict[str, Any],
        tokenizer,  # Type depending on the tokenizer used
        eos_id: int,
        pad_id: int,
    ):
        dataset_params = params["dataset"]
        processing_params = params["processing"]
        self.tokenizer = tokenizer
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.max_seq_length = processing_params.pop("max_seq_length", 1024)

        self.mlm_fraction = dataset_params.pop("mlm_fraction", 0.15)
        self.max_predictions = math.ceil(
            self.mlm_fraction * self.max_seq_length
        )
        self.seed = processing_params.pop("seed", 42)
        self.rng = random.Random(self.seed + os.getpid())
        np.random.seed(self.seed + os.getpid())

        self.excluded_tokens = dataset_params.pop(
            "excluded_tokens",
            ['<cls>', '<pad>', '<eos>', '<unk>', '<null_1>', '<mask>'],
        )
        self.allowable_token_ids = self._get_allowable_token_ids()
        self.special_tokens_ids = {
            self.tokenizer.cls_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.unk_token_id,
        }

    def _get_allowable_token_ids(self) -> List[int]:
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

        Args:
            input_ids (List[int]): Original sequence of token IDs.

        Returns:
            Tuple[List[int], List[int], List[int], List[int]]:
                - Modified sequence with masked tokens.
                - Positions of the masked tokens.
                - Binary indicators (1s) for positions that were masked.
                - Original token IDs of the masked tokens for label purposes.
        """
        sequence = np.array(input_ids.copy())
        masked_positions = []
        masked_lm_labels = []
        labels = []
        indices_can_be_masked = [
            i
            for i, token_id in enumerate(input_ids)
            if token_id not in self.special_tokens_ids
        ]
        if not indices_can_be_masked:
            return sequence, [], [], []  # Adjust the return as needed

        # Calculate the number of tokens to mask
        num_tokens_to_mask = min(
            int(self.mlm_fraction * len(indices_can_be_masked)),
            self.max_predictions,
        )
        if num_tokens_to_mask == 0:
            return (
                sequence,
                [],
                [0] * self.max_predictions,
                labels,
            )  # Return if no tokens can be masked

        # Randomly select tokens to mask
        indices_to_mask = sorted(
            self.rng.sample(indices_can_be_masked, k=num_tokens_to_mask)
        )

        for pos in indices_to_mask:
            original_token_id = sequence[pos].copy()
            prob = self.rng.random()
            if prob < 0.8:
                sequence[pos] = self.tokenizer.mask_token_id
            elif prob < 0.9:
                # Ensure selected token is not a special token
                masked_token_id = np.random.choice(self.allowable_token_ids)
                sequence[pos] = masked_token_id
            # Store the original token ID as the label
            masked_positions.append(pos)
            masked_lm_labels.append(1)
            labels.append(original_token_id)

        # Pad the lists to reach max_predictions length
        while len(masked_positions) < self.max_predictions:
            masked_positions.append(0)
            masked_lm_labels.append(0)
            labels.append(0)

        return list(sequence), masked_positions, masked_lm_labels, labels

    def encode(self, data: str) -> Tuple[List[np.ndarray], Dict]:
        """
        Tokenize and encode the data for masked language modeling.

        Args:
            data (str): Text  data to encode.

        Returns:
            Tuple[List[np.ndarray], Dict]: Tuple of encoded features for masked language modeling and dataset stats.
        """

        raw_chars_count = len(data)
        raw_bytes_count = len(data.encode("utf-8"))

        # Initialize data_stats
        data_stats = {
            "discarded": 0,
            "processed": 1,
            "successful": 0,
            "raw_chars_count": raw_chars_count,
            "raw_bytes_count": raw_bytes_count,
            "num_pad_tokens": 0,
            "non_pad_tokens": 0,
            "num_masked_tokens": 0,
            "loss_valid_tokens": 0,
            "num_tokens": 0,
            "normalized_chars_count": raw_chars_count,
            "normalized_bytes_count": raw_bytes_count,
        }

        tokenized_data = self.tokenizer(
            data,
            max_length=self.max_seq_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
        )
        input_ids, attention_mask = (
            tokenized_data['input_ids'],
            tokenized_data['attention_mask'],
        )

        if input_ids == []:
            data_stats["discarded"] = 1
            return [], data_stats

        data_stats["successful"] = 1

        (
            input_ids,
            masked_lm_positions,
            masked_lm_labels,
            labels,
        ) = self.mask_single_sequence(input_ids)
        data_stats["non_pad_tokens"] = sum(
            1 for id in input_ids if id != self.pad_id
        )
        data_stats["num_pad_tokens"] = (
            self.max_seq_length - data_stats["non_pad_tokens"]
        )
        data_stats["num_tokens"] = self.max_seq_length
        data_stats["num_masked_tokens"] = data_stats["num_pad_tokens"] + len(
            masked_lm_positions
        )
        data_stats["loss_valid_tokens"] = sum(
            1 for pos in masked_lm_positions if pos != 0
        )

        # As HDF5 output requires the sizes of all arrays to be same, we will pad masked_lm_labels,
        # masked_lm_positions and labels to the same length as input_ids ie the max_seq_length.
        # As we only support CSV data loaders for MLM, we will put the indices to pick in the last token
        # of masked_lm_positions, masked_lm_labels, labels using which we will extract the right number
        # of masked values
        def pad_and_encode_combined(labels, positions, mlm_labels):
            # Combine the lists
            combined_list = labels + positions + mlm_labels

            # Calculate the padding length
            pad_length = self.max_seq_length - len(combined_list) - 1

            # Pad the list
            padded_list = (
                combined_list + [self.pad_id] * pad_length + [len(labels)]
            )

            return padded_list

        padded_array = pad_and_encode_combined(
            labels, masked_lm_positions, masked_lm_labels
        )
        stacked_values = np.stack(
            [
                input_ids,
                attention_mask,
                padded_array,
            ]
        )
        sample = np.expand_dims(stacked_values, axis=0)
        return sample, data_stats
