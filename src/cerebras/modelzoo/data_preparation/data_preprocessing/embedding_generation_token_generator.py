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
EmbeddingGenerationTokenGenerator Module

This module provides the EmbeddingGenerationTokenGenerator class which is designed to process
text data and create features suitable for embedding generation.

Usage:
    tokenizer = EmbeddingGenerationTokenGenerator(dataset_params, tokenizer, eos_id, pad_id)
    
    sample_sda = [
        {type': 'embedding', 'content': 'Sample text for processing.'},
        {'type': 'id', 'content': 12}
    ]
    tokenized_features = tokenizer.encode(sample_sda)
"""

from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from cerebras.modelzoo.data_preparation.data_preprocessing.utils import (
    EmbeddingStatCollector,
    EmbedGenList,
)


class EmbeddingGenerationTokenGenerator:
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
        self.max_seq_length = processing_params["max_seq_length"]
        self.use_ftfy = dataset_params.pop("use_ftfy", False)
        self.ftfy_normalizer = dataset_params.pop("ftfy_normalizer", "NFC")
        self.wikitext_detokenize = dataset_params.pop(
            "wikitext_detokenize", False
        )
        self.eos_id = eos_id
        self.eos_token = (
            self.tokenizer.pad_token_id
            if self.eos_id is None
            else self.tokenizer.convert_ids_to_tokens(self.eos_id)
        )
        self.pad_id = pad_id
        self.embedding_stat_collector = EmbeddingStatCollector(
            use_ftfy=self.use_ftfy,
            ftfy_normalizer=self.ftfy_normalizer,
            wikitext_detokenize=self.wikitext_detokenize,
            eos_id=self.eos_id,
            pad_id=self.pad_id,
        )

    def tokenize_context(self, content: str) -> List:
        '''
        Args:
            content: string which needs to be tokenized.

        Returns:
            tokenised output
            raw data statistics
            tokenized data statistics
        '''
        content, raw_data_stats = self.embedding_stat_collector.get_raw_stats(
            content
        )

        tokens = self.tokenizer(
            content,
            padding='max_length',
            max_length=self.max_seq_length,
            truncation=True,
        )

        tokenized_data_stats = (
            self.embedding_stat_collector.get_tokenized_stats(tokens)
        )

        return tokens, raw_data_stats, tokenized_data_stats

    def tokenize_data(
        self, semantic_data_array: List[Dict[str, Any]]
    ) -> Tuple[List[np.ndarray], Dict[str, int]]:
        """
        Tokenize the text and create features for embedding generation.

        Args:
            semantic_data_dict (Union[Dict[str, Any], List[Dict[str, Any]]]): Data to tokenize.

            The format for SDA is:
            [
               {type': 'embedding', 'content': 'text to embed'},
               {'type': 'id', 'content': id}
            ]

        Returns:
            Tuple[List[np.ndarray], Dict[str, int]]: Tuple of encoded features for embedding generation and dataset stats.

            The shape of the returned data ndarray is (1, 3, max_seq_length)
            The shape of the returned id ndarray is (1, 1)
        """
        # make sure the sda data is in the correct format
        EmbedGenList(sda=semantic_data_array)

        data_stats = defaultdict(int)
        for entry in semantic_data_array:
            if entry.get("type") == "embedding":
                tokens, raw_data_stats, tokenized_data_stats = (
                    self.tokenize_context(entry.get("content"))
                )
                data_stats = self.embedding_stat_collector.combine_stats(
                    data_stats, raw_data_stats, tokenized_data_stats
                )
            else:  # "id"
                embed_id = int(entry.get("content"))
        feature_order = ['input_ids', 'attention_mask', 'token_type_ids']
        data = np.array([tokens[key] for key in feature_order])
        embed_id = np.array([embed_id])

        data = np.expand_dims(data, axis=0)
        embed_id = np.expand_dims(embed_id, axis=0)
        return (data, embed_id), data_stats

    def encode(
        self, semantic_data_array: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """
        Tokenize and encode the data for embedded generation.

        Args:
            semantic_data_dict (Union[Dict[str, Any], List[Dict[str, Any]]]): Data to encode.

        Returns:
            Tuple[Dict[str, Any], Dict[str, int]]: Tuple of encoded features for embedded generation and dataset stats.
        """
        tokenized_data, data_stats = self.tokenize_data(semantic_data_array)

        if not tokenized_data:
            return {}, data_stats

        embedding_data, embed_id = tokenized_data
        # "data" field required by our map dataset
        data = {"data": embedding_data, "id": embed_id}

        return data, data_stats
