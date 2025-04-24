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
EmbeddingTrainingTokenGenerator Module

This module provides the EmbeddingTrainingTokenGenerator class which is designed to process
contexts data (in sda format) and create features suitable for embedding training.

Usage:
    tokenizer = EmbeddingTrainingTokenGenerator(dataset_params, tokenizer, eos_id, pad_id)
    sample_sda= [
        {'type':'question', 'content':'Sample Question'},
        {'type':'context_positive', 'content':'Sample positive context.'},
        {'type':'context_negative', 'content':'Sample negative context #1.'},
        {'type':'context_negative', 'content':'Sample negative context #2.'},
        ...
    ]
    tokenized_features = tokenizer.encode(sample_sda)
"""

from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from cerebras.modelzoo.data_preparation.data_preprocessing.utils import (
    EmbeddingStatCollector,
    EmbedTrainList,
)


class EmbeddingTrainingTokenGenerator:
    def __init__(
        self, params: Dict[str, Any], tokenizer: Any, eos_id: int, pad_id: int
    ):
        """
        Initialize the EmbeddingTrainingTokenGenerator class.

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

    def tokenize_content(self, content: str) -> List:
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
        Tokenize the text and create features for embedding training.

        Args:
            semantic_data_dict (Union[Dict[str, Any], List[Dict[str, Any]]]): Data to tokenize.

            The format for SDA is:
            example:
            [
                {'type':'question', 'content':.....},
                {'type':'context_positive', 'content':.....},
                {'type':'context_negative', 'content':.....},
                {'type':'context_negative', 'content':.....},
                ...
            ]

        Returns:
            Tuple[List[np.ndarray], Dict[str, int]]: Tuple of encoded features for contrastive-loss embedding training and dataset stats.

            The shape of the returned question_data ndarray is: (1, 3, max_seq_length)
            The shape of the returned context_data ndarray is: (1, 3, num_ctxs, max_seq_length)
            here num_ctx referes to 1(for positive ctx) and num_ctx-1(for all negative ctx)
            We currently assume that there is one positive context, so there are num_ctx-1 negative contexts
        """
        # make sure the sda data is in the correct format
        ## using a pydantic class verifier to verify the config file
        EmbedTrainList(sda=semantic_data_array)

        data_stats = defaultdict(int)

        question_features = []
        positive_context_features = []
        negative_context_features = []
        feature_map = {
            'question': question_features,
            'context_positive': positive_context_features,
            'context_negative': negative_context_features,
        }
        for entry in semantic_data_array:
            tokens, raw_data_stats, tokenized_data_stats = (
                self.tokenize_content(entry.get('content'))
            )

            if entry['type'] == 'context_negative':
                tokens["token_type_ids"] = [1 for _ in tokens['token_type_ids']]

            feature_map[entry['type']].append(tokens)
            data_stats = self.embedding_stat_collector.combine_stats(
                data_stats, raw_data_stats, tokenized_data_stats
            )

        data_stats['discarded_files'] = 0
        data_stats['processed_files'] = 1
        data_stats['successful_files'] = 1

        feature_order = ['input_ids', 'attention_mask', 'token_type_ids']
        # Currently we only support a single question
        # hence we are taking the first element of question_features
        question_data = np.array(
            [question_features[0][key] for key in feature_order]
        )

        context_features = positive_context_features + negative_context_features
        num_ctx = len(context_features)
        ctx_input_ids = np.array(
            [context_features[i]['input_ids'] for i in range(num_ctx)]
        )
        ctx_attention_mask = np.array(
            [context_features[i]['attention_mask'] for i in range(num_ctx)]
        )
        ctx_token_type_ids = np.array(
            [context_features[i]['token_type_ids'] for i in range(num_ctx)]
        )
        context_data = np.stack(
            (ctx_input_ids, ctx_attention_mask, ctx_token_type_ids), axis=0
        )

        question_data = np.expand_dims(question_data, axis=0)
        context_data = np.expand_dims(context_data, axis=0)
        return (question_data, context_data), data_stats

    def encode(
        self, semantic_data_array: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """
        Tokenize and encode the data for embedding training.

        Args:
            semantic_data_dict (Union[Dict[str, Any], List[Dict[str, Any]]]): Data to encode.

        Returns:
            Tuple[Dict[str, Any], Dict[str, int]]: Tuple of encoded features for embedding training and dataset stats.
        """
        tokenized_data, data_stats = self.tokenize_data(semantic_data_array)

        if not tokenized_data:
            return {}, data_stats

        question_data, context_data = tokenized_data
        # "data" field required by our map dataset
        data = {"data": question_data, "context_data": context_data}

        return data, data_stats
