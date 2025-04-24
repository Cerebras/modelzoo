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
This module provides the VSLFinetuningTokenGenerator class, which extends
the FinetuningTokenGenerator for processing tokenized text data specifically
for variable-length sequence summarization (VSLS). The class includes methods
for processing chunks of tokenized text, encoding documents for text summarization,
and optimizing the representation of tokenized data by merging shorter sequences
within a specified maximum sequence length.
"""

import copy
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from cerebras.modelzoo.data_preparation.data_preprocessing.finetuning_token_generator import (
    FinetuningTokenGenerator,
)
from cerebras.modelzoo.data_preparation.data_preprocessing.utils import (
    get_data_stats,
    setup_warning_logging,
)


class VSLFinetuningTokenGenerator(FinetuningTokenGenerator):
    """
    Token generator for variable-length sequence summarization (VSLS).
    Extends FinetuningTokenGenerator with additional functionality for VSLS.
    """

    def __init__(self, params, tokenizer, eos_id, pad_id):
        """
        Initialize VSLFinetuningTokenGenerator with dataset parameters,
        tokenizer, and token IDs.
        """
        super(VSLFinetuningTokenGenerator, self).__init__(
            params, tokenizer, eos_id, pad_id
        )
        setup_params = params["setup"]
        warning_log_dir = os.path.join(
            setup_params.get("output_dir", "./output")
        )
        self.logger = setup_warning_logging(warning_log_dir, __name__)
        self.position_ids_dtype = params["dataset"].pop(
            "position_ids_dtype", "int32"
        )
        self.features = [
            "input_ids",
            "attention_mask",
            "labels",
            "attention_span",
            "position_ids",
        ]

    def create_features_finetuning_vsl(
        self,
        bin,
    ):
        """Given a list of VSL sequences, generate input features and labels.

        Args:
            bin (list(sequence)): list of VSL sequences.

        Returns:
            Tuple containing features and labels
        """
        input_ids, input_mask, labels, attention_span, position_ids = (
            [],
            [],
            [],
            [],
            [],
        )
        num_bins = len(bin)
        for i, data in enumerate(bin):
            token_ids, token_mask = data.get("token_ids"), data.get(
                "input_mask"
            )
            input_ids.extend(token_ids)
            labels.extend(token_ids)
            input_mask.extend(token_mask)
            if i != num_bins - 1:
                sample_len = len(token_ids)
                attention_span.extend(list(range(sample_len - 1, -1, -1)))
                position_ids.extend(list(range(sample_len)))
            else:
                ## We will be chopping of the last token id in the last bin
                sample_len = len(token_ids) - 1
                attention_span.extend(list(range(sample_len - 1, -1, -1)))
                position_ids.extend(list(range(sample_len)))

        input_ids = input_ids[:-1]
        labels = labels[1:]
        input_mask = input_mask[1:]
        # padding
        num_pad = self.max_seq_length - len(input_ids)
        padding = [self.pad_id] * num_pad
        input_ids.extend(padding)
        labels.extend(padding)

        padding = [0] * num_pad
        input_mask.extend(padding)
        attention_span.extend(padding)
        position_ids.extend(padding)

        # assertions to ensure correct output shapes
        assert (
            len(input_ids) == self.max_seq_length
            and len(labels) == self.max_seq_length
            and len(input_mask) == self.max_seq_length
            and len(attention_span) == self.max_seq_length
            and len(position_ids) == self.max_seq_length
        ), "Wrong sequence length"

        input_ids = getattr(np, self.input_ids_dtype)(input_ids)
        input_mask = getattr(np, self.input_mask_dtype)(input_mask)

        if self.inverted_mask:
            input_mask = np.equal(input_mask, 0).astype(self.input_mask.dtype)

        labels = getattr(np, self.input_mask_dtype)(labels)
        attention_span = getattr(np, self.position_ids_dtype)(attention_span)
        position_ids = getattr(np, self.position_ids_dtype)(position_ids)
        result = {
            "data": np.stack(
                [input_ids, input_mask, labels, attention_span, position_ids]
            )
        }
        return result

    def process_chunks(
        self, tokenized_data: List[List[tuple]]
    ) -> Tuple[List[Any], int]:
        """
        Process chunks of tokenized text and return processed features along with
        the total padding added.

        Args:
            tokenized_data (List[List[tuple]]): List of tokenized text chunks,
                where each chunk is represented as a list of (prompt, completion) tuples.

        Returns:
            Tuple[List[Any], int]: Tuple containing a list of processed results
                and the total number of padding tokens added.
        """
        results = defaultdict(list)  # List to store processed results
        data_stats = defaultdict(int)
        stats_checkpoint_list = []

        for vsl_list in tokenized_data:
            processed = self.create_features_finetuning_vsl(vsl_list)
            if processed != []:
                stats = get_data_stats(
                    processed["data"],
                    self.pad_id,
                    self.eos_id,
                    self.max_seq_length,
                )

                results["data"].append(
                    np.expand_dims(processed["data"], axis=0)
                )
                for key in stats:
                    data_stats[key] += stats[key]
                data_stats["num_sequences_before_packing"] += len(vsl_list)
                stats_checkpoint_list.append(copy.deepcopy(data_stats))

        return results, stats_checkpoint_list

    def create_features_finetuning_vsl(self, bin):
        """Given a list of VSL sequences, generate input features and labels.

        Args:
            bin (list(sequence)): list of VSL sequences.
            max_sequence_length (int): Maximum sequence length for data writes.
            pad_id (int): Id for pad token. Defaults to `0`.
            sep_id (int): Id for separator token. Defaults to `None`.
            inverted_mask (bool): Invert mask if specified for runtime execution.
                Defaults to `False`.
            input_ids_dtype (str): Dtype as string for input ids.
                Defaults to `int32`.
            input_mask_dtype (str): Dtype as string for input mask.
                Defaults to `int32`.
            labels_dtype (str): Dtype as string for labels. Defaults to `int32`.
            attention_span_dtype (str): Dtype as string for keys attention span in VSL.
                Defaults to `int32`.
            position_ids_dtype (str): Dtype as string for position ids in VSL.
                Defaults to `int32`.

        Returns:
            Tuple containing features and labels
        """
        input_ids, input_mask, labels, attention_span, position_ids = (
            [],
            [],
            [],
            [],
            [],
        )
        num_bins = len(bin)
        for i, data in enumerate(bin):
            token_ids, token_mask = data.get("token_ids"), data.get(
                "input_mask"
            )
            input_ids.extend(token_ids)
            labels.extend(token_ids)
            input_mask.extend(token_mask)
            if i != num_bins - 1:
                sample_len = len(token_ids)
                attention_span.extend(list(range(sample_len - 1, -1, -1)))
                position_ids.extend(list(range(sample_len)))
            else:
                ## We will be chopping of the last token id in the last bin
                sample_len = len(token_ids) - 1
                attention_span.extend(list(range(sample_len - 1, -1, -1)))
                position_ids.extend(list(range(sample_len)))

        input_ids = input_ids[:-1]
        labels = labels[1:]
        input_mask = input_mask[1:]
        # padding
        num_pad = self.max_seq_length - len(input_ids)
        padding = [self.pad_id] * num_pad
        input_ids.extend(padding)
        labels.extend(padding)

        padding = [0] * num_pad
        input_mask.extend(padding)
        attention_span.extend(padding)
        position_ids.extend(padding)

        # assertions to ensure correct output shapes
        assert (
            len(input_ids) == self.max_seq_length
            and len(labels) == self.max_seq_length
            and len(input_mask) == self.max_seq_length
            and len(attention_span) == self.max_seq_length
            and len(position_ids) == self.max_seq_length
        ), "Wrong sequence length"

        input_ids = getattr(np, self.input_ids_dtype)(input_ids)
        input_mask = getattr(np, self.input_mask_dtype)(input_mask)

        if self.inverted_mask:
            input_mask = np.equal(input_mask, 0).astype(input_mask.dtype)

        labels = getattr(np, self.input_ids_dtype)(labels)
        attention_span = getattr(np, self.position_ids_dtype)(attention_span)
        position_ids = getattr(np, self.position_ids_dtype)(position_ids)
        result = {
            "data": np.stack(
                [input_ids, input_mask, labels, attention_span, position_ids]
            )
        }
        return result

    def encode(
        self, semantic_data_array: List[Dict[str, Any]]
    ) -> Tuple[List, Dict]:
        """
        Tokenize and encode the document for text summarization.

        Args:
            data: Union[List[Dict], Tuple]: Contains data either as a tuple of prompt, completion or a multi turn dialogue

        Returns:
            Tuple[List[tuple],Dict: List of tokenized data  and a stats dictionary
        """

        data, raw_data_stats = self._encode(semantic_data_array)
        if data == {}:
            return {}, raw_data_stats

        total_len = len(data.get("data").get("token_ids", []))
        discarded_files = 0
        if total_len > self.max_seq_length:
            self.logger.warning(
                "prompt_ids + completion_ids > max_sequence_length, skipping this example..."
            )
            discarded_files += 1
            data = {}
        elif total_len < self.min_sequence_len:
            self.logger.warning(
                "prompt_ids + completion_ids < min_sequence_len, skipping this example..."
            )
            discarded_files += 1
            data = {}

        raw_data_stats["discarded_files"] = discarded_files
        raw_data_stats["successful_files"] = 1 - discarded_files

        return data, raw_data_stats

    def append_within_max_length(self, tokenized_data) -> List[List[tuple]]:
        """
        Optimize representation of tokenized data by merging shorter sequences
        within the specified maximum sequence length.
        Args:
            tokenized_data (List[List[tuple]]): List of tokenized text data
                where each inner list contains (prompt, completion) tuples.
        Returns:
            List[List[tuple]]: Optimized list after merging shorter sequences.
        """
        result = []
        lengths_lookup = [
            len(sublist.get("token_ids", []))
            for sublist in tokenized_data["data"]
        ]
        tokenized_data = [[data] for data in tokenized_data["data"]]

        # Start from the end of the main list and iterate backwards
        i = len(tokenized_data) - 1
        while i > 0:
            current_sublist_length = lengths_lookup[i]
            # Check each preceding sublist starting from i-1
            for j in range(i - 1, -1, -1):
                combined_length = current_sublist_length + lengths_lookup[j]
                if combined_length <= self.max_seq_length:
                    tokenized_data[j].extend(tokenized_data[i])
                    lengths_lookup[j] = combined_length
                    del tokenized_data[i]
                    del lengths_lookup[i]
                    break  # Break as we've appended the current sublist
            i -= 1
        return tokenized_data
