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
This module provides the VSLPretrainingTokenGenerator class, extending
PretrainingTokenGenerator for advanced processing of tokenized text data tailored
for variable-length sequence language modeling (VSLLM). Includes methods for
processing chunks of tokenized text, optimizing representation of tokenized
data by merging shorter sequences within a specified maximum sequence length,
and tokenizing text for auto-regressive language modeling.
"""

import copy
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from cerebras.modelzoo.data_preparation.data_preprocessing.pretraining_token_generator import (
    PretrainingTokenGenerator,
)
from cerebras.modelzoo.data_preparation.data_preprocessing.utils import (
    append_eos_to_multiple_semantic_regions,
    get_data_stats,
    setup_warning_logging,
)


class VSLPretrainingTokenGenerator(PretrainingTokenGenerator):
    """
    Processes tokenized text data, specifically for VSLLM. Extends
    PretrainingTokenGenerator by handling text tokenization, feature creation,
    and optimizing representation of tokenized data for language modeling tasks.

    Attributes:
        fold_long_doc (bool): Whether to fold long documents.
        position_ids_dtype (str): Data type for position IDs in tokenized output.

    Args:
        params (dict): Parameters for the dataset and model.
        tokenizer: Tokenizer instance for text tokenization.
        eos_id (int): End-of-sequence ID.
        pad_id (int): Padding ID.
    """

    def __init__(
        self, params: Dict[str, Any], tokenizer: Any, eos_id: int, pad_id: int
    ):
        """
        Initialize VSLPretrainingTokenGenerator with dataset parameters,
        tokenizer, and token IDs.
        """
        super().__init__(params, tokenizer, eos_id, pad_id)
        setup_params = params["setup"]
        warning_log_dir = os.path.join(
            setup_params.get("output_dir", "./output")
        )
        self.logger = setup_warning_logging(warning_log_dir, __name__)
        self.fold_long_doc = params["dataset"].pop("fold_long_doc", True)
        self.position_ids_dtype = params["dataset"].pop(
            "position_ids_dtype", "int32"
        )
        self.pack_sequences = False
        self.features += ["attention_span", "position_ids"]

    def create_features_vsl_mlm(
        self,
        bin,
    ):
        """Given a list of VSL sequences, generate input features and labels.

        Args:
            bin (list(sequence)): list of VSL sequences.

        Returns:
            Tuple containing features and labels
        """
        input_ids, attention_span, position_ids = [], [], []
        for sample in bin:
            input_ids.extend(sample)
            attention_span.extend(list(range(len(sample) - 1, -1, -1)))
            position_ids.extend(list(range(len(sample))))

        input_ids, masked_lm_positions, masked_lm_mask, labels = (
            self.mask_single_sequence(input_ids)
        )

        num_pad = self.max_seq_length - len(input_ids)
        input_mask = [1] * len(input_ids) + [0] * num_pad
        attention_span = attention_span + [0] * num_pad
        position_ids = position_ids + [0] * num_pad
        input_ids = input_ids + [self.pad_id] * num_pad

        if not self.mlm_with_gather:
            labels = labels + [self.ignore_index] * num_pad

        # assertions to ensure correct output shapes
        assert (
            len(input_ids) == self.max_seq_length
            and len(input_mask) == self.max_seq_length
            and len(attention_span) == self.max_seq_length
            and len(position_ids) == self.max_seq_length
        ), "Wrong sequence length"

        if self.inverted_mask:
            input_mask = [elem == 0 for elem in input_mask]

        return (
            input_ids,
            input_mask,
            attention_span,
            position_ids,
            masked_lm_positions,
            masked_lm_mask,
            labels,
        )

    def create_features_auto_lm_vsl(
        self,
        bin,
    ):
        """Given a list of VSL sequences, generate input features and labels.

        Args:
            bin (list(sequence)): list of VSL sequences.
            pad_id (int): Id for pad token. Defaults to `0`.

        Returns:
            Tuple containing features and labels
        """
        input_ids, labels, attention_span, position_ids = [], [], [], []
        input_mask = []
        for i, sample in enumerate(bin):
            input_ids.extend(sample[0])
            input_mask.extend(sample[1])
            labels.extend(sample[2])
            sample_len = len(sample[0])
            if i == len(bin) - 1:
                attention_span.extend(list(range(sample_len - 2, -1, -1)))
                position_ids.extend(list(range(sample_len - 1)))
            else:
                attention_span.extend(list(range(sample_len - 1, -1, -1)))
                position_ids.extend(list(range(sample_len)))

        input_ids = input_ids[:-1]
        input_mask = input_mask[:-1]
        labels = labels[1:]
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
        attention_span = getattr(np, self.input_ids_dtype)(attention_span)
        position_ids = getattr(np, self.position_ids_dtype)(position_ids)

        return np.stack(
            [input_ids, input_mask, labels, attention_span, position_ids]
        )

    def process_chunks(
        self, tokenized_data: List[List[Any]]
    ) -> Tuple[List[Any], dict]:
        if self.mlm:
            return self.process_chunks_mlm(tokenized_data)
        else:
            return self.process_chunks_nextwordpred(tokenized_data)

    def process_chunks_mlm(
        self, tokenized_data: List[List[Any]]
    ) -> Tuple[List[Any], dict]:
        """
        Processes chunks of tokenized text and returns processed features
        along with statistics about padding and tokens.

        Args:
            tokenized_data (List[List[Any]]): Tokenized text chunks as a list.

        Returns:
            Tuple[List[Any], dict]: Processed results and statistics.
        """
        results = {
            "data": [],
            "labels": [],
        }  # List to store the processed results
        stats = defaultdict(int)
        stats_checkpoint_list = []

        for tokenized_text_chunks in tokenized_data:
            (
                input_ids,
                input_mask,
                attention_span,
                position_ids,
                masked_lm_positions,
                masked_lm_mask,
                labels,
            ) = self.create_features_vsl_mlm(tokenized_text_chunks)

            lvt = len(labels) - labels.count(self.ignore_index)
            stats["num_masked_tokens"] += len(labels) - lvt

            data = np.stack(
                [input_ids, input_mask, attention_span, position_ids], axis=0
            ).reshape(-1, 4, self.max_seq_length)
            if self.mlm_with_gather:
                labels = np.stack(
                    [labels, masked_lm_positions, masked_lm_mask], axis=0
                ).reshape(-1, 3, self.max_predictions)
            else:
                labels = np.stack([labels], axis=0).reshape(
                    -1, 1, self.max_seq_length
                )

            results["data"].append(data)
            results["labels"].append(labels)

            pad_index = np.where(np.array(input_ids) == self.pad_id)[0]
            p_i = int(pad_index[0] if len(pad_index) > 0 else len(input_ids))
            num_pad = self.max_seq_length - p_i

            stats["loss_valid_tokens"] += lvt
            stats["num_pad_tokens"] += num_pad
            stats["non_pad_tokens"] += self.max_seq_length - num_pad
            stats["num_tokens"] += self.max_seq_length
            stats["n_examples"] += 1
            stats["num_sequences_before_packing"] += len(tokenized_text_chunks)
            stats_checkpoint_list.append(copy.deepcopy(stats))

        if len(results["data"]) == 0:
            data = {}
        else:
            data = results

        return data, stats_checkpoint_list

    def process_chunks_nextwordpred(
        self, tokenized_data: List[List[Any]]
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """
        Processes chunks of tokenized text and returns processed features
        along with statistics about padding and tokens.

        Args:
            tokenized_data (List[List[Any]]): Tokenized text chunks as a list.

        Returns:
            Tuple[Dict[str, Any], Dict[str, int]]: Processed results and statistics.
        """
        results = {"data": []}  # List to store the processed results
        stats = defaultdict(int)
        stats_checkpoint_list = []

        for tokenized_text_chunks in tokenized_data:
            eos_len = 1 if self.eos_id is not None else 0
            tokenized_text_chunks_len = sum(
                (len(one_d_list) - eos_len)
                for one_d_list in tokenized_text_chunks[0]
            )
            processed = self.create_features_auto_lm_vsl(
                tokenized_text_chunks,
            )
            if processed.size != 0:
                processed_stats = get_data_stats(
                    processed, self.pad_id, self.eos_id, self.max_seq_length
                )
                stats["num_sequences_before_packing"] += len(
                    tokenized_text_chunks
                )

                for key in processed_stats:
                    stats[key] += processed_stats[key]
                processed = np.expand_dims(processed, axis=0)
                results["data"].append(processed)
                stats_checkpoint_list.append(copy.deepcopy(stats))

        if len(results["data"]) == 0:
            data = {}
        else:
            data = results

        return data, stats_checkpoint_list

    def tokenize_data(
        self, semantic_data_array: List[Dict[str, Any]]
    ) -> Tuple[List[np.ndarray], Dict[str, int]]:
        """
        Tokenizes the given text and creates features suitable for
        auto-regressive language modeling. Handles end-of-sequence addition,
        sequence length adjustments, and document folding for long documents.

        Args:
            semantic_data_dict (Union[Dict[str, Any], List[Dict[str, Any]]]): The data to tokenize.

        Returns:
            Tuple[List[np.ndarray], Dict[str, int]]: Tokenized and processed text features
                                                     and statistics.
        """
        region_data, raw_data_stats = self.parse_semantic_data_array(
            semantic_data_array
        )
        if not region_data:
            return {}, raw_data_stats

        semantic_regions = region_data.get("semantic_regions")

        data, image_paths = (
            region_data.get("data"),
            region_data.get("image_paths"),
        )

        if data == "":
            return {}, raw_data_stats

        if self.mlm:
            tokenized_data = self.tokenizer(
                data,
                max_length=self.max_seq_length,
                truncation=True,
                padding=False,
                return_attention_mask=True,
            )
            input_ids = tokenized_data['input_ids']

            raw_data_stats["processed_files"] = 1
            raw_data_stats["discarded_files"] = 0
            if len(input_ids) == 0:
                raw_data_stats["discarded_files"] = 1
            else:
                raw_data_stats["successful_files"] = 1

            return {"data": [input_ids]}, raw_data_stats

        tokenized_data = self.tokenizer(
            data,
            return_offsets_mapping=True,
        )

        if len(semantic_regions) > 0:
            append_eos_to_multiple_semantic_regions(
                data,
                self.data_ranges,
                self.eos_token,
                self.image_token,
                False,
            )

        tokenized_semantic_region_list = self.get_segment_indices(
            tokenized_data,
            semantic_regions,
        )
        data = {
            "tokenized_data": tokenized_data,
            "image_paths": image_paths,
            "tokenized_semantic_regions": tokenized_semantic_region_list,
        }

        doc_list = self.chop_doc_into_msl(data)
        results, tokenized_data_stats = self.process_docs(doc_list)
        data_stats = tokenized_data_stats.copy()
        data_stats.update(raw_data_stats)
        return results, data_stats

    def create_features_pretraining(
        self,
        doc,
        token_modality_idx=None,
    ):
        input_ids = doc.get("input_ids")
        total_len = len(input_ids)
        if total_len < self.min_sequence_len:
            self.logger.warning(
                "Length of token ids < min_sequence_len, skipping this example..."
            )
            return []

        input_mask, attention_mask = doc.get("loss_mask"), doc.get(
            "attention_mask"
        )

        if not self.is_multimodal and self.rng.random() < self.short_seq_prob:
            input_ids = input_ids[
                0 : self.rng.randint(2, self.max_seq_length - 1)
            ]
            input_mask = input_mask[0 : len(input_ids)]
            attention_mask = attention_mask[0 : len(input_ids)]

        labels = input_ids
        assert (
            len(input_ids) == len(labels)
            and len(labels) == len(input_mask)
            and len(input_mask) == len(attention_mask)
        ), "Wrong sequence length"

        # Create features dictionary
        features = {
            "input_ids": getattr(np, self.input_ids_dtype)(input_ids),
            "labels": getattr(np, self.input_ids_dtype)(labels),
        }
        input_mask = getattr(np, self.input_mask_dtype)(input_mask)
        attention_mask = getattr(np, self.input_ids_dtype)(attention_mask)
        if self.inverted_mask:
            input_mask = np.equal(input_mask, 0).astype(self.input_mask_dtype)

        return np.stack(
            [
                features["input_ids"],
                input_mask,
                features["labels"],
            ]
        )

    def append_within_max_length(
        self, tokenized_data: Dict[str, List[List[Any]]]
    ) -> List[List[List[Any]]]:
        """
        Optimizes the representation of tokenized data by merging shorter sequences
        within the specified maximum sequence length (MSL). This function can handle
        tokenized data structured either as a list of lists of lists of integers or
        as a list of lists of NumPy arrays of integers.

        The function converts a 3D list into a modified 3D structure where each
        innermost list or array is treated as a separate 2D list. It then merges
        these 2D lists or arrays if their combined length is within the maximum
        sequence length (MSL).

        Args:
            tokenized_data (Dict[str, List[List[Any]]]): A dictionary containing the
                tokenized text data under the "data" key. The data can be a 3D list
                of tokenized sequences, where sequences are either lists of integers
                or NumPy arrays of integers.

        Returns:
            List[List[List[Any]]]: An optimized 3D list after merging shorter sequences
            within the maximum sequence length. The output maintains the structure
            of grouped sequences that adhere to the specified MSL.
        """
        tokenized_data = tokenized_data["data"]

        def get_sequence_length(data_element):
            """
            Helper function to determine the length of a sequence based on its type.
            Handles both lists of integers and numpy arrays.
            """
            if isinstance(data_element, list):
                return len(data_element)
            elif isinstance(data_element, np.ndarray):
                return data_element.shape[1]
            else:
                raise ValueError(
                    "Unsupported data type for sequence length calculation."
                )

        def group_arrays_by_msl(tokenized_data, msl):
            grouped_arrays = []
            current_group = []
            current_length = 0

            # Flatten the list of lists into a single list of arrays or lists
            flat_data = [item for sublist in tokenized_data for item in sublist]

            # Sort the flat_data by sequence length in descending order
            flat_data.sort(key=lambda x: get_sequence_length(x), reverse=True)

            for data_element in flat_data:
                num_tokens = get_sequence_length(data_element)

                # If adding this array or list exceeds the MSL, finalize the current group and start a new one
                if current_length + num_tokens > msl:
                    if current_group:  # Ensure the current group is not empty
                        grouped_arrays.append(current_group)
                    current_group = []
                    current_length = 0

                # Add the current element to the group
                current_group.append(data_element)
                current_length += num_tokens

            # Add the last group if it has any elements
            if current_group:
                grouped_arrays.append(current_group)

            return grouped_arrays

        tokenized_data_by_group = group_arrays_by_msl(
            tokenized_data, self.max_seq_length
        )
        return tokenized_data_by_group
