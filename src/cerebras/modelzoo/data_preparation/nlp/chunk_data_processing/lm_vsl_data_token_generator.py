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
This module provides the VSLLMDataTokenGenerator class, extending
LMDataTokenGenerator for advanced processing of tokenized text data tailored
for variable-length sequence language modeling (VSLLM). Includes methods for
processing chunks of tokenized text, optimizing representation of tokenized
data by merging shorter sequences within a specified maximum sequence length,
and tokenizing text for auto-regressive language modeling.
"""

from typing import Any, List, Tuple

import numpy as np

from cerebras.modelzoo.data_preparation.nlp.chunk_data_processing.lm_data_token_generator import (
    LMDataTokenGenerator,
)
from cerebras.modelzoo.data_preparation.nlp.hdf5_preprocessing.utils import (
    create_features_auto_lm_vsl,
)


class VSLLMDataTokenGenerator(LMDataTokenGenerator):
    """
    Processes tokenized text data, specifically for VSLLM. Extends
    LMDataTokenGenerator by handling text tokenization, feature creation,
    and optimizing representation of tokenized data for language modeling tasks.

    Attributes:
        use_vsl (bool): Usage of variable sequence length logic.
        fold_long_doc (bool): Whether to fold long documents.
        position_ids_dtype (str): Data type for position IDs in tokenized output.

    Args:
        params (dict): Parameters for the dataset and model.
        tokenizer: Tokenizer instance for text tokenization.
        eos_id (int): End-of-sequence ID.
        pad_id (int): Padding ID.
    """

    use_vsl = True

    def __init__(self, params: dict, tokenizer, eos_id: int, pad_id: int):
        """
        Initialize VSLLMDataTokenGenerator with dataset parameters,
        tokenizer, and token IDs.
        """
        super(VSLLMDataTokenGenerator, self).__init__(
            params, tokenizer, eos_id, pad_id
        )
        self.fold_long_doc = params["dataset"].pop("fold_long_doc", True)
        self.position_ids_dtype = params["dataset"].pop(
            "position_ids_dtype", "int32"
        )
        self.pack_sequences = False

    def process_chunks(
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
        results = []  # List to store the processed results
        stats = {
            "loss_valid_tokens": 0,
            "num_tokens": 0,
            "num_pad_tokens": 0,
            "non_pad_tokens": 0,
            "num_masked_tokens": 0,
        }

        for tokenized_text_chunks in tokenized_data:
            eos_len = 1 if self.eos_id is not None else 0
            tokenized_text_chunks_len = sum(
                (len(one_d_list) - eos_len)
                for one_d_list in tokenized_text_chunks
            )
            num_pad = self.max_seq_length - tokenized_text_chunks_len
            processed = create_features_auto_lm_vsl(
                tokenized_text_chunks,
                self.max_seq_length,
                num_pad,
                pad_id=self.pad_id,
                inverted_mask=self.inverted_mask,
                input_ids_dtype=self.input_ids_dtype,
                input_mask_dtype=self.input_mask_dtype,
                labels_dtype=self.input_ids_dtype,
                attention_span_dtype=self.position_ids_dtype,
                position_ids_dtype=self.position_ids_dtype,
            )
            if processed != []:
                loss_valid_tokens = int(processed[1, :].sum())
                stats["num_pad_tokens"] += num_pad
                stats["non_pad_tokens"] += self.max_seq_length - num_pad
                stats["num_masked_tokens"] += (
                    self.max_seq_length - loss_valid_tokens
                )
                stats["loss_valid_tokens"] += loss_valid_tokens
                stats["num_tokens"] += len(processed[0])
                results.append(processed)

        return [results], stats

    def tokenize_text_auto_lm(self, text: str) -> Tuple[List[np.ndarray], dict]:
        """
        Tokenizes the given text and creates features suitable for
        auto-regressive language modeling. Handles end-of-sequence addition,
        sequence length adjustments, and document folding for long documents.

        Args:
            text (str): The text to tokenize.

        Returns:
            Tuple[List[np.ndarray], dict]: Tokenized and processed text features
                                           and statistics.
        """
        results = []  # List to store the processed results
        stats = {
            "loss_valid_tokens": 0,
            "num_tokens": 0,
            "num_pad_tokens": 0,
            "non_pad_tokens": 0,
            "num_masked_tokens": 0,
            "discarded_files": 0,
        }
        # tokenize text
        tokenized_text = self.tokenize_text(text)
        if self.eos_id is not None:
            tokenized_text += [self.eos_id]

        tokenized_text_len = len(tokenized_text)
        if tokenized_text_len < self.min_sequence_len:
            return results, stats

        if self.rng.random() < self.short_seq_prob:
            tokenized_text = tokenized_text[
                0 : self.rng.randint(2, self.max_seq_length)
            ]
            tokenized_text_len = len(tokenized_text)

        if tokenized_text_len > self.max_seq_length + 1:
            if not self.fold_long_doc:
                return results, stats

        tokenized_text_chunks = [
            tokenized_text[i : i + self.max_seq_length + 1]
            for i in range(0, len(tokenized_text), self.max_seq_length)
        ]

        # update prefix if last chunk is < max_seq_length
        num_tokens_last_chunk = len(tokenized_text_chunks[-1])
        if num_tokens_last_chunk < 2:
            _ = tokenized_text_chunks.pop(-1)

        return tokenized_text_chunks, stats

    def append_within_max_length(
        self, tokenized_data: List[List[List[Any]]]
    ) -> List[List[List[Any]]]:
        """
        Optimizes representation of tokenized data by merging shorter sequences
        within the specified maximum sequence length. Converts 3D list to a
        modified 3D structure where each innermost list is treated as a separate
        2D list, then merges these 2D lists if their combined length is within
        the max sequence length.

        Args:
            tokenized_data (List[List[List[Any]]]): 3D list of tokenized text data.

        Returns:
            List[List[List[Any]]]: Optimized 3D list after merging shorter sequences.
        """

        def convert_3d_to_modified_3d(tokenized_data):
            # First, flatten the 3D list to a 2D list
            flattened_2d_list = []
            for two_d_list in tokenized_data:
                for one_d_list in two_d_list:
                    flattened_2d_list.append(one_d_list)

            # Then, convert each list in the flattened 2D list to a 2D list
            # within a new 3D list
            new_3d_list = []
            for one_d_list in flattened_2d_list:
                new_2d_list = [one_d_list]
                new_3d_list.append(new_2d_list)

            return new_3d_list

        tokenized_data = convert_3d_to_modified_3d(tokenized_data)
        eos_len = 1 if self.eos_id is not None else 0
        # Precompute combined length of all lists in each 2D list
        combined_lengths = [
            sum(len(one_d_list) - eos_len for one_d_list in two_d_list)
            for two_d_list in tokenized_data
        ]

        indices_to_remove = set()

        # Iterate over each 2D list in the 3D list in reverse order
        for i in range(len(tokenized_data) - 1, 0, -1):
            # Use the precomputed length
            current_combined_length = combined_lengths[i]
            # Check if combined length of current 2D list is less than max_seq_length
            if current_combined_length < self.max_seq_length:
                # Look for a previous 2D list to merge with
                for j in range(i - 1, -1, -1):
                    # Use the precomputed length
                    total_combined_length = (
                        current_combined_length + combined_lengths[j]
                    )

                    # Check if combined length of both 2D lists is within max_seq_length
                    if total_combined_length <= self.max_seq_length:
                        # If so, merge current 2D list into the previous 2D list
                        tokenized_data[j].extend(tokenized_data[i])
                        # Update combined length for merged 2D list
                        combined_lengths[j] += combined_lengths[i]

                        # Instead of deleting immediately, add the index to the set
                        indices_to_remove.add(i)
                        break  # Exit inner loop as merge is done

        # Delete the elements after the loop is done
        # Convert indices_to_remove to a list and sort in reverse order to ensure indices remain correct while deleting
        for index in sorted(indices_to_remove, reverse=True):
            del tokenized_data[index]
            del combined_lengths[index]

        return tokenized_data
