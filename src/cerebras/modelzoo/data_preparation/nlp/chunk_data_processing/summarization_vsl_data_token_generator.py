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
This module provides the VSLSummarizationTokenGenerator class, which extends
the SummarizationTokenGenerator for processing tokenized text data specifically
for variable-length sequence summarization (VSLS). The class includes methods
for processing chunks of tokenized text, encoding documents for text summarization,
and optimizing the representation of tokenized data by merging shorter sequences
within a specified maximum sequence length.
"""

import logging
from typing import Any, Dict, List, Tuple

import ftfy
import numpy as np

from cerebras.modelzoo.data_preparation.nlp.chunk_data_processing.summarization_data_token_generator import (
    SummarizationTokenGenerator,
)
from cerebras.modelzoo.data_preparation.nlp.hdf5_preprocessing.utils import (
    wikitext_detokenizer,
)

logger = logging.getLogger("utils")
logger.setLevel(logging.INFO)


def create_features_summarization_vsl(
    bin,
    max_sequence_length,
    pad_id=0,
    eos_id=0,
    eos_after_prompt=False,
    sep_id=None,
    completion_prefix_mask_len=0,
    inverted_mask=False,
    input_ids_dtype="int32",
    input_mask_dtype="int32",
    labels_dtype="int32",
    attention_span_dtype="int32",
    position_ids_dtype="int32",
):
    """Given a list of VSL sequences, generate input features and labels.

    Args:
        bin (list(sequence)): list of VSL sequences.
        max_sequence_length (int): Maximum sequence length for data writes.
        pad_id (int): Id for pad token. Defaults to `0`.
        eos_id (int): Id for end of sequence token. Defaults to `0`.
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
    for seq in bin:
        token_ids, token_mask = [], []
        for prompt_ids, completion_ids in seq:
            if eos_after_prompt:
                prompt_ids = prompt_ids + [eos_id]
            if sep_id is not None:
                prompt_ids = prompt_ids + [sep_id]
            completion_ids += [eos_id]
            token_ids += prompt_ids + completion_ids

            token_mask += [0] * (len(prompt_ids) - 1)
            token_mask += [0] * completion_prefix_mask_len
            # start prediction on the last prompt token (including if it's sep or eos) or the last completion prefix token
            token_mask += [1]
            token_mask += [1] * (
                len(completion_ids) - completion_prefix_mask_len - 1
            )
            # don't want to learn from predicting next prompt after end of completion
            token_mask += [0]

        input_ids.extend(token_ids[:-1])
        labels.extend(token_ids[1:])
        input_mask.extend(token_mask[:-1])
        sample_len = len(token_ids) - 1
        attention_span.extend(list(range(sample_len - 1, -1, -1)))
        position_ids.extend(list(range(sample_len)))

    # padding
    num_pad = max_sequence_length - len(input_ids)
    padding = [pad_id] * num_pad
    input_ids.extend(padding)
    labels.extend(padding)

    padding = [0] * num_pad
    input_mask.extend(padding)
    attention_span.extend(padding)
    position_ids.extend(padding)

    # assertions to ensure correct output shapes
    assert (
        len(input_ids) == max_sequence_length
        and len(labels) == max_sequence_length
        and len(input_mask) == max_sequence_length
        and len(attention_span) == max_sequence_length
        and len(position_ids) == max_sequence_length
    ), "Wrong sequence length"

    input_ids = getattr(np, input_ids_dtype)(input_ids)
    input_mask = getattr(np, input_mask_dtype)(input_mask)

    if inverted_mask:
        input_mask = np.equal(input_mask, 0).astype(input_mask.dtype)

    labels = getattr(np, labels_dtype)(labels)
    attention_span = getattr(np, attention_span_dtype)(attention_span)
    position_ids = getattr(np, position_ids_dtype)(position_ids)
    return np.stack(
        [input_ids, input_mask, labels, attention_span, position_ids]
    )


class VSLSummarizationTokenGenerator(SummarizationTokenGenerator):
    """
    Token generator for variable-length sequence summarization (VSLS).
    Extends SummarizationTokenGenerator with additional functionality for VSLS.
    """

    use_vsl = True

    def __init__(self, params, tokenizer, eos_id, pad_id):
        """
        Initialize VSLSummarizationTokenGenerator with dataset parameters,
        tokenizer, and token IDs.
        """
        super(VSLSummarizationTokenGenerator, self).__init__(
            params, tokenizer, eos_id, pad_id
        )
        self.position_ids_dtype = params["dataset"].pop(
            "position_ids_dtype", "int32"
        )
        self.eos_after_prompt = params["dataset"].pop("eos_after_prompt", False)

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
        results = []  # List to store processed results
        data_stats = {
            "loss_valid_tokens": 0,
            "num_tokens": 0,
            "num_pad_tokens": 0,
            "non_pad_tokens": 0,
            "num_masked_tokens": 0,
        }
        sep_len = 1 if self.sep_id is not None else 0
        eos_end_len = 1 if self.eos_after_prompt else 0

        for vsl_list in tokenized_data:
            processed = create_features_summarization_vsl(
                vsl_list,
                self.max_seq_length,
                pad_id=self.pad_id,
                eos_id=self.eos_id,
                eos_after_prompt=self.eos_after_prompt,
                sep_id=self.sep_id,
                completion_prefix_mask_len=self.comp_prefix_toks_len,
                inverted_mask=self.inverted_mask,
                input_ids_dtype=self.input_ids_dtype,
                input_mask_dtype=self.input_mask_dtype,
                labels_dtype=self.input_ids_dtype,
                attention_span_dtype=self.position_ids_dtype,
                position_ids_dtype=self.position_ids_dtype,
            )
            if processed != []:
                loss_valid_tokens = int(processed[1, :].sum())
                num_tokens = int(processed[0, :].shape[0])
                non_pad_tokens = int(
                    np.logical_and(
                        processed[0, :] != self.eos_id,
                        processed[0, :] != self.pad_id,
                    ).sum()
                )
                num_pad = int((processed[0, :] == self.pad_id).sum())
                data_stats["num_pad_tokens"] += num_pad
                data_stats["non_pad_tokens"] += non_pad_tokens
                data_stats["num_masked_tokens"] += (
                    self.max_seq_length - loss_valid_tokens
                )
                data_stats["loss_valid_tokens"] += loss_valid_tokens
                data_stats["num_tokens"] += num_tokens
                results.append(processed)

        return [results], data_stats

    def encode(self, doc: List[tuple]) -> Tuple[List[tuple], Dict]:
        """
        Tokenize and encode the document for text summarization.

        Args:
            doc (List[tuple]): Contains a list of (prompt, completion) data to encode.

        Returns:
            Tuple[List[tuple],Dict: List of tokenized tuples (prompt, completion) and a stats dictionary
        """

        raw_chars_count = 0
        raw_bytes_count = 0
        files_processed = 1
        discarded_files = 0
        normalized_chars_count = 0
        normalized_bytes_count = 0

        total_len = 0
        tokens = []

        for i, (prompt, completion) in enumerate(doc):
            if not self.check_valid_doc(prompt, completion):
                continue
            raw_chars_count += len(prompt) + len(completion)
            raw_bytes_count += len(prompt.encode("utf-8")) + len(
                completion.encode("utf-8")
            )

            if self.use_ftfy:
                prompt = ftfy.fix_text(
                    prompt, normalization=self.ftfy_normalizer
                )
                completion = ftfy.fix_text(
                    completion, normalization=self.ftfy_normalizer
                )

            if self.wikitext_detokenize:
                prompt = wikitext_detokenizer(prompt)
                completion = wikitext_detokenizer(completion)

            normalized_chars_count = len(prompt) + len(completion)
            normalized_bytes_count = len(prompt.encode("utf-8")) + len(
                completion.encode("utf-8")
            )

            prompt_encoded = self.tokenize_text(prompt)
            completion_encoded = self.tokenize_text(completion)
            prompt_encoded, completion_encoded = self.prepend_prefix(
                prompt_encoded, completion_encoded, i
            )
            total_len += (
                len(prompt_encoded)
                + len(completion_encoded)
                + int(self.eos_after_prompt)
            )
            total_len += 1  # for internal eos tokens after completion
            tokens.append((prompt_encoded, completion_encoded))
            if self.sep_id is not None:
                total_len += 1

        total_len -= 1  # but we will remove the last eos token to create input/label pairs
        if total_len > self.max_seq_length:
            logger.warning(
                "prompt_ids + completion_ids > max_sequence_length, skipping this example..."
            )
            discarded_files += 1
            tokens = []
        if total_len < self.min_sequence_len:
            logger.warning(
                "prompt_ids + completion_ids < min_sequence_len, skipping this example..."
            )
            discarded_files += 1
            tokens = []

        data_stats = {
            "discarded": discarded_files,
            "processed": files_processed,
            "successful": files_processed - discarded_files,
            "raw_chars_count": raw_chars_count,
            "raw_bytes_count": raw_bytes_count,
            "normalized_chars_count": normalized_chars_count,
            "normalized_bytes_count": normalized_bytes_count,
            "num_pad_tokens": 0,
            "non_pad_tokens": 0,
            "num_masked_tokens": 0,
            "loss_valid_tokens": 0,
            "num_tokens": 0,
        }

        return tokens, data_stats

    def append_within_max_length(
        self, tokenized_data: List[List[tuple]]
    ) -> List[List[List[tuple]]]:
        """
        Optimize representation of tokenized data by merging shorter sequences
        within the specified maximum sequence length.

        Args:
            tokenized_data (List[List[tuple(List, List)]]): List of tokenized text data
                where each inner list contains (prompt, completion) tuples.

        Returns:
            List[List[List[tuple]]]: Optimized list after merging shorter sequences.
        """
        sep_len = 1 if self.sep_id is not None else 0
        eos_end_len = 1 if self.eos_after_prompt else 0
        # Create a lookup for storing the length of each sublist
        lengths_lookup = [
            sum(
                len(prompt) + len(completion) + sep_len + eos_end_len
                for prompt, completion in sublist
            )
            + len(sublist)
            - 1
            for sublist in tokenized_data
        ]

        tokenized_data = [[entry] for entry in tokenized_data]
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
