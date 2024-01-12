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
SummarizationTokenGenerator Module

This module provides the SummarizationTokenGenerator class which is designed to tokenize
prompt/completion data and create features suitable for summarization tasks. The class
utilizes the BPETokenizer from the modelzoo.transformers.data_processing.tokenizers
package for tokenization.

Usage:
    tokenizer = SummarizationTokenizer(dataset_params,max_sequence_length,tokenizer)
    tokenized_features = tokenizer.encode(("prompt_text","completion_text"))
"""

import logging
from typing import List

import ftfy
import numpy as np

from modelzoo.transformers.data_processing.scripts.hdf5_preprocessing.utils import (
    create_features_summarization,
    wikitext_detokenizer,
)

logger = logging.getLogger("utils")
logger.setLevel(logging.INFO)


def create_features_summarization(
    prompt_ids,
    completion_ids,
    max_sequence_length,
    eos_id=0,
    sep_id=None,
    pad_id=0,
    min_len=10,
    inverted_mask=False,
    input_ids_dtype="int32",
    input_mask_dtype="int32",
    labels_dtype="int32",
):
    """
    Given a list of prompt_ids and completion_ids, generate input sequence
    and labels.

    Args:
        prompt_ids (sequence): List containing token ids for the prompt to
            create features,labels and input mask from.
        completion_ids (sequence): List containing token ids for the completion
            create features,labels and input mask from.
        max_sequence_length (int): Maximum sequence length for data writes.
        eos_id (int): Id for end of sequence token. Defaults to `0`.
        sep_id (int): Id for separator token. Defaults to `None`.
        pad_id (int): Id for pad token. Defaults to `0`.
        min_len (int): Minimum length of token_ids to be considered a valid
            sequence.
        inverted_mask (bool): Invert mask if specified for runtime execution.
            Defaults to `False`.
        input_ids_dtype (str): Dtype as string for input ids.
            Defaults to `int32`.
        input_mask_dtype (str): Dtype as string for input mask.
            Defaults to `int32`.
        labels_dtype (str): Dtype as string for labels. Defaults to `int32`.
    """

    # extra <EOS>
    total_len = len(prompt_ids) + len(completion_ids) + 1
    if sep_id is not None:
        total_len += 1
    if total_len > max_sequence_length:
        logger.warning(
            "prompt_ids + completion_ids > max_sequence_length, skipping this example..."
        )
        return ([], 0, 0, 0, 0)
    if total_len < min_len:
        logger.warning(
            "prompt_ids + completion_ids < min_sequence_len, skipping this example..."
        )
        return ([], 0, 0, 0, 0)

    token_ids = prompt_ids
    if sep_id is not None:
        token_ids = token_ids + [sep_id]
    token_ids = token_ids + completion_ids + [eos_id]

    token_mask = [0] * (len(prompt_ids))
    if sep_id is not None:
        token_mask += [1]
    else:
        # if no sep_id, prediction starts at the last token of prompt_ids
        token_mask[-1] = 1
    token_mask += [1] * len(completion_ids)
    token_mask += [0]  # EOS

    # add padding
    token_ids_pad = max_sequence_length + 1 - len(token_ids)
    input_mask_pad = max_sequence_length - len(token_mask)

    token_ids.extend([pad_id] * token_ids_pad)
    token_mask.extend([0] * input_mask_pad)

    input_ids = token_ids[:-1]
    labels = token_ids[1:]

    assert (
        len(input_ids) == max_sequence_length
        and len(labels) == max_sequence_length
        and len(token_mask) == max_sequence_length
    ), "Wrong sequence length"

    features = dict()
    features["input_ids"] = getattr(np, input_ids_dtype)(input_ids)
    features["input_mask"] = getattr(np, input_mask_dtype)(token_mask)

    if inverted_mask:
        features["input_mask"] = np.equal(features["input_mask"], 0).astype(
            features["input_mask"].dtype
        )
    labels = getattr(np, labels_dtype)(labels)
    loss_valid_tokens = int(features["input_mask"].sum())
    num_tokens = int(features["input_ids"].shape[0])
    # This function is a modified copy of a function with same name written in
    # hdf5_preprocessing utils. The difference here is the we are returning
    # number of pad and masked tokens in addition to the original data.
    return (
        np.stack([features["input_ids"], features["input_mask"], labels]),
        token_ids_pad,
        (max_sequence_length - len(completion_ids)),
        loss_valid_tokens,
        num_tokens,
    )


class SummarizationTokenGenerator:
    def __init__(self, params, tokenizer, eos_id, pad_id):
        """
        Initialize the SummarizationTokenizer class.

        Args:
            vocab_file (str): Path to the vocabulary file.
            encoder_file (str): Path to the encoder file.
            max_seq_length (int, optional): Maximum sequence length. Defaults to 2048.
        """
        dataset_params = params["dataset"]
        processing_params = params["processing"]
        self.files_per_record = processing_params.pop(
            "files_per_record", 50000
        )  ## redundant param
        self.tokenizer = tokenizer

        self.use_ftfy = dataset_params.pop("use_ftfy", False)
        self.ftfy_normalizer = dataset_params.pop("ftfy_normalizer", "NFC")
        self.wikitext_detokenize = dataset_params.pop(
            "wikitext_detokenize", False
        )
        self.input_ids_dtype = dataset_params.pop("input_ids_dtype", "int32")
        self.input_mask_dtype = dataset_params.pop("input_mask_dtype", "int32")
        self.sep_token = dataset_params.pop("sep_token", None)
        self.sep_id = None
        if self.sep_token:
            self.sep_id = self.tokenizer.get_token_id(self.sep_token)
        self.inverted_mask = dataset_params.pop("inverted_mask", False)
        self.min_sequence_len = dataset_params.pop("min_sequence_len", 10)

        self.max_seq_length = processing_params.pop("max_seq_length", 2048)
        self.eos_id = eos_id
        self.pad_id = pad_id

    def tokenize_text(self, text: str) -> List[int]:
        """
        Tokenize the provided text.

        Args:
            text (str): Text to tokenize.

        Returns:
            List[int]: List of token IDs.
        """
        return self.tokenizer.encode(text)

    def encode(self, doc: tuple) -> List[np.ndarray]:
        """
        Tokenize and encode the doc for text summarization.

        Args:
            doc (tuple): Contains prompt, completion data to encode

        Returns:
            List[np.ndarray]: Encoded features for text summarization.
        """
        prompt, completion = doc

        raw_chars_count = len(prompt) + len(completion)
        raw_bytes_count = len(prompt.encode("utf-8")) + len(
            completion.encode("utf-8")
        )
        files_processed = 0
        discarded_files = 0
        normalized_chars_count = raw_chars_count
        normalized_bytes_count = raw_bytes_count
        num_pad_tokens = 0
        num_masked_tokens = 0

        if self.use_ftfy:
            prompt = ftfy.fix_text(prompt, normalization=self.ftfy_normalizer)
            completion = ftfy.fix_text(
                completion, normalization=self.ftfy_normalizer
            )
            normalized_chars_count = len(prompt) + len(completion)
            normalized_bytes_count = len(prompt.encode("utf-8")) + len(
                completion.encode("utf-8")
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
        (
            sample,
            num_pad_tokens,
            num_masked_tokens,
            loss_valid_tokens,
            num_tokens,
        ) = create_features_summarization(
            prompt_encoded,
            completion_encoded,
            self.max_seq_length,
            self.eos_id,
            self.sep_id,
            self.pad_id,
            min_len=self.min_sequence_len,
            inverted_mask=self.inverted_mask,
            input_ids_dtype=self.input_ids_dtype,
            input_mask_dtype=self.input_mask_dtype,
            labels_dtype=self.input_ids_dtype,
        )
        if sample == []:
            discarded_files += 1
        else:
            files_processed += 1
            sample = np.expand_dims(sample, axis=0)

        data_stats = {
            "discarded": discarded_files,
            "processed": files_processed,
            "successful": files_processed - discarded_files,
            "raw_chars_count": raw_chars_count,
            "raw_bytes_count": raw_bytes_count,
            "num_pad_tokens": num_pad_tokens,
            "num_masked_tokens": num_masked_tokens,
            "loss_valid_tokens": loss_valid_tokens,
            "num_tokens": num_tokens,
            "normalized_chars_count": normalized_chars_count,
            "normalized_bytes_count": normalized_bytes_count,
        }

        return sample, data_stats

    def get_token_id(self, token: str) -> int:
        """
        Get the token ID for the given token.

        Args:
            token (str): Token for which the ID is needed.

        Returns:
            int: Token ID.
        """
        return self.tokenizer.get_token_id(token)
