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
from typing import Dict, List, Tuple

import ftfy
import numpy as np

from cerebras.modelzoo.data_preparation.nlp.hdf5_preprocessing.utils import (
    wikitext_detokenizer,
)

logger = logging.getLogger("summarization_token_generator")
logger.setLevel(logging.INFO)


def create_features_summarization(
    prompt_ids,
    completion_ids,
    max_sequence_length,
    eos_id=0,
    sep_id=None,
    pad_id=0,
    completion_prefix_mask_len=0,
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
        completion_prefix_mask_len (int): This stores the number of tokens which
        the completion prefix takes.
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

    # This function is a modified copy of a function with same name written in
    # hdf5_preprocessing utils. The difference here is the we are returning
    # number of pad tokens,masked tokens, loss valid tokens and total number of tokens in addition to the original data.

    # extra <EOS>
    total_len = len(prompt_ids) + len(completion_ids) + 1
    if sep_id is not None:
        total_len += 1
    if total_len > max_sequence_length:
        logger.warning(
            "prompt_ids + completion_ids > max_sequence_length, skipping this example..."
        )
        return []
    if total_len < min_len:
        logger.warning(
            "prompt_ids + completion_ids < min_sequence_len, skipping this example..."
        )
        return []

    token_ids = []
    if sep_id is not None:
        prompt_ids = prompt_ids + [sep_id]
    completion_ids += [eos_id]
    token_ids = prompt_ids + completion_ids

    token_mask = [0] * (len(prompt_ids) - 1)
    ## mask the completion prefix tokens
    token_mask += [0] * completion_prefix_mask_len
    # start prediction on the last prompt token (including if it's sep or eos) or the last completion prefix token
    token_mask += [1]
    token_mask += [1] * (len(completion_ids) - completion_prefix_mask_len - 1)
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

    return np.stack([features["input_ids"], features["input_mask"], labels])


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
        self.default_add_bos = False
        if (
            hasattr(self.tokenizer, "add_bos_token")
            and self.tokenizer.add_bos_token
        ):
            self.default_add_bos = True

        self.inverted_mask = dataset_params.pop("inverted_mask", False)
        self.min_sequence_len = dataset_params.pop("min_sequence_len", 10)

        self.max_seq_length = processing_params.pop("max_seq_length", 2048)
        self.eos_id = eos_id
        self.pad_id = pad_id

        self.prompt_prefix = params["dataset"].pop("prompt_prefix", None)
        self.completion_prefix = params["dataset"].pop(
            "completion_prefix", None
        )
        self.tokenize_prefix_tags()

    def tokenize_prefix_tags(self):
        if self.prompt_prefix is not None:
            self.prompt_prefix_toks = self.tokenize_text(self.prompt_prefix)
            self.prompt_prefix_toks_len = len(self.prompt_prefix_toks)
        else:
            self.prompt_prefix_toks_len = 0

        if self.completion_prefix is not None:
            self.comp_prefix_toks = self.tokenize_text(self.completion_prefix)
            self.comp_prefix_toks_len = len(self.comp_prefix_toks)
            if self.default_add_bos:
                self.comp_prefix_toks_len -= 1  ## Remove the default bos token
        else:
            self.comp_prefix_toks_len = 0

    def prepend_prefix(
        self,
        prompt_ids: List[int],
        completion_ids: List[int],
        index: int,
    ) -> Tuple[List[int], List[int]]:
        """
        Prepends prefixes to prompt ids and completion ids and manages the beginning of sentence (BOS) tokens.

        :param prompt_ids: A list of integer IDs representing the prompt.
        :param completion_ids: A list of integer IDs representing the completion.
        :param index: The index indicating the position of the sequence being processed.

        return: A tuple of two lists: the updated prompt_ids and completion_ids.
        """

        if self.default_add_bos:
            prompt_ids_suffix = (
                prompt_ids[1:]
                if index > 0 or self.prompt_prefix
                else prompt_ids
            )
            completion_ids_suffix = completion_ids[1:]
        else:
            prompt_ids_suffix = prompt_ids
            completion_ids_suffix = completion_ids

        if self.prompt_prefix is not None:
            prompt_prefix_tokens = (
                self.prompt_prefix_toks[1:]
                if self.default_add_bos and index > 0
                else self.prompt_prefix_toks
            )
            prompt_ids = prompt_prefix_tokens + prompt_ids_suffix
        else:
            prompt_ids = prompt_ids_suffix

        if self.completion_prefix is not None:
            completion_prefix_tokens = (
                self.comp_prefix_toks[1:]
                if self.default_add_bos
                else self.comp_prefix_toks
            )
            completion_ids = completion_prefix_tokens + completion_ids_suffix
        else:
            completion_ids = completion_ids_suffix

        return prompt_ids, completion_ids

    def tokenize_text(self, text: str) -> List[int]:
        """
        Tokenize the provided text.

        Args:
            text (str): Text to tokenize.

        Returns:
            List[int]: List of token IDs.
        """
        return self.tokenizer.encode(text)

    def check_valid_doc(self, prompt, completion):
        doc_field = None
        if prompt == "" and completion == "":
            doc_field = "both prompt and completion"
        elif prompt == "":
            doc_field = "prompt"
        elif completion == "":
            doc_field = "completion"

        if doc_field:
            logger.warning(f"{doc_field} is empty. Skipping this doc...")
            return False
        else:
            return True

    def encode(self, doc: List[Tuple]) -> Tuple[List[np.ndarray], Dict]:
        """
        Tokenize and encode the doc for text summarization.

        Args:
            List[tuple]: Contains  a list of prompt, completion data to encode

        Returns:
            -> Tuple[List[np.ndarray], Dict]: Tuple of encoded features for text summarization and dataset stats
        """

        raw_chars_count = 0
        raw_bytes_count = 0
        files_processed = 0
        discarded_files = 0
        normalized_chars_count = 0
        normalized_bytes_count = 0
        num_pad_tokens = 0
        non_pad_tokens = 0
        num_masked_tokens = 0
        loss_valid_tokens = 0
        num_tokens = 0
        sample_list = []
        for i, (prompt, completion) in enumerate(doc):
            if not self.check_valid_doc(prompt, completion):
                continue
            raw_chars_count += len(prompt) + len(completion)
            raw_bytes_count += len(prompt.encode("utf-8")) + len(
                completion.encode("utf-8")
            )
            files_processed += 1

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

            normalized_chars_count += len(prompt) + len(completion)
            normalized_bytes_count += len(prompt.encode("utf-8")) + len(
                completion.encode("utf-8")
            )
            prompt_encoded = self.tokenize_text(prompt)
            completion_encoded = self.tokenize_text(completion)
            prompt_encoded, completion_encoded = self.prepend_prefix(
                prompt_encoded, completion_encoded, i
            )

            sample = create_features_summarization(
                prompt_encoded,
                completion_encoded,
                self.max_seq_length,
                self.eos_id,
                self.sep_id,
                self.pad_id,
                self.comp_prefix_toks_len,
                min_len=self.min_sequence_len,
                inverted_mask=self.inverted_mask,
                input_ids_dtype=self.input_ids_dtype,
                input_mask_dtype=self.input_mask_dtype,
                labels_dtype=self.input_ids_dtype,
            )
            if sample == []:
                discarded_files += 1
            else:
                num_loss_valid_tokens = int(sample[1, :].sum())
                num_pad_tokens += int((sample[0, :] == self.pad_id).sum())
                non_pad_tokens += int(
                    np.logical_and(
                        sample[0, :] != self.eos_id, sample[0, :] != self.pad_id
                    ).sum()
                )
                num_masked_tokens += self.max_seq_length - num_loss_valid_tokens
                loss_valid_tokens += num_loss_valid_tokens
                num_tokens += int(sample[0, :].shape[0])

                sample = np.expand_dims(sample, axis=0)
                if sample_list != []:
                    sample_list = np.stack([sample_list, sample], axis=0)
                else:
                    sample_list = sample

        data_stats = {
            "discarded": discarded_files,
            "processed": files_processed,
            "successful": files_processed - discarded_files,
            "raw_chars_count": raw_chars_count,
            "raw_bytes_count": raw_bytes_count,
            "normalized_chars_count": normalized_chars_count,
            "normalized_bytes_count": normalized_bytes_count,
            "num_pad_tokens": num_pad_tokens,
            "non_pad_tokens": non_pad_tokens,
            "num_masked_tokens": num_masked_tokens,
            "loss_valid_tokens": loss_valid_tokens,
            "num_tokens": num_tokens,
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
