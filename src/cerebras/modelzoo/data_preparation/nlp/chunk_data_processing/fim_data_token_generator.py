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
FIMTokenGenerator Module

This module offers the FIMTokenGenerator class, an extension of the
LMDataTokenGenerator class, tailored for fill in the middle (FIM) tasks.

Usage:
    from your_module_name import FIMTokenGenerator

    # Initialize the token generator with the required parameters
    tokenizer = FIMTokenGenerator(params, tokenizer_impl, eos_id, pad_id)

    # Tokenize and encode text data
    tokenized_data, stats = tokenizer.encode("Your sample text to process.")
"""

import logging
from typing import Dict, List, Tuple

import ftfy
import numpy as np

from cerebras.modelzoo.data_preparation.nlp.chunk_data_processing.lm_data_token_generator import (
    LMDataTokenGenerator,
)
from cerebras.modelzoo.data_preparation.nlp.hdf5_preprocessing.utils import (
    check_fim_special_tokens,
    fim,
    handle_bos_token_default,
    wikitext_detokenizer,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class FIMTokenGenerator(LMDataTokenGenerator):
    def __init__(self, params, tokenizer, eos_id, pad_id):
        """
        Initialize the FIMPreprocessor class.
        Args:
            params (args): Params from config file
        """
        super(FIMTokenGenerator, self).__init__(
            params, tokenizer, eos_id, pad_id
        )
        processing_params = params["processing"]

        self.fim_rate = processing_params.pop("fim_rate", None)
        self.spm_rate = processing_params.pop("spm_rate", None)

        # Ensures that FIM tokens are specified in config, and that
        # the specified tokens are actually in the tokenizer
        check_fim_special_tokens(params, self.tokenizer)

        # Some tokenizers use BOS ID at the beginning and others do not.
        # Here we get a flag for whether to use BOS by default
        # and the BOS id if needed.
        self.default_bos_token, self.opt_bos_tok_id = handle_bos_token_default(
            self.tokenizer
        )

        self.suffix_tok_id = self.tokenizer.encode(
            params['processing'].get("fim_suffix_tok")
        )[-1]
        self.prefix_tok_id = self.tokenizer.encode(
            params['processing'].get("fim_prefix_tok")
        )[-1]
        self.middle_tok_id = self.tokenizer.encode(
            params['processing'].get("fim_middle_tok")
        )[-1]

    def encode(self, data: str) -> Tuple[List[np.ndarray], Dict]:
        """
        Tokenize and encode the data for auto-regressive language modeling.

        Args:
            data (str): Text to tokenize

        Returns:
            Tuple[List[np.ndarray], Dict]: Tuple of encoded features for auto-regressive language modeling and dataset stats.
        """
        raw_chars_count = len(data)
        raw_bytes_count = len(data.encode("utf-8"))
        files_processed = 1
        discarded_files = 0
        normalized_chars_count = raw_chars_count
        normalized_bytes_count = raw_bytes_count

        if self.use_ftfy:
            data = ftfy.fix_text(data, normalization=self.ftfy_normalizer)
            normalized_chars_count = len(data)
            normalized_bytes_count = len(data.encode("utf-8"))
        if self.wikitext_detokenize:
            data = wikitext_detokenizer(data)
            normalized_chars_count = len(data)
            normalized_bytes_count = len(data.encode("utf-8"))

        tokenized_text, stats = self.tokenize_text_auto_lm(data)
        result = []
        # Reset the stats for pad tokens and masked tokens and recompute for FIM
        num_masked_tokens = 0
        num_pad_tokens = 0
        loss_valid_tokens = 0
        num_tokens = 0
        discarded_files = stats["discarded_files"] + stats["empty_chunks"]
        if tokenized_text == []:
            discarded_files += 1
        for i, sample in enumerate(tokenized_text):
            if sample != []:
                sample = fim(
                    sample,
                    i,
                    self.tokenizer,
                    self.fim_rate,
                    self.spm_rate,
                    self.suffix_tok_id,
                    self.prefix_tok_id,
                    self.middle_tok_id,
                    self.pad_id,
                    self.eos_id,
                    self.opt_bos_tok_id,
                )
                num_pad_tokens += (sample[0, :] == self.pad_id).sum()
                num_masked_tokens += (sample[1, :] == 0).sum()
                loss_valid_tokens += sample[1, :].sum()
                num_tokens += sample.shape[1]
            else:
                discarded_files += 1
            result.append(sample)

        data_stats = {
            "discarded": discarded_files,
            "processed": files_processed,
            "successful": files_processed - discarded_files,
            "raw_chars_count": raw_chars_count,
            "raw_bytes_count": raw_bytes_count,
            "num_pad_tokens": int(num_pad_tokens),
            "num_masked_tokens": int(num_masked_tokens),
            "loss_valid_tokens": int(loss_valid_tokens),
            "num_tokens": int(num_tokens),
            "normalized_chars_count": normalized_chars_count,
            "normalized_bytes_count": normalized_bytes_count,
        }

        return result, data_stats
