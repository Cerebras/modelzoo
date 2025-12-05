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
PretrainingTokenGenerator class, tailored for fill in the middle (FIM) tasks.

Usage:
    from your_module_name import FIMTokenGenerator

    # Initialize the token generator with the required parameters
    tokenizer = FIMTokenGenerator(params, tokenizer_impl, eos_id, pad_id)

    # Tokenize and encode text data
    tokenized_data, stats = tokenizer.encode("Your sample text to process.")
"""

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from cerebras.modelzoo.data_preparation.data_preprocessing.pretraining_token_generator import (
    PretrainingTokenGenerator,
)
from cerebras.modelzoo.data_preparation.data_preprocessing.utils import (
    check_fim_special_tokens,
    fim,
    get_data_stats,
    handle_bos_token_default,
)


class FIMTokenGenerator(PretrainingTokenGenerator):
    def __init__(self, params, tokenizer, eos_id, pad_id):
        """
        Initialize the FIMTokenGenerator class.
        Args:
            params (Dict[str, Any]): Params from config file.
            tokenizer: Tokenizer instance.
            eos_id (int): End of sequence token ID.
            pad_id (int): Padding token ID.
        """
        super(FIMTokenGenerator, self).__init__(
            params, tokenizer, eos_id, pad_id
        )
        dataset_params = params.get("dataset", {})
        processing_params = params.get("processing", {})
        self.fim_rate = dataset_params.pop("fim_rate", None)
        self.spm_rate = dataset_params.pop("spm_rate", None)

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
            dataset_params.pop("fim_suffix_tok", None)
        )[-1]
        self.prefix_tok_id = self.tokenizer.encode(
            dataset_params.pop("fim_prefix_tok", None)
        )[-1]
        self.middle_tok_id = self.tokenizer.encode(
            dataset_params.pop("fim_middle_tok", None)
        )[-1]

    def encode(
        self, semantic_data_array: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """
        Tokenize and encode the data for auto-regressive language modeling.

        Args:
            semantic_data_array (Union[Dict[str, Any], List[Dict[str, Any]]]): Data to encode.

        Returns:
            Tuple[Dict[str, Any], Dict[str, int]]: Tuple of encoded features for auto-regressive language modeling and dataset stats.
        """
        tokenized_data, data_stats = self.tokenize_data(semantic_data_array)
        if not tokenized_data:
            return {}, data_stats
        tokenized_data = tokenized_data["data"]
        result = []
        # Reset the stats for pad tokens and masked tokens and recompute for FIM
        num_masked_tokens = 0
        num_pad_tokens = 0
        loss_valid_tokens = 0
        num_tokens = 0
        tokenized_data_stats = defaultdict(int)
        for i, sample in enumerate(tokenized_data):
            if len(sample) != 0:
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
                sample_data_stats = get_data_stats(
                    sample, self.pad_id, self.eos_id, self.max_seq_length
                )
                for key in sample_data_stats:
                    tokenized_data_stats[key] += sample_data_stats[key]
                result.append(sample)

        if not result:
            data = {}
        else:
            data = {"data": result}
            data_stats.update(tokenized_data_stats)
        return data, data_stats
