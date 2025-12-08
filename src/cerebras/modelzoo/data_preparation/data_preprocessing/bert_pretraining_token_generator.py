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
BertPretrainingTokenGenerator Module

This module provides the BertPretrainingTokenGenerator class which is designed to process
text data and create features suitable for BERT pretraining tasks. It inherits from
PretrainingTokenGenerator and implements BERT-specific tokenization and feature creation.

Usage:
    tokenizer = BertPretrainingTokenGenerator(params, tokenizer, eos_id, pad_id)
    tokenized_features = tokenizer.encode("Sample text for processing.")
"""

from typing import Any, Dict

import numpy as np

from cerebras.modelzoo.data_preparation.data_preprocessing.pretraining_token_generator import (
    PretrainingTokenGenerator,
)


class BertPretrainingTokenGenerator(PretrainingTokenGenerator):
    """
    A class for generating tokens for pretraining BERT models.
    Inherits from PretrainingTokenGenerator and implements specific methods for BERT.
    """

    def __init__(
        self,
        params: Dict[str, Any],
        tokenizer: Any,
        eos_id: int = None,
        pad_id: int = None,
        **kwargs: Any,
    ):
        """
        Initializes the BertPretrainingTokenGenerator with the given parameters.

        Args:
            params (Dict[str, Any]): Configuration parameters for the token generator.
            tokenizer (Any): Tokenizer instance to use for encoding text.
            eos_id (int, optional): End of sequence token ID. Defaults to None.
            pad_id (int, optional): Padding token ID. Defaults to None.
            **kwargs (Any): Additional keyword arguments for customization.
        """
        super().__init__(
            params,
            tokenizer,
            eos_id=eos_id,
            pad_id=pad_id,
        )
        self.is_bert = True
        self.features = [
            "input_ids",
            "attention_mask",
            "token_type_ids",
        ]
        if self.is_multimodal:
            raise NotImplementedError(
                "BertPretrainingTokenGenerator does not support multimodal data."
            )
        if self.mlm:
            raise NotImplementedError(
                f"MLM is not tested for BERT pretraining. "
                f"Please set `mlm` to False."
            )
        if self.split_text_to_tokenize:
            raise NotImplementedError(
                f"split_text_to_tokenize is not supported for BERT pretraining. "
                f"Please set `split_text_to_tokenize` to False."
            )
        if self.pack_sequences:
            self.logger.warning(
                f"pack_sequences is not supported for BERT pretraining. "
                f"Please set `pack_sequences` to False."
            )
            self.pack_sequences = False

    def chop_doc_into_msl(self, data):
        doc_list = []
        tokenized_data = data.get("tokenized_data")
        tokenized_semantic_regions = data.get("tokenized_semantic_regions")

        # Get all tokens from tokenized data
        all_tokens = tokenized_data["input_ids"]
        all_token_type_ids = tokenized_data["token_type_ids"]
        all_attention_mask = tokenized_data["attention_mask"]

        # Create loss mask based on semantic regions using vectorized operations
        all_loss_mask = np.zeros(len(all_tokens), dtype=np.int32)
        for region in tokenized_semantic_regions:
            start_idx, end_idx = region["indices"]
            loss_weight = region["loss_weight"]
            end_idx = min(end_idx, len(all_tokens))
            all_loss_mask[start_idx:end_idx] = loss_weight
        all_loss_mask = all_loss_mask.tolist()

        if self.truncate_to_msl:
            # Single chunk truncated to self.max_seq_length
            chunk_tokens = all_tokens[: self.max_seq_length]
            chunk_token_type_ids = all_token_type_ids[: self.max_seq_length]
            chunk_loss_mask = all_loss_mask[: self.max_seq_length]
            chunk_attention_mask = all_attention_mask[: self.max_seq_length]

            doc_list.append(
                {
                    "input_ids": chunk_tokens,
                    "token_type_ids": chunk_token_type_ids,
                    "loss_mask": chunk_loss_mask,
                    "attention_mask": chunk_attention_mask,
                }
            )
        else:
            # Create chunks of self.max_seq_length
            total_len = len(all_tokens)
            for i in range(0, total_len, self.max_seq_length):
                chunk_tokens = all_tokens[i : i + self.max_seq_length]
                chunk_token_type_ids = all_token_type_ids[
                    i : i + self.max_seq_length
                ]
                chunk_loss_mask = all_loss_mask[i : i + self.max_seq_length]
                chunk_attention_mask = all_attention_mask[
                    i : i + self.max_seq_length
                ]

                doc_list.append(
                    {
                        "input_ids": chunk_tokens,
                        "token_type_ids": chunk_token_type_ids,
                        "loss_mask": chunk_loss_mask,
                        "attention_mask": chunk_attention_mask,
                    }
                )

        return doc_list

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

        input_mask = doc.get("loss_mask")
        attention_mask = doc.get("attention_mask")
        token_type_ids = doc.get("token_type_ids")

        # Add padding
        num_pad = self.max_seq_length - len(input_ids)
        padding = [self.pad_id] * num_pad
        input_ids.extend(padding)
        # # labels.extend(padding)

        input_mask_padding = [0] * num_pad
        input_mask.extend(input_mask_padding)
        attention_mask_padding = [0] * num_pad
        attention_mask.extend(attention_mask_padding)
        token_type_ids_padding = [token_type_ids[-1]] * num_pad
        token_type_ids.extend(token_type_ids_padding)

        assert (
            len(input_ids) == self.max_seq_length
            and len(token_type_ids) == self.max_seq_length
            and len(input_mask) == self.max_seq_length
            and len(attention_mask) == self.max_seq_length
        ), "Wrong sequence length"

        # Create features dictionary
        features = {
            "input_ids": getattr(np, self.input_ids_dtype)(input_ids),
        }
        input_mask = getattr(np, self.input_mask_dtype)(input_mask)
        attention_mask = getattr(np, self.input_ids_dtype)(attention_mask)
        token_type_ids = getattr(np, self.input_ids_dtype)(token_type_ids)

        if self.inverted_mask:
            input_mask = np.equal(input_mask, 0).astype(self.input_mask_dtype)

        return np.stack([features["input_ids"], attention_mask, token_type_ids])

    def tokenize_data(self, semantic_data_array):
        region_data, raw_data_stats = self.parse_semantic_data_array(
            semantic_data_array
        )
        if not region_data:
            return {}, raw_data_stats

        semantic_regions = region_data.get("semantic_regions")

        data = region_data.get("data")

        if data == "":
            return {}, raw_data_stats

        tokenized_data = self.tokenizer(
            data,
            return_offsets_mapping=True,
        )

        # TODO: Check if BERT needs such a feature, commenting out till then
        # if len(semantic_regions) > 0:
        #     append_eos_to_multiple_semantic_regions(
        #         data,
        #         self.data_ranges,
        #         self.eos_token,
        #         self.image_token,
        #         False,
        #     )

        tokenized_semantic_region_list = self.get_segment_indices(
            tokenized_data,
            semantic_regions,
        )

        data = {
            "tokenized_data": tokenized_data,
            "tokenized_semantic_regions": tokenized_semantic_region_list,
        }

        doc_list = self.chop_doc_into_msl(data)
        results, tokenized_data_stats = self.process_docs(doc_list)
        data_stats = tokenized_data_stats.copy()
        data_stats.update(raw_data_stats)

        return results, data_stats
