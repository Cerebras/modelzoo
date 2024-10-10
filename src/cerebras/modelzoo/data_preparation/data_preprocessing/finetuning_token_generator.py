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

import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import ftfy
import numpy as np

from cerebras.modelzoo.data_preparation.data_preprocessing.utils import (
    append_eos_to_multiple_semantic_regions,
    find_region_in_formatted_string,
    find_token_range,
    wikitext_detokenizer,
)

logger = logging.getLogger(__name__)


def create_features_finetuning(
    tokenized_data,
    tokenized_semantic_region_list,
    max_sequence_length,
    return_attention_mask=False,
    min_len=10,
):

    token_ids = tokenized_data["input_ids"]
    total_len = len(token_ids)

    if total_len > max_sequence_length:
        logger.warning(
            "Length of token ids > max_sequence_length, skipping this example..."
        )
        return {}
    if total_len < min_len:
        logger.warning(
            "Length of token ids < min_sequence_len, skipping this example..."
        )
        return {}

    def loss_mask_region():
        input_mask = [0] * len(token_ids)
        attention_mask = None
        if return_attention_mask:
            attention_mask = [1] * len(token_ids)
        for i, semantic_region in enumerate(tokenized_semantic_region_list):
            region_name = semantic_region.get("indices")
            start_idx, end_idx = semantic_region.get("indices")
            region_loss_mask = semantic_region.get("loss_weight", 0)
            region_attention_mask = semantic_region.get("attention_mask", 1)
            for idx in range(start_idx, end_idx):
                if idx >= len(token_ids):
                    break
                input_mask[idx] = region_loss_mask
                if return_attention_mask:
                    attention_mask[idx] = region_attention_mask
            if (
                return_attention_mask
                and i == len(tokenized_semantic_region_list) - 1
                and region_name != "image"
            ):
                attention_mask = attention_mask[:-1]

        return input_mask, attention_mask

    input_mask, attention_mask = loss_mask_region()
    if return_attention_mask:
        return {
            "token_ids": token_ids,
            "input_mask": input_mask,
            "attention_mask": attention_mask,
        }
    else:
        return {"token_ids": token_ids, "input_mask": input_mask}


def pad_to_msl(
    data,
    max_sequence_length,
    pad_id,
    inverted_mask=False,
    input_ids_dtype="int32",
    input_mask_dtype="int32",
    labels_dtype="int32",
):

    token_ids, input_mask, attention_mask = (
        data.get("token_ids"),
        data.get("input_mask"),
        data.get("attention_mask", None),
    )
    input_ids = token_ids[:-1]
    labels = token_ids[1:]
    input_mask = input_mask[1:]
    # Calculate padding lengths
    num_pad = max_sequence_length - len(input_ids)
    # Add padding
    input_ids.extend([pad_id] * num_pad)
    input_mask.extend([0] * num_pad)
    labels.extend([pad_id] * num_pad)
    if attention_mask is not None:
        num_pad = max_sequence_length - len(attention_mask)
        attention_mask.extend([0] * num_pad)
        attention_mask = getattr(np, input_ids_dtype)(attention_mask)
        assert (
            len(attention_mask) == max_sequence_length
        ), "Wrong sequence length"
        attention_mask = np.equal(attention_mask, 0).astype(input_mask_dtype)

    # Ensure lengths are consistent
    assert (
        len(input_ids) == max_sequence_length
        and len(labels) == max_sequence_length
        and len(input_mask) == max_sequence_length
    ), "Wrong sequence length"

    # Create features dictionary
    features = {
        "input_ids": getattr(np, input_ids_dtype)(input_ids),
        "labels": getattr(np, labels_dtype)(labels),
    }
    input_mask = getattr(np, input_mask_dtype)(input_mask)

    if inverted_mask:
        input_mask = np.equal(input_mask, 0).astype(input_mask_dtype)

    if attention_mask is not None:
        return np.stack(
            [
                features["input_ids"],
                input_mask,
                features["labels"],
                attention_mask,
            ]
        )
    else:
        return np.stack([features["input_ids"], input_mask, features["labels"]])


class FinetuningTokenGenerator:
    def __init__(self, params, tokenizer, eos_id, pad_id):
        dataset_params = params["dataset"]
        processing_params = params["processing"]
        self.tokenizer = tokenizer
        self.is_multimodal = dataset_params.pop("is_multimodal", False)
        self.use_ftfy = dataset_params.pop("use_ftfy", False)
        self.ftfy_normalizer = dataset_params.pop("ftfy_normalizer", "NFC")
        self.wikitext_detokenize = dataset_params.pop(
            "wikitext_detokenize", False
        )
        self.input_ids_dtype = dataset_params.pop("input_ids_dtype", "int32")
        self.input_mask_dtype = dataset_params.pop("input_mask_dtype", "int32")
        default_sep_token = (
            self.tokenizer.sep_token if self.tokenizer.sep_token else "<|sep|>"
        )
        self.sep_token = dataset_params.pop("sep_token", default_sep_token)
        self.inverted_mask = dataset_params.pop("inverted_mask", False)
        self.min_sequence_len = dataset_params.pop("min_sequence_len", 10)
        self.max_seq_length = processing_params.pop("max_seq_length", 2048)
        self.chat_template = processing_params.pop("chat_template", None)
        if self.chat_template:
            self.tokenizer.chat_template = self.chat_template
        self.eos_id = eos_id
        self.eos_token = (
            self.tokenizer.convert_ids_to_tokens(self.pad_id)
            if self.eos_id is None
            else self.tokenizer.convert_ids_to_tokens(self.eos_id)
        )
        self.pad_id = pad_id
        self.sample_features = ["input_ids", "attention_mask", "labels"]
        self.semantic_loss_weight = processing_params.pop(
            "semantic_loss_weight", {}
        )
        self.semantic_drop_mask = processing_params.pop(
            "semantic_drop_mask", {}
        )
        self.end_of_turn_tok = processing_params.pop("end_of_turn_token", None)

    def clean_text(self, data: str) -> str:
        """
        Clean the provided text.

        Args:
            data (str): Text to clean.

        Returns:
            str: Cleaned text.
        """
        if self.use_ftfy:
            data = ftfy.fix_text(data, normalization=self.ftfy_normalizer)
        if self.wikitext_detokenize:
            data = wikitext_detokenizer(data)

        return data

    def get_tokenized_semantic_regions(
        self, formatted_data, tokenized_data, text_semantic_regions
    ):

        tokenized_semantic_region_list = []
        starting_offset_index = 0
        for text_semantic_region in text_semantic_regions:
            tokenized_semantic_region = find_token_range(
                text_semantic_region,
                tokenized_data["offset_mapping"],
                starting_offset_index,
            )
            start_token_idx, end_token_idx = tokenized_semantic_region[
                "indices"
            ]
            if text_semantic_region.get("handle_turn_token", False):
                tokenized_semantic_region["indices"] = (
                    start_token_idx,
                    end_token_idx + 1,
                )
            starting_offset_index = tokenized_semantic_region.get("indices")[1]
            tokenized_semantic_region_list.append(tokenized_semantic_region)
        return tokenized_semantic_region_list

    def get_data_stats(self, sample: np.ndarray) -> Dict[str, int]:
        """
        Get data statistics from the sample.

        Args:
            sample (np.ndarray): Tokenized sample.

        Returns:
            Dict[str, int]: Data statistics.
        """
        stats = defaultdict(int)
        if sample == []:
            return stats
        stats["num_pad_tokens"] = int((sample[0, :] == self.pad_id).sum())
        stats["non_pad_tokens"] = int(
            np.logical_and(
                sample[0, :] != self.eos_id, sample[0, :] != self.pad_id
            ).sum()
        )
        stats["loss_valid_tokens"] = int(sample[1, :].sum())
        stats["num_masked_tokens"] = (
            self.max_seq_length - stats["loss_valid_tokens"]
        )
        stats["num_tokens"] = int(sample[0, :].shape[0])

        return stats

    def parse_semantic_data_array(
        self, semantic_data_array: List[Dict[str, Any]]
    ) -> Tuple[Tuple[List[str], List[Dict[str, str]]], Dict[str, int]]:

        if not semantic_data_array:
            return {}, {}
        role = semantic_data_array[0].get("type")
        is_chat_data = not (role == "prompt" or role == "completion")
        if is_chat_data:
            conversation_data = []
        else:
            instruction_data = ""
        text_semantic_regions = []
        instruction_length = 0
        stats = {
            "raw_chars_count": 0,
            "raw_bytes_count": 0,
            "normalized_chars_count": 0,
            "normalized_bytes_count": 0,
            "total_raw_docs": 1,
            "raw_docs_skipped": 0,
        }

        global_idx = 0
        for turn in semantic_data_array:
            role = turn["type"]
            semantic_loss_weight = turn.get("semantic_loss_weight")
            semantic_drop_mask = turn.get("semantic_drop_mask")
            if semantic_loss_weight is not None:
                assert len(semantic_loss_weight) == len(
                    turn["content"]
                ), " The length of semantic loss mask must match the number of regions"
            if semantic_drop_mask is not None:
                assert len(semantic_drop_mask) == len(
                    turn["content"]
                ), " The length of semantic loss mask must match the number of regions"
            content_parts = []
            for i, part in enumerate(turn["content"]):

                include_tags = part.pop("include_tags", False)
                region_key = list(part.keys())[0]
                region_val = part.get(region_key)
                if not region_val:
                    logger.warning(
                        f"Missing {role} section in the data. Skipping this example "
                    )
                    stats["raw_docs_skipped"] = 1
                    return {}, stats
                stats["raw_chars_count"] += len(region_val)
                stats["raw_bytes_count"] += len(region_val.encode("utf-8"))
                cleaned_region_val = self.clean_text(region_val)
                stats["normalized_chars_count"] += len(cleaned_region_val)
                stats["normalized_bytes_count"] += len(
                    cleaned_region_val.encode("utf-8")
                )

                if not semantic_loss_weight:
                    loss_weight = self.semantic_loss_weight.get(region_key)
                    if not loss_weight:
                        ## set default weights
                        loss_weight = (
                            1
                            if (role == "assistant" or role == "completion")
                            else 0
                        )
                else:
                    loss_weight = semantic_loss_weight[i]
                if not semantic_drop_mask:
                    drop_region = self.semantic_drop_mask.get(region_key, False)
                else:
                    drop_region = semantic_drop_mask[i]

                if not drop_region:
                    if include_tags:
                        cleaned_region_val = (
                            f"<{region_key}>"
                            + cleaned_region_val
                            + f"</{region_key}>"
                        )
                    if not is_chat_data:
                        current_semantic_region = {
                            "indices": (
                                instruction_length,
                                instruction_length + len(cleaned_region_val),
                            ),
                            "region_modality": region_key,
                            "region_len": len(cleaned_region_val),
                            "loss_weight": loss_weight,
                        }
                        instruction_length += len(cleaned_region_val)
                        content = cleaned_region_val
                    else:
                        region_identifier = f"<{global_idx}_{region_key}>"
                        content = region_identifier + cleaned_region_val
                        current_semantic_region = {
                            "region_modality": region_key,
                            "region_identifier": region_identifier,
                            "region_len": len(cleaned_region_val),
                            "loss_weight": loss_weight,
                        }

                    text_semantic_regions.append(current_semantic_region)
                    content_parts.append(content)
                global_idx += 1

            content = ''.join(content_parts)

            if is_chat_data:
                conversation_data.append({"role": role, "content": content})
            else:
                if role == "prompt":
                    instruction_data = content + (
                        self.sep_token if self.sep_token else ""
                    )
                    instruction_length += (
                        len(self.sep_token) if self.sep_token else 0
                    )
                elif role == "completion":
                    instruction_data += content + (
                        self.eos_token if self.eos_token else ""
                    )
                    instruction_length += (
                        len(self.eos_token) if self.eos_token else 0
                    )

        if not is_chat_data:
            conversation_data = instruction_data
        transformed_data = {
            "conversation_data": conversation_data,
            "text_semantic_regions": text_semantic_regions,
            "is_chat_data": is_chat_data,
        }
        return transformed_data, stats

    def tokenize_data(self, semantic_data_array):

        data, raw_data_stats = self.parse_semantic_data_array(
            semantic_data_array
        )
        conversation_data, is_chat_data = data.get(
            "conversation_data"
        ), data.get("is_chat_data")
        text_semantic_regions = data.get("text_semantic_regions")
        if not conversation_data:
            return {}, raw_data_stats

        if is_chat_data:
            formatted_data = self.tokenizer.apply_chat_template(
                conversation_data, tokenize=False
            )
            formatted_data, text_semantic_regions = (
                find_region_in_formatted_string(
                    text_semantic_regions, formatted_data
                )
            )
            tokenized_data = self.tokenizer(
                formatted_data,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
        else:
            formatted_data = conversation_data
            tokenized_data = self.tokenizer(
                formatted_data,
                return_offsets_mapping=True,
            )
        text_semantic_regions = append_eos_to_multiple_semantic_regions(
            formatted_data,
            text_semantic_regions,
            self.end_of_turn_tok if self.end_of_turn_tok else self.eos_token,
            None,
            is_chat_data,
        )
        tokenized_semantic_region_list = self.get_tokenized_semantic_regions(
            formatted_data,
            tokenized_data,
            text_semantic_regions,
        )
        data = {
            "tokenized_data": tokenized_data,
            "tokenized_semantic_regions": tokenized_semantic_region_list,
        }
        return data, raw_data_stats

    def _encode(self, semantic_data_array):

        data, raw_data_stats = self.tokenize_data(semantic_data_array)

        if not data:
            return {}, raw_data_stats
        tokenized_conversation_data = data.get("tokenized_data")
        tokenized_semantic_regions = data.pop("tokenized_semantic_regions")
        sample = create_features_finetuning(
            tokenized_conversation_data,
            tokenized_semantic_regions,
            self.max_seq_length,
            min_len=self.min_sequence_len,
        )
        discarded_files = 0
        if sample == {}:
            discarded_files += 1
            data = {}
        else:
            data = {"data": sample}

        data_stats = {
            "total_raw_docs": 1,
            "raw_docs_skipped": 0,
            "discarded": discarded_files,
            "processed": 1,
            "successful": 1 - discarded_files,
            "raw_chars_count": raw_data_stats["raw_chars_count"],
            "raw_bytes_count": raw_data_stats["raw_bytes_count"],
            "normalized_chars_count": raw_data_stats["normalized_chars_count"],
            "normalized_bytes_count": raw_data_stats["normalized_bytes_count"],
        }

        return data, data_stats

    def encode(
        self, semantic_data_array: List[Dict]
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Tokenize and encode the doc for text summarization.

        Args:
            data (Dict): Contains a semantic data dict returned from a format hook

        Returns:
            -> Tuple[List[np.ndarray], Dict]: Tuple of encoded features for text summarization and dataset stats
        """

        data, raw_data_stats = self._encode(semantic_data_array)
        if data == {}:
            return {}, raw_data_stats

        padded_data = pad_to_msl(
            data.get("data"),
            self.max_seq_length,
            self.pad_id,
            inverted_mask=self.inverted_mask,
            input_ids_dtype=self.input_ids_dtype,
            input_mask_dtype=self.input_mask_dtype,
            labels_dtype=self.input_ids_dtype,
        )

        if data.get('attention_mask', None):
            self.sample_features.append('attention_mask')

        if data.get('attention_mask', None):
            self.sample_features.append('attention_mask')

        data = {"data": np.expand_dims(padded_data, axis=0)}

        tokenized_data_stats = self.get_data_stats(padded_data)
        data_stats = {
            "total_raw_docs": 1,
            "raw_docs_skipped": 0,
            "discarded": raw_data_stats["discarded"],
            "processed": 1,
            "successful": 1 - raw_data_stats["discarded"],
            "raw_chars_count": raw_data_stats["raw_chars_count"],
            "raw_bytes_count": raw_data_stats["raw_bytes_count"],
            "normalized_chars_count": raw_data_stats["normalized_chars_count"],
            "normalized_bytes_count": raw_data_stats["normalized_bytes_count"],
            "num_pad_tokens": tokenized_data_stats["num_pad_tokens"],
            "non_pad_tokens": tokenized_data_stats["non_pad_tokens"],
            "num_masked_tokens": tokenized_data_stats["num_masked_tokens"],
            "loss_valid_tokens": tokenized_data_stats["loss_valid_tokens"],
            "num_tokens": tokenized_data_stats["num_tokens"],
        }

        return data, data_stats
