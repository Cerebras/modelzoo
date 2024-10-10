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
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from cerebras.modelzoo.data_preparation.data_preprocessing.utils import (
    append_eos_to_multiple_semantic_regions,
    find_region_in_formatted_string,
)

from cerebras.modelzoo.data_preparation.data_preprocessing.finetuning_token_generator import (  # noqa
    FinetuningTokenGenerator,
    create_features_finetuning,
)

logger = logging.getLogger(__name__)


def create_features_multimodal(
    data,
    token_modality_idx,
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
    padding = [pad_id] * num_pad
    input_ids.extend(padding)
    labels.extend(padding)

    padding = [0] * num_pad

    input_mask.extend(padding)

    num_pad = max_sequence_length - len(attention_mask)
    attention_mask.extend([0] * num_pad)

    # Ensure lengths are consistent
    assert (
        len(input_ids) == max_sequence_length
        and len(labels) == max_sequence_length
        and len(input_mask) == max_sequence_length
        and len(attention_mask) == max_sequence_length
    ), "Wrong sequence length"

    # Create features dictionary
    features = {
        "input_ids": getattr(np, input_ids_dtype)(input_ids),
        "labels": getattr(np, labels_dtype)(labels),
    }
    input_mask = getattr(np, input_mask_dtype)(input_mask)

    if inverted_mask:
        input_mask = np.equal(input_mask, 0).astype(input_mask_dtype)
    attention_mask = getattr(np, input_ids_dtype)(attention_mask)
    key_padding_mask = np.equal(attention_mask, 0).astype(input_mask_dtype)

    return np.stack(
        [
            features["input_ids"],
            input_mask,
            features["labels"],
            key_padding_mask,
            token_modality_idx,
        ]
    )


class MultiModalFinetuningTokenGenerator(FinetuningTokenGenerator):
    def __init__(self, params, tokenizer, eos_id, pad_id):
        super(MultiModalFinetuningTokenGenerator, self).__init__(
            params, tokenizer, eos_id, pad_id
        )
        dataset_params = params["dataset"]
        processing_params = params["processing"]

        self.sample_features = [
            "input_ids",
            "attention_mask",
            "labels",
            "key_padding_mask",
            "token_modality_idx",
        ]
        self.image_token = dataset_params.pop(
            "image_token", "<special_image_token>"
        )
        self.image_dir = dataset_params.pop("image_dir", None)
        self.max_num_img = dataset_params.pop("max_num_img", 1)
        self.num_patches = dataset_params.pop("num_patches", 1)
        self.image_token_id = -1
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': [self.image_token]}
        )
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(
            self.image_token
        )
        self.sample_features = [
            "input_ids",
            "attention_mask",
            "labels",
            "key_padding_mask",
            "token_modality_idx",
        ]
        self.image_ids = [
            pad_id
        ] * self.num_patches  # Hardcoded to pad_id for now
        self.semantic_attention_mask = processing_params.pop(
            "semantic_attention_mask", {}
        )

    def parse_semantic_data_array(
        self, data: List[Dict[str, Any]]
    ) -> Tuple[Tuple[List[str], List[Dict[str, str]]], Dict[str, int]]:

        if not data:
            return {}, {}
        image_paths = []
        role = data[0].get("type")
        is_chat_data = not (role == "prompt" or role == "completion")
        if is_chat_data:
            conversation_data = []
        else:
            instruction_data = ""
        image_regions = []
        text_semantic_regions = []
        stats = {
            "raw_chars_count": 0,
            "raw_bytes_count": 0,
            "normalized_chars_count": 0,
            "normalized_bytes_count": 0,
            "total_raw_docs": 1,
            "raw_docs_skipped": 0,
        }
        global_idx = 0
        instruction_length = 0
        for turn in data:
            role = turn["type"]
            semantic_loss_weight = turn.get("semantic_loss_weight")

            semantic_drop_mask = turn.get("semantic_drop_mask")
            semantic_attention_mask = turn.get("semantic_attention_mask")
            if semantic_loss_weight is not None:
                assert len(semantic_loss_weight) == len(
                    turn["content"]
                ), " The length of semantic loss mask must match the number of regions"
            if semantic_drop_mask is not None:
                assert len(semantic_drop_mask) == len(
                    turn["content"]
                ), " The length of semantic loss mask must match the number of regions"
            if semantic_attention_mask is not None:
                assert len(semantic_attention_mask) == len(
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

                if region_key != "image":
                    cleaned_region_val = self.clean_text(region_val)
                    stats["raw_chars_count"] += len(region_val)
                    stats["raw_bytes_count"] += len(region_val.encode("utf-8"))
                    stats["normalized_chars_count"] += len(cleaned_region_val)
                    stats["normalized_bytes_count"] += len(
                        cleaned_region_val.encode("utf-8")
                    )
                else:
                    cleaned_region_val = region_val

                if not semantic_loss_weight:
                    loss_weight = self.semantic_loss_weight.get(region_key)
                    if not loss_weight:
                        ## set default weights
                        loss_weight = (
                            1
                            if (
                                (role == "assistant" or role == "completion")
                                and region_key != "image"
                            )
                            else 0
                        )
                else:
                    loss_weight = semantic_loss_weight[i]
                if not semantic_drop_mask:
                    drop_region = self.semantic_drop_mask.get(region_key, False)
                else:
                    drop_region = semantic_drop_mask[i]
                if not semantic_attention_mask:
                    attention_mask = self.semantic_attention_mask.get(
                        region_key, True
                    )
                else:
                    attention_mask = semantic_attention_mask[i]
                attention_mask = 1 if attention_mask else 0

                if region_key != "image":
                    if not drop_region and cleaned_region_val != "":
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
                                    instruction_length
                                    + len(cleaned_region_val),
                                ),
                                "region_modality": region_key,
                                "region_len": len(cleaned_region_val),
                                "loss_weight": loss_weight,
                                "attention_mask": attention_mask,
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
                                "attention_mask": attention_mask,
                            }

                        text_semantic_regions.append(current_semantic_region)
                        content_parts.append(content)
                else:
                    if not drop_region:
                        image_regions.append(
                            {
                                "region_modality": region_key,
                                "loss_weight": loss_weight,
                                "attention_mask": attention_mask,
                            }
                        )
                        image_paths.append(cleaned_region_val)
                        if include_tags:
                            content = (
                                f"<{region_key}>"
                                + self.image_token
                                + f"</{region_key}>"
                            )
                        else:
                            content = self.image_token
                        instruction_length += len(content)
                        content_parts.append(content)
                global_idx += 1
            content = ''.join(content_parts)
            if is_chat_data:
                conversation_data.append({"role": role, "content": content})
            else:
                if role == "prompt":
                    instruction_data = (
                        content + self.sep_token if self.sep_token else ""
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

        # Validate image paths
        for i, path in enumerate(image_paths):
            if path:
                full_path = os.path.join(self.image_dir, path)
                if not os.path.exists(full_path):
                    logger.warning(
                        f"Image with path - {full_path} does not exist. Hence skipping this."
                    )
                    stats["raw_docs_skipped"] = 1
                    return {}, stats
                else:
                    image_paths[i] = path.encode(encoding='utf-8')

        if not is_chat_data:
            conversation_data = instruction_data

        transformed_data = {
            "conversation_data": conversation_data,
            "image_paths": image_paths,
            "text_semantic_regions": text_semantic_regions,
            "image_regions": image_regions,
            "is_chat_data": is_chat_data,
        }
        return transformed_data, stats

    def tokenize_data(self, semantic_data_array):

        data, raw_data_stats = self.parse_semantic_data_array(
            semantic_data_array
        )
        conversation_data, image_paths, is_chat_data = (
            data.get("conversation_data"),
            data.get("image_paths"),
            data.get("is_chat_data"),
        )
        text_semantic_regions, image_regions = data.get(
            "text_semantic_regions"
        ), data.get("image_regions", [])
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
            self.image_token,
            is_chat_data,
        )
        new_input_ids = []
        new_offset_mapping = []
        new_attention_mask = []
        image_indices = []
        img_data_loc = []
        image_index = 0
        for id, offset, attention in zip(
            tokenized_data["input_ids"],
            tokenized_data['offset_mapping'],
            tokenized_data["attention_mask"],
        ):
            if id == self.image_token_id:
                new_input_ids.extend(self.image_ids)
                new_offset_mapping.extend([offset] * len(self.image_ids))
                new_attention_mask.extend([1] * len(self.image_ids))
                image_end_pos = len(new_input_ids)
                image_start_pos = image_end_pos - len(self.image_ids)
                if len(img_data_loc) >= self.max_num_img:
                    logger.warning(
                        "Sample contains more images than max_num_img. Skipping this."
                    )
                    return {}, raw_data_stats
                img_data_loc.append((image_start_pos, image_end_pos))
                loss_weight, attention_mask = image_regions[image_index].get(
                    "loss_weight"
                ), image_regions[image_index].get("attention_mask")
                image_indices.append(
                    {
                        "indices": (image_start_pos, image_end_pos),
                        "loss_weight": loss_weight,
                        "attention_mask": attention_mask,
                    }
                )
                image_index += 1
            else:
                new_input_ids.append(id)
                new_offset_mapping.append(offset)
                new_attention_mask.append(attention)
        tokenized_data['input_ids'] = new_input_ids
        tokenized_data['offset_mapping'] = new_offset_mapping
        tokenized_data['attention_mask'] = new_attention_mask
        tokenized_semantic_region_list = self.get_tokenized_semantic_regions(
            formatted_data, tokenized_data, text_semantic_regions
        )
        tokenized_semantic_region_list.extend(image_indices)
        data = {
            "tokenized_data": tokenized_data,
            "image_paths": image_paths,
            "img_data_loc": img_data_loc,
            "tokenized_semantic_regions": tokenized_semantic_region_list,
        }

        return data, raw_data_stats

    def _encode(self, semantic_data_array):

        data, raw_data_stats = self.tokenize_data(semantic_data_array)
        if not data:
            return {}, raw_data_stats
        tokenized_conversation_data, image_paths = data.get(
            "tokenized_data"
        ), data.get("image_paths")
        tokenized_semantic_regions = data.pop("tokenized_semantic_regions")
        sample = create_features_finetuning(
            tokenized_conversation_data,
            tokenized_semantic_regions,
            self.max_seq_length,
            return_attention_mask=True,
            min_len=self.min_sequence_len,
        )

        discarded_files = 0
        if sample == {}:
            discarded_files += 1
            data = {}
        else:
            data = {
                "data": sample,
                "img_path": image_paths,
                "img_data_loc": data.get("img_data_loc"),
            }

        data_stats = {
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
        token_modality_idx = np.zeros(self.max_seq_length)
        image_data_positions = data.get("img_data_loc")
        img_data_loc = np.full(
            (1, self.max_num_img, self.num_patches), self.max_seq_length
        )
        for i, (start_img_pos, end_img_pos) in enumerate(image_data_positions):
            img_data_loc[0, i] = list(range(start_img_pos, end_img_pos))
            token_modality_idx[start_img_pos:end_img_pos] = 1

        padded_data = create_features_multimodal(
            data.get("data"),
            token_modality_idx,
            self.max_seq_length,
            self.pad_id,
            inverted_mask=self.inverted_mask,
            input_ids_dtype=self.input_ids_dtype,
            input_mask_dtype=self.input_mask_dtype,
            labels_dtype=self.input_ids_dtype,
        )

        has_img = False
        image_paths = data.get("img_path", [])
        if image_paths:
            num_images = len(image_paths)
            image_paths += [None] * (self.max_num_img - num_images)
            has_img = True
        else:
            image_paths = [None] * (self.max_num_img)

        data = {
            "data": np.expand_dims(padded_data, axis=0),
            "img_path": np.array(image_paths, dtype="S").reshape(1, -1),
            "has_img": np.array([[has_img]], dtype=np.bool_),
            "img_data_loc": img_data_loc,
        }

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
