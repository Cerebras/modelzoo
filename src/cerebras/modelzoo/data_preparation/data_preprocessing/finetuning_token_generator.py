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

import os
from typing import Any, Dict, List, Tuple

import numpy as np

from cerebras.modelzoo.data_preparation.data_preprocessing.utils import (
    append_eos_to_multiple_semantic_regions,
    clean_text,
    default_chat_template,
    find_region_in_formatted_string,
    find_token_range,
    get_data_stats,
    setup_warning_logging,
    truncate_sequence,
)


class FinetuningTokenGenerator:
    def __init__(self, params, tokenizer, eos_id, pad_id):
        dataset_params = params.get("dataset", {})
        processing_params = params.get("processing")
        setup_params = params.get("setup")
        warning_log_dir = os.path.join(
            setup_params.get("output_dir", "./output")
        )
        self.logger = setup_warning_logging(warning_log_dir, __name__)
        self.tokenizer = tokenizer
        self.is_multimodal = dataset_params.pop("is_multimodal", False)
        default_sep_token = (
            self.tokenizer.sep_token if self.tokenizer.sep_token else "<|sep|>"
        )
        self.sep_token = dataset_params.pop("sep_token", default_sep_token)
        self.chat_template = dataset_params.pop("chat_template", None)
        self.truncate_to_msl = dataset_params.pop("truncate_to_msl", None)
        self.use_vsl = dataset_params.pop("use_vsl", False)

        self.use_ftfy = processing_params.pop("use_ftfy", True)
        self.ftfy_normalizer = processing_params.pop("ftfy_normalizer", "NFC")
        self.wikitext_detokenize = processing_params.pop(
            "wikitext_detokenize", False
        )
        self.input_ids_dtype = processing_params.pop("input_ids_dtype", "int32")
        self.input_mask_dtype = processing_params.pop(
            "input_mask_dtype", "int32"
        )
        self.inverted_mask = processing_params.pop("inverted_mask", False)
        self.min_sequence_len = processing_params.pop("min_sequence_len", 10)
        self.max_seq_length = processing_params.pop("max_seq_length", 2048)

        if self.chat_template:
            self.tokenizer.chat_template = self.chat_template

        if self.tokenizer.chat_template is None:
            self.logger.warning(
                "Tokenizer doesn't have a chat template, setting a default chat template.."
            )
            self.tokenizer.chat_template = default_chat_template()

        self.eos_id = eos_id
        self.eos_token = (
            self.tokenizer.convert_ids_to_tokens(self.pad_id)
            if self.eos_id is None
            else self.tokenizer.convert_ids_to_tokens(self.eos_id)
        )
        self.pad_id = pad_id
        self.features = ["input_ids", "attention_mask", "labels"]
        self.semantic_loss_weight = processing_params.pop(
            "semantic_loss_weight", {}
        )
        self.semantic_drop_mask = processing_params.pop(
            "semantic_drop_mask", {}
        )
        self.end_of_turn_tok = processing_params.pop("end_of_turn_token", None)
        self.image_token = None
        self.semantic_attention_mask = processing_params.pop(
            "semantic_attention_mask", {}
        )
        if self.is_multimodal:
            self.features = [
                "text_input_ids",
                "loss_mask",
                "labels",
                "key_padding_mask",
                "token_modality_idx",
            ]
            self.image_token = dataset_params.pop(
                "image_token", "<special_image_token>"
            )
            self.image_dir = params["setup"].pop("image_dir", None)
            self.max_num_img = dataset_params.pop("max_num_img", 1)
            self.num_patches = dataset_params.pop("num_patches", 1)
            self.register_special_image_token = dataset_params.pop(
                "register_special_image_token", True
            )
            self.use_single_image_token = dataset_params.pop(
                "use_single_image_token", False
            )
            if self.use_single_image_token:
                self.num_patches = 1
                self.logger.info(
                    f"'num_patches' is set to 1 when using 'use_single_image_token'"
                )

            if (
                self.image_token
                and self.image_token not in self.tokenizer.get_vocab()
                and self.register_special_image_token
            ):
                self.tokenizer.add_special_tokens(
                    {'additional_special_tokens': [self.image_token]}
                )
            self.image_token_id = self.tokenizer.convert_tokens_to_ids(
                self.image_token
            )
            if self.use_single_image_token and self.image_token_id is None:
                raise ValueError(
                    f"Image token is not in tokenizer vocab. Please choose an existing token or register as special token."
                )

            self.image_ids = (
                [self.image_token_id]
                if self.use_single_image_token
                else [pad_id] * self.num_patches
            )
        else:
            self.features = ["input_ids", "attention_mask", "labels"]

        if self.truncate_to_msl:
            self.prompt_truncation_mode = self.truncate_to_msl.pop(
                "prompt_truncation_mode", None
            )
            self.max_turn_length = self.truncate_to_msl.pop(
                "max_turn_length", self.max_seq_length
            )

            if self.prompt_truncation_mode not in ['keep_start', 'keep_end']:
                self.logger.warning(
                    "Invalid truncation mode set - setting default mode as 'keep_end'."
                )
                self.prompt_truncation_mode = 'keep_end'

    def create_features_finetuning(
        self,
        tokenized_data,
        tokenized_semantic_region_list,
        truncation_params,
        return_attention_mask=False,
    ):

        token_ids = tokenized_data["input_ids"]
        total_len = len(token_ids)

        if total_len > self.max_seq_length:
            max_turn_length, truncate_to_msl, prompt_truncation_mode = (
                truncation_params
            )
            if self.truncate_to_msl is not None:
                tokenized_semantic_region_list, token_ids = truncate_sequence(
                    token_ids,
                    tokenized_semantic_region_list,
                    self.max_seq_length,
                    max_turn_length,
                    prompt_truncation_mode,
                )

                if len(token_ids) == 0:
                    self.logger.warning(
                        "Amount of truncation required is greater than what is available to truncate, skipping this example..."
                    )
                    return {}
            else:
                self.logger.warning(
                    "Length of token ids > max_sequence_len and truncation is not set, skipping this example..."
                )
                return {}

        if total_len < self.min_sequence_len:
            self.logger.warning(
                "Length of token ids < min_sequence_len, skipping this example..."
            )
            return {}

        def loss_mask_region():
            input_mask = [0] * len(token_ids)
            attention_mask = None
            if return_attention_mask:
                attention_mask = [1] * len(token_ids)
            for i, semantic_region in enumerate(tokenized_semantic_region_list):
                region_modality = semantic_region.get("region_modality")
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
                    and region_modality != "image"
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

    def pad_to_msl(self, data):

        token_ids, input_mask, attention_mask = (
            data.get("token_ids"),
            data.get("input_mask"),
            data.get("attention_mask", None),
        )
        input_ids = token_ids[:-1]
        labels = token_ids[1:]
        input_mask = input_mask[1:]
        # Calculate padding lengths
        num_pad = self.max_seq_length - len(input_ids)
        # Add padding
        input_ids.extend([self.pad_id] * num_pad)
        input_mask.extend([0] * num_pad)
        labels.extend([self.pad_id] * num_pad)
        if attention_mask is not None:
            num_pad = self.max_seq_length - len(attention_mask)
            attention_mask.extend([0] * num_pad)
            attention_mask = getattr(np, self.input_ids_dtype)(attention_mask)
            assert (
                len(attention_mask) == self.max_seq_length
            ), "Wrong sequence length"
            attention_mask = np.equal(attention_mask, 0).astype(
                self.input_mask_dtype
            )

        # Ensure lengths are consistent
        assert (
            len(input_ids) == self.max_seq_length
            and len(labels) == self.max_seq_length
            and len(input_mask) == self.max_seq_length
        ), "Wrong sequence length"

        # Create features dictionary
        features = {
            "input_ids": getattr(np, self.input_ids_dtype)(input_ids),
            "labels": getattr(np, self.input_ids_dtype)(labels),
        }
        input_mask = getattr(np, self.input_mask_dtype)(input_mask)

        if self.inverted_mask:
            input_mask = np.equal(input_mask, 0).astype(self.input_mask_dtype)

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
            return np.stack(
                [features["input_ids"], input_mask, features["labels"]]
            )

    def create_features_multimodal(
        self,
        data,
        token_modality_idx,
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
        num_pad = self.max_seq_length - len(input_ids)
        # Add padding
        padding = [self.pad_id] * num_pad
        input_ids.extend(padding)
        labels.extend(padding)

        padding = [0] * num_pad

        input_mask.extend(padding)

        num_pad = self.max_seq_length - len(attention_mask)
        attention_mask.extend([0] * num_pad)

        # Ensure lengths are consistent
        assert (
            len(input_ids) == self.max_seq_length
            and len(labels) == self.max_seq_length
            and len(input_mask) == self.max_seq_length
            and len(attention_mask) == self.max_seq_length
        ), "Wrong sequence length"

        # Create features dictionary
        features = {
            "input_ids": getattr(np, self.input_ids_dtype)(input_ids),
            "labels": getattr(np, self.input_ids_dtype)(labels),
        }
        input_mask = getattr(np, self.input_mask_dtype)(input_mask)

        if self.inverted_mask:
            input_mask = np.equal(input_mask, 0).astype(self.input_mask_dtype)
        attention_mask = getattr(np, self.input_ids_dtype)(attention_mask)
        key_padding_mask = np.equal(attention_mask, 0).astype(
            self.input_mask_dtype
        )

        return np.stack(
            [
                features["input_ids"],
                input_mask,
                features["labels"],
                key_padding_mask,
                token_modality_idx,
            ]
        )

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
            tokenized_semantic_region['role'] = text_semantic_region['role']
            if text_semantic_region.get("handle_turn_token", False):
                tokenized_semantic_region["indices"] = (
                    start_token_idx,
                    end_token_idx + 1,
                )
            starting_offset_index = tokenized_semantic_region.get("indices")[1]
            tokenized_semantic_region_list.append(tokenized_semantic_region)
        return tokenized_semantic_region_list

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
        image_paths = []
        image_regions = []
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
        for turn in semantic_data_array:
            role = turn["type"]
            semantic_loss_weight = turn.get("semantic_loss_weight")
            semantic_drop_mask = turn.get("semantic_drop_mask")
            semantic_attention_mask = turn.get("semantic_attention_mask")
            if semantic_loss_weight is not None and len(
                semantic_loss_weight
            ) != len(turn["content"]):
                raise ValueError(
                    " The length of semantic loss mask must match the number of regions"
                )
            if semantic_drop_mask is not None and len(
                semantic_drop_mask
            ) != len(turn["content"]):
                raise ValueError(
                    " The length of semantic drop mask must match the number of regions"
                )
            if semantic_attention_mask is not None and len(
                semantic_attention_mask
            ) != len(turn["content"]):
                raise ValueError(
                    " The length of semantic attention mask must match the number of regions"
                )
            content_parts = []
            for i, part in enumerate(turn["content"]):
                include_tags = part.pop("include_tags", False)
                region_key = list(part.keys())[0]
                region_val = part.get(region_key)
                if not region_val:
                    self.logger.warning(
                        f"Missing {role} section in the data. Skipping this example "
                    )
                    stats["raw_docs_skipped"] = 1
                    return {}, stats
                if region_key != "image":
                    cleaned_region_val = clean_text(
                        region_val,
                        self.use_ftfy,
                        self.wikitext_detokenize,
                        self.ftfy_normalizer,
                    )
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
                                role == "assistant"
                                or role == "completion"
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
                                "role": role,
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
                                "role": role,
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
                                "role": role,
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

        if self.is_multimodal:
            # Validate image paths
            for i, path in enumerate(image_paths):
                if path:
                    full_path = os.path.join(self.image_dir, path)
                    if not os.path.exists(full_path):
                        self.logger.warning(
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
        if self.is_multimodal:
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
                        self.logger.warning(
                            "Sample contains more images than max_num_img. Skipping this."
                        )
                        return {}, raw_data_stats
                    img_data_loc.append((image_start_pos, image_end_pos))
                    loss_weight, attention_mask = image_regions[
                        image_index
                    ].get("loss_weight"), image_regions[image_index].get(
                        "attention_mask"
                    )
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
            formatted_data,
            tokenized_data,
            text_semantic_regions,
        )
        if self.is_multimodal:
            tokenized_semantic_region_list.extend(image_indices)
        data = {
            "tokenized_data": tokenized_data,
            "image_paths": image_paths if self.is_multimodal else None,
            "img_data_loc": img_data_loc if self.is_multimodal else None,
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

        if self.truncate_to_msl is not None:
            truncation_params = (
                self.max_turn_length,
                self.truncate_to_msl,
                self.prompt_truncation_mode,
            )
        else:
            truncation_params = (None, None, None)

        sample = self.create_features_finetuning(
            tokenized_conversation_data,
            tokenized_semantic_regions,
            truncation_params,
            return_attention_mask=self.is_multimodal,
        )
        discarded_files = 0
        if sample == {}:
            discarded_files += 1
            data = {}
        else:
            if self.is_multimodal:
                data = {
                    "data": sample,
                    "img_path": image_paths,
                    "img_data_loc": data.get("img_data_loc"),
                }
            else:
                data = {"data": sample}

        data_stats = {
            "total_raw_docs": 1,
            "raw_docs_skipped": 0,
            "discarded_files": discarded_files,
            "processed_files": 1,
            "successful_files": 1 - discarded_files,
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

        if not self.is_multimodal:
            padded_data = self.pad_to_msl(
                data.get("data"),
            )
            data = {"data": np.expand_dims(padded_data, axis=0)}
        else:
            token_modality_idx = np.zeros(self.max_seq_length)
            image_data_positions = data.get("img_data_loc")
            img_data_loc = np.full(
                (1, self.max_num_img, self.num_patches), self.max_seq_length
            )
            for i, (start_img_pos, end_img_pos) in enumerate(
                image_data_positions
            ):
                img_data_loc[0, i] = np.arange(start_img_pos, end_img_pos)
                token_modality_idx[start_img_pos:end_img_pos] = 1

            padded_data = self.create_features_multimodal(
                data.get("data"),
                token_modality_idx,
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

        tokenized_data_stats = get_data_stats(
            padded_data, self.pad_id, self.eos_id, self.max_seq_length
        )
        data_stats = tokenized_data_stats.copy()
        data_stats.update(raw_data_stats)

        return data, data_stats
