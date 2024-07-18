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
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from cerebras.modelzoo.data_preparation.data_preprocessing.pretraining_token_generator import (
    PretrainingTokenGenerator,
)
from cerebras.modelzoo.data_preparation.data_preprocessing.utils import (
    append_eos_to_multiple_semantic_regions,
    find_token_range,
)

logger = logging.getLogger(__name__)


def create_features_multimodal_pretraining(
    doc,
    token_modality_idx,
    max_sequence_length,
    pad_id,
    eos_id,
    min_len=10,
    inverted_mask=False,
    input_ids_dtype="int32",
    input_mask_dtype="int32",
    labels_dtype="int32",
):
    tokenized_semantic_region_list = doc.get("tokenized_semantic_regions")
    token_ids = doc.get("token_ids")

    total_len = len(token_ids)

    if total_len < min_len:
        logger.warning(
            "Length of token ids < min_sequence_len, skipping this example..."
        )
        return []

    def loss_mask_region():
        input_mask = [0] * len(token_ids)
        attention_mask = [1] * len(token_ids)
        for i, semantic_region in enumerate(tokenized_semantic_region_list):
            region_name = semantic_region.get("region_name")
            start_idx, end_idx = semantic_region.get("indices")
            region_loss_mask = semantic_region.get("loss_weight", 0)
            region_attention_mask = semantic_region.get("attention_mask", 1)
            for idx in range(start_idx, end_idx):
                if idx >= len(token_ids):
                    break
                input_mask[idx] = region_loss_mask
                attention_mask[idx] = region_attention_mask
            if (
                i == len(tokenized_semantic_region_list) - 1
                and region_name != "image"
            ):
                attention_mask = attention_mask[:-1]

        return input_mask, attention_mask

    input_mask, attention_mask = loss_mask_region()
    input_ids = token_ids[:-1]
    labels = token_ids[1:]
    input_mask = input_mask[1:]

    # Add padding
    num_pad = max_sequence_length - len(input_ids)
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
    attention_mask = getattr(np, input_ids_dtype)(attention_mask)
    if inverted_mask:
        input_mask = np.equal(input_mask, 0).astype(input_mask_dtype)

    # NOTE this is because our internal stack requires the inverted mask and
    # doesn't do the inversion internally
    key_padding_mask = np.equal(attention_mask, 0).astype(input_mask.dtype)

    return np.stack(
        [
            features["input_ids"],
            input_mask,
            features["labels"],
            key_padding_mask,
            token_modality_idx,
        ]
    )


class MultiModalPretrainingTokenGenerator(PretrainingTokenGenerator):
    def __init__(self, params, tokenizer, eos_id, pad_id):
        super(MultiModalPretrainingTokenGenerator, self).__init__(
            params, tokenizer, eos_id, pad_id
        )
        dataset_params = params["dataset"]
        processing_params = params["processing"]
        self.image_token = dataset_params.pop(
            "image_token", "<special_image_token>"
        )
        self.image_dir = dataset_params.pop("image_dir", None)
        self.max_num_img = dataset_params.pop("max_num_img", 1)
        self.num_patches = dataset_params.pop("num_patches", 1)
        self.image_token_id = -1
        if (
            self.image_token
            and self.image_token not in self.tokenizer.get_vocab()
        ):
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
        self.semantic_loss_weight = processing_params.pop(
            "semantic_loss_weight", {}
        )
        self.semantic_drop_mask = processing_params.pop(
            "semantic_drop_mask", {}
        )
        self.semantic_attention_mask = processing_params.pop(
            "semantic_attention_mask", {}
        )
        self.include_image_tag = False
        self.data_ranges = []
        self.eos_token = (
            self.tokenizer.pad_token_id
            if self.eos_id is None
            else self.tokenizer.convert_ids_to_tokens(self.eos_id)
        )

    def get_data_ranges(
        self, semantic_regions, formatted_data: str
    ) -> Tuple[
        List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]
    ]:
        """
        Get data ranges for the conversation data.

        Args:
            conversation_data (List[Dict[str, str]]): List of conversation data.
            formatted_data (str): Formatted conversation data.

        Returns:
            Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]: Ranges for system, user, and assistant data.
        """
        lower = self.tokenizer.init_kwargs.get('do_lower_case', False)
        formatted_data = formatted_data.lower() if lower else formatted_data
        string_search_idx = 0
        for content in semantic_regions:
            region_name = content.get("region_name")
            region_identifier = content.get("region_identifier", "")
            region_len = content.get("region_len")
            loss_weight = content.get("loss_weight")
            attention_mask = content.get("attention_mask", None)
            region_identifier_start_idx = formatted_data.find(
                region_identifier.lower() if lower else region_identifier,
                string_search_idx,
            )
            formatted_data = formatted_data.replace(region_identifier, "")
            start_idx = region_identifier_start_idx
            end_idx = start_idx + region_len
            string_search_idx = end_idx
            self.data_ranges.append(
                {
                    "region_name": region_name,
                    "indices": (start_idx, end_idx),
                    "loss_weight": loss_weight,
                    "attention_mask": attention_mask,
                }
            )

        return formatted_data

    def chop_doc_into_msl(self, data):

        doc_list = []
        image_index = 0
        curr_doc = []
        curr_img_paths = []
        start_token_idx = 0

        tokenized_text_data, image_paths = data.get("tokenized_data"), data.get(
            "image_paths"
        )
        tokenized_semantic_regions = data.get("tokenized_semantic_regions")
        start_doc_idx = tokenized_semantic_regions[0].get("indices")[0]
        input_ids = []
        has_img = False
        image_data_positions = []

        image_start_tokens, image_end_tokens = [], []
        if self.include_image_tag:
            image_start_tokens = self.tokenizer(
                f"<{image}>", add_special_tokens=False
            )
            image_end_tokens = self.tokenizer(
                f"</{image}>", add_special_tokens=False
            )

        for region in tokenized_semantic_regions:
            region_start_idx, region_end_idx = region.get(
                "indices"
            )  ## big doc indices
            len_region_remaining = region_end_idx - region_start_idx
            region_name = region.get("region_name")
            loss_weight = region.get("loss_weight")
            region_attention_mask = region.get("attention_mask")
            if region_name != "image":
                ## Chop text into multiple doc's
                while (
                    self.max_seq_length + 1
                ) <= start_doc_idx + len_region_remaining:

                    end_token_idx = (
                        start_token_idx
                        + (self.max_seq_length + 1)
                        - start_doc_idx
                    )
                    indices = (start_doc_idx, self.max_seq_length + 1)
                    input_ids.extend(
                        tokenized_text_data.get("input_ids")[
                            start_token_idx:end_token_idx
                        ]
                    )
                    curr_doc.append(
                        {
                            "region_name": region_name,
                            "indices": indices,
                            "loss_weight": loss_weight,
                            "attention_mask": region_attention_mask,
                        }
                    )
                    doc_list.append(
                        {
                            "token_ids": input_ids,
                            "tokenized_semantic_regions": curr_doc,
                            "image_paths": curr_img_paths,
                            "image_data_positions": image_data_positions,
                            "has_img": has_img,
                        }
                    )
                    curr_img_paths = []
                    image_data_positions = []
                    has_img = False
                    len_region_remaining -= (
                        self.max_seq_length + 1
                    ) - start_doc_idx
                    start_doc_idx = 0
                    start_token_idx = end_token_idx
                    curr_doc = []
                    input_ids = []
                if len_region_remaining < (self.max_seq_length + 1):

                    indices = (
                        start_doc_idx,
                        start_doc_idx + len_region_remaining,
                    )
                    end_token_idx = start_token_idx + len_region_remaining
                    input_ids.extend(
                        tokenized_text_data.get("input_ids")[
                            start_token_idx:end_token_idx
                        ]
                    )
                    curr_doc.append(
                        {
                            "region_name": region_name,
                            "indices": indices,
                            "loss_weight": loss_weight,
                            "attention_mask": region_attention_mask,
                        }
                    )
                    start_doc_idx = (start_doc_idx + len_region_remaining) % (
                        self.max_seq_length + 1
                    )
                    start_token_idx = end_token_idx
            else:
                image_path = image_paths[image_index]
                has_img = True
                ## Check if image or other regions can fit in the previous partially filled  region
                if (
                    start_doc_idx
                    + len_region_remaining
                    + len(image_start_tokens)
                    + len(image_end_tokens)
                    < self.max_seq_length + 1
                ):
                    if self.include_image_tag:
                        start_doc_idx += len(image_start_tokens)
                        input_ids.extend(image_start_tokens)
                    indices = (
                        start_doc_idx,
                        (start_doc_idx + len_region_remaining),
                    )
                    start_doc_idx += len_region_remaining
                    input_ids.extend(
                        tokenized_text_data.get("input_ids")[
                            start_token_idx:region_end_idx
                        ]
                    )
                    if self.include_image_tag:
                        start_doc_idx += len(image_end_tokens)
                        input_ids.extend(image_end_tokens)
                    image_data_positions.append((indices[0], indices[1]))
                    curr_doc.append(
                        {
                            "region_name": region_name,
                            "indices": indices,
                            "loss_weight": loss_weight,
                            "attention_mask": region_attention_mask,
                        }
                    )
                    curr_img_paths.append(image_path)
                    start_token_idx = region_end_idx
                else:
                    if curr_doc != []:
                        doc_list.append(
                            {
                                "token_ids": input_ids,
                                "tokenized_semantic_regions": curr_doc,
                                "image_paths": curr_img_paths,
                                "image_data_positions": image_data_positions,
                                "has_img": has_img,
                            }
                        )
                    curr_doc = []
                    image_data_positions = []
                    input_ids = []

                    has_img = True
                    curr_img_paths = [image_path]
                    start_doc_idx = 0
                    if self.include_image_tag:
                        start_doc_idx += len(image_start_tokens)
                        input_ids.extend(image_start_tokens)
                    assert (
                        len_region_remaining <= self.max_seq_length + 1
                    ), f"{region_name} region and the region tags if included cannot be split into multiple msl's. Increase the msl or decrease the length of region "
                    indices = (
                        start_doc_idx,
                        start_doc_idx + len_region_remaining,
                    )

                    image_data_positions.append((indices[0], indices[1]))
                    start_doc_idx += len_region_remaining
                    curr_doc.append(
                        {
                            "region_name": region_name,
                            "indices": indices,
                            "loss_weight": loss_weight,
                            "attention_mask": region_attention_mask,
                        }
                    )
                    input_ids.extend(
                        tokenized_text_data.get("input_ids")[
                            start_token_idx:region_end_idx
                        ]
                    )
                    if self.include_image_tag:
                        start_doc_idx += len(image_end_tokens)
                        input_ids.extend(image_end_tokens)
                    start_doc_idx = start_doc_idx % (self.max_seq_length + 1)
                    start_token_idx = region_end_idx
                image_index += 1
        if curr_doc != []:
            doc_list.append(
                {
                    "token_ids": input_ids,
                    "tokenized_semantic_regions": curr_doc,
                    "image_paths": curr_img_paths,
                    "image_data_positions": image_data_positions,
                    "has_img": has_img,
                }
            )
        return doc_list

    def get_segment_indices(
        self,
        formatted_data,
        tokenized_data: List[Tuple[int, int]],
        image_region_list: List,
    ):
        """
        Get segment indices for the data ranges.

        Args:
            data_ranges (Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]): Data ranges for system, user, and assistant.
            offset_mapping (List[Tuple[int, int]]): Offset mapping of the tokenized data.
        """
        tokenized_semantic_region_list = []
        image_index = 0
        text_index = 0
        tokenized_semantic_region = None
        while text_index < len(self.data_ranges) or image_index < len(
            image_region_list
        ):
            if text_index < len(self.data_ranges) and image_index < len(
                image_region_list
            ):
                if not tokenized_semantic_region:
                    text_data_range = self.data_ranges[text_index]
                    region_name = text_data_range.get("region_name")
                    tokenized_semantic_region = find_token_range(
                        text_data_range, tokenized_data["offset_mapping"]
                    )
                    tokenized_semantic_region["region_name"] = region_name
                image_region = image_region_list[image_index]
                if (
                    tokenized_semantic_region.get("indices")[1]
                    <= image_region.get("indices")[0]
                ):  ## text end index less than image start index
                    tokenized_semantic_region_list.append(
                        tokenized_semantic_region
                    )
                    tokenized_semantic_region = None
                    text_index += 1
                else:
                    tokenized_semantic_region_list.append(image_region)
                    image_index += 1
            elif text_index < len(self.data_ranges):
                if not tokenized_semantic_region:
                    text_data_range = self.data_ranges[text_index]
                    region_name = text_data_range.get("region_name")
                    tokenized_semantic_region = find_token_range(
                        text_data_range, tokenized_data["offset_mapping"]
                    )
                    tokenized_semantic_region["region_name"] = region_name
                tokenized_semantic_region_list.append(tokenized_semantic_region)
                tokenized_semantic_region = None
                text_index += 1
            elif image_index < len(image_region_list):
                image_region = image_region_list[image_index]
                tokenized_semantic_region_list.append(image_region)
                image_index += 1

        return tokenized_semantic_region_list

    def parse_semantic_data_array(
        self, semantic_data_array: List[Dict[str, Any]]
    ) -> Tuple[Tuple[List[str], List[Dict[str, str]]], Dict[str, int]]:

        image_paths = []
        image_regions = []
        text_semantic_regions = []
        self.data_ranges = []
        stats = {
            "raw_chars_count": 0,
            "raw_bytes_count": 0,
            "normalized_chars_count": 0,
            "normalized_bytes_count": 0,
        }

        formatted_data = ""

        for entry in semantic_data_array:
            semantic_loss_weight = entry.get("semantic_loss_weight")
            semantic_drop_mask = entry.get("semantic_drop_mask")
            semantic_attention_mask = entry.get("semantic_attention_mask")
            if semantic_loss_weight is not None:
                assert len(semantic_loss_weight) == len(
                    entry["content"]
                ), " The length of semantic loss mask must match the number of regions"
            if semantic_drop_mask is not None:
                assert len(semantic_drop_mask) == len(
                    entry["content"]
                ), " The length of semantic loss mask must match the number of regions"
            if semantic_attention_mask is not None:
                assert len(semantic_attention_mask) == len(
                    entry["content"]
                ), " The length of semantic loss mask must match the number of regions"
            content_parts = []
            global_idx = 0
            for i, part in enumerate(entry["content"]):

                region_key = list(part.keys())[0]
                region_val = part[region_key]
                if not region_val:
                    continue
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

                include_tags = part.pop("include_tags", False)
                if not semantic_loss_weight:
                    loss_weight = self.semantic_loss_weight.get(region_key)
                    if not loss_weight:
                        ## set default weights
                        loss_weight = 1 if region_key != "image" else 0
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
                if region_key != "image":  ## hardcoding name of image
                    if not drop_region and cleaned_region_val != "":
                        if include_tags:
                            cleaned_region_val = (
                                f"<{region_key}>"
                                + cleaned_region_val
                                + f"</{region_key}>"
                            )
                        region_identifier = f"<{global_idx}_{region_key}>"
                        text_semantic_regions.append(
                            {
                                "region_name": region_key,
                                "region_identifier": region_identifier,
                                "region_len": len(cleaned_region_val),
                                "loss_weight": loss_weight,
                                "attention_mask": attention_mask,
                            }
                        )
                        content = region_identifier + cleaned_region_val
                        content_parts.append(content)

                else:
                    self.include_image_tag = include_tags
                    if not drop_region:
                        image_regions.append(
                            {
                                "region_name": region_key,
                                "loss_weight": loss_weight,
                                "attention_mask": attention_mask,
                            }
                        )
                        image_paths.append(cleaned_region_val)
                        content = self.image_token
                        content_parts.append(content)
                global_idx += 1

            formatted_data += ''.join(content_parts)

        # Validate image paths
        for i, path in enumerate(image_paths):
            if path:
                full_path = os.path.join(self.image_dir, path)
                if not os.path.exists(full_path):
                    logger.warning(
                        f"Image with path - {full_path} does not exist. Hence skipping this."
                    )
                    return None, stats
                else:
                    image_paths[i] = path.encode(encoding='utf-8')
        transformed_data = {
            "text_data": formatted_data,
            "image_paths": image_paths,
            "text_semantic_regions": text_semantic_regions,
            "image_regions": image_regions,
        }
        return transformed_data, stats

    def tokenize_data(self, semantic_data_array):

        data, raw_data_stats = self.parse_semantic_data_array(
            semantic_data_array
        )
        if not data:
            return {}, raw_data_stats
        text_data, image_paths = (
            data.get("text_data"),
            data.get("image_paths"),
        )
        text_semantic_regions, image_regions = data.get(
            "text_semantic_regions"
        ), data.get("image_regions", [])
        image_indices = []

        if text_data == "":
            return {}, raw_data_stats

        text_data = self.get_data_ranges(text_semantic_regions, text_data)

        tokenized_data = self.tokenizer(
            text_data,
            return_offsets_mapping=True,
        )

        if len(self.data_ranges) > 0:
            append_eos_to_multiple_semantic_regions(
                text_data,
                self.data_ranges,
                self.eos_token,
                self.image_token,
                False,
            )

        new_input_ids = []
        new_offset_mapping = []
        new_attention_mask = []
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
                loss_weight, attention_mask = image_regions[image_index].get(
                    "loss_weight"
                ), image_regions[image_index].get("attention_mask")
                image_indices.append(
                    {
                        "region_name": "image",
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

        tokenized_semantic_region_list = self.get_segment_indices(
            text_data, tokenized_data, image_indices
        )
        data = {
            "tokenized_data": tokenized_data,
            "image_paths": image_paths,
            "tokenized_semantic_regions": tokenized_semantic_region_list,
        }
        return data, raw_data_stats

    def process_docs(self, doc_list):

        results = defaultdict(list)
        tokenized_data_stats = defaultdict(int)
        for doc_idx, doc in enumerate(doc_list):
            has_img = False
            if doc.get("token_ids", []) == []:
                tokenized_data_stats["discarded"] += 1
                continue
            image_paths, image_data_positions = doc.pop("image_paths"), doc.pop(
                "image_data_positions"
            )
            has_img = doc.pop("has_img")

            token_modality_idx = np.zeros(self.max_seq_length)
            img_data_loc = np.full(
                (self.max_num_img, self.num_patches), self.max_seq_length
            )
            image_index = 0
            for start_img_pos, end_img_pos in image_data_positions:
                if self.max_num_img <= image_index:
                    break
                img_data_loc[image_index] = list(
                    range(start_img_pos, end_img_pos)
                )
                token_modality_idx[start_img_pos:end_img_pos] = 1
                image_index += 1
            if self.max_num_img <= image_index:
                tokenized_data_stats["discarded"] += 1
                logger.warning(
                    "Sequence has more images than maximum allowed images. Skipping this"
                )
                continue

            sample = create_features_multimodal_pretraining(
                doc,
                token_modality_idx,
                self.max_seq_length,
                self.pad_id,
                self.eos_id,
                min_len=self.min_sequence_len,
                inverted_mask=self.inverted_mask,
                input_ids_dtype=self.input_ids_dtype,
                input_mask_dtype=self.input_mask_dtype,
                labels_dtype=self.input_ids_dtype,
            )
            if sample == []:
                tokenized_data_stats["processed"] += 1
                tokenized_data_stats["discarded"] += 1
                continue

            if image_paths:
                num_images = len(image_paths)
                image_paths += [None] * (self.max_num_img - num_images)
                has_img = True
            else:
                image_paths = [None] * (self.max_num_img)

            sample_stats = self.get_data_stats(sample)
            for key in sample_stats:
                tokenized_data_stats[key] += sample_stats[key]
            tokenized_data_stats["processed"] += 1
            tokenized_data_stats["successful"] += 1
            data = {
                "data": sample,
                "img_path": np.array(image_paths, dtype="S"),
                "has_img": np.array([has_img], dtype=np.bool_),
                "img_data_loc": img_data_loc,
            }
            for key, value in data.items():
                results[key].append(value)
        return results, tokenized_data_stats

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
        data, raw_data_stats = self.tokenize_data(semantic_data_array)
        if not data:
            return {}, raw_data_stats
        doc_list = self.chop_doc_into_msl(data)
        results, tokenized_data_stats = self.process_docs(doc_list)
        data_stats = {
            "discarded": tokenized_data_stats["discarded"],
            "processed": tokenized_data_stats["processed"],
            "successful": tokenized_data_stats["successful"],
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
        return results, data_stats
