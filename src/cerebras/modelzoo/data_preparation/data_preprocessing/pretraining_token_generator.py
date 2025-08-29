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
PretrainingTokenGenerator Module

This module provides the PretrainingTokenGenerator class which is designed to process
text data and create features suitable for language modeling tasks.

Usage:
    tokenizer = PretrainingTokenGenerator(dataset_params, max_sequence_length, tokenizer)
    tokenized_features = tokenizer.encode("Sample text for processing.")
"""

import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from cerebras.modelzoo.data_preparation.data_preprocessing.utils import (
    append_eos_to_multiple_semantic_regions,
    clean_text,
    find_token_range,
    get_data_stats,
    setup_warning_logging,
    split_text_and_tokenize,
)


class PretrainingTokenGenerator:
    def __init__(
        self, params: Dict[str, Any], tokenizer: Any, eos_id: int, pad_id: int
    ):
        """
        Initialize the PretrainingTokenGenerator class.

        Args:
            params (Dict[str, Any]): Parameters for the dataset and processing.
            tokenizer (Any): Tokenizer to use for tokenization.
            eos_id (int): End-of-sequence token ID.
            pad_id (int): Padding token ID.
        """
        dataset_params = params.get("dataset", {})
        processing_params = params["processing"]
        setup_params = params["setup"]
        warning_log_dir = os.path.join(
            setup_params.get("output_dir", "./output")
        )
        self.logger = setup_warning_logging(warning_log_dir, __name__)
        self.tokenizer = tokenizer

        self.training_objective = dataset_params.pop("training_objective", None)
        self.mlm = (
            (self.training_objective == 'mlm')
            if self.training_objective is not None
            else False
        )
        self.use_vsl = dataset_params.pop("use_vsl", False)

        self.use_ftfy = processing_params.pop("use_ftfy", True)
        self.ftfy_normalizer = processing_params.pop("ftfy_normalizer", "NFC")
        self.wikitext_detokenize = processing_params.pop(
            "wikitext_detokenize", False
        )
        self.min_sequence_len = processing_params.pop("min_sequence_len", 10)
        self.input_ids_dtype = processing_params.pop("input_ids_dtype", "int32")
        self.input_mask_dtype = processing_params.pop(
            "input_mask_dtype", "int32"
        )
        self.inverted_mask = processing_params.pop("inverted_mask", False)
        self.seed = processing_params.pop("seed", 0)
        np.random.seed(self.seed)
        self.max_seq_length = processing_params.pop("max_seq_length", 2048)
        self.short_seq_prob = processing_params.pop("short_seq_prob", 0.0)
        self.semantic_drop_mask = processing_params.pop(
            "semantic_drop_mask", {}
        )

        self.split_text_to_tokenize = dataset_params.pop(
            "split_text_to_tokenize", False
        )
        if self.split_text_to_tokenize:
            self.chunk_len_to_split = dataset_params.pop(
                "chunk_len_to_split", 2000
            )
            self.remove_bos_in_chunks = dataset_params.pop(
                "remove_bos_in_chunks", False
            )
        self.eos_id = eos_id
        self.pad_id = pad_id
        if self.pad_id:
            self.pad_token = self.tokenizer.convert_ids_to_tokens(self.pad_id)

        self.rng = random.Random()
        self.rng.seed(self.seed)
        self.prefix = []
        self.prefix_doc = None

        # Multimodal parameters initialization
        self.is_multimodal = dataset_params.pop("is_multimodal", False)
        self.features = (
            [
                "text_input_ids",
                "loss_mask",
                "labels",
                "key_padding_mask",
                "token_modality_idx",
            ]
            if self.is_multimodal
            else [
                "input_ids",
                "attention_mask",
                "labels",
            ]
        )
        ##MLM fields
        if self.mlm:
            import math

            self.mlm_fraction = dataset_params.pop("mlm_fraction", 0.15)
            self.max_predictions = math.ceil(
                self.mlm_fraction * self.max_seq_length
            )
            self.mlm_with_gather = dataset_params.pop("mlm_with_gather", False)
            self.ignore_index = dataset_params.pop(
                "ignore_index", -100
            )  # default value for torch.nn.CrossEntropyLoss
            self.excluded_tokens = dataset_params.pop(
                "excluded_tokens",
                ['<cls>', '<pad>', '<eos>', '<unk>', '<null_1>', '<mask>'],
            )
            self.allowable_token_ids = self.get_allowable_token_ids()
            self.special_tokens_ids = {
                self.tokenizer.cls_token_id,
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.unk_token_id,
            }

            if self.mlm_with_gather:
                self.features.extend(["masked_lm_positions", "masked_lm_mask"])

        self.pack_sequences = dataset_params.pop(
            "pack_sequences", False if self.is_multimodal else True
        )
        self.image_token = dataset_params.pop(
            "image_token", "<special_image_token>"
        )

        self.image_dir = params["setup"].pop("image_dir", None)
        self.max_num_img = dataset_params.pop("max_num_img", 1)
        self.num_patches = dataset_params.pop("num_patches", 1)
        self.image_token_id = -1
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
            and self.is_multimodal
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
            [pad_id] * self.num_patches if self.is_multimodal else []
        )
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

        if not self.is_multimodal and self.rng.random() < self.short_seq_prob:
            input_ids = input_ids[
                0 : self.rng.randint(2, self.max_seq_length - 1)
            ]
            input_mask = input_mask[0 : len(input_ids)]
            attention_mask = attention_mask[0 : len(input_ids)]

        input_mask, attention_mask = doc.get("loss_mask"), doc.get(
            "attention_mask"
        )
        labels = input_ids[1:]
        input_ids = input_ids[:-1]
        attention_mask = attention_mask[:-1]
        input_mask = input_mask[1:]

        # Add padding
        num_pad = self.max_seq_length - len(input_ids)
        padding = [self.pad_id] * num_pad
        input_ids.extend(padding)
        labels.extend(padding)

        padding = [0] * num_pad
        input_mask.extend(padding)
        attention_mask.extend([0] * num_pad)

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
        attention_mask = getattr(np, self.input_ids_dtype)(attention_mask)
        if self.inverted_mask:
            input_mask = np.equal(input_mask, 0).astype(self.input_mask_dtype)

        # NOTE this is because our internal stack requires the inverted mask and
        # doesn't do the inversion internally
        if self.is_multimodal:
            key_padding_mask = np.equal(attention_mask, 0).astype(
                input_mask.dtype
            )

        return (
            np.stack(
                [
                    features["input_ids"],
                    input_mask,
                    features["labels"],
                    key_padding_mask,
                    token_modality_idx,
                ]
            )
            if self.is_multimodal
            else np.stack(
                [
                    features["input_ids"],
                    input_mask,
                    features["labels"],
                ]
            )
        )

    def create_features_auto_lm(
        self,
        token_ids: List[int],
    ) -> np.ndarray:
        """Given a list of token_ids, generate input sequence and labels.

        Args:
            token_ids (List[int]): List containing token ids for creating features,
                labels and input mask from.

        Returns:
            np.ndarray: Array containing features, labels, and input mask.
        """

        if not len(token_ids) >= self.min_sequence_len:
            self.logger.warning(
                f"token_ids must have at least {self.min_sequence_len} elements, skipping this example..."
            )
            return []

        if self.rng.random() < self.short_seq_prob:
            token_ids = token_ids[
                0 : self.rng.randint(2, self.max_seq_length - 1)
            ]

        input_ids = token_ids[:-1]
        labels = token_ids[1:]
        input_mask = [1] * len(labels)

        # padding
        num_pad = self.max_seq_length - len(input_ids)
        padding = [self.pad_id] * num_pad

        input_ids.extend(padding)
        labels.extend(padding)
        input_mask.extend([0] * num_pad)

        # assertions to ensure correct output shapes
        assert (
            len(input_ids) == self.max_seq_length
            and len(labels) == self.max_seq_length
            and len(input_mask) == self.max_seq_length
        ), "Wrong sequence length"

        # create feature dict
        features = dict()
        features["input_ids"] = getattr(np, self.input_ids_dtype)(input_ids)
        features["input_mask"] = getattr(np, self.input_mask_dtype)(input_mask)

        if self.inverted_mask:
            features["input_mask"] = np.equal(features["input_mask"], 0).astype(
                features["input_mask"].dtype
            )
        labels = getattr(np, self.input_ids_dtype)(labels)

        return np.stack([features["input_ids"], features["input_mask"], labels])

    def chop_doc_into_msl(self, data):
        doc_list = []
        tokenized_data = data.get("tokenized_data")
        tokenized_semantic_regions = data.get("tokenized_semantic_regions")
        image_paths = data.get("image_paths", [])
        image_index = 0
        max_len = self.max_seq_length + 1  # Including space for EOS token

        last_chunk_tokens = []
        last_chunk_loss_mask = []
        last_chunk_attention_mask = []
        last_chunk_img_paths = []
        last_chunk_has_img = False
        last_chunk_image_positions = []

        if self.pack_sequences and self.prefix_doc:
            last_chunk_tokens = self.prefix_doc['input_ids']
            last_chunk_loss_mask = self.prefix_doc['loss_mask']
            last_chunk_attention_mask = self.prefix_doc['attention_mask']
            last_chunk_img_paths = self.prefix_doc['image_paths']
            last_chunk_has_img = self.prefix_doc['has_img']
            last_chunk_image_positions = self.prefix_doc['image_data_positions']
            self.prefix_doc = None

        for idx, region in enumerate(tokenized_semantic_regions):
            modality = region["region_modality"]
            loss_weight = region["loss_weight"]
            attn_mask_value = region["attention_mask"]
            start_idx, end_idx = region["indices"]
            orig_idx = (start_idx, end_idx)
            start_idx = 0 if idx == 0 else start_idx
            tokenized_semantic_regions[idx]['indices'] = (start_idx, end_idx)
            tokens = tokenized_data["input_ids"][start_idx:end_idx]
            region_len = len(tokens)
            # Generate loss_mask and attention_mask for the region
            region_loss_mask = [loss_weight] * region_len
            region_attention_mask = [attn_mask_value] * region_len

            if modality != "image":
                # Combine last_chunk_* variables with current tokens
                tokens = last_chunk_tokens + tokens
                region_loss_mask = last_chunk_loss_mask + region_loss_mask
                region_attention_mask = (
                    last_chunk_attention_mask + region_attention_mask
                )

                region_len = len(
                    tokens
                )  # Update region_len after concatenation

                # Split text region into chunks fitting max_len
                chunks = [
                    (
                        tokens[i : i + max_len],
                        region_loss_mask[i : i + max_len],
                        region_attention_mask[i : i + max_len],
                    )
                    for i in range(0, region_len, max_len)
                ]

                # Determine the number of complete chunks
                num_chunks = len(chunks)
                if len(chunks[-1][0]) < max_len:
                    num_complete_chunks = num_chunks - 1
                else:
                    num_complete_chunks = num_chunks

                # Process complete chunks
                for idx in range(num_complete_chunks):
                    chunk_tokens, chunk_loss_mask, chunk_attention_mask = (
                        chunks[idx]
                    )
                    if idx == 0 and last_chunk_tokens:
                        assert len(chunk_tokens) == len(
                            chunk_loss_mask
                        ), "Length of input ids and loss is different"
                        # First chunk may have image data from previous last_chunk
                        doc_list.append(
                            {
                                "input_ids": chunk_tokens,
                                "loss_mask": chunk_loss_mask,
                                "attention_mask": chunk_attention_mask,
                                "image_paths": last_chunk_img_paths,
                                "has_img": last_chunk_has_img,
                                "image_data_positions": last_chunk_image_positions,
                            }
                        )
                        # Reset image data after first chunk
                        last_chunk_img_paths = []
                        last_chunk_has_img = False
                        last_chunk_image_positions = []
                    else:
                        assert len(chunk_tokens) == len(
                            chunk_loss_mask
                        ), "Length of input ids and loss is different"
                        # Subsequent chunks without image data
                        doc_list.append(
                            {
                                "input_ids": chunk_tokens,
                                "loss_mask": chunk_loss_mask,
                                "attention_mask": chunk_attention_mask,
                                "image_paths": [],  # No images in subsequent chunks
                                "has_img": False,
                                "image_data_positions": [],
                            }
                        )

                # Update last_chunk_* variables for the next iteration
                if num_complete_chunks < num_chunks:
                    # The last chunk is incomplete; store it for the next iteration
                    last_chunk_tokens = chunks[-1][0]
                    last_chunk_loss_mask = chunks[-1][1]
                    last_chunk_attention_mask = chunks[-1][2]
                    # last_chunk_img_paths and last_chunk_has_img remain reset
                else:
                    # All chunks are complete; reset last_chunk_* variables
                    last_chunk_tokens = []
                    last_chunk_loss_mask = []
                    last_chunk_attention_mask = []
                    # last_chunk_img_paths and last_chunk_has_img remain reset
            else:
                # Handle image region
                image_path = image_paths[image_index]
                image_index += 1
                image_tokens = tokens  # Image tokens should not be split
                image_loss_mask = region_loss_mask
                image_attention_mask = region_attention_mask
                image_len = len(image_tokens)
                combined_len = len(last_chunk_tokens) + image_len

                if combined_len < max_len - 1:
                    # Add image tokens to last_chunk
                    start_idx = len(last_chunk_tokens)
                    last_chunk_tokens.extend(image_tokens)
                    last_chunk_loss_mask.extend(image_loss_mask)
                    last_chunk_attention_mask.extend(image_attention_mask)
                    end_idx = len(last_chunk_tokens)
                    last_chunk_img_paths += [image_path]
                    last_chunk_has_img = True
                    image_indices = (
                        orig_idx if idx == 0 else (start_idx, end_idx)
                    )
                    last_chunk_image_positions.append(image_indices)
                else:
                    # Finalize last_chunk
                    if last_chunk_tokens:
                        assert len(last_chunk_tokens) == len(
                            last_chunk_loss_mask
                        ), "Length of input ids and loss is different"
                        doc_list.append(
                            {
                                "input_ids": last_chunk_tokens,
                                "loss_mask": last_chunk_loss_mask,
                                "attention_mask": last_chunk_attention_mask,
                                "image_paths": last_chunk_img_paths,
                                "has_img": last_chunk_has_img,
                                "image_data_positions": last_chunk_image_positions,
                            }
                        )
                    # Start new last_chunk with image tokens if they fit
                    if image_len < max_len - 1:
                        last_chunk_tokens = image_tokens
                        last_chunk_loss_mask = image_loss_mask
                        last_chunk_attention_mask = image_attention_mask
                        last_chunk_img_paths = [image_path]
                        last_chunk_has_img = True
                        last_chunk_image_positions = [(0, image_len)]
                    else:
                        # Image tokens exceed max_len; cannot split images
                        raise ValueError(
                            "Image tokens exceed maximum sequence length."
                        )

        # Append any remaining last_chunk to doc_list
        if last_chunk_tokens:
            assert len(last_chunk_tokens) == len(
                last_chunk_loss_mask
            ), "Length of input ids and loss is different"
            doc = {
                "input_ids": last_chunk_tokens,
                "loss_mask": last_chunk_loss_mask,
                "attention_mask": last_chunk_attention_mask,
                "image_paths": last_chunk_img_paths,
                "has_img": last_chunk_has_img,
                "image_data_positions": last_chunk_image_positions,
            }
            if not self.pack_sequences:
                doc_list.append(doc)
            else:
                self.prefix_doc = doc

        return doc_list

    def get_segment_indices(
        self,
        tokenized_data: Dict[str, int],
        semantic_region_list: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Get segment indices for the data ranges.

        Args:
            tokenized_data (List[Tuple[int, int]]): Tokenized data with offset mappings.
            text_semantic_region_list (List[Dict[str, Any]]): List of text semantic regions with region details.

        Returns:
            List[Dict[str, Any]]: List of tokenized semantic regions and image regions with their indices.
        """
        tokenized_semantic_region_list = []
        tokenized_semantic_region = None
        starting_offset_index = 0
        for region in semantic_region_list:
            region_name = region.get("region_modality")
            tokenized_semantic_region = find_token_range(
                region,
                tokenized_data["offset_mapping"],
                starting_offset_index,
            )
            tokenized_semantic_region["region_modality"] = region_name
            starting_offset_index = tokenized_semantic_region["indices"][1]
            tokenized_semantic_region_list.append(tokenized_semantic_region)

        return tokenized_semantic_region_list

    def get_allowable_token_ids(self) -> List[int]:
        """Generate a list of token IDs that can be masked."""
        excluded_token_ids = {
            self.tokenizer.convert_tokens_to_ids(tok)
            for tok in self.excluded_tokens
            if tok in self.tokenizer.get_vocab()
        }
        allowable_token_ids = [
            tok_id
            for tok, tok_id in self.tokenizer.get_vocab().items()
            if tok_id not in excluded_token_ids
        ]
        return list(allowable_token_ids)

    def mask_single_sequence(
        self, input_ids: List[int]
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Masks tokens in a single sequence according to the MLM strategy.
        When self.mlm_with_gather is False, the returning len(labels) == len(input_ids)
        When self.mlm_with_gather is True, the returning len(labels) == self.max_predictions

        Args:
            input_ids (List[int]): Original sequence of token IDs.

        Returns:
            Tuple[List[int], List[int], List[int], List[int]]:
                - input_ids: Modified sequence with masked tokens.
                - masked_lm_positions: Positions of the masked tokens, empty if not self.mlm_with_gather.
                - masked_lm_mask: Binary indicators (1s) for positions that were masked, empty if not self.mlm_with_gather.
                - labels: Original token IDs of the masked tokens for label purposes.
        """
        sequence = np.array(input_ids.copy())
        masked_lm_positions = []
        masked_lm_mask = []
        labels = (
            [] if self.mlm_with_gather else [self.ignore_index] * len(input_ids)
        )
        indices_can_be_masked = [
            i
            for i, token_id in enumerate(input_ids)
            if token_id not in self.special_tokens_ids
        ]

        # Calculate the number of tokens to mask
        num_tokens_to_mask = min(
            int(self.mlm_fraction * len(indices_can_be_masked)),
            self.max_predictions,
        )
        if num_tokens_to_mask > 0:
            # Randomly select tokens to mask
            indices_to_mask = sorted(
                self.rng.sample(indices_can_be_masked, k=num_tokens_to_mask)
            )

            for pos in indices_to_mask:
                original_token_id = sequence[pos].copy()
                prob = self.rng.random()
                if prob < 0.8:  # 80% of the time, replace with [MASK]
                    sequence[pos] = self.tokenizer.mask_token_id
                elif prob < 0.9:  # 10% of the time, replace with a random token
                    # Ensure selected token is not a special token
                    masked_token_id = np.random.choice(self.allowable_token_ids)
                    sequence[pos] = masked_token_id
                elif prob <= 1.0:
                    pass  # 10% of the time, keep the original token

                # Store the original token ID as the label
                if self.mlm_with_gather:
                    masked_lm_positions.append(pos)
                    masked_lm_mask.append(1)
                    labels.append(original_token_id)
                else:
                    labels[pos] = original_token_id

        if self.mlm_with_gather:
            # Pad the lists to reach max_predictions length
            num_paddings = self.max_predictions - len(masked_lm_positions)
            masked_lm_positions = masked_lm_positions + [0] * num_paddings
            masked_lm_mask = masked_lm_mask + [0] * num_paddings
            labels = labels + [self.ignore_index] * num_paddings

        return list(sequence), masked_lm_positions, masked_lm_mask, labels

    def process_chunks(
        self, tokenized_text_chunks: List[List[int]]
    ) -> Tuple[List[np.ndarray], Dict[str, int]]:
        """
        Processes chunks of tokenized text and returns processed features along with the total padding added.

        Args:
            tokenized_text_chunks (List[List[int]]): A list of tokenized text chunks, where each chunk is represented as a list of integers.

        Returns:
            Tuple[List[np.ndarray], Dict[str, int]]: A tuple containing a list of processed results and dataset stats.
        """
        results = {"data": []}  # List to store the processed results
        stats = defaultdict(int)
        # Iterate over each chunk in the tokenized text chunks
        for chunk in tokenized_text_chunks:
            # Process the chunk and get the processed result and number of padding tokens added
            processed = self.create_features_auto_lm(
                chunk,
            )

            # If the processed chunk is not empty, add the results to the list and update the total padding
            if len(processed) != 0:
                processed_stats = get_data_stats(
                    processed, self.pad_id, self.eos_id, self.max_seq_length
                )
                for key in processed_stats:
                    stats[key] += processed_stats[key]
                results["data"].append(processed)

        # Return the list of processed results and data stats
        return results, stats

    def process_chunks_mlm(
        self, tokenized_text_chunks: List[List[int]]
    ) -> Tuple[List[Any], Dict]:
        """
        Processes chunks of tokenized text and returns processed features along with the total padding added.

        Args:
        tokenized_text_chunks (List[List[int]]): A list of tokenized text chunks, where each chunk is represented as a list of integers.

        Returns:
        Tuple[List[Any], Dict]: A tuple containing a list of processed results and dataset stats.
        """

        results = {
            'data': [],
            'labels': [],
        }  # List to store the processed result

        stats = defaultdict(int)

        masked_lm_positions_list = []
        masked_lm_mask_list = []
        input_id_list = []
        labels_list = []
        attention_mask_list = []
        # Iterate over each chunk in the tokenized text chunks
        for chunk in tokenized_text_chunks:
            input_ids, masked_lm_positions, masked_lm_mask, labels = (
                self.mask_single_sequence(chunk)
            )
            num_pad = self.max_seq_length - len(input_ids)
            attention_mask = [1] * len(input_ids) + [0] * num_pad
            input_ids = input_ids + [self.pad_id] * num_pad

            input_id_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)
            masked_lm_positions_list.append(masked_lm_positions)
            masked_lm_weights_list.append(masked_lm_weights)

            lvt = len(labels) - labels.count(self.ignore_index)
            processed_stats = get_data_stats(
                np.expand_dims(np.array(input_ids), 0),
                self.pad_id,
                self.eos_id,
                self.max_seq_length,
                lvt,
            )
            for key in processed_stats:
                stats[key] += processed_stats[key]

        if len(tokenized_text_chunks) > 0:
            results['data'] = np.stack(
                [np.array(input_id_list), np.array(attention_mask_list)], axis=1
            )
            if self.mlm_with_gather:
                results['labels'] = np.stack(
                    [
                        np.array(labels_list),
                        np.array(masked_lm_positions_list),
                        np.array(masked_lm_weights_list),
                    ],
                    axis=1,
                )
            else:
                results['labels'] = np.stack(
                    [np.array(labels_list)],
                    axis=1,
                )

        # Return the list of processed results and data stats
        return results, stats

    def process_mlm(self, text_data, raw_data_stats):
        tokenized_data = self.tokenizer(
            text_data,
            max_length=self.max_seq_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
        )
        input_ids, attention_mask = (
            tokenized_data['input_ids'],
            tokenized_data['attention_mask'],
        )
        tokenized_data_stats = dict()
        results = dict()

        tokenized_data_stats["processed_files"] = 1
        tokenized_data_stats["discarded_files"] = 0
        if len(input_ids) == 0:
            tokenized_data_stats["discarded_files"] = 1
            return {"data": [], "labels": []}, tokenized_data_stats

        tokenized_data_stats["successful_files"] = 1

        input_ids, masked_lm_positions, masked_lm_mask, labels = (
            self.mask_single_sequence(input_ids)
        )

        results['data'] = np.stack(
            [np.array(input_ids), np.array(attention_mask)], axis=0
        ).reshape(1, 2, self.max_seq_length)

        if self.mlm_with_gather:
            results['labels'] = np.stack(
                [
                    np.array(labels),
                    np.array(masked_lm_positions),
                    np.array(masked_lm_mask),
                ],
                axis=0,
            ).reshape(1, 3, self.max_predictions)
        else:
            results['labels'] = np.stack(
                [
                    np.array(labels),
                ],
                axis=0,
            ).reshape(1, 1, self.max_seq_length)

        tokenized_data_stats["non_pad_tokens"] = sum(
            1 for id in input_ids if id != self.pad_id
        )
        tokenized_data_stats["num_pad_tokens"] = (
            self.max_seq_length - tokenized_data_stats["non_pad_tokens"]
        )
        tokenized_data_stats["num_tokens"] = self.max_seq_length
        tokenized_data_stats["num_masked_tokens"] = input_ids.count(
            self.tokenizer.mask_token_id
        )
        tokenized_data_stats["loss_valid_tokens"] = len(labels) - labels.count(
            self.ignore_index
        )
        tokenized_data_stats["n_examples"] = 1
        tokenized_data_stats.update(raw_data_stats)

        return results, tokenized_data_stats

    def process_single_semantic_region(self, text_data, raw_data_stats):
        discarded_files = 0
        # tokenize text
        if self.split_text_to_tokenize:
            tokenized_text = split_text_and_tokenize(
                text_data,
                self.tokenizer,
                max_tok_len=self.chunk_len_to_split,
                remove_bos_in_chunks=self.remove_bos_in_chunks,
            )
        else:
            tokenized_text = self.tokenizer.encode(text_data)

        if self.eos_id is not None:
            tokenized_text += [self.eos_id]
        all_text = self.prefix + tokenized_text
        tokenized_text_chunks = [
            all_text[i : i + self.max_seq_length + 1]
            for i in range(0, len(all_text), self.max_seq_length)
        ]
        # reset prefix
        self.prefix = []
        # update prefix if last chunk is < max_seq_length
        num_tokens_last_chunk = len(tokenized_text_chunks[-1])

        if self.pack_sequences:
            if num_tokens_last_chunk < self.max_seq_length + 1:
                last_chunk = tokenized_text_chunks.pop(-1)
                self.prefix.extend(last_chunk)
        elif num_tokens_last_chunk < 2:
            _ = tokenized_text_chunks.pop(-1)
        results, tokenized_data_stats = self.process_chunks(
            tokenized_text_chunks
        )
        if len(results["data"]) == 0 and len(self.prefix) == 0:
            discarded_files += 1
        tokenized_data_stats["discarded_files"] = discarded_files
        tokenized_data_stats["processed_files"] = 1
        tokenized_data_stats["successful_files"] = (
            tokenized_data_stats["processed_files"]
            - tokenized_data_stats["discarded_files"]
        )
        tokenized_data_stats.update(raw_data_stats)
        return results, tokenized_data_stats

    def tokenize_data(self, semantic_data_array):
        region_data, raw_data_stats = self.parse_semantic_data_array(
            semantic_data_array
        )
        if not region_data:
            return {}, raw_data_stats

        semantic_regions = region_data.get("semantic_regions")

        data, image_paths = (
            region_data.get("data"),
            region_data.get("image_paths"),
        )

        if data == "":
            return {}, raw_data_stats

        if self.mlm:
            return self.process_mlm(data, raw_data_stats)

        if (self.split_text_to_tokenize) and not self.is_multimodal:
            return self.process_single_semantic_region(data, raw_data_stats)

        if self.split_text_to_tokenize:
            raise ValueError(
                f"Multiple semantic region is not supported with `split_text_to_tokenize`."
            )

        tokenized_data = self.tokenizer(
            data,
            return_offsets_mapping=True,
        )

        if self.is_multimodal and not self.use_single_image_token:
            # Convert input_ids to numpy array
            input_ids = np.array(tokenized_data['input_ids'])

            # Replace image_token_id with pad_id
            input_ids[input_ids == self.image_token_id] = self.pad_id

            # Convert back to list
            tokenized_data['input_ids'] = input_ids.tolist()

        if len(semantic_regions) > 0:
            append_eos_to_multiple_semantic_regions(
                data,
                self.data_ranges,
                self.eos_token,
                self.image_token,
                False,
            )

        tokenized_semantic_region_list = self.get_segment_indices(
            tokenized_data,
            semantic_regions,
        )
        data = {
            "tokenized_data": tokenized_data,
            "image_paths": image_paths,
            "tokenized_semantic_regions": tokenized_semantic_region_list,
        }

        doc_list = self.chop_doc_into_msl(data)
        results, tokenized_data_stats = self.process_docs(doc_list)
        data_stats = tokenized_data_stats.copy()
        data_stats.update(raw_data_stats)

        return results, data_stats

    def parse_semantic_data_array(
        self, semantic_data_array: List[Dict[str, Any]]
    ) -> Tuple[Tuple[List[str], List[Dict[str, str]]], Dict[str, int]]:

        if not semantic_data_array:
            return {}, {}

        image_paths = []
        semantic_regions = []
        stats = {
            "raw_chars_count": 0,
            "raw_bytes_count": 0,
            "normalized_chars_count": 0,
            "normalized_bytes_count": 0,
        }

        formatted_data = ""
        formatted_data_length = 0
        for entry in semantic_data_array:
            semantic_loss_weight = entry.get("semantic_loss_weight")
            semantic_drop_mask = entry.get("semantic_drop_mask")
            semantic_attention_mask = entry.get("semantic_attention_mask")
            if semantic_loss_weight is not None and len(
                semantic_loss_weight
            ) != len(entry["content"]):
                raise ValueError(
                    "The length of semantic loss mask must match the number of regions"
                )
            if semantic_drop_mask is not None and len(
                semantic_drop_mask
            ) != len(entry["content"]):
                raise ValueError(
                    "The length of semantic drop mask must match the number of regions"
                )
            if semantic_attention_mask is not None and len(
                semantic_attention_mask
            ) != len(entry["content"]):
                raise ValueError(
                    "The length of semantic attention mask must match the number of regions"
                )

            content_parts = []
            global_idx = 0
            for i, part in enumerate(entry["content"]):

                region_key = list(part.keys())[0]
                region_val = part[region_key]
                if not region_val:
                    continue
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
                                f"<|{region_key}|>"
                                + cleaned_region_val
                                + f"<|{region_key}|>"
                            )
                        semantic_regions.append(
                            {
                                "indices": (
                                    formatted_data_length,
                                    formatted_data_length
                                    + len(cleaned_region_val),
                                ),
                                "region_modality": region_key,
                                "region_len": len(cleaned_region_val),
                                "loss_weight": loss_weight,
                                "attention_mask": attention_mask,
                            }
                        )
                        formatted_data_length += len(cleaned_region_val)
                        content_parts.append(cleaned_region_val)

                else:
                    if not drop_region:
                        image_paths.append(cleaned_region_val)
                        # Store the pad id for image region and handle `include_tags`
                        if self.use_single_image_token:
                            patches = self.image_token
                        else:
                            patches = self.num_patches * [self.image_token]
                            patches = ''.join(patches)

                        if include_tags:
                            patches = (
                                f"<|{region_key}|>"
                                + patches
                                + f"<|{region_key}|>"
                            )
                        patch_len = len(patches)
                        semantic_regions.append(
                            {
                                "indices": (
                                    formatted_data_length,
                                    formatted_data_length + patch_len,
                                ),
                                "region_modality": region_key,
                                "loss_weight": loss_weight,
                                "attention_mask": attention_mask,
                            }
                        )
                        formatted_data_length += patch_len
                        content_parts.append(patches)
                global_idx += 1

            formatted_data += ''.join(content_parts)

        if (
            self.is_multimodal
            and self.max_num_img
            and len(image_paths) > self.max_num_img
        ):
            self.logger.warning(
                f"Document more images than max_num_img. Skipping this doc..."
            )
            stats["raw_docs_skipped"] = 1
            return {}, stats

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
        transformed_data = {
            "data": formatted_data,
            "image_paths": image_paths,
            "semantic_regions": semantic_regions,
        }
        return transformed_data, stats

    def process_docs(self, doc_list):
        results = defaultdict(list)
        tokenized_data_stats = defaultdict(int)
        tokenized_data_stats["discarded_files"] = 0
        if len(doc_list) == 0:
            tokenized_data_stats["processed_files"] += 1
            if self.prefix_doc is None:
                tokenized_data_stats["discarded_files"] += 1
            else:
                tokenized_data_stats["successful_files"] += 1
            return {}, tokenized_data_stats

        # Add eos at the end.
        # TODO: Find a better way to handle this?
        last_doc = doc_list[-1]
        if len(last_doc.get("input_ids", [])) != 0 and self.eos_id != None:
            if last_doc['input_ids'][-1] != self.eos_id:
                if len(last_doc['input_ids']) < self.max_seq_length + 1:
                    last_doc['input_ids'].append(self.eos_id)
                    last_doc['loss_mask'].append(last_doc['loss_mask'][-1])
                    last_doc['attention_mask'].append(
                        last_doc['attention_mask'][-1]
                    )
                else:
                    last_doc['input_ids'][-1] = self.eos_id

        for doc_idx, doc in enumerate(doc_list):
            has_img = False
            if len(doc.get("input_ids", [])) == 0:
                continue

            token_modality_idx = (
                np.zeros(self.max_seq_length) if self.is_multimodal else None
            )
            image_paths, image_data_positions = doc.pop(
                "image_paths", None
            ), doc.pop("image_data_positions", None)
            has_img = doc.pop("has_img", None)
            img_data_loc = None
            if self.is_multimodal:
                img_data_loc = np.full(
                    (self.max_num_img, self.num_patches), self.max_seq_length
                )

                assert (
                    len(image_data_positions) <= self.max_num_img
                ), "Number of images should be <= max_num_images"

                # Preallocate img_data_loc as a list of arrays to avoid dynamic resizing
                for image_index, (start_img_pos, end_img_pos) in enumerate(
                    image_data_positions
                ):
                    img_data_loc[image_index] = np.arange(
                        start_img_pos, end_img_pos
                    )

                    # Efficiently update the token_modality_idx using vectorized assignment
                    token_modality_idx[start_img_pos:end_img_pos] = 1

            sample = self.create_features_pretraining(
                doc,
                token_modality_idx,
            )
            if len(sample) == 0:
                continue

            if self.is_multimodal:
                if image_paths:
                    num_images = len(image_paths)
                    image_paths += [None] * (self.max_num_img - num_images)
                    has_img = True
                else:
                    image_paths = [None] * (self.max_num_img)

            if not self.use_vsl:
                ## Sample stats for vsl are computed after packing is done.
                sample_stats = get_data_stats(
                    sample, self.pad_id, self.eos_id, self.max_seq_length
                )
                for key in sample_stats:
                    tokenized_data_stats[key] += sample_stats[key]
            data = (
                {
                    "data": sample,
                    "img_path": np.array(image_paths, dtype="S"),
                    "has_img": np.array([has_img], dtype=np.bool_),
                    "img_data_loc": img_data_loc,
                }
                if self.is_multimodal
                else {
                    "data": sample,
                }
            )
            for key, value in data.items():
                results[key].append(value)

        tokenized_data_stats["processed_files"] += 1
        if len(results.get("data", [])) == 0:
            tokenized_data_stats["discarded_files"] += 1
        else:
            tokenized_data_stats["successful_files"] += 1

        return results, tokenized_data_stats

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
        if len(tokenized_data.get("data", [])) == 0:
            return {}, data_stats
        else:
            if self.is_multimodal:
                data = tokenized_data
            else:
                if not self.mlm:
                    data = {'data': tokenized_data['data']}
                else:
                    data = {'data': tokenized_data['data']}
                    if not self.use_vsl:
                        data['labels'] = tokenized_data['labels']

        return data, data_stats

    def encode_leftover_prefix(
        self, prefix: List[np.ndarray]
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """
        Processes the leftover prefix which is a list of ndarray tokens into chunks based
        on max sequence length.

        The last chunk is handled specifically if it's shorter than the max sequence
        length. If the last chunk has less than two tokens, it's discarded.

        Args:
            prefix (List[np.ndarray]): The prefix list of token arrays to process.

        Returns:
            Tuple[Dict[str, Any], Dict[str, int]]: A tuple containing the processed token chunks as
            a list of ndarrays and the dataset stats.
        """
        if (self.split_text_to_tokenize or self.mlm) and not self.is_multimodal:
            tokenized_text_chunks = (
                [
                    prefix[i : i + self.max_seq_length]
                    for i in range(0, len(prefix), self.max_seq_length)
                ]
                if self.mlm
                else [
                    prefix[i : i + self.max_seq_length + 1]
                    for i in range(0, len(prefix), self.max_seq_length)
                ]
            )

            # Handle last chunk if shorter than max_seq_length
            num_tokens_last_chunk = len(tokenized_text_chunks[-1])
            if num_tokens_last_chunk < self.max_seq_length + 1:
                _ = tokenized_text_chunks.pop(-1)
            elif num_tokens_last_chunk < 2:
                _ = tokenized_text_chunks.pop(-1)

            results, stats = self.process_chunks(tokenized_text_chunks)
            if len(results["data"]) == 0:
                return {}, stats
            data = results

            return data, stats

        # Handle prefix doc
        if not prefix:
            return {}, {}

        doc_list = prefix
        results, tokenized_data_stats = self.process_docs(doc_list)
        if results == {}:
            return {}, {}

        data_stats = {
            "num_pad_tokens": tokenized_data_stats["num_pad_tokens"],
            "non_pad_tokens": tokenized_data_stats["non_pad_tokens"],
            "num_masked_tokens": tokenized_data_stats["num_masked_tokens"],
            "loss_valid_tokens": tokenized_data_stats["loss_valid_tokens"],
            "num_tokens": tokenized_data_stats["num_tokens"],
            "n_examples": tokenized_data_stats.get("n_examples", 0),
        }
        return results, data_stats
