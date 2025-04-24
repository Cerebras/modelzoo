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
Processor for PyTorch BERT training.
"""

import csv
import os
import random
from typing import Any, List, Literal, Optional, Union

import numpy as np
import torch
from pydantic import Field, PositiveInt, field_validator

from cerebras.modelzoo.common.input_utils import (
    bucketed_batch,
    get_streaming_batch_size,
)
from cerebras.modelzoo.config import DataConfig
from cerebras.modelzoo.config.types import AliasedPath, ValidatedPath
from cerebras.modelzoo.data.common.input_utils import (
    get_data_for_task,
    num_tasks,
    shard_list_interleaved,
    task_id,
)
from cerebras.modelzoo.data.nlp.bert.bert_utils import (
    build_vocab,
    create_masked_lm_predictions,
    get_meta_data,
    parse_text,
)


class BertCSVDynamicMaskDataProcessorConfig(DataConfig):
    data_processor: Literal["BertCSVDynamicMaskDataProcessor"]

    data_dir: Union[ValidatedPath, List[ValidatedPath]] = ...
    "Path to the data files to use."

    batch_size: PositiveInt = ...
    "The batch size."

    disable_nsp: bool = False
    "Whether Next Sentence Prediction (NSP) objective is disabled."

    dynamic_mlm_scale: bool = False
    "Whether to dynamically scale the loss."

    buckets: Optional[List[int]] = None
    """
    A list of bucket boundaries. If set to None, then no
    bucketing will happen, and data will be batched normally. If set to
    a list, then data will be grouped into `len(buckets) + 1` buckets. A
    sample `s` will go into bucket `i` if
    `buckets[i-1] <= element_length_fn(s) < buckets[i]` where 0 and inf are
    the implied lowest and highest boundaries respectively. `buckets` must
    be sorted and all elements must be non-zero.
    """

    mask_whole_word: bool = False
    "Flag to whether mask the entire word."

    do_lower: bool = False
    "Flag to lower case the texts."

    vocab_file: AliasedPath = ...
    "Path to the vocabulary file."

    oov_token: str = "[UNK]"
    "Out of vocabulary token."

    mask_token: str = "[MASK]"
    "Mask token."

    document_separator_token: str = "[SEP]"
    "Seperator token."

    exclude_from_masking: List[str] = ["[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    "Tokens that should be excluded from being masked."

    max_sequence_length: int = ...

    max_predictions_per_seq: int = ...

    masked_lm_prob: float = 0.15

    gather_mlm_labels: bool = True

    labels_pad_id: int = 0

    input_pad_id: int = 0

    attn_mask_pad_id: int = 0

    segment_pad_id: int = 0

    shuffle: bool = True
    "Whether or not to shuffle the dataset."

    shuffle_seed: Optional[int] = None
    "The seed used for deterministic shuffling."

    shuffle_buffer: Optional[int] = None
    """
    Buffer size to shuffle samples across.
    If None and shuffle is enabled, 10*batch_size is used.
    """

    num_workers: int = 0
    "The number of PyTorch processes used in the dataloader."

    prefetch_factor: Optional[int] = 10
    "The number of batches to prefetch in the dataloader."

    persistent_workers: bool = True
    "Whether or not to keep workers persistent between epochs."

    drop_last: bool = True
    "Whether to drop last batch of epoch if it's an incomplete batch."

    # The following fields are deprecated and unused.
    # They will be removed in the future once all configs have been fixed
    vocab_size: Optional[Any] = Field(default=None, deprecated=True)
    num_examples: Optional[Any] = Field(default=None, deprecated=True)
    steps: Optional[Any] = Field(default=None, deprecated=True)
    whole_word_masking: Optional[Any] = Field(default=None, deprecated=True)

    def post_init(self, context):
        super().post_init(context)

        if not self.num_workers:
            self.prefetch_factor = None  # the default value in DataLoader
            self.persistent_workers = False

    @field_validator("disable_nsp", mode="after")
    @classmethod
    def get_disable_nsp(cls, disable_nsp, info):
        if info.context:
            model_config = info.context.get("model", {}).get("config")

            if hasattr(model_config, "disable_nsp"):
                return model_config.disable_nsp

        return disable_nsp

    @field_validator("vocab_file", mode="after")
    @classmethod
    def get_vocab_file(cls, vocab_file):
        if not os.path.exists(vocab_file):
            raise ValueError(f"Vocab file does not exist: {vocab_file}")
        return os.path.abspath(vocab_file)


class BertCSVDynamicMaskDataProcessor(torch.utils.data.IterableDataset):
    """
    Reads csv files containing the input text tokens, adds MLM features
    on the fly.
    """

    def __init__(self, config: BertCSVDynamicMaskDataProcessorConfig):
        super().__init__()

        # Input params.
        self.meta_data = get_meta_data(config.data_dir)

        self.meta_data_values = list(self.meta_data.values())
        self.meta_data_filenames = list(self.meta_data.keys())
        # Please note the appending of [0]
        self.meta_data_values_cum_sum = np.cumsum([0] + self.meta_data_values)

        self.num_examples = sum(map(int, self.meta_data.values()))
        self.disable_nsp = config.disable_nsp
        self.batch_size = get_streaming_batch_size(config.batch_size)

        self.num_batches = self.num_examples // self.batch_size
        assert (
            self.num_batches > 0
        ), "Dataset does not contain enough samples for one batch. Please choose a smaller batch size"

        self.num_tasks = num_tasks()
        self.task_id = task_id()
        self.num_batch_per_task = self.num_batches // self.num_tasks

        assert (
            self.num_batch_per_task > 0
        ), "Dataset cannot be evenly distributed across the given tasks. Please choose fewer tasks to run with"

        self.num_examples_per_task = self.num_batch_per_task * self.batch_size
        self.files_in_task = get_data_for_task(
            self.task_id,
            self.meta_data_values_cum_sum,
            self.num_examples_per_task,
            self.meta_data_values,
            self.meta_data_filenames,
        )

        self.shuffle = config.shuffle
        self.shuffle_seed = config.shuffle_seed
        if config.shuffle_buffer is None:
            self.shuffle_buffer = 10 * self.batch_size
        else:
            self.shuffle_buffer = config.shuffle_buffer
        self.mask_whole_word = config.mask_whole_word
        self.do_lower = config.do_lower
        self.dynamic_mlm_scale = config.dynamic_mlm_scale
        self.buckets = config.buckets

        # Multi-processing params.
        self.num_workers = config.num_workers
        self.drop_last = config.drop_last
        self.prefetch_factor = config.prefetch_factor
        self.persistent_workers = config.persistent_workers

        # Get special tokens and tokens that should not be masked.
        self.special_tokens = {
            "oov_token": config.oov_token,
            "mask_token": config.mask_token,
            "document_separator_token": config.document_separator_token,
        }
        self.exclude_from_masking = config.exclude_from_masking

        if self.do_lower:
            self.special_tokens = {
                key: value.lower() for key, value in self.special_tokens.items()
            }

            self.exclude_from_masking = list(
                map(lambda token: token.lower(), self.exclude_from_masking)
            )

        # Get vocab file and size.
        self.vocab, self.vocab_size = build_vocab(
            config.vocab_file, self.do_lower, self.special_tokens["oov_token"]
        )

        # Init tokenizer.
        self.tokenize = self.vocab.forward

        # Getting indices for special tokens.
        self.special_tokens_indices = {
            key: self.tokenize([value])[0]
            for key, value in self.special_tokens.items()
        }

        self.exclude_from_masking_ids = [
            self.tokenize([token])[0] for token in self.exclude_from_masking
        ]

        # We create a pool with tokens that can be used to randomly replace input tokens
        # for BERT MLM task.
        self.replacement_pool = list(
            set(range(self.vocab_size)) - set(self.exclude_from_masking_ids)
        )

        # Padding indices.
        # See https://huggingface.co/transformers/glossary.html#labels.
        self.labels_pad_id = config.labels_pad_id
        self.input_pad_id = config.input_pad_id
        self.attn_mask_pad_id = config.attn_mask_pad_id
        if not self.disable_nsp:
            self.segment_pad_id = config.segment_pad_id

        # Max sequence lengths size params.
        self.max_sequence_length = config.max_sequence_length
        self.max_predictions_per_seq = config.max_predictions_per_seq
        self.masked_lm_prob = config.masked_lm_prob
        self.gather_mlm_labels = config.gather_mlm_labels

        # Store params.
        self.data_buffer = []
        self.csv_files_per_task_per_worker = []
        self.processed_buffers = 0

    def load_buffer(self):
        """
        Generator to read the data in chunks of size of `data_buffer`.

        :returns: Yields the data stored in the `data_buffer`.

        """
        self.processed_buffers = 0
        self.data_buffer = []

        while self.processed_buffers < len(self.csv_files_per_task_per_worker):
            (
                current_file_path,
                num_examples,
                start_id,
            ) = self.csv_files_per_task_per_worker[self.processed_buffers]

            with open(current_file_path, "r", newline="") as fin:
                data_reader = csv.DictReader(fin)

                for row_id, row in enumerate(data_reader):
                    if start_id <= row_id < start_id + num_examples:
                        self.data_buffer.append(row)
                    else:
                        continue

                    if len(self.data_buffer) == self.shuffle_buffer:
                        if self.shuffle:
                            self.rng.shuffle(self.data_buffer)

                        for ind in range(len(self.data_buffer)):
                            yield self.data_buffer[ind]
                        self.data_buffer = []

                self.processed_buffers += 1

        if self.shuffle:
            self.rng.shuffle(self.data_buffer)

        for ind in range(len(self.data_buffer)):
            yield self.data_buffer[ind]
        self.data_buffer = []

    def __len__(self):
        # Returns the len of dataset on the task process
        if not self.drop_last:
            return (
                self.num_examples_per_task + self.batch_size - 1
            ) // self.batch_size
        elif self.buckets is None:
            return self.num_examples_per_task // self.batch_size
        else:
            # give an under-estimate in case we don't fully fill some buckets
            length = self.num_examples_per_task // self.batch_size
            length -= len(self.buckets)
            return length

    def get_single_item(self):
        """
        Iterating over the data to construct input features.

        :return: A tuple with training features:
            * np.array[int.32] input_ids: Numpy array with input token indices.
                Shape: (`max_sequence_length`).
            * np.array[int.32] labels: Numpy array with labels.
               Shape: (`max_sequence_length`).
            * np.array[int.32] attention_mask
               Shape: (`max_sequence_length`).
            * np.array[int.32] token_type_ids: Numpy array with segment indices.
               Shape: (`max_sequence_length`).
            * np.array[int.32] next_sentence_label: Numpy array with labels for NSP task.
               Shape: (1).
            * np.array[int.32] masked_lm_mask: Numpy array with a mask of
               predicted tokens.
               Shape: (`max_predictions`)
               `0` indicates the non masked token, and `1` indicates the masked token.
        """
        # Iterate over the data rows to create input features.
        for data_row in self.load_buffer():
            # `data_row` is a dict with keys:
            # ["tokens", "segment_ids", "is_random_next"].
            tokens = parse_text(data_row["tokens"], do_lower=self.do_lower)

            if self.disable_nsp:
                # truncate tokens to MSL
                tokens = tokens[: self.max_sequence_length]
            else:
                assert (
                    len(tokens) <= self.max_sequence_length
                ), "When using NSP head, make sure that len(tokens) <= MSL."

            (
                input_ids,
                labels,
                attention_mask,
                masked_lm_mask,
            ) = create_masked_lm_predictions(
                tokens,
                self.max_sequence_length,
                self.special_tokens_indices["mask_token"],
                self.max_predictions_per_seq,
                self.input_pad_id,
                self.attn_mask_pad_id,
                self.labels_pad_id,
                self.tokenize,
                self.vocab_size,
                self.masked_lm_prob,
                self.rng,
                self.exclude_from_masking,
                self.mask_whole_word,
                self.replacement_pool,
            )
            features = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if self.gather_mlm_labels:
                # Gather MLM positions
                _mlm_positions = np.nonzero(masked_lm_mask)[0]
                _num_preds = len(_mlm_positions)
                gathered_mlm_positions = np.zeros(
                    (self.max_predictions_per_seq,), dtype=np.int32
                )
                gathered_mlm_positions[:_num_preds] = _mlm_positions
                gathered_labels = np.zeros(
                    (self.max_predictions_per_seq,), dtype=np.int32
                )
                gathered_labels[:_num_preds] = labels[_mlm_positions]
                gathered_mlm_mask = np.zeros(
                    (self.max_predictions_per_seq,), dtype=np.int32
                )
                gathered_mlm_mask[:_num_preds] = masked_lm_mask[_mlm_positions]
                features["labels"] = gathered_labels
                features["masked_lm_mask"] = gathered_mlm_mask
                features["masked_lm_positions"] = gathered_mlm_positions
            else:
                features["labels"] = labels
                features["masked_lm_mask"] = masked_lm_mask

            if not self.disable_nsp:
                next_sentence_label = np.zeros((1,), dtype=np.int32)

                token_type_ids = (
                    np.ones((self.max_sequence_length,), dtype=np.int32)
                    * self.segment_pad_id
                )

                segment_ids = data_row["segment_ids"].strip("[]").split(", ")
                token_type_ids[: len(segment_ids)] = list(map(int, segment_ids))
                next_sentence_label[0] = int(data_row["is_random_next"])
                features["token_type_ids"] = token_type_ids
                features["next_sentence_label"] = next_sentence_label

            yield features

    def __iter__(self):
        batched_dataset = bucketed_batch(
            self.get_single_item(),
            self.batch_size,
            buckets=self.buckets,
            element_length_fn=lambda feats: np.sum(feats["attention_mask"]),
            drop_last=self.drop_last,
            seed=self.shuffle_seed,
        )
        for batch in batched_dataset:
            if self.dynamic_mlm_scale:
                scale = self.batch_size / torch.sum(batch["masked_lm_mask"])
                batch["mlm_loss_scale"] = scale.expand(self.batch_size, 1)
            yield batch

    def _worker_init_fn(self, worker_id):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            # Single-process
            worker_id = 0
            num_workers = 1

        self.processed_buffers = 0
        if self.shuffle_seed is not None:
            self.shuffle_seed += worker_id + 1
        self.rng = random.Random(self.shuffle_seed)

        # Shard the data across multiple processes.
        self.csv_files_per_task_per_worker = shard_list_interleaved(
            self.files_in_task, worker_id, num_workers
        )
        if self.shuffle:
            self.rng.shuffle(self.csv_files_per_task_per_worker)

    def create_dataloader(self):
        """
        Classmethod to create the dataloader object.
        """
        if self.num_workers:
            dataloader = torch.utils.data.DataLoader(
                self,
                batch_size=None,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=self.persistent_workers,
                worker_init_fn=self._worker_init_fn,
            )
        else:
            dataloader = torch.utils.data.DataLoader(self, batch_size=None)
            self._worker_init_fn(0)

        return dataloader
