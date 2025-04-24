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
Processor for PyTorch BERT fine tuning - Summarization.
"""

import csv
import os
from typing import List, Literal, Optional, Union

import numpy as np
import torch
from pydantic import PositiveInt, field_validator

from cerebras.modelzoo.common.input_utils import get_streaming_batch_size
from cerebras.modelzoo.config import DataConfig
from cerebras.modelzoo.config.types import AliasedPath, ValidatedPath
from cerebras.modelzoo.data.common.input_utils import (
    check_sharding_sanity,
    get_data_for_task,
    num_tasks,
    task_id,
)
from cerebras.modelzoo.data.nlp.bert.bert_utils import (
    build_vocab,
    get_meta_data,
    shard_and_shuffle_data,
)
from cerebras.modelzoo.data_preparation.utils import (
    convert_str_to_int_list,
    pad_input_sequence,
)


class BertSumCSVDataProcessorConfig(DataConfig):
    data_processor: Literal["BertSumCSVDataProcessor"]

    data_dir: Union[ValidatedPath, List[ValidatedPath]] = ...
    "Path to the data files to use."

    vocab_file: Optional[AliasedPath] = None
    "Path to the vocabulary file."

    batch_size: PositiveInt = ...
    "The batch size."

    max_sequence_length: int = ...

    max_cls_tokens: int = ...

    pad_id: Optional[int] = None

    mask_whole_word: bool = False
    "Flag to whether mask the entire word."

    do_lower: bool = False
    "Flag to lower case the texts."

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

    @field_validator("vocab_file", mode="after")
    @classmethod
    def get_vocab_file(cls, vocab_file, info):
        if info.context:
            model_config = info.context.get("model", {}).get("config")

            if model_config is not None:
                if vocab_file is None:
                    vocab_file = model_config.vocab_file

                if vocab_file != model_config.vocab_file:
                    raise ValueError(
                        f"Vocab file from model and input do not match."
                        f"\n\tmodel vocab file: {model_config.vocab_file}"
                        f"\n\tinput vocab file: {vocab_file}"
                    )

        if vocab_file is None:
            raise ValueError("Vocab file must be provided.")
        if not os.path.exists(vocab_file):
            raise ValueError(f"Vocab file does not exist: {vocab_file}")
        return os.path.abspath(vocab_file)

    def post_init(self, context):
        super().post_init(context)

        if not self.num_workers:
            self.prefetch_factor = None  # the default value in DataLoader
            self.persistent_workers = False


class BertSumCSVDataProcessor(torch.utils.data.IterableDataset):
    """
    Reads csv file containing the `input_token_ids`, and `label_ids`.
    Creates `attention_mask` and `segment_ids` on the fly
    :param <dict> params: dict containing input parameters for creating dataset.
    """

    def __init__(self, config: BertSumCSVDataProcessorConfig):
        super().__init__()

        # Input params.
        self.meta_data = get_meta_data(config.data_dir)

        self.meta_data_values = list(self.meta_data.values())
        self.meta_data_filenames = list(self.meta_data.keys())
        # Please note the appending of [0], 0 for the header
        self.meta_data_values_cum_sum = np.cumsum([0] + self.meta_data_values)

        self.num_examples = sum(map(int, self.meta_data.values()))
        self.batch_size = get_streaming_batch_size(config.batch_size)

        self.num_batches = self.num_examples // self.batch_size
        assert (
            self.num_batches
        ), "Dataset does not contain enough samples for one batch. Please choose a smaller batch size."
        self.num_tasks = num_tasks()
        self.num_batch_per_task = self.num_batches // self.num_tasks

        assert (
            self.num_batch_per_task
        ), "Dataset cannot be evenly distributed across the given tasks. Please choose fewer tasks to run with."

        self.num_examples_per_task = self.num_batch_per_task * self.batch_size
        self.files_in_task = get_data_for_task(
            task_id(),
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

        # Multi-processing params.
        self.num_workers = config.num_workers
        self.drop_last = config.drop_last
        self.prefetch_factor = config.prefetch_factor
        self.persistent_workers = config.persistent_workers

        # Check that our sharding will produce at least one batch
        check_sharding_sanity(
            [num_examples for _, num_examples, _ in self.files_in_task],
            self.batch_size,
            self.num_workers,
            self.drop_last,
        )

        self.special_tokens = {
            "oov_token": "[UNK]",
            "class_token": "[CLS]",
            "pad_token": "[PAD]",
            "document_separator_token": "[SEP]",
        }
        if self.do_lower:
            self.special_tokens = {
                key: value.lower() for key, value in self.special_tokens.items()
            }

        # Get vocab file and size.
        self.vocab_file = config.vocab_file
        self.vocab, self.vocab_size = build_vocab(
            self.vocab_file, self.do_lower, self.special_tokens["oov_token"]
        )

        # Init tokenizer.
        self.tokenize = self.vocab.forward

        # Getting indices for special tokens.
        self.special_tokens_indices = {
            key: self.tokenize([value])[0]
            for key, value in self.special_tokens.items()
        }

        # Padding indices.
        if config.pad_id is None:
            self.pad_id = self.special_tokens_indices["pad_token"]
        else:
            self.pad_id = config.pad_id

        assert (
            self.pad_id >= 0
        ), f"`pad_id` must be non-negative, got {self.pad_id}"

        self.max_sequence_length = config.max_sequence_length
        self.max_cls_tokens = config.max_cls_tokens

        self.csv_files_per_task_per_worker = []
        self.processed_buffers = 0

    def create_dataloader(self):
        """
        Classmethod to create the dataloader object.
        """
        dataloader = torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            prefetch_factor=self.prefetch_factor if self.num_workers else None,
            persistent_workers=(
                self.persistent_workers if self.num_workers else False
            ),
        )
        return dataloader

    def load_buffer(self):
        """
        Generator to read the data in chunks of size of `data_buffer`.

        :returns: Yields the data stored in the `data_buffer`.
        """
        data_buffer = []
        while self.processed_buffers < len(self.csv_files_per_task_per_worker):
            (
                current_file_path,
                num_examples,
                start_id,
            ) = self.csv_files_per_task_per_worker[self.processed_buffers]
            with open(current_file_path, "r", newline="") as fid:
                data_reader = csv.DictReader(
                    fid, delimiter=",", quoting=csv.QUOTE_MINIMAL
                )
                for row_id, row in enumerate(data_reader):
                    if row_id < start_id:
                        continue
                    if row_id >= start_id + num_examples:
                        break

                    if not self.shuffle:
                        yield row
                    else:
                        if len(data_buffer) < self.shuffle_buffer:
                            data_buffer.append(row)
                        else:
                            index = self.rng.randrange(self.shuffle_buffer)
                            yield data_buffer[index]
                            data_buffer[index] = row

                self.processed_buffers += 1

        if self.shuffle:
            self.rng.shuffle(data_buffer)

        for ind in range(len(data_buffer)):
            yield data_buffer[ind]
        data_buffer = []

    def __len__(self):
        """
        Returns the length of the dataset on each slurm task.
        """
        return self.num_examples_per_task

    def __iter__(self):
        """
        Iterator over the data to construct input feature.

        :return: A tuple with training feature:
            * np.array[int.32] input_ids: Numpy array with input token indices.
                Shape: (`max_sequence_length`).
            * np.array[int.32] labels: Numpy array with labels.
               Shape: (`max_cls_tokens`).
            * np.array[int.32] attention_mask
               Shape: (`max_sequence_length`).
            * np.array[int.32] token_type_ids: Numpy array with segment indices.
               Shape: (`max_sequence_length`).
            * np.array[int.32] cls_indices: Numpy array with class indices.
               Shape: (`max_cls_tokens`).
            * np.array[float.32] cls_weights: Numpy array with class weights.
               Shape: (`max_cls_tokens`).
        """
        (
            self.processed_buffers,
            self.csv_files_per_task_per_worker,
            self.shuffle_seed,
            self.rng,
        ) = shard_and_shuffle_data(
            self.files_in_task,
            self.shuffle,
            self.shuffle_seed,
        )

        # Iterate over the data rows to create input feature.
        for data_row in self.load_buffer():
            # `data_row` is a dict with keys:
            # ["input_token_ids", "labels", "segment_ids", "cls_indices"].
            (
                input_token_ids,
                labels,
                segment_ids,
                cls_indices,
            ) = tuple(map(convert_str_to_int_list, data_row.values()))
            feature = create_bertsum_feature(
                input_token_ids,
                segment_ids,
                cls_indices,
                labels,
                self.max_sequence_length,
                self.max_cls_tokens,
                self.pad_id,
            )

            yield feature


def create_bertsum_feature(
    input_ids,
    segment_ids,
    cls_indices,
    labels,
    max_sequence_length,
    max_cls_tokens,
    pad_id,
):
    """
    Creates the feature dict for bertsum model after applying padding.

    :param list input_ids: Token ids to pad.
    :param list segment_ids: Segment ids to pad.
    :param list cls_indices: Class ids to pad.
    :param list labels: Labels to pad.
    :param int max_sequence_length: Maximum sequence length.
    :param int max_cls_tokens: Max class tokens.
    :param int pad_id: Padding id.
    :param callable tokenize: Method to tokenize the input sequence.

    :return: dict for feature which includes keys:
            * 'input_tokens': Numpy array with input token indices.
                shape: (`max_sequence_length`), dtype: int32.
            * 'attention_mask': Numpy array with attention mask.
               shape: (`max_sequence_length`), dtype: int32.
            * 'token_type_ids': Numpy array with segment ids.
               shape: (`max_sequence_length`), dtype: int32.
            * 'labels': Numpy array with labels.
               shape: (`max_cls_tokens`), dtype: int32.
            * 'cls_indices': Numpy array with class indices.
               Shape: (`max_cls_tokens`).
            * 'cls_weights': Numpy array with class weights.
               Shape: (`max_cls_tokens`).
    """
    input_ids = pad_input_sequence(input_ids, pad_id, max_sequence_length)
    labels = pad_input_sequence(labels, pad_id, max_cls_tokens)
    input_mask = np.not_equal(input_ids, pad_id).astype(np.int32)
    segment_ids = pad_input_sequence(segment_ids, pad_id, max_sequence_length)
    cls_indices = pad_input_sequence(cls_indices, pad_id, max_cls_tokens)
    cls_weights = np.not_equal(cls_indices, pad_id).astype(np.float32)

    feature = {
        "input_ids": input_ids,
        "token_type_ids": segment_ids,
        "attention_mask": input_mask,
        "labels": labels,
        "cls_indices": cls_indices,
        "cls_weights": cls_weights,
    }

    return feature
