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
from cerebras.modelzoo.config.types import ValidatedPath
from cerebras.modelzoo.data.common.input_utils import (
    get_data_for_task,
    num_tasks,
    shard_list_interleaved,
    task_id,
)
from cerebras.modelzoo.data.nlp.bert.bert_utils import get_meta_data


class BertCSVDataProcessorConfig(DataConfig):
    data_processor: Literal["BertCSVDataProcessor"]

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

    shuffle: bool = False
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

    prefetch_factor: Optional[int] = 2
    "The number of batches to prefetch in the dataloader."

    persistent_workers: bool = False
    "Whether or not to keep workers persistent between epochs."

    drop_last: bool = True
    "Whether to drop last batch of epoch if it's an incomplete batch."

    # The following fields are deprecated and unused.
    # They will be removed in the future once all configs have been fixed
    vocab_size: Optional[Any] = Field(default=None, deprecated=True)
    vocab_file: Optional[Any] = Field(default=None, deprecated=True)
    whole_word_masking: Optional[Any] = Field(default=None, deprecated=True)
    max_predictions_per_seq: Optional[Any] = Field(
        default=None, deprecated=True
    )
    do_lower: Optional[Any] = Field(default=None, deprecated=True)
    masked_lm_prob: Optional[Any] = Field(default=None, deprecated=True)
    max_sequence_length: Optional[Any] = Field(default=None, deprecated=True)
    max_position_embeddings: Optional[Any] = Field(
        default=None, deprecated=True
    )

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


class BertCSVDataProcessor(torch.utils.data.IterableDataset):
    """Reads csv files containing the input text tokens, and MLM features."""

    def __init__(self, config: BertCSVDataProcessorConfig):
        super().__init__()

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
        self.dynamic_mlm_scale = config.dynamic_mlm_scale
        self.buckets = config.buckets

        # Multi-processing params.
        self.num_workers = config.num_workers
        self.drop_last = config.drop_last
        self.prefetch_factor = config.prefetch_factor
        self.persistent_workers = config.persistent_workers

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
            length -= self.batch_size * (len(self.buckets) + 1)
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

        def make_features(
            data_row,
            feature_names,
            required_features=False,
            dtype=np.int32,
        ):
            if required_features:
                absent_features = [
                    feature
                    for feature in feature_names
                    if feature not in data_row
                ]
                if absent_features:
                    raise ValueError(
                        f"{absent_features} are required features, but absent in the dataset"
                    )

            return {
                feature: np.array(eval(data_row[feature]), dtype=dtype)
                for feature in feature_names
                if feature in data_row
            }

        # Iterate over the data rows to create input features.
        for data_row in self.load_buffer():
            # `data_row` is a dict with keys:
            features = make_features(
                data_row,
                [
                    "input_ids",
                    "attention_mask",
                    "labels",
                ],
                required_features=True,
            )
            features.update(
                make_features(
                    data_row, ["masked_lm_weights", "masked_lm_positions"]
                )
            )
            if "masked_lm_weights" in features:
                # Stored as masked_lm_weights, but really masked_lm_mask
                features["masked_lm_mask"] = features["masked_lm_weights"]
                features.pop("masked_lm_weights")

            if not self.disable_nsp:
                features.update(
                    make_features(
                        data_row, ["next_sentence_label", "token_type_ids"]
                    )
                )
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
