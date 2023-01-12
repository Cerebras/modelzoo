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

import numpy as np
import torch

from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch.input_utils import bucketed_batch
from modelzoo.common.pytorch.run_utils import half_dtype_instance
from modelzoo.transformers.pytorch.bert.input.utils import get_meta_data
from modelzoo.transformers.pytorch.input_utils import (
    get_data_for_task,
    num_tasks,
    shard_list_interleaved,
    task_id,
)


class BertCSVDataProcessor(torch.utils.data.IterableDataset):
    """
    Reads csv files containing the input text tokens, and MLM features.
    :param <dict> params: dict containing input parameters for creating dataset.
    Expects the following fields:

    - "data_dir" (string): path to the data files to use.
    - "batch_size" (int): Batch size.
    - "shuffle" (bool): Flag to enable data shuffling.
    - "shuffle_seed" (int): Shuffle seed.
    - "shuffle_buffer" (int): Shuffle buffer size.
    - "repeat" (bool): Flag to enable data repeat.
    - "dynamic_mlm_scale" (bool): Flag to dynamically scale the loss.
    - "num_workers" (int):  How many subprocesses to use for data loading.
    - "drop_last" (bool): If True and the dataset size is not divisible
       by the batch size, the last incomplete batch will be dropped.
    - "prefetch_factor" (int): Number of samples loaded in advance by each worker.
    - "persistent_workers" (bool): If True, the data loader will not shutdown
       the worker processes after a dataset has been consumed once.
    - "mixed_precision" (bool): Casts input mask to fp16 if set to True.
      Otherwise, the generated mask is float32.
    """

    def __init__(self, params):
        super(BertCSVDataProcessor, self).__init__()

        self.use_cs = cm.use_cs()
        # Input params.
        self.meta_data = get_meta_data(params["data_dir"])

        self.meta_data_values = list(self.meta_data.values())
        self.meta_data_filenames = list(self.meta_data.keys())
        # Please note the appending of [0]
        self.meta_data_values_cum_sum = np.cumsum([0] + self.meta_data_values)

        self.num_examples = sum(map(int, self.meta_data.values()))
        self.disable_nsp = params.get("disable_nsp", False)
        self.batch_size = params["batch_size"]

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

        self.shuffle = params.get("shuffle", True)
        self.shuffle_seed = params.get("shuffle_seed", None)
        self.shuffle_buffer = params.get("shuffle_buffer", 10 * self.batch_size)
        self.dynamic_mlm_scale = params.get("dynamic_mlm_scale", False)
        self.buckets = params.get("buckets", None)

        # Multi-processing params.
        self.num_workers = params.get("num_workers", 0)
        self.drop_last = params.get("drop_last", True)
        self.prefetch_factor = params.get("prefetch_factor", 2)
        self.persistent_workers = params.get("persistent_workers", False)

        # Store params.
        self.mp_type = (
            half_dtype_instance.half_dtype
            if params.get("mixed_precision")
            else torch.float32
        )
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

        # Iterate over the data rows to create input features.
        for data_row in self.load_buffer():
            # `data_row` is a dict with keys:
            features = {
                "input_ids": np.array(
                    eval(data_row["input_ids"]), dtype=np.int32
                ),
                "masked_lm_mask": np.array(
                    # Stored as masked_lm_weights, but really masked_lm_mask
                    eval(data_row["masked_lm_weights"]),
                    dtype=np.int32,
                ),
                "masked_lm_positions": np.array(
                    eval(data_row["masked_lm_positions"]), dtype=np.int32
                ),
                "attention_mask": np.array(
                    eval(data_row["attention_mask"]), dtype=np.int32
                ),
                "labels": np.array(eval(data_row["labels"]), dtype=np.int32),
            }

            if not self.disable_nsp:
                features["next_sentence_label"] = np.array(
                    eval(data_row["next_sentence_label"]), dtype=np.int32
                )
                features["token_type_ids"] = np.array(
                    eval(data_row["token_type_ids"]), dtype=np.int32
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
                batch["mlm_loss_scale"] = scale.expand(self.batch_size, 1).to(
                    half_dtype_instance.half_dtype
                )
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

    def create_dataloader(self, is_training=True):
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
