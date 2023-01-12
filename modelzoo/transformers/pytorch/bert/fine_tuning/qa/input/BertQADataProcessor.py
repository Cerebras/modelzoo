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
Processor for PyTorch BERT fine tuning - SQuAD (QA)
"""
import csv
import logging

import numpy as np
import torch

from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch.run_utils import half_dtype_instance
from modelzoo.transformers.data_processing.utils import convert_str_to_int_list
from modelzoo.transformers.pytorch.bert.input.utils import (
    get_meta_data,
    shard_and_shuffle_data,
)
from modelzoo.transformers.pytorch.input_utils import get_data_for_task, task_id


class BertQADataProcessor(torch.utils.data.IterableDataset):
    """
    Reads csv file containing the input token ids, and label_ids. 
    Creates attention_masks and sedment_ids on the fly

    Args:
        params: dict containing input parameters for creating dataset.
    Expects the following fields:

    - "data_dir" (str or list of str): Path to the metadata files.
    - "batch_size" (int): Batch size.
    - "shuffle" (bool): Flag to enable data shuffling.
    - "shuffle_buffer" (int): Shuffle buffer size.
    - "shuffle_seed" (int): Shuffle seed.
    - "num_workers" (int): Number of PyTorch data workers (see PyTorch docs).
    - "prefetch_factor" (int): How much data to prefetch.
       for better performance (see PyTorch docs).
    - "persistent_workers" (bool): For multi-worker dataloader controls if the
       workers are recreated at the end of each epoch ((see PyTorch docs).
    - "max_sequence_length" (int): Maximum sequence length for the model.
    """

    def __init__(self, params):
        super(BertQADataProcessor, self).__init__()

        self.use_cs = cm.use_cs()
        # Input params.
        self.meta_data = get_meta_data(params["data_dir"])

        self.meta_data_values = list(self.meta_data.values())
        self.meta_data_filenames = list(self.meta_data.keys())
        # Please note the appending of [0]
        self.meta_data_values_cum_sum = np.cumsum([0] + self.meta_data_values)

        self.num_examples = sum(map(int, self.meta_data.values()))
        self.batch_size = params["batch_size"]

        self.num_batches = self.num_examples // self.batch_size
        assert (
            self.num_batches > 0
        ), "Dataset does not contain enough samples for one batch. Please choose a smaller batch size"
        self.num_tasks = cm.num_streamers() if cm.is_streamer() else 1
        self.num_batch_per_task = self.num_batches // self.num_tasks

        assert (
            self.num_batch_per_task > 0
        ), "Dataset cannot be evenly distributed across the given tasks. Please choose fewer tasks to run with"

        self.num_examples_per_task = self.num_batch_per_task * self.batch_size
        self.files_in_task = get_data_for_task(
            task_id(),
            self.meta_data_values_cum_sum,
            self.num_examples_per_task,
            self.meta_data_values,
            self.meta_data_filenames,
        )

        self.shuffle = params.get("shuffle", True)
        self.shuffle_seed = params.get("shuffle_seed", None)
        self.shuffle_buffer = params.get("shuffle_buffer", 10 * self.batch_size)

        # Multi-processing params.
        self.num_workers = params.get("num_workers", 0)
        self.drop_last = params.get("drop_last", True)
        self.prefetch_factor = params.get("prefetch_factor", 10)
        self.persistent_workers = params.get("persistent_workers", True)

        if self.num_workers > len(self.files_in_task):
            reduced_workers = len(self.files_in_task)
            if task_id() == 0:
                logging.warning(
                    f"Not enough samples in dataset to shard across "
                    f"{self.num_workers} workers. Reducing the num_workers  "
                    f"to: {reduced_workers}. This will be applied to all "
                    f"{self.num_tasks} task(s)."
                )
            self.num_workers = reduced_workers

        # Store params.
        self.mp_type = (
            half_dtype_instance.half_dtype
            if params.get("mixed_precision")
            else torch.float32
        )
        self.data_buffer = []
        self.csv_files_per_task_per_worker = []
        self.processed_buffers = 0

        self.max_sequence_length = params["max_sequence_length"]

    def create_dataloader(self, is_training=True):
        """
        Classmethod to create the dataloader object.
        """
        if self.num_workers:
            dataloader = torch.utils.data.DataLoader(
                self,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=self.persistent_workers,
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, drop_last=self.drop_last,
            )

        return dataloader

    def load_buffer(self):
        """
        Generator to read the data in chunks of size of data_buffer.

        Returns: Yields the data stored in the data_buffer.
        """
        self.data_buffer = []
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
        """
        Returns the length of the dataset on task process.
        """
        return self.num_examples_per_task

    def __iter__(self):
        """
        Iterator over the data to construct input features.

        Return: A tuple with training features:
            * np.array[int.32] input_ids: Numpy array with input token indices.
                Shape: (`max_sequence_length`).
            * np.array[int.32] labels: Numpy array with labels.
               Shape: (`max_sequence_length`).
            * np.array[int.32] attention_mask
               Shape: (`max_sequence_length`).
            * np.array[int.32] token_type_ids: Numpy array with segment indices.
               Shape: (`max_sequence_length`).
        """
        (
            self.processed_buffers,
            self.csv_files_per_task_per_worker,
            self.shuffle_seed,
            self.rng,
        ) = shard_and_shuffle_data(
            self.files_in_task, self.shuffle, self.shuffle_seed,
        )

        # Iterate over the data rows to create input features.
        for data_row in self.load_buffer():
            data_parsed = parse_qa_row(data_row)
            features = create_qa_features(data_parsed, self.max_sequence_length)
            yield features


def parse_qa_row(data_row):
    """
    Processing of a row in the CSV file.

    Args:
        dict data_row: dictionary with input values.

    Returns: tuple: (list of parsed tokens, List of labels).
    """

    data = {
        "input_ids": convert_str_to_int_list(data_row["input_ids"]),
        "attention_mask": convert_str_to_int_list(data_row["input_mask"]),
        "token_type_ids": convert_str_to_int_list(data_row["segment_ids"]),
    }

    do_prediction = False if data_row["start_position"] else True

    if do_prediction:
        # During prediction, these ids are used to match windows
        # with specific examples.
        data["unique_ids"] = int(data_row["unique_id"])
    else:
        # During prediction, there are no labels saved
        data["labels"] = [
            int(data_row["start_position"]),
            int(data_row["end_position"]),
        ]

    return data


def create_qa_features(data, max_sequence_length):
    """
    Creates features dictionary of numpy arrays.
    """
    features = dict()
    for k, v in data.items():
        features[k] = np.array(v, dtype=np.int32)
    if "labels" in features:
        features["label_weights"] = (
            features["labels"] < max_sequence_length
        ).astype(np.float16)
    else:
        features["labels"] = np.array([0, 0], dtype=np.float16)
        features["label_weights"] = np.array([0, 0], dtype=np.float16)
    return features
