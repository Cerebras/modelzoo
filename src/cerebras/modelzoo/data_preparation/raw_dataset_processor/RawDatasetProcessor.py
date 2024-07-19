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
    This is Dataset process for processing Raw data set on the fly
    This contains methods for loading the dataset, tokenizing the dataset
    and all data transformations are handled as part of the collator function
"""

import random
from typing import Any, Dict, Iterator, List

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.data.common.input_utils import (
    num_tasks,
    shard_list_contiguous,
    task_id,
)
from cerebras.modelzoo.data_preparation.data_preprocessing.data_preprocessor import (
    DataPreprocessor,
)
from cerebras.modelzoo.data_preparation.raw_dataset_processor.utils import (
    Reader,
)


@registry.register_datasetprocessor("RawDatasetProcessor")
class RawDatasetProcessor(torch.utils.data.IterableDataset):
    def __init__(self, params: Dict[str, Any]):
        super(RawDatasetProcessor, self).__init__()
        self.params = params
        self.preprocessing_params = self.params.get("preprocessing", None)
        self.dataset_processor = DataPreprocessor(self.preprocessing_params)
        self.features_list = self.preprocessing_params["processing"].get(
            "features_list", ["input_ids", "attention_mask", "labels"]
        )
        self.num_workers = params.get("num_workers", 0)
        self.drop_last = params.get("drop_last", True)
        if self.num_workers == 0:
            self.prefetch_factor = None
        else:
            self.prefetch_factor = params.get("prefetch_factor", 10)
        self.persistent_workers = params.get("persistent_workers", True)
        self.reader = None
        self.batch_size = params.get("batch_size", None)
        self.seed = self.params.pop("seed", None)
        self.rng = random.Random(self.seed)
        self.reader = Reader(
            self.dataset_processor.input_files,
            keys=self.dataset_processor.data_keys,
            format_hook_fn=self.dataset_processor.format_hook_fn,
        )
        self.num_tasks = num_tasks()
        self.task_id = task_id()
        self.input_files_in_this_task = shard_list_contiguous(
            self.dataset_processor.input_files, self.task_id, self.num_tasks
        )

    def _worker_init_fn(self, worker_id: int):
        """
        Initialization function for each worker in a DataLoader.

        Args:
            worker_id (int): The ID of the current worker.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            # Single-process
            worker_id = 0
            num_workers = 1

        if self.seed is not None:
            # Use a unique seed for each worker.
            random.seed(self.seed + worker_id)

        # Shard the data files between workers
        self.input_files_in_this_worker = shard_list_contiguous(
            self.input_files_in_this_task, worker_id, num_workers
        )

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        """
        Returns an iterator over the items of the class.

        Returns:
            Iterator[Dict[str, np.ndarray]]: An iterator yielding dictionaries with string keys
                and NumPy array values.
        """
        return self.get_next_item()

    def get_next_item(self) -> Iterator[Dict[str, np.ndarray]]:
        """
        Returns the next item in the iteration.

        This function iterates over the data stream from the reader, tokenizes the data,
        and yields dictionaries containing features as keys and NumPy arrays as values.

        Returns:
            Iterator[Dict[str, np.ndarray]]: An iterator yielding dictionaries with string keys
            and NumPy array values.
        """
        for data in self.reader.stream_data():
            data_array = self.dataset_processor.format_hook_fn(data)

            # Tokenize the data and get stats
            tokenized_data, stats = (
                self.dataset_processor.token_generator.encode(data_array)
            )
            # Continue to next iteration if "data" key is not present
            if "data" not in tokenized_data.keys():
                continue
            # Iterate through the tokenized data and yield feature dictionary
            for d in tokenized_data["data"]:
                yield {
                    feature: np.array(d[i], np.int32)
                    for i, feature in enumerate(self.features_list)
                }

    def collate_fn(self, batch: List[Dict[str, np.ndarray]]) -> Any:
        """
        Collates a list of dictionaries into a batch

        Args:
            batch (List[Dict[str, np.ndarray]]): A list of dictionaries, where each dictionary
                contains string keys and NumPy array values.
        Returns:
            Any: The collated batch.
        """
        if self.dataset_processor.shuffle:
            random.shuffle(batch)
        return default_collate(batch)

    def create_dataloader(self) -> DataLoader:
        """
        Classmethod to create the dataloader object.

        Returns:
            DataLoader: A DataLoader object for the dataset.
        """
        # Create the DataLoader object with the specified parameters
        dataloader = DataLoader(
            self,
            batch_size=self.batch_size,  # Number of samples per batch
            drop_last=self.drop_last,  # Drop the last incomplete batch if the dataset size is not divisible by the batch size
            collate_fn=self.collate_fn,  # Function to merge a list of samples to form a mini-batch
            num_workers=self.num_workers,  # Number of subprocesses to use for data loading
            prefetch_factor=(
                self.prefetch_factor if self.num_workers > 0 else None
            ),  # Number of samples loaded in advance by each worker
            persistent_workers=(
                self.persistent_workers if self.num_workers > 0 else False
            ),  # Keep worker processes alive after they finish their tasks
            worker_init_fn=(
                self._worker_init_fn
                if self.num_workers > 0 and self.seed is not None
                else None
            ),  # Function to initialize the worker process
        )
        # set self.data_partitions in case self.num_workers == 0
        if self.num_workers == 0:
            self._worker_init_fn(0)

        return dataloader
