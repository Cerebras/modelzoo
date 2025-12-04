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

import logging
import os
import random
from typing import Any, Dict, Iterator, List, Literal, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, default_collate

from cerebras.modelzoo.config import DataConfig
from cerebras.modelzoo.config.types import ValidatedPath
from cerebras.modelzoo.data.common.input_utils import (
    num_tasks,
    shard_list_contiguous,
    task_id,
)
from cerebras.modelzoo.data.vision.preprocessing import get_preprocess_transform
from cerebras.modelzoo.data_preparation.data_preprocessing.data_preprocessor import (
    DataPreprocessor,
)
from cerebras.modelzoo.data_preparation.raw_dataset_processor.utils import (
    Reader,
)

LOGGER = logging.getLogger(__name__)


class RawDatasetProcessorConfig(DataConfig):
    """Configuration class for RawDatasetProcessor."""

    data_processor: Literal["RawDatasetProcessor"]
    "The name of the dataprocessor. Must be set to `RawDatasetProcessor`."

    batch_size: int = ...
    """
    Indicates global batch size of the model. This differs from `micro_batch_size`.
    Global batch size refers to the number of samples 
    fed to the model in total for forward and backward pass before
    weights are updated. A global batch size exceeding
    device memory is split into smaller `micro_batch_sizes` and 
    forward and backward pass are executed and accummulated before final update.
    """

    """ The dataset preprocessing configuration. """

    ##TODO: Create a config class for preprocessing as well
    preprocessing: dict = ...
    "Dict of params to be passed to DataPreprocessor class."

    shuffle: bool = True
    "Shuffle samples if True."

    shuffle_seed: int = 0
    "Seed to use for shuffling to ensure reproducibility."

    num_workers: int = 0
    "The number of PyTorch processes used in the dataloader."

    prefetch_factor: Optional[int] = 10
    "The number of batches to prefetch in the dataloader."

    persistent_workers: bool = True
    "Whether or not to keep workers persistent between epochs."

    drop_last: bool = True
    "Whether to drop the last incomplete batch."

    def post_init(self, context):
        if not self.num_workers:
            self.prefetch_factor = None  # the default value in DataLoader
            self.persistent_workers = False


class MultimodalRawDatasetProcessorConfig(RawDatasetProcessorConfig):
    """Multimodal Configuration class for RawDatasetProcessor."""

    data_processor: Literal["MultimodalRawDatasetProcessor"]
    "The name of the dataprocessor. Must be set to `MultimodalRawDatasetProcessor`."

    image_data_size: List[int] = ...
    "The final C x H x W shape of the image."

    transforms: List[dict] = ...
    "List of transformations to apply to images."

    img_data_dir: ValidatedPath = ...
    "The directory containing the image data."


class RawDatasetProcessor(torch.utils.data.IterableDataset):
    def __init__(self, config: RawDatasetProcessorConfig):
        if isinstance(config, dict):
            config = RawDatasetProcessorConfig(**config)
        super(RawDatasetProcessor, self).__init__()
        self.config = config
        self.batch_size = self.config.batch_size
        self.preprocessing_params = self.config.preprocessing

        self.dataset_processor = DataPreprocessor(self.preprocessing_params)
        self.features_list = self.dataset_processor.token_generator.features

        self.num_workers = self.config.num_workers
        self.prefetch_factor = self.config.prefetch_factor
        self.persistent_workers = self.config.persistent_workers

        self.reader = Reader(
            self.dataset_processor.input_files,
            keys=self.dataset_processor.data_keys,
            read_hook_fn=self.dataset_processor.read_hook_fn,
        )

        self.shuffle_seed = self.config.shuffle_seed
        self.rng = random.Random(self.shuffle_seed)
        self.input_files_in_this_task = shard_list_contiguous(
            self.dataset_processor.input_files, task_id(), num_tasks()
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

        if self.shuffle_seed is not None:
            # Use a unique seed for each worker.
            random.seed(self.shuffle_seed + worker_id)

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
            data_array = self.dataset_processor.read_hook_fn(data)
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
            drop_last=self.config.drop_last,  # Drop the last incomplete batch if the dataset size is not divisible by the batch size
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
                if self.num_workers > 0 and self.shuffle_seed is not None
                else None
            ),  # Function to initialize the worker process
        )
        # set self.data_partitions in case self.num_workers == 0
        if self.num_workers == 0:
            self._worker_init_fn(0)

        return dataloader


class MultimodalRawDatasetProcessor(RawDatasetProcessor):
    """Dataset processor for multimodal data (e.g., image data)."""

    def __init__(self, config: MultimodalRawDatasetProcessorConfig):
        if isinstance(config, dict):
            config = MultimodalRawDatasetProcessorConfig(**config)
        super(MultimodalRawDatasetProcessor, self).__init__(config)
        self.img_data_dir = self.config.img_data_dir
        self.image_data_size = self.config.image_data_size
        self.transforms = get_preprocess_transform(
            {
                "transforms": self.config.transforms,
            }
        )
        self.image_data_size = self.config.image_data_size
        self.transforms = get_preprocess_transform(
            {
                "transforms": self.config.transforms,
            }
        )

    def preprocess_img(self, path_list):

        img_list = []
        for img_paths in path_list:
            imgs_per_sample_list = []
            ## iterate over all the image paths in 1 data sample
            for path in img_paths:
                path = path.decode("utf-8")
                if path != "None":
                    image_path = os.path.join(self.img_data_dir, path)
                    image = Image.open(image_path).convert("RGB")
                else:
                    image = Image.new(
                        mode="RGB",
                        size=(self.image_data_size[2], self.image_data_size[1]),
                    )
                imgs_per_sample_list.append(self.transforms(image).unsqueeze(0))
            imgs_per_sample = torch.cat(
                imgs_per_sample_list, dim=0
            )  ## shape - max_num_img * C * H * W
            img_list.append(imgs_per_sample.unsqueeze(0))

        img = torch.cat(
            img_list, dim=0
        )  ## shape - batch_size * max_num_img * C * H * W
        return img

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
            data_array = self.dataset_processor.read_hook_fn(data)
            # Tokenize the data and get stats
            tokenized_data, stats = (
                self.dataset_processor.token_generator.encode(data_array)
            )

            # Continue to next iteration if "data" key is not present
            if "data" not in tokenized_data.keys():
                continue

            # Apply image transformation
            tokenized_data['image_data'] = self.preprocess_img(
                tokenized_data['img_path']
            )
            for i in range(len(tokenized_data["data"])):
                data = {
                    feature: np.array(
                        tokenized_data["data"][i][feature_idx], np.int32
                    )
                    for feature_idx, feature in enumerate(self.features_list)
                }
                data.update(
                    {
                        "image_data": tokenized_data["image_data"][i],
                        "image_data_loc": tokenized_data["img_data_loc"][i],
                    }
                )
                yield data
