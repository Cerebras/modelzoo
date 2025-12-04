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

# Adapted from: https://github.com/ad12/meddlr/blob/main/meddlr/ops/categorical.py#L13 (0f146c9)
# and https://github.com/ad12/meddlr/blob/main/meddlr/data/data_utils.py#L271 (13ff7ca)

import json
import os.path as osp
import random
from typing import Any, Literal, Optional

import h5py
import numpy as np
import torch
from pydantic import Field
from torchvision import transforms

from cerebras.modelzoo.data.vision.segmentation.Hdf5BaseIterDataProcessor import (
    Hdf5BaseIterDataProcessor,
    Hdf5BaseIterDataProcessorConfig,
)
from cerebras.modelzoo.data.vision.segmentation.preprocessing_utils import (
    normalize_tensor_transform,
)
from cerebras.modelzoo.data.vision.transforms import create_transform


class SkmDataProcessorConfig(Hdf5BaseIterDataProcessorConfig):
    data_processor: Literal["SkmDataProcessor"]

    echo_type: Literal[
        "echo1", "echo2", "echo1-echo2-mc", "root_sum_of_squares"
    ] = "echo1"

    aggregate_cartilage: bool = True

    split: Literal["train", "val"] = "train"

    overfit: Optional[Any] = Field(default=None, deprecated=True)


class SkmDataProcessor(Hdf5BaseIterDataProcessor):
    """
    A SKM-TEA MRI DICOM Track (Stanford MRI Dataset) Data Processor class for U-Net Segmentation.
    This class includes data preprocessing and transforms that are necessary for utilizing the
    SkmDicomDataset class for training models.

    Currently supports masks (segmentation) and does NOT support bounding boxes (detection).

    Args:
        params (dict): YAML config file with adaptable model and data configurations
    """

    def __init__(self, config: SkmDataProcessorConfig):
        super(SkmDataProcessor, self).__init__(config)
        self.echo_type = config.echo_type
        self.split = config.split
        self.aggregate_cartilage = config.aggregate_cartilage

    def _shard_files(self, is_training=False):
        split = "train" if is_training else "val"
        with open(
            osp.join(self.data_dir, f"annotations/v1.0.0/{split}.json")
        ) as f:
            dataset_dict = json.load(f)
            files = [
                osp.join(self.data_dir, "image_files", data["file_name"])
                for data in dataset_dict["images"]
            ]
            if not files:
                raise RuntimeError("No hdf5 datasets found")

        assert (
            len(files) >= self.num_tasks
        ), f"Number of h5 files {len(files)} should atleast be equal to the number of Slurm tasks {self.num_tasks}, to correctly shard the dataset between the streamers"

        # one example per file
        self.files_in_this_task = sorted(files)
        self.num_examples_in_this_task = len(self.files_in_this_task)

        assert (
            self.num_examples_in_this_task >= self.num_workers * self.batch_size
        ), f"The number of examples on this worker={self.num_examples_in_this_task} is lesser than batch size(={self.batch_size}) * num_workers(={self.num_workers}). Please consider reducing the number of workers (or) increasing the number of samples in files (or) reducing the batch size"

        if self.shuffle:
            random.seed(self.shuffle_seed)
            random.shuffle(self.files_in_this_task)

    def _load_buffer(self, data_partitions):
        for file_path in data_partitions:
            with h5py.File(file_path, mode="r") as h5_file:
                mask = h5_file["seg"][()]  # (6, 160, 512, 512)
                if self.echo_type in ["echo1-echo2-mc", "root_sum_of_squares"]:
                    echo1 = h5_file["echo1"][()]
                    echo2 = h5_file["echo2"][()]

                    if self.echo_type == "echo1-echo2-mc":
                        image = np.stack((echo1, echo2), axis=-1)
                    elif self.echo_type == "root_sum_of_squares":
                        # rss in fp32 to avoid underflow/overflow since echo1, echo2 are np.int16
                        image = (
                            (echo1.astype(np.float32) ** 2)
                            + (echo2.astype(np.float32) ** 2)
                        ) ** 0.5

                elif self.echo_type in ["echo1", "echo2"]:
                    image = h5_file[self.echo_type][()]
                yield (image, mask)

    def __iter__(self):
        for image, mask in self._load_buffer(self.data_partitions):
            image, mask = self.transform_image_and_mask(image, mask)
            yield image, mask

    def _shard_dataset(self, worker_id, num_workers):
        per_worker_partition = np.array_split(
            self.files_in_this_task, num_workers
        )[worker_id]
        return per_worker_partition

    def preprocess_image(self, image):
        # converts to (C, X, Y, Z) tensor
        def to_tensor(x):
            x = torch.as_tensor(np.array(x), dtype=torch.float32)
            if self.echo_type != "echo1-echo2-mc":
                x = x.unsqueeze(-1)
            return x.permute(3, 2, 0, 1)

        to_tensor_transform = transforms.Lambda(lambda x: to_tensor(x))

        def normalize(x):
            # normalize echo channel separately
            for echo_channel in range(len(x)):
                x[echo_channel] = normalize_tensor_transform(
                    x[echo_channel],
                    normalize_data_method=self.normalize_data_method,
                )
            return x

        normalize_transform = transforms.Lambda(lambda x: normalize(x))

        to_dtype_transform = create_transform(
            {"name": "to_dtype", "mp_type": self.mp_type}
        )

        transforms_list = [
            to_tensor_transform,
            normalize_transform,
            to_dtype_transform,
        ]
        image = transforms.Compose(transforms_list)(image)
        return image

    def preprocess_mask(self, mask):
        def aggregate_classes(one_hot_labels):
            """Aggregate medial and laterial tissues into one class"""
            patellar_labels = one_hot_labels[:, :, :, 0]
            femoral_labels = one_hot_labels[:, :, :, 1]
            tibial_labels = np.logical_or(
                one_hot_labels[:, :, :, 2], one_hot_labels[:, :, :, 3]
            )
            meniscus_labels = np.logical_or(
                one_hot_labels[:, :, :, 4], one_hot_labels[:, :, :, 5]
            )
            aggregated_labels = np.stack(
                (
                    patellar_labels,
                    femoral_labels,
                    tibial_labels,
                    meniscus_labels,
                ),
                axis=-1,
            )
            return aggregated_labels

        to_aggregate_transform = transforms.Lambda(
            lambda x: aggregate_classes(x)
        )

        def to_categorical(one_hot_labels):
            # labels are one-hot encodings where all zeros represents the background
            # add additional index for background class
            one_hot_labels = np.insert(one_hot_labels, 0, False, axis=-1)
            categorical_labels = np.argmax(one_hot_labels, axis=-1)
            return categorical_labels

        to_categorical_transform = transforms.Lambda(
            lambda x: to_categorical(x)
        )

        to_tensor_transform = transforms.ToTensor()

        to_dtype_transform = create_transform(
            {"name": "to_dtype", "mp_type": torch.int32}
        )

        transforms_list = (
            [to_aggregate_transform] if self.aggregate_cartilage else []
        )
        transforms_list.extend(
            [
                to_categorical_transform,
                to_tensor_transform,
                to_dtype_transform,
            ]
        )
        mask = transforms.Compose(transforms_list)(mask)
        return mask

    def transform_image_and_mask(self, image, mask):
        "Preprocess the masks and images"
        image = self.preprocess_image(image)
        mask = self.preprocess_mask(mask)

        return (image, mask)
