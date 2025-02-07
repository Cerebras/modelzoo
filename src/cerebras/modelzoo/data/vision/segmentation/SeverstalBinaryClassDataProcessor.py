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

import os
import tempfile
from typing import List, Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import VisionDataset

import cerebras.pytorch as cstorch
import cerebras.pytorch.distributed as dist
from cerebras.modelzoo.data.vision.segmentation.UNetDataProcessor import (
    UNetDataProcessor,
    UNetDataProcessorConfig,
)
from cerebras.modelzoo.data.vision.utils import create_worker_cache


class SeverstalBinaryClassDataset(VisionDataset):
    def __init__(
        self,
        root,
        train_test_split,
        class_id_to_consider,
        split="train",
        transforms=None,
        transform=None,
        target_transform=None,
        use_worker_cache=False,
    ):
        super(SeverstalBinaryClassDataset, self).__init__(
            root, transforms, transform, target_transform
        )
        self.train_test_split = train_test_split
        assert class_id_to_consider <= 4, "Maximum 4 available classes."
        self.class_id_to_consider = class_id_to_consider

        if use_worker_cache and dist.is_streamer():
            if not cstorch.use_cs():
                raise RuntimeError(
                    "use_worker_cache not supported for non-CS runs"
                )
            else:
                self.root = create_worker_cache(self.root)

        self.data_dir = self.root

        if split not in ["train", "val"]:
            raise ValueError(
                f"Invalid value={split} passed to `split` argument. "
                f"Valid are 'train' or 'val'"
            )
        self.split = split

        self.images_dir, self.csv_file_path = self._get_data_dirs()
        train_dataframe, val_dataframe = self._process_csv_file()

        if split == "train":
            self.data = train_dataframe
        elif split == "val":
            self.data = val_dataframe

    def _get_data_dirs(self):

        images_dir = os.path.join(self.root, "train_images")
        csv_file = os.path.join(self.root, "train.csv")

        return images_dir, csv_file

    def _process_csv_file(self):
        """
        Function to read contents to csv file and make dataset splits
        """
        csv_data = pd.read_csv(self.csv_file_path)
        csv_data = csv_data[csv_data["ClassId"] == self.class_id_to_consider]

        self.total_rows = len(csv_data.index)

        try:
            temp_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
            self.class_id_dataset_path = temp_file.name
        finally:
            temp_file.close()

        csv_data.to_csv(self.class_id_dataset_path, index=True)

        # Get train-test splits.
        train_rows = int(np.floor(self.train_test_split * self.total_rows))

        train_data = csv_data[:train_rows]
        val_data = csv_data[train_rows:]

        return train_data, val_data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item.
        """
        image_filename, class_id, encoded_pixels = self.data.iloc[index]
        image_file_path = os.path.join(self.images_dir, image_filename)
        image = Image.open(image_file_path).convert("L")  # PILImage

        # (W, H) = (1600, 256) is the standard image size for this dataset
        _W = 1600
        _H = 256
        target = torch.zeros(_W * _H, dtype=torch.int32)
        rle_list = encoded_pixels.split()  # Run Length Encoding

        if rle_list[0] != "-1":
            rle_numbers = [int(x) for x in rle_list]
            start_pixels = rle_numbers[::2]
            lengths = rle_numbers[1::2]
            # EncodedPixels are numbered from top to bottom,
            # then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc
            # Refer to: https://www.kaggle.com/c/severstal-steel-defect-detection/overview/evaluation

            for start, lgth in zip(start_pixels, lengths):
                start_loc = start - 1  # Since one-based encoding
                target[start_loc : start_loc + lgth] = 1

        target = torch.reshape(target, (_W, _H))
        target = torch.transpose(target, 0, 1)
        target = torch.unsqueeze(target, dim=0)  # outshape: (C, H, W)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.data.index)


class SeverstalBinaryClassDataProcessorConfig(UNetDataProcessorConfig):
    data_processor: Literal["SeverstalBinaryClassDataProcessor"]

    use_worker_cache: bool = False

    train_test_split: float = ...

    class_id: int = ...

    max_image_shape: List[int] = [256, 1600]

    split: Literal["train", "val"] = "train"
    "Dataset split."


class SeverstalBinaryClassDataProcessor(UNetDataProcessor):
    def __init__(self, config: SeverstalBinaryClassDataProcessorConfig):
        super(SeverstalBinaryClassDataProcessor, self).__init__(config)

        self.split = config.split
        self.shuffle = self.shuffle and (self.split == "train")

        self.use_worker_cache = config.use_worker_cache
        self.train_test_split = config.train_test_split
        self.class_id_to_consider = config.class_id

        self.image_shape = config.image_shape  # of format (H, W, C)
        self._tiling_image_shape = self.image_shape  # out format: (H, W, C)

        # Tiling param:
        # If `image_shape` < 1K x 2K, do not tile.
        # If `image_shape` > 1K x 2K in any dimension,
        #   first resize image to min(img_shape, max_image_shape)
        #   and then tile to target height and width specified in yaml
        # self.max_image_shape: (H, W)
        self.max_image_shape = config.max_image_shape
        self.image_shape = self._update_image_shape()
        (
            self.tgt_image_height,
            self.tgt_image_width,
            self.channels,
        ) = self.image_shape

    def _update_image_shape(self):

        # image_shape is of format (H, W, C)
        image_shape = []
        for i in range(2):
            image_shape.append(
                min(self.image_shape[i], self.max_image_shape[i])
            )
        image_shape = (
            image_shape + self.image_shape[-1:]
        )  # Output shape format (H, W, C)

        return image_shape

    def create_dataset(self):
        dataset = SeverstalBinaryClassDataset(
            root=self.data_dir,
            train_test_split=self.train_test_split,
            class_id_to_consider=self.class_id_to_consider,
            split=self.split,
            transforms=self.transform_image_and_mask,
            use_worker_cache=self.use_worker_cache,
        )
        return dataset

    def preprocess_mask(self, mask):

        # Resize
        resize_transform = transforms.Resize(
            [self.tgt_image_height, self.tgt_image_width],
            interpolation=transforms.InterpolationMode.NEAREST,
        )
        tile_transform = self.get_tile_transform()
        transforms_list = [resize_transform, tile_transform]
        mask = transforms.Compose(transforms_list)(mask)

        return mask
