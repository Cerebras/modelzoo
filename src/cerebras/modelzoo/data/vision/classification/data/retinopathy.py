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

import csv
import os
from typing import Any, Literal, Optional

import cv2
import numpy as np
from PIL import Image
from pydantic import Field
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.vision import VisionDataset

from cerebras.modelzoo.data.vision.classification.dataset_factory import (
    VisionClassificationProcessor,
    VisionClassificationProcessorConfig,
)


class DiabeticRetinopathy(VisionDataset):
    _TARGET_PIXELS = {
        "original": None,
        "1M": 1000000,
        "250K": 250000,
        "btgraham-300": 300,
    }
    _LABEL_FILE = {
        "train": "trainLabels.csv",
        "val": "retinopathy_solution.csv",
        "test": "retinopathy_solution.csv",
    }

    def __init__(
        self,
        root,
        split="train",
        config="btgraham-300",
        use_heavy_train_aug=False,
        transform=None,
        target_transform=None,
    ):
        super().__init__(
            os.path.join(root, "retinopathy"),
            transform=transform,
            target_transform=target_transform,
        )
        if not os.path.exists(self.root):
            raise RuntimeError(
                "Dataset not found. Download from "
                "https://www.kaggle.com/c/diabetic-retinopathy-detection/data"
            )
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        verify_str_arg(config, "config", self._TARGET_PIXELS.keys())
        self._use_heavy_aug = use_heavy_train_aug

        if config == "btgraham-300":
            self.img_processing = _btgraham_processing
        else:
            self.img_processing = _resize_image_if_necessary

        self.target_pixels = self._TARGET_PIXELS[config]

        self._image_files = []
        self.labels = []

        label_file = os.path.join(self.root, self._LABEL_FILE[split])
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                content = csv.reader(f, delimiter=",")
                next(content, None)  # skip the headers
                for line in content:
                    if split == "train":
                        self._image_files.append(line[0])
                        self.labels.append(float(line[1]))
                    else:
                        assert len(line) == 3
                        if (split == "val" and line[2] == "Public") or (
                            split == "test" and line[2] == "Private"
                        ):
                            self._image_files.append(line[0])
                            self.labels.append(float(line[1]))
        else:
            if split == "train" or split == "val":
                raise RuntimeError(
                    f"Missing label file {label_file} for {split} split"
                )

            for img_file in os.listdir(os.path.join(self.root), split):
                if img_file.endswith(".jpeg"):
                    self._image_files.append(img_file[:-5])
            self.labels = [-1] * len(self._image_files)

    def __getitem__(self, idx):
        image_file = self._image_files[idx]
        split_dir = "train" if self._split == "train" else "test"
        image = cv2.imread(
            os.path.join(self.root, split_dir, f"{image_file}.jpeg"), flags=3
        )
        image = self.img_processing(image, self.target_pixels).astype("uint8")
        # convert the color from BGR (cv2 format) to RGB (PIL format)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self._use_heavy_aug:
            image = _heavy_data_augmentation(np.array(image))

        image = Image.fromarray(image.astype("uint8"), 'RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self._image_files)


def _btgraham_processing(image, target_pixels, crop_to_radius=False):
    """Process an image as the winner of the 2015 Kaggle competition."""
    image = _scale_radius_size(image, target_radius_size=target_pixels)
    image = _subtract_local_average(image, target_radius_size=target_pixels)
    image = _mask_and_crop_to_radius(
        image,
        target_radius_size=target_pixels,
        radius_mask_ratio=0.9,
        crop_to_radius=crop_to_radius,
    )
    return image


def _resize_image_if_necessary(image, target_pixels=None):
    """Resize an image to have (roughly) the given number of target pixels."""
    if target_pixels is None:
        return image
    # Get image height and width.
    height, width, _ = image.shape
    actual_pixels = height * width
    if actual_pixels > target_pixels:
        factor = np.sqrt(target_pixels / actual_pixels)
        image = cv2.resize(image, dsize=None, fx=factor, fy=factor)
    return image


def _scale_radius_size(image, target_radius_size):
    x = image[image.shape[0] // 2, :, :].sum(axis=1)
    r = (x > x.mean() / 10.0).sum() / 2.0
    if r < 1.0:
        # Some images in the dataset are corrupted, causing the radius heuristic
        # to fail. In these cases, just assume that the radius is the height of
        # the original image.
        r = image.shape[0] / 2.0
    s = target_radius_size / r
    return cv2.resize(image, dsize=None, fx=s, fy=s)


def _subtract_local_average(image, target_radius_size):
    image_blurred = cv2.GaussianBlur(image, (0, 0), target_radius_size / 30)
    image = cv2.addWeighted(image, 4, image_blurred, -4, 128)
    return image


def _mask_and_crop_to_radius(
    image, target_radius_size, radius_mask_ratio=0.9, crop_to_radius=False
):
    """Mask and crop image to the given radius ratio."""
    mask = np.zeros(image.shape)
    center = (image.shape[1] // 2, image.shape[0] // 2)
    radius = int(target_radius_size * radius_mask_ratio)
    cv2.circle(
        mask, center=center, radius=radius, color=(1, 1, 1), thickness=-1
    )
    image = image * mask + (1 - mask) * 128
    if crop_to_radius:
        x_max = min(image.shape[1] // 2 + radius, image.shape[1])
        x_min = max(image.shape[1] // 2 - radius, 0)
        y_max = min(image.shape[0] // 2 + radius, image.shape[0])
        y_min = max(image.shape[0] // 2 - radius, 0)
        image = image[y_min:y_max, x_min:x_max, :]
    return image


def _sample_heavy_data_augmentation_parameters():
    # Scale image +/- 10%.
    s = np.random.uniform(-0.1, 0.1)
    # Rotate image [0, 2pi).
    a = np.random.uniform(0.0, 2.0 * np.pi)
    # Vertically shear image +/- 20%.
    b = np.random.uniform(-0.2, 0.2) + a
    # Horizontal and vertial flipping.
    flip = [-1.0, 1.0]
    np.random.shuffle(flip)
    hf = flip[0]
    np.random.shuffle(flip)
    vf = flip[0]
    # Relative x,y translation.
    dx = np.random.uniform(-0.1, 0.1)
    dy = np.random.uniform(-0.1, 0.1)
    return s, a, b, hf, vf, dx, dy


def _heavy_data_augmentation(image):
    height = float(image.shape[0])
    width = float(image.shape[1])

    # sample data augmentation parameters
    s, a, b, hf, vf, dx, dy = _sample_heavy_data_augmentation_parameters()
    # Rotation + scale.
    c00 = (1 + s) * np.cos(a)
    c01 = (1 + s) * np.sin(a)
    c10 = (s - 1) * np.sin(b)
    c11 = (1 - s) * np.cos(b)
    # Horizontal and vertial flipping.
    c00 = c00 * hf
    c01 = c01 * hf
    c10 = c10 * vf
    c11 = c11 * vf
    # Convert x,y translation to absolute values.
    dx = width * dx
    dy = height * dy
    # Convert affine matrix to TF's transform. Matrix is applied w.r.t. the
    # center of the image.
    cy = height / 2.0
    cx = width / 2.0
    affine_matrix = np.array(
        [
            [c00, c01, (1.0 - c00) * cx - c01 * cy + dx],
            [c10, c11, (1.0 - c11) * cy - c10 * cx + dy],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    affine_inv_matrix = np.linalg.inv(affine_matrix)
    # Since background is grey in these configs, put in pixels in [-1, 1]
    # range to avoid artifacts from the affine transformation.
    image = image.astype(float)
    image = (image / 127.5) - 1.0
    # Apply the affine transformation.
    image = np.matmul(image, affine_inv_matrix)

    # Put pixels back to [0, 255] range and cast to uint8, since this is what
    # our preprocessing pipeline usually expects.
    image = (1.0 + image) * 127.5
    return image


class DiabeticRetinopathyProcessorConfig(VisionClassificationProcessorConfig):
    data_processor: Literal["DiabeticRetinopathyProcessor"]

    use_worker_cache: bool = ...

    split: Literal["train", "val", "test"] = "train"
    "Dataset split."

    config: str = "btgraham-300"

    num_classes: Optional[Any] = Field(None, deprecated=True)


class DiabeticRetinopathyProcessor(VisionClassificationProcessor):
    def __init__(self, config: DiabeticRetinopathyProcessorConfig):
        super().__init__(config)
        self.split = config.split
        self.config = config.config
        self.shuffle = self.shuffle and (self.split == "train")
        self.num_classes = 5

    def create_dataset(self):
        use_training_transforms = self.split == "train"
        transform, target_transform = self.process_transform(
            use_training_transforms
        )
        dataset = DiabeticRetinopathy(
            root=self.data_dir,
            split=self.split,
            config=self.config,
            transform=transform,
            target_transform=target_transform,
        )
        return dataset
