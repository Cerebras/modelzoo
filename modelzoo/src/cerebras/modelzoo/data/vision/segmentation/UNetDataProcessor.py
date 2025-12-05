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

import logging
from typing import List, Literal, Optional, Union

import matplotlib.pyplot as plt
import torch
from pydantic import field_validator
from torchvision import transforms

import cerebras.pytorch as cstorch
from cerebras.modelzoo.common.input_utils import get_streaming_batch_size
from cerebras.modelzoo.config import DataConfig
from cerebras.modelzoo.config.types import ValidatedPath
from cerebras.modelzoo.data.vision.segmentation.preprocessing_utils import (
    adjust_brightness_transform,
    normalize_tensor_transform,
    rotation_90_transform,
    tile_image_transform,
)
from cerebras.modelzoo.data.vision.utils import (
    FastDataLoader,
    ShardedSampler,
    num_tasks,
    task_id,
)


class UNetDataProcessorConfig(DataConfig):
    data_processor: Literal["UNetDataProcessor"]

    data_dir: Union[ValidatedPath, List[ValidatedPath]] = ...

    num_classes: Optional[int] = None

    loss: Optional[Literal["bce", "multilabel_bce", "ssce", "ssce_dice"]] = None

    normalize_data_method: Optional[
        Literal["zero_centered", "zero_one", "standard_score"]
    ] = None

    shuffle: bool = True

    shuffle_seed: Optional[int] = None

    augment_data: bool = True

    batch_size: int = ...

    num_workers: int = 0

    image_shape: Optional[List[int]] = None

    drop_last: bool = True

    prefetch_factor: Optional[int] = 10

    persistent_workers: bool = True

    use_fast_dataloader: bool = True

    duplicate_act_worker_data: bool = False

    convert_to_onehot: Optional[bool] = None

    @field_validator("convert_to_onehot")
    @classmethod
    def set_convert_to_onehot(cls, convert_to_onehot, info):
        if info.context:
            model_config = info.context.get("model", {}).get("config")
            if convert_to_onehot is None:
                convert_to_onehot = model_config.loss == "multilabel_bce"

        return convert_to_onehot

    def post_init(self, context):
        model_config = context.get("model", {}).get("config")

        if model_config is not None:
            if hasattr(model_config, "image_shape"):
                self.image_shape = model_config.image_shape

            if hasattr(model_config, "num_classes"):
                self.num_classes = model_config.num_classes

            if "loss" in self.model_fields_set:
                logging.warning(
                    "Loss cannot be set in data configuration. "
                    "Defaulting to value in model configuration."
                )
            if hasattr(model_config, "loss"):
                self.loss = model_config.loss

        if any(
            x is None
            for x in [
                self.image_shape,
                self.num_classes,
                self.loss,
            ]
        ):
            raise ValueError(
                "image_shape, num_classes and loss must "
                "be configured from the model config."
            )


class UNetDataProcessor:
    def __init__(self, config: UNetDataProcessorConfig):
        self.data_dir = config.data_dir

        self.num_classes = config.num_classes

        self.loss_type = config.loss
        self.normalize_data_method = config.normalize_data_method

        self.shuffle_seed = config.shuffle_seed
        if self.shuffle_seed is not None:
            torch.manual_seed(self.shuffle_seed)

        self.augment_data = config.augment_data
        self.batch_size = get_streaming_batch_size(config.batch_size)
        self.shuffle = config.shuffle

        # Multi-processing params.
        self.num_workers = config.num_workers
        self.drop_last = config.drop_last
        self.prefetch_factor = config.prefetch_factor
        self.persistent_workers = config.persistent_workers

        self.mp_type = cstorch.amp.get_floating_point_dtype()

        # Using Faster Dataloader for mapstyle dataset.
        self.use_fast_dataloader = config.use_fast_dataloader

        # Each activation worker can access entire dataset when True
        self.duplicate_act_worker_data = config.duplicate_act_worker_data

    def create_dataset():
        raise NotImplementedError(
            "create_dataset must be implemented in a child class!!"
        )

    def create_dataloader(self):
        dataset = self.create_dataset()

        generator_fn = torch.Generator(device="cpu")
        if self.shuffle_seed is not None:
            generator_fn.manual_seed(self.shuffle_seed)

        self.disable_sharding = False
        samples_per_task = len(dataset) // num_tasks()
        if self.batch_size > samples_per_task:
            print(
                f"Dataset size: {len(dataset)} too small for num_tasks:"
                f" {num_tasks} and batch_size: {self.batch_size},"
                " using duplicate data for activation workers..."
            )
            self.disable_sharding = True

        if self.shuffle:
            if self.duplicate_act_worker_data or self.disable_sharding:
                # Multiples activation workers, each sending same data in different
                # order since the dataset is extremely small
                if self.shuffle_seed is None:
                    seed = task_id()
                else:
                    seed = self.shuffle_seed + task_id()

                generator_fn.manual_seed(seed)
                data_sampler = torch.utils.data.RandomSampler(
                    dataset, generator=generator_fn
                )
            else:
                data_sampler = ShardedSampler(
                    dataset, self.shuffle, self.shuffle_seed, self.drop_last
                )
        else:
            data_sampler = torch.utils.data.SequentialSampler(dataset)

        if self.use_fast_dataloader:
            dataloader_fn = FastDataLoader
            print("-- Using FastDataloader -- ")
        else:
            dataloader_fn = torch.utils.data.DataLoader
            print("-- Using torch.utils.data.DataLoader -- ")

        if self.num_workers:
            dataloader = dataloader_fn(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=self.persistent_workers,
                drop_last=self.drop_last,
                generator=generator_fn,
                sampler=data_sampler,
            )
        else:
            dataloader = dataloader_fn(
                dataset,
                batch_size=self.batch_size,
                drop_last=self.drop_last,
                generator=generator_fn,
                sampler=data_sampler,
            )
        return dataloader

    def transform_image_and_mask(self, image, mask):
        image = self.preprocess_image(image)
        mask = self.preprocess_mask(mask)

        if self.augment_data:
            do_horizontal_flip = torch.rand(size=(1,)).item() > 0.5
            # n_rots in range [0, 3)
            n_rotations = torch.randint(low=0, high=3, size=(1,)).item()

            if self.tgt_image_height != self.tgt_image_width:
                # For a rectangle image
                n_rotations = n_rotations * 2

            augment_transform_image = self.get_augment_transforms(
                do_horizontal_flip=do_horizontal_flip,
                n_rotations=n_rotations,
                do_random_brightness=True,
            )
            augment_transform_mask = self.get_augment_transforms(
                do_horizontal_flip=do_horizontal_flip,
                n_rotations=n_rotations,
                do_random_brightness=False,
            )

            image = augment_transform_image(image)
            mask = augment_transform_mask(mask)

        # Handle dtypes and mask shapes based on `loss_type`
        # and `mixed_precsion`

        if self.loss_type == "bce":
            mask = mask.to(self.mp_type)
        elif self.loss_type == "multilabel_bce":
            mask = torch.squeeze(mask, 0)
            # Only long tensors are accepted by one_hot fcn.
            mask = mask.to(torch.long)

            # out shape: ((D), H, W, num_classes)
            mask = torch.nn.functional.one_hot(
                mask, num_classes=self.num_classes
            )
            # out shape: (num_classes, (D), H, W)
            mask_axes = [_ for _ in range(len(mask.shape))]
            mask = torch.permute(mask, mask_axes[-1:] + mask_axes[0:-1])
            mask = mask.to(self.mp_type)

        elif self.loss_type == "ssce":
            # out shape: ((D), H, W) with each value in [0, num_classes)
            mask = torch.squeeze(mask, 0)

            mask = mask.to(torch.int32)
        if cstorch.amp.mixed_precision():
            image = image.to(self.mp_type)

        return image, mask

    def preprocess_image(self, image):

        # converts to (C, (D), H, W) format.
        to_tensor_transform = transforms.PILToTensor()

        # Resize and convert to torch.Tensor
        resize_pil_transform = transforms.Resize(
            [self.tgt_image_height, self.tgt_image_width],
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )
        # Tiling when image shape qualifies
        tile_transform = self.get_tile_transform()

        # Normalize
        normalize_transform = transforms.Lambda(
            lambda x: normalize_tensor_transform(
                x, normalize_data_method=self.normalize_data_method
            )
        )
        transforms_list = [
            to_tensor_transform,
            resize_pil_transform,
            tile_transform,
            normalize_transform,
        ]
        image = transforms.Compose(transforms_list)(image)
        return image

    def preprocess_mask(self, mask):
        tile_transform = self.get_tile_transform()
        return tile_transform(mask)

    def get_augment_transforms(
        self, do_horizontal_flip, n_rotations, do_random_brightness
    ):

        augment_transforms_list = []
        if do_horizontal_flip:
            horizontal_flip_transform = transforms.Lambda(
                lambda x: transforms.functional.hflip(x)
            )
            augment_transforms_list.append(horizontal_flip_transform)

        if n_rotations > 0:
            rotation_transform = transforms.Lambda(
                lambda x: rotation_90_transform(x, num_rotations=n_rotations)
            )
            augment_transforms_list.append(rotation_transform)

        if do_random_brightness:
            brightness_transform = transforms.Lambda(
                lambda x: adjust_brightness_transform(x, p=0.5, delta=0.2)
            )
            augment_transforms_list.append(brightness_transform)

        return transforms.Compose(augment_transforms_list)

    @property
    def tiling_image_shape(self):
        if not hasattr(self, "_tiling_image_shape"):
            raise AttributeError(
                "_tiling_image_shape not defined. "
                + "Please set it in __init__ method of DataProcessor of child class. "
                + "Format is (H, W, C)"
            )
        return self._tiling_image_shape

    def get_tile_transform(self):
        tiling_image_height, tiling_image_width = (
            self.tiling_image_shape[0],
            self.tiling_image_shape[1],
        )
        tile_transform = transforms.Lambda(
            lambda x: tile_image_transform(
                x, tiling_image_height, tiling_image_width
            )
        )
        return tile_transform


def visualize_dataset(dataset, num_samples=3):
    figure = plt.figure(figsize=(10, 10))
    rows, cols = num_samples, 2
    for i in range(1, cols * rows + 1, 2):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.axis("off")
        plt.imshow(img.permute(1, 2, 0) / torch.max(img))

        figure.add_subplot(rows, cols, i + 1)
        plt.axis("off")
        plt.imshow(label / torch.max(label))
