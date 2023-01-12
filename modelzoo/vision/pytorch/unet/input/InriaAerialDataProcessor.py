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
import os
import random

import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import VisionDataset

from modelzoo.common.pytorch.run_utils import half_dtype_instance
from modelzoo.vision.pytorch.input.classification.dataset_factory import (
    VisionSubset,
)
from modelzoo.vision.pytorch.input.transforms import LambdaWithParam
from modelzoo.vision.pytorch.input.utils import (
    FastDataLoader,
    ShardedSampler,
    num_tasks,
    task_id,
)
from modelzoo.vision.pytorch.unet.input.preprocessing_utils import (
    adjust_brightness_transform,
    normalize_tensor_transform,
    rotation_90_transform,
)


class InriaAerialDataset(VisionDataset):
    def __init__(
        self,
        root,
        split="train",
        transforms=None,
        transform=None,
        target_transform=None,
    ):
        super(InriaAerialDataset, self).__init__(
            root, transforms, transform, target_transform
        )

        if split not in ["train", "val", "test"]:
            raise ValueError(
                f"Invalid value={split} passed to `split` argument. "
                f"Valid are 'train' or 'val' or 'test' "
            )
        self.split = split

        if split == "test" and target_transform is not None:
            raise ValueError(
                "split {split} has no mask images and hence target_transform should be None. Got {target_tranform}"
            )

        self.data_dir = os.path.join(self.root, self.split)
        self.image_dir = os.path.join(self.data_dir, "images")
        self.mask_dir = os.path.join(self.data_dir, "gt")
        self.file_list = sorted(os.listdir(self.image_dir))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item.
        """

        image_file_path = os.path.join(self.image_dir, self.file_list[index])
        image = Image.open(image_file_path).convert("L")  # PILImage

        mask_file_path = os.path.join(self.mask_dir, self.file_list[index])
        target = Image.open(mask_file_path)  # PILImage
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self):
        return len(self.file_list)


class InriaAerialDataProcessor:
    def __init__(self, params):
        self.data_dir = params["data_dir"]

        self.num_classes = params["num_classes"]
        self.image_shape = params["image_shape"]  # of format (H, W, C)
        self.duplicate_act_worker_data = params.get(
            "duplicate_act_worker_data", False
        )

        self.loss_type = params["loss"]
        self.normalize_data_method = params.get("normalize_data_method")

        self.shuffle_seed = params.get("shuffle_seed", None)
        if self.shuffle_seed:
            torch.manual_seed(self.shuffle_seed)

        self.augment_data = params.get("augment_data", True)
        self.batch_size = params["batch_size"]
        self.shuffle = params.get("shuffle", True)

        # Multi-processing params.
        self.num_workers = params.get("num_workers", 0)
        self.drop_last = params.get("drop_last", True)
        self.prefetch_factor = params.get("prefetch_factor", 10)
        self.persistent_workers = params.get("persistent_workers", True)

        self.mixed_precision = params.get("mixed_precision")
        if self.mixed_precision:
            self.mp_type = half_dtype_instance.half_dtype
        else:
            self.mp_type = torch.float32

        # Debug params:
        self.overfit = params.get("overfit", False)
        # default is that each activation worker sends `num_workers`
        # batches so total batch_size * num_act_workers * num_pytorch_workers samples
        self.overfit_num_batches = params.get(
            "overfit_num_batches", num_tasks() * self.num_workers
        )
        self.random_indices = params.get("overfit_indices", None)
        if self.overfit:
            logging.info(f"---- Overfitting {self.overfit_num_batches}! ----")

        # Using Faster Dataloader for mapstyle dataset.
        self.use_fast_dataloader = params.get("use_fast_dataloader", False)

    def create_dataset(self, is_training):

        split = "train" if is_training else "val"
        dataset = InriaAerialDataset(
            root=self.data_dir,
            split=split,
            transforms=self.transform_image_and_mask,
        )

        if self.overfit:
            random.seed(self.shuffle_seed)
            if self.random_indices is None:
                indices = random.sample(
                    range(0, len(dataset)),
                    self.overfit_num_batches * self.batch_size,
                )
            else:
                indices = self.random_indices

            dataset = VisionSubset(dataset, indices)
            print(f"---- Overfitting {indices}! ----")

        return dataset

    def create_dataloader(self, is_training=False):
        dataset = self.create_dataset(is_training)
        shuffle = self.shuffle and is_training
        generator_fn = torch.Generator(device="cpu")
        if self.shuffle_seed is not None:
            generator_fn.manual_seed(self.shuffle_seed)

        if shuffle:
            if self.duplicate_act_worker_data:
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
                    dataset, shuffle, self.shuffle_seed, self.drop_last
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

    def _apply_normalization(
        self, image, normalize_data_method, *args, **kwargs
    ):
        return normalize_tensor_transform(image, normalize_data_method)

    def preprocess_image(self, image):

        # converts to (C, H, W) format. In this case (1, H, W)
        to_tensor_transform = transforms.PILToTensor()

        # Normalize
        normalize_transform = LambdaWithParam(
            self._apply_normalization, self.normalize_data_method
        )

        transforms_list = [
            to_tensor_transform,
            normalize_transform,
        ]
        image = transforms.Compose(transforms_list)(image)
        return image

    def preprocess_mask(self, mask):
        to_tensor_transform = transforms.PILToTensor()
        normalize_transform = LambdaWithParam(
            self._apply_normalization, "zero_one"
        )
        transforms_list = [
            to_tensor_transform,
            normalize_transform,
        ]
        mask = transforms.Compose(transforms_list)(
            mask
        )  # output of shape (1, 5000, 5000)
        return mask

    def transform_image_and_mask(self, image, mask):
        image = self.preprocess_image(image)
        mask = self.preprocess_mask(mask)

        if self.augment_data:
            do_horizontal_flip = torch.rand(size=(1,)).item() > 0.5
            # n_rots in range [0, 3)
            n_rotations = torch.randint(low=0, high=3, size=(1,)).item()

            if self.image_shape[0] != self.image_shape[1]:  # H != W
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
        if self.mixed_precision:
            image = image.to(self.mp_type)

        return image, mask

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
