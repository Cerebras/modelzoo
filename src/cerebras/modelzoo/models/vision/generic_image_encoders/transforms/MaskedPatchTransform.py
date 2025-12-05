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
import math
import os
import random

import numpy as np
import torch
from torchvision.transforms import transforms
from torchvision.utils import save_image

import cerebras.pytorch as cstorch
from cerebras.modelzoo.data.vision.transforms import create_transform
from cerebras.modelzoo.layers.utils import patchify_helper, unpatchify_helper
from cerebras.modelzoo.models.vision.generic_image_encoders.base.BaseSSLImageTransform import (
    BaseSSLImageTransform,
)
from cerebras.modelzoo.models.vision.generic_image_encoders.transforms.config import (
    MaskedPatchTransformConfig,
)


class MaskedPatchTransform(BaseSSLImageTransform):
    """
    A transform class for creating blockwise masks on images for self-supervised image encoder training. The masking scheme is as follows:
    1. For (1 - mask_probability ) proportion of images, apply no masking and leave image unchanged.
    2. For mask_probability proportion of images, for each image we sample from a uniform distribution to determine proportion of patches to mask.
    """

    def __init__(self, config: MaskedPatchTransformConfig):
        if isinstance(config, dict):
            config = MaskedPatchTransformConfig(**config)

        self.config = config

        self.image_height, self.image_width = config.image_size
        self.patch_height, self.patch_width = config.patch_size
        self.height = self.image_height // self.patch_height
        self.width = self.image_width // self.patch_width

        self.n_tokens = self.height * self.width
        self.min_num_patches = config.min_num_patches
        self.max_num_patches = config.mask_probability * self.n_tokens
        self.mask_probability = config.mask_probability
        self.mask_ratio_tuple = config.mask_ratio_tuple
        self.composed_transform = config.composed_transform
        self.log_aspect_ratio = (
            math.log(config.min_aspect),
            math.log(config.max_aspect),
        )

        self.addnl_transform = create_transform(
            {
                "name": "to_dtype",
                "mp_type": cstorch.amp.get_floating_point_dtype(),
            }
        )

        self.transform = self.create_image_transform(config.transform_list)

    @property
    def output_keys(self):
        return self.config.output_keys

    def create_image_transform(self, transform_list):
        if transform_list is None:
            return None

        _transforms = []

        for tx in transform_list:
            _transforms.append(create_transform(tx))

        final_transform = _transforms + [self.addnl_transform]
        final_transform = transforms.Compose(final_transform)

        logging.debug(
            f"The following sequence is used to transform data:\n{final_transform}"
        )

        return final_transform

    def __call__(self, image):
        # This would contain things like Gaussian blur, to_tensor etc.
        # If we have another transform before this we don't want to repeat
        # the augmentations. For example in DinoV2 those transforms are
        # already part of ImageRandomMultiCropTransform
        if not self.composed_transform and self.transform:
            image = self.transform(image)
        return image

    def get_shape(self):
        return self.height, self.width

    # copied from DinoV2 repo
    # https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/data/masking.py#L49
    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            # The aspect ratio is to create more interesting block-masks than purely
            # square masks, given in Algorithm 1 from BEIT paper
            # https://arxiv.org/pdf/2106.08254
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    # copied from DinoV2 repo
    # https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/data/masking.py#L73
    def mask_gen(self, num_masking_patches):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask

    # copied from DinoV2 repo
    # https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/data/collate.py#L10
    def collate_fn(self, batch):
        """
        Takes in a batch of images and calls self.mask_gen to create patch masks for each image in the batch according to the masking probability scheme described in the __init__ function docstring.

        Args:
            batch (list): A list of tuples where each tuple contains an image tensor and its corresponding label.

        Returns:
            dict: A dictionary containing:
                - "collated_masks" (torch.Tensor): A tensor of boolean masks with shape [B, N], where B is the batch size and N is the number of patches.
                - "image" (torch.Tensor, optional): A tensor of batched images with shape [B, C, H, W] if self.composed_transform is False.

        """
        B = len(batch)

        # batch: [(image, label), ..., (image, label)]
        # collated_batch: [image-tensor, label-tensor]
        # image-tensor: [BS, C, H, W]
        if not self.composed_transform:
            collated_batch = torch.utils.data.default_collate(batch)

        N = self.n_tokens
        n_samples_masked = int(B * self.mask_probability)
        probs = torch.linspace(*self.mask_ratio_tuple, n_samples_masked + 1)
        upperbound = 0
        masks_list = []
        for i in range(0, n_samples_masked):
            prob_min = probs[i]
            prob_max = probs[i + 1]
            masks_list.append(
                torch.BoolTensor(
                    self.mask_gen(int(N * random.uniform(prob_min, prob_max)))
                )
            )
            upperbound += int(N * prob_max)
        for i in range(n_samples_masked, B):
            masks_list.append(torch.BoolTensor(self.mask_gen(0)))

        random.shuffle(masks_list)

        collated_masks = torch.stack(masks_list).flatten(1)
        out = {"collated_masks": collated_masks}
        if not self.composed_transform:
            out["image"] = collated_batch[0]

        return out

    def apply_patches(self, images, patches):
        """
        Apply patches to the given images by setting the specified patches to zero. Used as a helper in
        visualizing the images.

        Args:
            images (torch.Tensor): A tensor of shape [B, C, H, W] representing a batch of images,
                where B is the batch size, C is the number of channels, H is the height, and W is the width.
            patches (torch.Tensor): A tensor of shape [B, num_patches] indicating the patches to be applied.
                Each element specifies which patches in the corresponding image batch to set to zero.

        Returns:
            torch.Tensor: A tensor of shape [B, C, H, W] with the specified patches set to zero.
        """
        image_patches = patchify_helper(
            images, [self.patch_height, self.patch_width]
        )
        image_patches[patches] = 0.0
        bs, channels, image_height, image_width = images.shape
        return unpatchify_helper(
            image_patches,
            [channels, image_height, image_width],
            [self.patch_height, self.patch_width],
        )

    def visualize_transform(self, batch):
        batch_size = batch["image"].shape[0]
        image_dir = os.path.join(
            os.path.dirname(__file__), "visualize_MaskedPatchTransform"
        )
        logging.info(
            f"Batch visualization with `visualize_MaskedPatchTransform` saved at {image_dir}"
        )
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        images = self.apply_patches(batch["image"], batch["collated_masks"])
        for i in range(batch_size):
            img_path = os.path.join(image_dir, f"sample_{i}.jpg")
            save_image(images[i], img_path)
