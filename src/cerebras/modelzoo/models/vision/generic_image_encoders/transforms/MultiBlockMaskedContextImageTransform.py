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
from logging import getLogger

_GLOBAL_SEED = 0
logger = getLogger()

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
    MultiBlockMaskedContextImageTransformConfig,
)


# Multiblock masking as in case of I-JEPA
class MultiBlockMaskedContextImageTransform(BaseSSLImageTransform):
    def __init__(self, config: MultiBlockMaskedContextImageTransformConfig):
        if isinstance(config, dict):
            config = MultiBlockMaskedContextImageTransformConfig(**config)

        self.config = config

        self.image_size = config.image_size

        self.addnl_transform = create_transform(
            {
                "name": "to_dtype",
                "mp_type": cstorch.amp.get_floating_point_dtype(),
            }
        )

        self.height, self.width = (
            config.image_size[0] // config.patch_size[0],
            config.image_size[1] // config.patch_size[1],
        )

        self.transform = self.create_image_transform(config.transform)

        # Find max number of patches in a mask for a
        # given mask_scale and aspect_ratio

        self.max_num_mask_patches_predictor = self._max_mask_patches(
            config.predictor_mask_scale, config.predictor_aspect_ratio
        )

        self.max_num_mask_patches_encoder = self._max_mask_patches(
            config.encoder_mask_scale, config.encoder_aspect_ratio
        )

        self.collate_class_obj = MultiBlockMaskCollator(
            image_size=self.image_size,
            patch_size=config.patch_size,
            num_encoder_masks=config.num_encoder_masks,
            encoder_mask_scale=config.encoder_mask_scale,
            encoder_aspect_ratio=config.encoder_aspect_ratio,
            max_num_mask_patches_encoder=self.max_num_mask_patches_encoder,
            num_predictor_masks=config.num_predictor_masks,
            predictor_mask_scale=config.predictor_mask_scale,
            predictor_aspect_ratio=config.predictor_aspect_ratio,
            max_num_mask_patches_predictor=self.max_num_mask_patches_predictor,
            min_mask_patches=config.min_mask_patches,
            allow_overlap=config.allow_overlap,
        )

    @property
    def output_keys(self):
        return self.config.output_keys

    def create_image_transform(self, transform_list):
        _transforms = []

        for tx in transform_list:
            _transforms.append(create_transform(tx))

        final_transform = _transforms + [self.addnl_transform]
        final_transform = transforms.Compose(final_transform)

        logging.debug(
            f"The following sequence is used to transform data:\n{final_transform}"
        )

        return final_transform

    def _max_mask_patches(self, scale, aspect_ratio_scale):
        max_area = -float("inf")
        for sc in scale:
            for ar in aspect_ratio_scale:
                max_keep = int(self.height * self.width * sc)

                h = int(round(math.sqrt(max_keep * ar)))
                w = int(round(math.sqrt(max_keep / ar)))

                h = min(self.height - 1, h)
                w = min(self.width - 1, w)

                if h * w > max_area:
                    max_area = h * w

        return max_area

    def __call__(self, image):
        if self.transform:
            image = self.transform(image)
        return image

    def collate_fn(self, batch):
        (
            collated_batch,
            collated_masks_enc,
            collated_masks_pred,
            num_valid_mask_encoder,
            num_valid_mask_predictor,
            loss_mask,
        ) = self.collate_class_obj(batch)

        output = {
            "image": collated_batch[0],
            "labels": collated_batch[1],
            "encoder_mask_idx": torch.stack(
                collated_masks_enc, dim=1
            ),  # shape: (bsz, num_encoder_masks, max_num_mask_patches_encoder)
            "predictor_mask_idx": torch.stack(
                collated_masks_pred, dim=1
            ),  # shape: (bsz, num_predictor_masks, max_num_mask_patches_predictor)
            "num_valid_mask_encoder": num_valid_mask_encoder,  # shape: (bsz, num_encoder_masks)
            "num_valid_mask_predictor": num_valid_mask_predictor,  # shape: (bsz, num_predictor_masks)
            "loss_mask": loss_mask.unsqueeze(
                3
            ),  # (bsz, num_predictor_masks, max_num_mask_patches_predictor, 1)
        }

        return output

    def _visualize_apply_mask(
        self, images, mask_idx, patch_size, mask_valid_len=None
    ):
        bsz = mask_idx.shape[0]
        num_masks = mask_idx.shape[1]

        num_patches_height = images.shape[2] // patch_size[0]
        num_patches_width = images.shape[3] // patch_size[1]
        num_patches = num_patches_height * num_patches_width

        image_patches = patchify_helper(
            images, patch_size
        )  # bsz, num_patches, num_channels*patch_height*patch_width
        image_patches = image_patches.unsqueeze(1).repeat((1, num_masks, 1, 1))
        image_patches_cp = image_patches.clone()

        if mask_valid_len is not None:
            # For all `pad` index, replace them with a index at position (0)
            # which is part of indixes in mask
            _m = (
                torch.arange(mask_idx.shape[2])
                .reshape(1, 1, -1)
                .broadcast_to(mask_idx.shape)
            )
            _m = torch.where(
                _m >= mask_valid_len.unsqueeze(2), 1, 0
            )  # bsz, num_masks, max_num_mask_patches
            _m = _m * mask_idx[:, :, 0:1]
            mask_idx = mask_idx + _m

        masked_patches = torch.ones_like(image_patches) * -100
        index = mask_idx.unsqueeze(3).repeat(1, 1, 1, image_patches.shape[-1])

        image_patches_cp.scatter_(dim=2, index=index, src=masked_patches)
        image_patches[image_patches_cp != -100] = 0.0  # complement mask

        masked_patches = image_patches.reshape(bsz * num_masks, num_patches, -1)
        unpatchified_images = unpatchify_helper(
            masked_patches, images.shape[1:], patch_size
        )
        unpatchified_images = unpatchified_images.reshape(
            bsz, num_masks, *images.shape[1:]
        )
        return unpatchified_images

    def visualize_transform(self, batch):
        images = batch["image"]
        encoder_mask_idx = batch["encoder_mask_idx"]
        predictor_mask_idx = batch["predictor_mask_idx"]

        num_valid_mask_encoder = batch["num_valid_mask_encoder"]
        num_valid_mask_predictor = batch["num_valid_mask_predictor"]

        batch_size = images.shape[0]
        image_path = os.path.join(
            os.path.dirname(__file__),
            "visualize_MultiBlockMaskedContextImageTransform.jpg",
        )
        logging.info(
            f"Batch visualization with `MultiBlockMaskedContextImageTransform` saved at {image_path}"
        )

        mask_images_encoder = self._visualize_apply_mask(
            images.clone(),
            encoder_mask_idx,
            self.collate_class_obj.patch_size,
            num_valid_mask_encoder,
        )
        mask_images_predictor = self._visualize_apply_mask(
            images.clone(),
            predictor_mask_idx,
            self.collate_class_obj.patch_size,
            num_valid_mask_predictor,
        )
        masked_images = torch.cat(
            [
                images.unsqueeze(1),
                mask_images_encoder,
                mask_images_predictor,
            ],
            dim=1,
        )
        masked_images = masked_images.reshape(-1, *images.shape[1:])
        save_image(
            masked_images,
            image_path,
            padding=4,
            pad_value=1.0,
            normalize=True,
            nrow=masked_images.shape[0] // batch_size,
        )


# copied from I-JEPA: https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/src/masks/multiblock.py
class MultiBlockMaskCollator(object):

    def __init__(
        self,
        image_size=(224, 224),
        patch_size=(16, 16),
        num_encoder_masks=1,
        encoder_mask_scale=(0.2, 0.8),
        encoder_aspect_ratio=(1.0, 1.0),
        max_num_mask_patches_encoder=None,
        num_predictor_masks=2,
        predictor_mask_scale=(0.2, 0.8),
        predictor_aspect_ratio=(0.3, 3.0),
        max_num_mask_patches_predictor=None,
        min_mask_patches=4,
        allow_overlap=False,
    ):
        super(MultiBlockMaskCollator, self).__init__()
        if not isinstance(image_size, (tuple, list)):
            image_size = (image_size,) * 2
        self.patch_size = patch_size
        self.height, self.width = (
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
        )
        self.encoder_mask_scale = encoder_mask_scale
        self.encoder_aspect_ratio = encoder_aspect_ratio
        self.num_encoder_masks = num_encoder_masks

        self.predictor_mask_scale = predictor_mask_scale
        self.predictor_aspect_ratio = predictor_aspect_ratio
        self.num_predictor_masks = num_predictor_masks

        self.min_mask_patches = (
            min_mask_patches  # minimum number of patches to keep
        )
        self.allow_overlap = (
            allow_overlap  # whether to allow overlap b/w enc and pred masks
        )

        self.max_num_mask_patches_predictor = (
            self.height * self.width
            if max_num_mask_patches_predictor is None
            else max_num_mask_patches_predictor
        )
        self.max_num_mask_patches_encoder = (
            self.height * self.width
            if max_num_mask_patches_encoder is None
            else max_num_mask_patches_encoder
        )
        self._itr_counter = -1

    def step(self):
        self._itr_counter += 1
        return self._itr_counter

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        return (h, w)

    def _sample_block_mask(self, b_size, generator, acceptable_regions=None):
        h, w = b_size

        def constrain_mask(mask, tries=0):
            """Helper to restrict given mask to a set of acceptable regions."""
            N = max(int(len(acceptable_regions) - tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]

        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            top = torch.randint(0, self.height - h, (1,), generator=generator)
            left = torch.randint(0, self.width - w, (1,), generator=generator)
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top : top + h, left : left + w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_mask_patches
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(
                        f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"'
                    )
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones(
            (self.height, self.width), dtype=torch.int32
        )
        mask_complement[top : top + h, left : left + w] = 0
        # --
        return mask, mask_complement

    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask.
        '''
        B = len(batch)

        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        # Choose a single (h, w) for all masks
        # in this batch
        p_size = self._sample_block_size(
            generator=g,
            scale=self.predictor_mask_scale,
            aspect_ratio_scale=self.predictor_aspect_ratio,
        )
        e_size = self._sample_block_size(
            generator=g,
            scale=self.encoder_mask_scale,
            aspect_ratio_scale=self.encoder_aspect_ratio,
        )

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(B):

            masks_p, masks_C = [], []
            for _ in range(self.num_predictor_masks):
                mask, mask_C = self._sample_block_mask(p_size, generator=g)
                masks_p.append(mask)
                masks_C.append(mask_C)
                # Ensures all masks are of the same shape
                # i.e same number of mask patches
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions = None
            except Exception as e:
                logger.warning(f'Encountered exception in mask-generator {e}')

            masks_e = []
            for _ in range(self.num_encoder_masks):
                mask, _ = self._sample_block_mask(
                    e_size, generator=g, acceptable_regions=acceptable_regions
                )
                masks_e.append(mask)
                # Ensures all masks are of the same shape
                # i.e same number of mask patches
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        collated_masks_pred = [
            [cm[:min_keep_pred] for cm in cm_list]
            for cm_list in collated_masks_pred
        ]
        collated_masks_pred = torch.utils.data.default_collate(
            collated_masks_pred
        )
        # --
        collated_masks_enc = [
            [cm[:min_keep_enc] for cm in cm_list]
            for cm_list in collated_masks_enc
        ]
        collated_masks_enc = torch.utils.data.default_collate(
            collated_masks_enc
        )

        # Pad mask_patch_idx to max_value
        num_valid_mask_encoder = torch.tensor(
            [_mask.shape[1] for _mask in collated_masks_enc], dtype=torch.int32
        )
        num_valid_mask_encoder = num_valid_mask_encoder.repeat(
            (B, 1)
        )  # (B, num_encoder_masks)

        num_valid_mask_predictor = torch.tensor(
            [_mask.shape[1] for _mask in collated_masks_pred], dtype=torch.int32
        )
        num_valid_mask_predictor = num_valid_mask_predictor.repeat(
            (B, 1)
        )  # (B, num_predictor_masks

        padded_collated_masks_enc = []
        for enc_mask in collated_masks_enc:
            if enc_mask.shape[1] < self.max_num_mask_patches_encoder:
                pad_mask = torch.zeros(
                    B,
                    self.max_num_mask_patches_encoder - enc_mask.shape[1],
                    dtype=enc_mask.dtype,
                )
                padded_collated_masks_enc.append(
                    torch.cat((enc_mask, pad_mask), dim=1)
                )
            else:
                padded_collated_masks_enc.append(enc_mask)

        padded_collated_masks_pred = []
        for pred_mask in collated_masks_pred:
            if pred_mask.shape[1] < self.max_num_mask_patches_predictor:
                pad_mask = torch.zeros(
                    B,
                    self.max_num_mask_patches_predictor - pred_mask.shape[1],
                    dtype=pred_mask.dtype,
                )
                padded_collated_masks_pred.append(
                    torch.cat((pred_mask, pad_mask), dim=1)
                )
            else:
                padded_collated_masks_pred.append(pred_mask)

        _mask = torch.arange(self.max_num_mask_patches_predictor).reshape(
            1, 1, -1
        )
        loss_mask = torch.where(
            _mask < num_valid_mask_predictor.unsqueeze(2), 1.0, 0.0
        )  # (bsz, num_pred_masks, max_num_mask_patches_pred)

        return (
            collated_batch,
            padded_collated_masks_enc,
            padded_collated_masks_pred,
            num_valid_mask_encoder,
            num_valid_mask_predictor,
            loss_mask,
        )
