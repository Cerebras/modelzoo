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

from cerebras.modelzoo.models.vision.generic_image_encoders.base.BaseSSLImageTransform import (
    BaseSSLImageTransform,
)
from cerebras.modelzoo.models.vision.generic_image_encoders.transforms.config import (
    Dinov2TransformConfig,
)


class Dinov2Transform(BaseSSLImageTransform):
    def __init__(self, config: Dinov2TransformConfig):
        if isinstance(config, dict):
            config = Dinov2TransformConfig(**config)

        self.config = config

        self.multi_crop_transform = config.multi_crop_transform()
        self.masked_patch_transform = config.masked_patch_transform()

    @property
    def output_keys(self):
        return self.config.output_keys

    def __call__(self, image):
        # the masks are created in the collate_fn. we can just perform
        # __call__ once so that we don't apply augmentations twice
        return self.multi_crop_transform(image)

    def collate_fn(self, batch):
        out = self.multi_crop_transform.collate_fn(batch)
        global_view_shape = out['global_view'].shape
        # flatten batch_size x num_global_crops to get a unique mask
        # for each global crop. if we just create batch_size number
        # of masks, they would be repeated for the different global crops,
        # which is not what we want
        masks = self.masked_patch_transform.collate_fn(
            out['global_view'].reshape(-1, *global_view_shape[2:])
        )['collated_masks']
        # we return (batch_size x num_global_crops) masks, but we have to
        # separate back out into separate dims because CS requires consistent
        # batch_size across all tensors
        out['collated_masks'] = masks.reshape(
            global_view_shape[0], global_view_shape[1], -1
        )
        return out

    def visualize_transform(self, batch):
        # modify the batch in-place to mask the patches
        for global_crop_i in range(self.multi_crop_transform.global_num_crops):
            # [B, num_global_crops, C, H, W]
            batch['global_view'][:, global_crop_i, :, :, :] = (
                self.masked_patch_transform.apply_patches(
                    batch['global_view'][:, global_crop_i, :, :, :],
                    batch['collated_masks'][:, global_crop_i, :],
                )
            )

        # then we can directly use multi-crop subclass visualizer
        self.multi_crop_transform.visualize_transform(batch)
