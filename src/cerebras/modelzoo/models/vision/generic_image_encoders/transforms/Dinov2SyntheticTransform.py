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

import torch

from cerebras.modelzoo.models.vision.generic_image_encoders.base.BaseSSLImageTransform import (
    BaseSSLImageTransform,
)
from cerebras.modelzoo.models.vision.generic_image_encoders.transforms.config import (
    Dinov2SyntheticTransformConfig,
)


class Dinov2SyntheticTransform(BaseSSLImageTransform):
    def __init__(self, config: Dinov2SyntheticTransformConfig):
        if isinstance(config, dict):
            config = Dinov2SyntheticTransformConfig(**config)

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

        self._itr_counter = -1

    def step(self):
        self._itr_counter += 1
        return self._itr_counter

    @property
    def output_keys(self):
        return self.config.output_keys

    def __call__(self, *args, **kwargs):
        return args, kwargs

    def collate_fn(self, batch):

        batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        generator = torch.Generator()
        generator.manual_seed(seed)

        B = batch["global_view"].shape[0] * batch["global_view"].shape[1]
        n_samples_masked = int(B * self.mask_probability)
        no_mask = B - n_samples_masked

        # n_samples_masked intervals
        probs = torch.linspace(*self.mask_ratio_tuple, n_samples_masked + 1)

        num_masked_tokens = torch.tensor(
            [
                int(
                    torch.rand((1))
                    .uniform_(probs[i], probs[i + 1], generator=generator)
                    .item()
                )
                for i in range(n_samples_masked)
            ]
        )
        num_masked_tokens = torch.clamp(
            num_masked_tokens, self.min_num_patches, self.max_num_patches
        )

        n_masks = torch.rand(
            (n_samples_masked, self.n_tokens), generator=generator
        ).argsort(dim=1)
        n_masks = n_masks < num_masked_tokens.reshape(-1, 1)
        non_masked = torch.zeros(no_mask, self.n_tokens).to(torch.bool)

        masks = torch.cat([n_masks, non_masked], dim=0)
        rand_idx = torch.randperm(masks.shape[0], generator=generator)
        # Shuffle masks
        masks = masks[rand_idx]
        masks = masks.reshape(
            batch["global_view"].shape[0], batch["global_view"].shape[1], -1
        )
        batch["collated_masks"] = masks
        return batch

    def visualize_transform(self, data):
        pass
