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

from typing import List, Optional

import torch

from cerebras.modelzoo.config import BaseConfig
from cerebras.modelzoo.layers.utils import patchify_helper


class MAEProcessorConfig(BaseConfig):
    image_size: Optional[List[int]] = None
    patch_size: Optional[List[int]] = None
    image_channels: Optional[int] = None
    mask_ratio: float = 0.75

    def post_init(self, context):
        super().post_init(context)

        model_config = context.get("model", {}).get("config")
        if model_config is not None:
            if hasattr(model_config, "image_size"):
                self.image_size = model_config.image_size

            if hasattr(model_config, "patch_size"):
                self.patch_size = model_config.patch_size

            if hasattr(model_config, "num_channels"):
                self.image_channels = model_config.num_channels

        if any(
            x is None
            for x in [self.image_size, self.patch_size, self.image_channels]
        ):
            raise ValueError(
                "image_size, patch_size, and image_channels must be provided "
                "or be configured from the model config."
            )


class MAEProcessor:
    def __init__(self, config: MAEProcessorConfig):
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.image_channels = config.image_channels
        # calculate embedding sequence length
        self.patchified_seq_len = (self.image_size[0] // self.patch_size[0]) * (
            self.image_size[1] // self.patch_size[1]
        )

        self.mask_ratio = config.mask_ratio
        self.encoder_portion = int(
            (1 - self.mask_ratio) * self.patchified_seq_len
        )

    def generate_masks(self):
        indices_permutation = torch.randperm(self.patchified_seq_len)
        indices_encoder = indices_permutation[: self.encoder_portion]
        indices_decoder = indices_permutation[self.encoder_portion :]
        indices_restore = torch.argsort(indices_permutation)

        loss_mask = torch.zeros(self.patchified_seq_len)
        loss_mask[indices_encoder] = 1

        return indices_encoder, indices_decoder, loss_mask, indices_restore

    def generate_partition(self, input_images_patchified, indices_decoder):
        batch_size, decoder_length = indices_decoder.shape
        indices_decoder = torch.broadcast_to(
            indices_decoder.unsqueeze(-1),
            (batch_size, decoder_length, input_images_patchified.shape[-1]),
        ).long()

        labels = torch.gather(input_images_patchified, 1, indices_decoder)
        return labels

    def mae_collate_fn(self, data):
        indices = [self.generate_masks() for _ in range(len(data))]
        indices_encoder = torch.stack([i[0] for i in indices])
        indices_decoder = torch.stack([i[1] for i in indices])
        # loss_mask = torch.stack([i[2] for i in indices])
        indices_restore = torch.stack([i[3] for i in indices])

        input_images = torch.stack([d[0] for d in data])  # [bs, c, h, w]
        input_images_patchified = patchify_helper(input_images, self.patch_size)

        # encoder_input_images_patchified = self.generate_partition(input_images_patchified, indices_encoder)
        mae_labels = self.generate_partition(
            input_images_patchified, indices_decoder
        )

        results = {}
        results["input_images"] = input_images
        results["indices_encoder"] = indices_encoder
        results["indices_decoder"] = indices_decoder
        # results["loss_mask"] = loss_mask
        results["mae_labels"] = mae_labels
        results["indices_restore"] = indices_restore

        return results
