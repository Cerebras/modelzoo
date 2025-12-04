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

from typing import Literal

import torch

from cerebras.modelzoo.common.utils.model.transformer_utils import (
    replace_with_zero_and_neg_inf,
)
from cerebras.modelzoo.models.vision.generic_image_encoders.utils import misc
from cerebras.modelzoo.models.vision.vision_transformer.ViTModel import (
    ViTModel,
    ViTModelConfig,
)


class IJEPAContextEncoderConfig(ViTModelConfig):
    name: Literal["IJEPAContextEncoder"]

    @property
    def __model_cls__(self):
        return IJEPAContextEncoder


class IJEPAContextEncoder(ViTModel):
    def __init__(self, config: IJEPAContextEncoderConfig):
        if isinstance(config, dict):
            config = IJEPAContextEncoderConfig(**config)

        super().__init__(config)

    def forward(
        self,
        input_image=None,
        input_image_embeddings=None,
        masks_encoder=None,
        num_valid_mask_patches_encoder=None,
    ):
        """
        input_image: torch.Tensor of shape (bsz, C, H, W) with single image per sample in batch
        input_image_embeddings: torch.Tensor indicating embeddings of images.
            Shape (bsz, MSL, H) i.e. single image per sample in batch (or)
        masks_encoder: torch.Tensor of shape (bsz, num_encoder_masks, max_num_mask_patches_encoder).
            Index to the positions in MSL in embeddings to be
            masked
        num_valid_mask_patches_encoder: torch.Tensor of shape (bsz, num_encoder_masks) indicate the
            valid number of patches in dimension 1 in `masks_encoder`.
        """
        if input_image is not None and input_image_embeddings is not None:
            raise ValueError(
                f"Only one of `input_image` or `input_image_embeddings` should be passed to model.forward"
            )

        if input_image_embeddings is None:
            input_image_embeddings = self.embedding_layer(input_image)

        if masks_encoder is not None:
            # Only supported for single image per sample scenario
            bsz, num_encoder_masks, max_num_mask_patches_encoder = (
                masks_encoder.shape
            )
            input_image_embeddings = misc.apply_mask(
                input_image_embeddings, masks_encoder
            )
            input_image_embeddings = input_image_embeddings.reshape(
                (bsz * num_encoder_masks, max_num_mask_patches_encoder, -1)
            )

        hidden_dim = input_image_embeddings.shape[-1]

        attn_mask = torch.zeros(
            (input_image_embeddings.shape[0], input_image_embeddings.shape[1]),
            dtype=input_image_embeddings.dtype,
            device=input_image_embeddings.device,
        )
        # (bsz*num_encoder_masks, max_num_mask_patches_encoder)
        if num_valid_mask_patches_encoder is not None:
            # Build attention mask, 0 at positions to attend
            # and -inf at positions to ignore
            _mask = torch.arange(
                input_image_embeddings.shape[1],
                device=num_valid_mask_patches_encoder.device,
                dtype=num_valid_mask_patches_encoder.dtype,
            ).reshape(
                1, -1
            )  # (1, max_num_mask_patches_encoder)
            attn_mask = torch.where(
                _mask >= num_valid_mask_patches_encoder.reshape(-1, 1),
                1.0,
                0.0,
            )  # (bsz*num_encoder_masks, max_num_mask_patches_encoder)

        attn_mask = replace_with_zero_and_neg_inf(attn_mask)[:, None, None, :]
        hidden_states, _ = self.encoder(
            input_image_embeddings, src_mask=attn_mask
        )

        if masks_encoder is not None:
            # Note that masks are only supported for single image case,
            hidden_states = hidden_states.reshape(
                masks_encoder.shape[0], masks_encoder.shape[1], -1, hidden_dim
            )  # (bsz, num_encoder_masks, max_num_mask_patches_encoder, hidden_dim)

        return hidden_states
