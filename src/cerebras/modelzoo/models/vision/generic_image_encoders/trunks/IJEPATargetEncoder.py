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

import torch.nn.functional as F

from cerebras.modelzoo.models.vision.generic_image_encoders.utils import misc
from cerebras.modelzoo.models.vision.vision_transformer.ViTModel import (
    ViTModel,
    ViTModelConfig,
)


class IJEPATargetEncoderConfig(ViTModelConfig):
    name: Literal["IJEPATargetEncoder"]

    @property
    def __model_cls__(self):
        return IJEPATargetEncoder


class IJEPATargetEncoder(ViTModel):
    def __init__(self, config: IJEPATargetEncoderConfig):
        if isinstance(config, dict):
            config = IJEPATargetEncoderConfig(**config)
        super().__init__(config)

    def forward(
        self,
        input_image=None,
        input_image_embeddings=None,
        masks_predictor=None,
        masks_encoder=None,
    ):
        """
        input_image: torch.Tensor of shape (bsz, C, H, W) with single image per sample in batch
        input_image_embeddings: torch.Tensor indicating embeddings of images.
            Shape (bsz, MSL, H) i.e. single image per sample in batch (or)
        masks_predictor: torch.Tensor of shape (bsz, num_predictor_masks, max_num_mask_patches_predictor).
            Index to the positions in MSL in embeddings to be masked.
            Corresponds to the patch indices to be predicted by IJEPATargetEncoder.
        masks_encoder: torch.Tensor of shape (bsz, num_encoder_masks, max_num_mask_patches_encoder).
            Index to the positions in MSL in embeddings to be masked.
            Corresponds to the patch index to be masked in the image and
            used as input by IJEPAContextEncoder.
        """
        hidden_states, _ = super().forward(input_image, input_image_embeddings)
        hidden_states = F.layer_norm(hidden_states, (hidden_states.size(-1),))
        bsz, msl, hidden_dim = hidden_states.shape
        num_pred_masks = masks_predictor.shape[1]  # noqa
        num_encoder_masks = masks_encoder.shape[1]
        hidden_states = misc.apply_mask(
            hidden_states, masks_predictor
        )  # shape (bsz, num_pred_masks, max_num_mask_patches_predictor, hidden_dim)
        hidden_states = hidden_states.repeat(
            (1, num_encoder_masks, 1, 1)
        )  # shape (bsz, num_pred_masks*num_encoder_masks, max_num_mask_patches_predictor, hidden_dim)

        return hidden_states
