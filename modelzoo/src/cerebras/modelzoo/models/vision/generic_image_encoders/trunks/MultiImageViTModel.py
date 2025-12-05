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

from cerebras.modelzoo.models.vision.vision_transformer.ViTModel import (
    ViTModel,
    ViTModelConfig,
)


class MultiImageViTModelConfig(ViTModelConfig):
    name: Literal["MultiImageViTModel"]
    "Name of the model. Must be set to `MultiImageViTModel`."

    @property
    def __model_cls__(self):
        return MultiImageViTModel


class MultiImageViTModel(ViTModel):
    def __init__(self, config: MultiImageViTModelConfig):
        if isinstance(config, dict):
            config = MultiImageViTModelConfig(**config)
        super().__init__(config)

        self.hidden_size = config.hidden_size

    def forward(
        self, input_image=None, input_image_embeddings=None, masks=None
    ):
        if input_image is not None and input_image_embeddings is not None:
            raise ValueError(
                f"Only one of `input_image` or `input_image_embeddings` should be passed to model.forward"
            )

        if input_image is not None:
            bsz, n_imgs, other_dims = (
                input_image.shape[0],
                input_image.shape[1],
                input_image.shape[2:],
            )
            input_image = input_image.view(bsz * n_imgs, *other_dims)
        elif input_image_embeddings is not None:
            bsz, n_imgs, other_dims = (
                input_image_embeddings.shape[0],
                input_image_embeddings.shape[1],
                input_image_embeddings.shape[2:],
            )
            input_image_embeddings = input_image_embeddings.view(
                bsz * n_imgs, *other_dims
            )

        if input_image_embeddings is None:
            input_image_embeddings = self.compute_input_embeddings(
                input_image, masks=masks
            )

        hidden_states, pooled_states = self.encoder(input_image_embeddings)

        # Reshape:
        # hidden_states of shape (bsz * n_imgs, num_patches, hidden_dim) ->
        # (bsz, n_imgs, num_patches, hidden_dim)
        hidden_states = hidden_states.reshape(
            bsz, n_imgs, -1, hidden_states.shape[-1]
        )
        pooled_states = pooled_states.reshape(bsz, n_imgs, -1)

        return hidden_states, pooled_states
