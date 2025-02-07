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
from torch import nn

from cerebras.modelzoo.models.multimodal.multimodal_base_model import (
    ModalityType,
)
from cerebras.modelzoo.models.multimodal.multimodal_utils import freeze_modules


class Llava(nn.Module):
    def __init__(
        self,
        image_model,
        text_model,
        image_start_idx,
        projector_image_model=None,
        projector_text_model=None,
        freeze=None,  # list of regex strings
        image_feature_select_layer_idx=None,
        image_feature_select_mode="patch",
    ):
        super(Llava, self).__init__()
        self._modalities = [ModalityType.IMAGE, ModalityType.TEXT]
        self.image_model = image_model
        self.text_model = text_model
        self.projector_image_model = projector_image_model
        self.projector_text_model = projector_text_model
        self.image_start_idx = image_start_idx
        self.image_feature_select_mode = image_feature_select_mode

        self.tie_weights()

        # Freeze specified parameters
        freeze_modules(self, freeze)

        self.image_feature_select_layer_idx = image_feature_select_layer_idx

    @property
    def modalities(self):
        return self._modalities

    def tie_weights(self):
        self.image_model.tie_weights()
        self.text_model.tie_weights()
        if self.projector_image_model and hasattr(
            self.projector_image_model, "tie_weights"
        ):
            self.projector_image_model.tie_weights()
        if self.projector_text_model and hasattr(
            self.projector_text_model, "tie_weights"
        ):
            self.projector_text_model.tie_weights()

    def reset_parameters(self):
        self.image_model.reset_parameters()
        self.text_model.reset_parameters()
        if self.projector_image_model and hasattr(
            self.projector_image_model, "reset_parameters"
        ):
            self.projector_image_model.reset_parameters()
        if self.projector_text_model and hasattr(
            self.projector_text_model, "reset_parameters"
        ):
            self.projector_text_model.reset_parameters()

    def forward(
        self,
        image_data=None,
        text_input_ids=None,
        image_embeddings=None,
        text_embeddings=None,
        attention_mask=None,
        tgt_key_padding_mask=None,  # 1 where pad and not attend
        attention_span=None,
        position_ids=None,
        img_start_idx=None,
    ):
        input_embeddings = self.compute_input_embeddings(
            image_data,
            image_embeddings,
            text_input_ids,
            text_embeddings,
            position_ids,
            img_start_idx,
        )

        logits = self.text_model(
            input_ids=None,
            attention_mask=attention_mask,  # Does nothing in decoder models fwd pass
            tgt_key_padding_mask=tgt_key_padding_mask,
            attention_span=attention_span,
            position_ids=position_ids,
            input_embeddings=input_embeddings,
        )
        return logits

    def compute_input_embeddings(
        self,
        image_data=None,
        image_embeddings=None,
        text_input_ids=None,
        text_embeddings=None,
        position_ids=None,
        img_start_idx=None,
    ):
        if image_data is not None and image_embeddings is not None:
            raise ValueError(
                f"Only one of `image_data` or `image_embeddings` should be passed to model.forward"
            )
        elif image_data is None and image_embeddings is None:
            raise ValueError(
                f"Both `image_data` or `image_embeddings` are None, "
                f"either one of the them should be passed to model.forward"
            )

        if text_input_ids is not None and text_embeddings is not None:
            raise ValueError(
                f"Only one of `text_input_ids` or `text_embeddings` should be passed to model.forward"
            )
        elif text_input_ids is None and text_embeddings is None:
            raise ValueError(
                f"Both `text_input_ids` or `text_embeddings` are None, "
                f"either one of the them should be passed to model.forward"
            )

        # Compute image_features by passing through embedding layer of image_model
        if image_data is not None:
            image_embeddings = self.image_model.compute_input_embeddings(
                image_data
            )

        image_features_w_cls = self.image_model.extract_features(
            image_embeddings, self.image_feature_select_layer_idx
        )

        if self.image_feature_select_mode == "patch":
            image_features = image_features_w_cls[
                :, 1:, :
            ]  # Remove CLS features
        else:
            image_features = image_features_w_cls

        num_patches = image_features.shape[1]

        # Pass image_embeddings through projector
        if self.projector_image_model is not None:
            image_features = self.projector_image_model(image_features)

        # Compute text embeddings
        if text_input_ids is not None:
            # If position_embeddings_type is `learned` or `fixed`, we'd like to
            # use them when computing embeddings for text_tokens
            # since the forward pass, when supplied with `input_embeddings`
            # does not call EmbeddingLayer fwd pass again.
            text_embeddings = self.text_model.compute_input_embeddings(
                text_input_ids, position_ids
            )

        if self.projector_text_model is not None:
            text_embeddings = self.projector_text_model(text_embeddings)

        # Replace patch positions with image_features
        # position_ids and key_padding_mask will ensure
        # appropriate positions to attend and
        # positional encoding
        if img_start_idx is None:
            # The `img_start_idx` is not provided, the <image> is at the beginning of the sentence.
            image_text_embeddings = torch.cat(
                (
                    text_embeddings[:, 0 : self.image_start_idx, :],
                    image_features,
                    text_embeddings[:, self.image_start_idx + num_patches :, :],
                ),
                dim=1,
            )
            return image_text_embeddings
        else:
            # The `img_start_idx` is not None, the location of <image> is arbitrary.
            index = torch.arange(
                0,
                num_patches,
                device=image_features.device,
                dtype=torch.float32,
            )
            index = index[None, :].broadcast_to(
                text_embeddings.shape[0], num_patches
            )
            index = (index + img_start_idx).to(torch.int64)
            index = index[:, :, None].broadcast_to(
                text_embeddings.shape[0], num_patches, text_embeddings.shape[-1]
            )
            text_embeddings.scatter_(
                1, index, image_features.to(text_embeddings.dtype)
            )
            return text_embeddings
