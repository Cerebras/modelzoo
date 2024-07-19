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

from cerebras.modelzoo.models.multimodal.multimodal_utils import (
    freeze_modules,
    init_component_model,
)


class MultimodalSimpleImageEncoders(nn.Module):
    r"""
    This class implements multimodality support for multiple image encoders.

    Arguments:
        image_model_list (:obj:`nn.ModuleList`, `required`):
            List of image encoders. Each image encoder contains a image model and a
            projection module.
        projection (:obj:`nn.Module`, `required`):
            Global projection model that combines patches across image encoders.
        image_layer_idx_list (:obj:`List`, `required`):
            Layer ID for feature selection in each image encoder.
        image_feature_selection_mode (:obj:`string`):
            Determines how to select features from ViTModel. In "patch" mode, the first cls token is ignored.
    """

    def __init__(
        self,
        image_model_list,
        projection,
        image_layer_idx_list,
        image_feature_select_mode,
    ):
        super().__init__()
        self.image_model_list = image_model_list
        self.projection = projection
        self.image_layer_idx_list = image_layer_idx_list
        self.image_feature_select_mode = image_feature_select_mode
        assert len(self.image_model_list) == len(self.image_layer_idx_list)

    # end def

    @staticmethod
    def build_model(params):
        def is_image_model(image_model_params):
            if (
                (type(image_model_params) == dict)
                and (len(image_model_params) == 1)
                and ('image_model' in image_model_params)
            ):
                return True

            return False

        # end def

        def is_global_projection(proj_model_params):
            if (
                (type(proj_model_params) == dict)
                and (len(proj_model_params) == 1)
                and ('global_image_projection' in proj_model_params)
            ):
                return True

            return False

        # end def

        def build_projection_model(proj_model_params):
            return init_component_model(proj_model_params)

        def build_image_model(image_model_params):
            assert (
                len(image_model_params) <= 2
            ), "Image model can only have an image encoder and an optional projection layer"
            image_layer_idx = -1
            if hasattr(image_model_params[0], "image_layer_idx"):
                image_layer_idx = image_model_params[0].image_layer_idx
            image_model = init_component_model(image_model_params[0])
            if len(image_model_params) > 1:
                projection = build_projection_model(image_model_params[1])
            else:
                projection = nn.Identity()

            image_model = nn.ModuleList([image_model, projection])
            return (image_layer_idx, image_model)

        # end def

        image_model_list = []
        projection = nn.Identity()
        image_layer_idx_list = []
        image_feature_select_mode = params.image_feature_select_mode
        if params.global_image_projection:
            projection = build_projection_model(params.global_image_projection)

        for model_params_entry in params.image_models:
            if is_image_model(model_params_entry):
                image_layer_idx, image_model = build_image_model(
                    model_params_entry['image_model']
                )
                image_model_list.append(image_model)
                image_layer_idx_list.append(image_layer_idx)
            # end if

        # end for

        image_model_list = nn.ModuleList(image_model_list)
        return MultimodalSimpleImageEncoders(
            image_model_list,
            projection,
            image_layer_idx_list,
            image_feature_select_mode,
        )

    def compute_image_features(self, image_data=None, image_embeddings=None):
        r"""
        Arguments:
            image_data (Tensor): Images input to the model with shape B x I x C x H_i x W_i, where
                B is batch size, I is number of images per sample.
            image_embeddings (Tensor): Image embeddings to the model with shape B x I x num_patches x H.
                where I is number of images per sample, and num_patches is number of patches in total across all image encoders + global projection,
                and H is hidden dimension. Note, 'image_data' and 'image_embeddings' are mutually exclusive.
        """
        if image_data is not None and image_embeddings is not None:
            raise ValueError(
                f"Only one of `image_data` or `image_embeddings` should be passed to model.forward"
            )
        elif image_data is None and image_embeddings is None:
            raise ValueError(
                f"Both `image_data` or `image_embeddings` are None, "
                f"either one of the them should be passed to model.forward"
            )
        out = []
        if image_data is not None:
            B, I, OD = (
                image_data.shape[0],
                image_data.shape[1],
                image_data.shape[2:],
            )
            image_data = image_data.view(B * I, *OD)

        for idx, model in enumerate(self.image_model_list):
            if image_data is not None:
                curr_image_embeddings = model[0].compute_input_embeddings(
                    image_data
                )
            else:
                curr_image_embeddings = image_embeddings
            image_features = model[0].extract_features(
                curr_image_embeddings, self.image_layer_idx_list[idx]
            )
            if self.image_feature_select_mode == "patch":
                image_features = image_features[:, 1:, :]  # Remove CLS features
            ## Projection layer
            image_features = model[1](image_features)
            out.append(image_features)

        image_features = torch.cat(out, dim=1)
        image_features = self.projection(image_features)
        _, OD = (image_features.shape[0], image_features.shape[1:])
        image_features = image_features.reshape(B, I, *OD)
        return image_features


class MultimodalSimple(nn.Module):
    r"""
    This class implements multimodality support for multiple image encoders, variable number of
    intermingled images.

    Arguments:
        image_model (:obj:`MultimodalSimpleImageEncoders`, `required`):
            List of image encoders + an optional global projection module. Each image encoder contains a image model and an optional
            projection module.
        text_model (:obj:`nn.Module`, `required`):
            The text model of multimodality model.
        freeze (:obj:`string`, `optional`, defaults to None):
            Filter to select which parameters are frozen. Note that regex patterns should be specified
            as single quotes in the yaml for escape codes.
    """

    def __init__(
        self,
        image_model,
        text_model,
        freeze=None,
    ):
        super().__init__()
        self.image_model = image_model
        self.text_model = text_model
        if freeze:
            freeze_modules(self, freeze)

    @staticmethod
    def build_model(params):
        model_params = params.model
        freeze = model_params.freeze
        image_model = MultimodalSimpleImageEncoders.build_model(
            model_params.image_model_list
        )
        text_model = init_component_model(model_params.text_model)
        return MultimodalSimple(image_model, text_model, freeze=freeze)

    def reset_parameters(self):
        self.image_model.reset_parameters()
        self.text_model.reset_parameters()

    def forward(
        self,
        image_data=None,  ## B x I x 3 x H_i x W_i
        text_input_ids=None,  ## B x S
        image_embeddings=None,  ## B x I num_patches X H
        text_embeddings=None,  ## B x S x H
        attention_mask=None,
        tgt_key_padding_mask=None,
        attention_span=None,
        image_data_loc=None,  ## B x I * num_patches
        token_modality_idx=None,  ## B x S
        position_ids=None,
    ):
        r"""
        Arguments:
            image_data (Tensor): Images input to the model with shape B x I x C x H_i x W_i, where
                B is batch size, I is number of images per sample.
            text_input_ids (Tensor): text indices with shape B x S, where S is max sequence length
            image_embeddings (Tensor): Image embeddings to the model with shape B x I x num_patches x H.
                where I is number of images per sample, and num_patches is number of patches in total across all image encoders + global projection,
                and H is hidden dimension. Note, 'image_data' and 'image_embeddings' are mutually exclusive.
            text_embeddings (Tensor): Image embeddings to the model with shape B x S x H. Note, 'text_input_ids' and
                'text_embeddings' are mutually exclusive.
            attention_mask (Tensor): Mask for padded positions with shape B x S (optional).
            tgt_key_padding_mask (Tensor): the mask for the tgt keys per batch with shape B x S (optional).
            attention_span (Tensor): Attention span of keys for VSL, has shape B x S (optional).
            image_data_loc (Tensor): The location indices for each image patch B x I x num_patches (required).
            token_modality_idx (Tensor): A tensor to specify a token's modality with shape B x S. 1 if this token is for image,
                and 0 if this token is for text (optional).
            position_ids (Tensor): position ids with shape B x S (optional).
        """
        if image_data_loc is None:
            raise ValueError(
                f"`image_data_loc` should be passed to model.forward and cannot be None"
            )

        modality_aligned_embeddings = self.compute_embeddings(
            image_data=image_data,
            image_embeddings=image_embeddings,
            text_input_ids=text_input_ids,
            text_embeddings=text_embeddings,
            position_ids=position_ids,
            image_data_loc=image_data_loc,
        )

        logits = self.text_model(
            input_embeddings=modality_aligned_embeddings,
            token_modality_idx=token_modality_idx,
            constant_pos_mask=token_modality_idx,
            input_ids=None,
            attention_mask=attention_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            attention_span=attention_span,
            position_ids=position_ids,
        )

        return logits

    def compute_embeddings(
        self,
        image_data_loc,
        image_data=None,
        image_embeddings=None,
        text_input_ids=None,
        text_embeddings=None,
        position_ids=None,
    ):

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
        image_features = self.image_model.compute_image_features(
            image_data=image_data, image_embeddings=image_embeddings
        )
        B, I, NP, H = image_features.shape
        image_features = image_features.reshape(B, I * NP, H)
        text_features = self.text_model.compute_input_embeddings(
            input_ids=text_input_ids, position_ids=position_ids
        )

        image_data_loc = image_data_loc.reshape(
            image_data_loc.shape[0], -1  # Shape: B, num_img * num_patches
        ).to(torch.int64)
        image_data_loc = torch.broadcast_to(
            image_data_loc.unsqueeze(-1), image_features.shape
        )
        # Concat a zero-valued column at the end of 'text_features' as a placehoder for fake/dummy image
        # features when scattering image features into text_features.
        # So we can put dummy image tokens to this zero-valued column, without polluting the rest of tokens.
        # This column will be removed after scatter.
        text_features = torch.cat(
            (
                text_features,
                torch.zeros(
                    text_features.shape[0],
                    1,
                    text_features.shape[-1],
                    device=text_features.device,
                    dtype=text_features.dtype,
                ),
            ),
            dim=1,
        )
        image_text_features = text_features.scatter(
            dim=1,
            index=image_data_loc,
            src=image_features.to(text_features.dtype),
        )
        image_text_features = image_text_features[:, :-1, :]

        return image_text_features
