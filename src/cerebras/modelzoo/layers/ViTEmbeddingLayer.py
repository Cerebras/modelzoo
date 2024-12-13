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

from cerebras.modelzoo.config import BaseConfig
from cerebras.modelzoo.layers.create_initializer import create_initializer
from cerebras.modelzoo.layers.utils import (
    get_2d_fixed_position_embeddings,
    patchify_helper,
)


class InterpolatePositionEmbeddingConfig(BaseConfig):
    antialias: bool = False
    "apply anti-aliasing when interpolating positional embeddings."

    interpolate_offset: float = 0.1
    "Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8"

    local_patch_dims: [int, int] = [7, 7]
    "Local crop patch dimensions"


class ViTEmbeddingLayer(nn.Module):
    def __init__(
        self,
        image_size=[224, 224],
        num_channels=3,
        patch_size=[16, 16],
        hidden_size=768,
        initializer_range=0.02,
        embedding_dropout_rate=0.0,
        projection_initializer=None,
        position_embedding_initializer=None,
        position_embedding_type="learned",
        use_conv_patchified_embedding=False,
        prepend_cls_token=False,
        cls_token_initializer=None,
        init_conv_like_linear=False,
        use_post_embed_layer_norm=False,
        layer_norm_epsilon=1.0e-5,
        use_embed_proj_bias=True,
        interpolate_position_embedding=None,
        use_masked_patches=False,
    ):
        super(ViTEmbeddingLayer, self).__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.position_embedding_type = position_embedding_type
        self.use_conv_patchified_embedding = use_conv_patchified_embedding
        self.prepend_cls_token = prepend_cls_token
        self.init_conv_like_linear = init_conv_like_linear
        self.use_post_embed_layer_norm = use_post_embed_layer_norm
        self.use_masked_patches = use_masked_patches
        if self.use_masked_patches:
            self.mask_token = nn.Parameter(torch.zeros(self.hidden_size))

        assert (
            self.image_size[0] % self.patch_size[0] == 0
            and self.image_size[1] % self.patch_size[1] == 0
        ), f"image size {self.image_size} is not divisible by patch_size {self.patch_size}"

        assert self.position_embedding_type in [
            None,
            "fixed",
            "learned",
        ], "Only `learned` or `fixed` position embeddings are supported for now."

        self.num_patches = [
            (self.image_size[0] // self.patch_size[0]),
            (self.image_size[1] // self.patch_size[1]),
        ]

        if use_conv_patchified_embedding:
            self.linear_proj = nn.Conv2d(
                self.num_channels,
                self.hidden_size,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                bias=use_embed_proj_bias,
            )
        else:
            self.embedding_size = (
                self.patch_size[0] * self.patch_size[1] * num_channels
            )
            self.linear_proj = nn.Linear(
                self.embedding_size, self.hidden_size, bias=use_embed_proj_bias
            )

        if self.position_embedding_type == "learned":
            num_position_embeddings = self.num_patches[0] * self.num_patches[1]
            if self.prepend_cls_token:
                num_position_embeddings += 1
            self.position_embeddings = nn.Embedding(
                num_position_embeddings, self.hidden_size
            )
        elif self.position_embedding_type == "fixed":  # fixed sin&cos
            position_embeddings = get_2d_fixed_position_embeddings(
                self.num_patches,
                self.hidden_size,
                add_cls_token=prepend_cls_token,
            )
            self.position_embeddings = torch.nn.Parameter(
                position_embeddings, requires_grad=False
            )

        self.default_initializer = {
            "name": "truncated_normal",
            "std": self.initializer_range,
            "mean": 0.0,
            "a": self.initializer_range * -2.0,
            "b": self.initializer_range * 2.0,
        }

        if self.prepend_cls_token:
            if cls_token_initializer is None:
                cls_token_initializer = self.default_initializer
            self.cls_token_initializer = cls_token_initializer
            self.cls_embedding = nn.Parameter(torch.zeros(self.hidden_size))
            self.cls_embedding_position_index = (
                self.num_patches[0] * self.num_patches[1]
            )  # seq_len + 1 - 1, cls pe is the last

        if projection_initializer is None:
            projection_initializer = self.default_initializer
        if position_embedding_initializer is None:
            position_embedding_initializer = self.default_initializer
        self.projection_initializer = projection_initializer
        self.position_embedding_initializer = position_embedding_initializer

        self.dropout_embd = nn.Dropout(embedding_dropout_rate)
        self.post_embed_ln = None
        if self.use_post_embed_layer_norm:
            self.post_embed_ln = nn.LayerNorm(
                hidden_size, eps=layer_norm_epsilon
            )

        if interpolate_position_embedding is not None:
            interpolation_matrix = self.create_bicubic_interpolation_matrix(
                interpolate_position_embedding
            )
            interpolation_matrix = interpolation_matrix.to(
                self.position_embeddings.weight.device
            )
            self.register_buffer(
                "interpolation_matrix", interpolation_matrix, persistent=False
            )
        else:
            self.interpolation_matrix = None

        self.__reset_parameters()

    def reset_parameters(self):
        self.__reset_parameters()

    def __reset_parameters(self):
        projection_initializer = create_initializer(self.projection_initializer)
        w = self.linear_proj.weight.data
        if self.use_conv_patchified_embedding and self.init_conv_like_linear:
            # Modifying fan-in fan-out by reshaping.
            # Bias set to zeros already
            projection_initializer(w.view([w.shape[0], -1]))
        else:
            projection_initializer(w)

        if hasattr(self.linear_proj, "bias") and hasattr(
            self.linear_proj.bias, "data"
        ):
            create_initializer("zeros")(self.linear_proj.bias.data)

        if self.prepend_cls_token:
            create_initializer(self.cls_token_initializer)(
                self.cls_embedding.data
            )
        if self.position_embedding_type == "learned":
            create_initializer(self.position_embedding_initializer)(
                self.position_embeddings.weight.data
            )
        if self.use_masked_patches:
            create_initializer("zeros")(self.mask_token.data)
        if hasattr(self.post_embed_ln, 'bias') and hasattr(
            self.post_embed_ln.bias, "data"
        ):
            self.post_embed_ln.bias.data.zero_()
        if hasattr(self.post_embed_ln, 'weight') and hasattr(
            self.post_embed_ln.weight, "data"
        ):
            self.post_embed_ln.weight.data.fill_(1.0)

    def create_bicubic_interpolation_matrix(self, interpolation_config):
        # Pre-calculate constant co-efficients to do bicubic interpolate
        gw, gh = self.num_patches  # global patch dims
        num_global_patches = gw * gh

        def get_kernel(position):
            lw, lh = interpolation_config.local_patch_dims  # local patch dims
            sx = (lw + interpolation_config.interpolate_offset) / gw
            sy = (lh + interpolation_config.interpolate_offset) / gh
            return nn.functional.interpolate(
                position,
                mode="bicubic",
                antialias=interpolation_config.antialias,
                scale_factor=(sx, sy),
            )

        assert self.position_embedding_type == "learned"
        T = torch.eye(num_global_patches, device="cpu").reshape(
            num_global_patches, 1, 1, gw, gh
        )
        mm = (
            torch.vmap(get_kernel, in_dims=0)(T)
            .reshape(num_global_patches, -1)
            .transpose(1, 0)
        )
        if self.prepend_cls_token:
            # Last Col: expand interpolation matrix to accomodate for cls token pos emb in embedding matrix
            # First Row: calculate cls token pos embedding with matmul
            mm = nn.functional.pad(
                mm, pad=(0, 1, 1, 0), mode='constant', value=0
            )
            mm[0, -1] = 1

        return mm

    def get_interpolated_position_embeddings(self, position_ids, dtype):
        # Interpolate position embeddings
        # Use matmul with pre-calculated co-efficients to do bicubic interpolate
        # instead of using nn.functional.interpolate(mode='bicubic')
        assert self.interpolation_matrix.shape[0] == position_ids.shape[1]

        interpolation_cst = self.interpolation_matrix.to(dtype)
        # Add batch dim
        interpolation_cst = interpolation_cst[None, :, :].broadcast_to(
            position_ids.shape[0], *interpolation_cst.shape
        )
        position_embeddings = torch.matmul(
            interpolation_cst, self.position_embeddings.weight
        )
        return position_embeddings

    def get_image_sequence_position_embeddings(self, embeddings, indices=None):
        # embeddings shape [batch_size, seq_len, hidden_size], shouldn't contain cls
        # indices shape [batch_size, seq_len]

        if indices is None:
            position_ids = torch.arange(
                embeddings.shape[1],
                device=embeddings.device,
                dtype=torch.float32,
            ).expand((embeddings.shape[0], -1))
            if self.prepend_cls_token:
                position_ids = torch.arange(
                    -1,
                    embeddings.shape[1],
                    device=embeddings.device,
                    dtype=torch.float32,
                ).expand((embeddings.shape[0], -1))
                mask = position_ids + self.cls_embedding_position_index + 1
                position_ids = torch.where(
                    mask != self.cls_embedding_position_index,
                    position_ids,
                    self.cls_embedding_position_index,
                )

            position_ids = position_ids.to(torch.int64)
        else:
            position_ids = indices

        if self.position_embedding_type == "learned":
            if (
                position_ids.shape[1]
                != self.position_embeddings.weight.shape[0]
                and self.interpolation_matrix is not None
            ):
                position_embeddings = self.get_interpolated_position_embeddings(
                    position_ids, dtype=embeddings.dtype
                )
            else:
                position_embeddings = self.position_embeddings(position_ids)
        elif self.position_embedding_type == "fixed":  # fixed
            position_ids = torch.broadcast_to(
                position_ids.unsqueeze(-1),
                (
                    position_ids.shape[0],
                    position_ids.shape[1],
                    embeddings.shape[-1],
                ),
            ).long()
            position_embeddings = torch.gather(
                self.position_embeddings.to(embeddings.dtype).expand(
                    position_ids.shape[0], -1, -1
                ),
                1,
                position_ids,
            )

        return position_embeddings

    def select_patches(self, patches, patch_indices=None):
        """Select from patches based on patch_indices

        Args:
            patches (Tensor): shape [batch_size, full_sequence_length, hidden_size]
            patch_indices (Tensor): shape [batch_size., subset_sequence_length]

        Returns:
            patches (Tensor): shape [batch_size, subset_sequence_length, hidden_size]
        """
        if patch_indices is None:
            return patches

        batch_size, subset_sequence_length = patch_indices.shape
        patch_indices = torch.broadcast_to(
            patch_indices.unsqueeze(-1),
            (batch_size, subset_sequence_length, patches.shape[-1]),
        ).long()

        patches = torch.gather(patches, 1, patch_indices)

        return patches

    def forward(self, input_images, patch_indices=None, masks=None):
        """Applies patching and linear projection to the input images.

        Args:
            input_images (Tensor): shape if use_conv_patchified_embedding ``[batch_size, num_channels, height, width]`` else ``[batch_size, sequence_len, embedding_size]``.
            patch_indices (Tensor): shape [batch_size, subset_seq_length]. If specified, embedding layer will select a subset of all image patches based on indices.
                This is used for applications like MAE. Default to None.

        Returns:
            image_embeddings (Tensor): shape ``[batch_size, sequence_length, hidden_size]``.
        """

        batch_size = input_images.shape[0]
        if self.use_conv_patchified_embedding:
            # conv projection
            image_embeddings = self.linear_proj(input_images)

            # reshape
            hidden_size = image_embeddings.shape[1]
            image_embeddings = image_embeddings.reshape(
                batch_size, hidden_size, -1
            ).transpose(
                1, 2
            )  # [bs, seq_length, hidden_size]
            image_embeddings = self.select_patches(
                image_embeddings, patch_indices=patch_indices
            )
        else:
            # patchify

            patchified_image = patchify_helper(input_images, self.patch_size)

            # this saves computation compared to the conv implementation because patch selection happens before linear_proj
            image_embeddings = self.select_patches(
                patchified_image, patch_indices=patch_indices
            )

            # linear projection
            image_embeddings = self.linear_proj(
                image_embeddings
            )  # [bs, seq_length, hidden_size]

        embeddings = image_embeddings

        # apply masks to patches for masked head in dinov2 like arch
        # Note masking is applied before position embedding
        if not self.use_masked_patches and masks is not None:
            raise ValueError(
                "Cannot use masks if setting use_masked_patches=False."
            )

        if masks is not None:
            masks_reshaped = masks.reshape(batch_size, -1, 1).to(torch.int)
            mask_token_bcasted = self.mask_token.to(
                embeddings.dtype
            ).broadcast_to(embeddings.shape)

            embeddings = (
                masks_reshaped * mask_token_bcasted
                + (1 - masks_reshaped) * embeddings
            )

        if self.prepend_cls_token:
            expanded_cls_embedding = self.cls_embedding.type_as(
                embeddings
            ).broadcast_to((batch_size, 1, self.hidden_size))

            embeddings = torch.cat([expanded_cls_embedding, embeddings], dim=1)

        if self.position_embedding_type is not None:
            image_and_cls_pe = self.get_image_sequence_position_embeddings(
                image_embeddings, indices=patch_indices
            )
            embeddings = embeddings + image_and_cls_pe

        embeddings = self.dropout_embd(embeddings)

        if self.post_embed_ln is not None:
            embeddings = self.post_embed_ln(embeddings)

        return embeddings
