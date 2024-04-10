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

from cerebras.modelzoo.layers.create_initializer import create_initializer
from cerebras.modelzoo.layers.utils import (
    get_2d_fixed_position_embeddings,
    patchify_helper,
)


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
        init_conv_like_linear=False,
        use_post_embed_layer_norm=False,
        layer_norm_epsilon=1.0e-5,
        use_embed_proj_bias=True,
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

        if self.prepend_cls_token:
            self.cls_embedding = nn.Parameter(torch.zeros(self.hidden_size))
            self.cls_embedding_position_index = (
                self.num_patches[0] * self.num_patches[1]
            )  # seq_len + 1 - 1, cls pe is the last

        self.default_initializer = {
            "name": "truncated_normal",
            "std": self.initializer_range,
            "mean": 0.0,
            "a": self.initializer_range * -2.0,
            "b": self.initializer_range * 2.0,
        }
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
            create_initializer(self.default_initializer)(
                self.cls_embedding.data
            )
        if self.position_embedding_type == "learned":
            create_initializer(self.position_embedding_initializer)(
                self.position_embeddings.weight.data
            )
        if hasattr(self.post_embed_ln, 'bias') and hasattr(
            self.post_embed_ln.bias, "data"
        ):
            self.post_embed_ln.bias.data.zero_()
        if hasattr(self.post_embed_ln, 'weight') and hasattr(
            self.post_embed_ln.weight, "data"
        ):
            self.post_embed_ln.weight.data.fill_(1.0)

    def get_image_sequence_position_embeddings(self, embeddings, indices=None):
        # embeddings shape [batch_size, seq_len, hidden_size], shouldn't contain cls
        # indices shape [batch_size, seq_len]

        if indices is None:
            position_ids = torch.arange(
                0,
                embeddings.shape[1],
                device=embeddings.device,
            ).expand((embeddings.shape[0], -1))
        else:
            position_ids = indices

        if self.position_embedding_type == "learned":
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

    def get_cls_token_position_embeddings(self, batch_size, dtype, device):
        if self.position_embedding_type == "learned":
            cls_indices = (
                torch.ones(
                    (batch_size, 1),
                    dtype=torch.int32,
                    device=device,
                )
                * self.cls_embedding_position_index
            )
            pe = self.position_embeddings(cls_indices)
        else:
            pe = (
                self.position_embeddings[self.cls_embedding_position_index :, :]
                .to(dtype)
                .expand(batch_size, -1, -1)
            )

        # [bs, 1, hidden_size]
        return pe

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

    def forward(self, input_images, patch_indices=None):
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
        if self.position_embedding_type is not None:
            image_pe = self.get_image_sequence_position_embeddings(
                image_embeddings, indices=patch_indices
            )
            embeddings = embeddings + image_pe

        if self.prepend_cls_token:
            expanded_cls_embedding = self.cls_embedding.type_as(
                image_embeddings
            ).broadcast_to((batch_size, 1, self.hidden_size))
            expanded_cls_position_embedding = (
                self.get_cls_token_position_embeddings(
                    batch_size,
                    image_embeddings.dtype,
                    expanded_cls_embedding.device,
                )
            )
            cls_embeddings = (
                expanded_cls_embedding + expanded_cls_position_embedding
            )

            embeddings = torch.cat([cls_embeddings, embeddings], dim=1)

        embeddings = self.dropout_embd(embeddings)

        if self.post_embed_ln is not None:
            embeddings = self.post_embed_ln(embeddings)

        return embeddings
