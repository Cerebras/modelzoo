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

import torch.nn as nn

from cerebras.modelzoo.layers.create_initializer import create_initializer
from cerebras.modelzoo.layers.RotaryPositionEmbeddingHelper import (
    RotaryPositionEmbeddingHelper,
)

from .AlibiPositionEmbeddingLayer import AlibiPositionEmbeddingLayer
from .FixedPositionEmbeddingLayer import FixedPositionEmbeddingLayer
from .LearnedPositionEmbeddingLayer import LearnedPositionEmbeddingLayer
from .RelativePositionEmbeddingLayer import RelativePositionEmbeddingLayer


class EmbeddingLayer(nn.Module):
    """
    Creates token and, optionally, position and segment embeddings.

    :param int vocab_size: Size of input vocabulary.
    :param int embedding_size: Dimension of the embedding space.
    :param Optional[int] pad_token_id: If specified, the entries at padding_idx
        do not contribute to the gradient; therefore,
        the embedding vector at padding_idx is not updated during training.
    :param int segment_embedding_size: Dimension of the embedding space for segment
        embeddings. Useful when factorized embeddings are used for tokens and
        so the size of the embedding space for segments differs from that for
        tokens. Defaults to the same value as embedding_size.
    :param Optional[str,Callable] embeddings_initializer: Token embeddings
        initializer. Defaults to 'uniform'.
    :param int max_position_embeddings: Maximum sequence length to train
        using model.
    :param str position_embedding_type: 'learned', 'fixed' or 'rotary'. Defaults to "learned",
        for 'rotary' embeddings, embeddings are not created at bottom but computed with key&query embeddings by RotaryPositionEmbeddingHelper
    :param int position_embedding_offset: Offset for position embeddings. Default to 0.
    :param Optional[int] min_timescale: The scale of the shortest sinusoid. Default to 1.0. (only need to be specified when position_embedding_type is fixed).
    :param Optional[int] max_timescale: The scale of the longest sinusoid. Default to 1.0e4. (only need to be specified when position_embedding_type is fixed).
    :param Optional[str,Callable] position_embeddings_initializer: Position
        embeddings initializer. Defaults to "uniform".
    :param Optional[int] num_segments: Number of segments for the segment
        embedding layer. Defaults to None, in which case the segment embedding
        layer is not created.
    :param Optional[str,Callable] segment_embeddings_initializer: Segment
        embeddings initializer. Defaults to "uniform".
    :param device (optional): Device to create the model parameters on, can be a cuda device or CS device.
    """

    def __init__(
        self,
        # Word Embedding Parameters
        vocab_size,
        embedding_size,
        pad_token_id=None,
        initializer="xavier_uniform",
        embeddings_initializer='uniform',
        device=None,
        # Positional Embedding Parameters:
        position_embedding_type="learned",
        max_position_embeddings=None,
        positional_embedding_size=None,
        position_embedding_offset=0,
        position_embeddings_initializer='uniform',
        # Fixed PE:
        min_timescale=1.0,
        max_timescale=1.0e4,
        # Learned PE:
        mask_padding_in_positional_embed=False,
        # Relative PE:
        num_heads=None,
        relative_attention_bias=None,
        num_relative_attention_buckets=32,
        bidirectional=False,
        # Rotary PE:
        rotary_dim=None,
        rope_theta=10000,
        pad_rope=False,
        # Alibi PE:
        alibi_slopes=None,
        alibi_trainable_slopes=False,
        # Rotary & Alibi PE:
        pos_scaling_factor=1.0,
        # Segment Embedding Parameters:
        num_segments=None,
        segment_embedding_size=None,
        segment_embeddings_initializer='uniform',
    ):
        super(EmbeddingLayer, self).__init__()

        if segment_embedding_size is None:
            segment_embedding_size = embedding_size

        if positional_embedding_size is None:
            positional_embedding_size = embedding_size
        self.embedding_size = embedding_size
        self.positional_embedding_size = positional_embedding_size
        self.segment_embedding_size = segment_embedding_size

        self.embeddings_initializer = embeddings_initializer
        self.position_embeddings_initializer = position_embeddings_initializer
        self.segment_embeddings_initializer = segment_embeddings_initializer
        self.position_embedding_offset = position_embedding_offset

        self.max_position_embeddings = max_position_embeddings
        self.position_embedding_type = position_embedding_type
        self.pad_token_id = pad_token_id
        self.mask_padding_in_positional_embed = mask_padding_in_positional_embed
        if self.mask_padding_in_positional_embed:
            assert (
                self.position_embedding_type == "learned"
            ), "Masking padding tokens in positional embeddings is only supported in learned position embeddings"

        self.word_embeddings = nn.Embedding(
            vocab_size,
            embedding_size,
            padding_idx=self.pad_token_id,
            device=device,
        )

        self.position_embeddings = None
        self.position_embed_helper = None
        if position_embedding_type is not None:
            assert position_embedding_offset >= 0, (
                f"Position embedding offset should be non-negative, but it is "
                f"{position_embedding_offset}."
            )

            if position_embedding_type != "learned":
                assert position_embedding_offset == 0, (
                    f"For {position_embedding_type} embeddings, position "
                    f"embedding offset must be 0, but it is "
                    f"{position_embedding_offset}."
                )

            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings should be specified.")

            if max_position_embeddings > 32768:
                raise ValueError("max_position_embeddings cannot exceed 32768")

            if (
                pos_scaling_factor != 1.0
                and self.position_embedding_type
                not in [
                    "rotary",
                    "alibi",
                ]
            ):
                raise ValueError(
                    "pos_scaling_factor is only supported for ALiBi and RoPE."
                )

            if position_embedding_type == "learned":
                self.position_embeddings = LearnedPositionEmbeddingLayer(
                    max_position_embeddings=max_position_embeddings,
                    positional_embedding_size=positional_embedding_size,
                    pad_token_id=pad_token_id,
                    position_embedding_offset=position_embedding_offset,
                    position_embeddings_initializer=position_embeddings_initializer,
                    mask_padding_in_positional_embed=mask_padding_in_positional_embed,
                    device=device,
                )
            elif position_embedding_type == "fixed":
                assert (
                    max_position_embeddings > 1
                ), "Max position embeddings of length 1 currently unsupported."
                self.position_embeddings = FixedPositionEmbeddingLayer(
                    max_position_embeddings,
                    positional_embedding_size,
                    min_timescale,
                    max_timescale,
                )
            elif self.position_embedding_type == "rotary":
                assert (
                    rotary_dim is not None
                ), "EmbeddingLayer requires rotary_dim when using rotary embeddings"

                self.position_embed_helper = RotaryPositionEmbeddingHelper(
                    self.max_position_embeddings,
                    rotary_dim,
                    base=rope_theta,
                    scaling_factor=pos_scaling_factor,
                    pad_fixed_pos_emb=pad_rope,
                )
            elif self.position_embedding_type == "relative":
                assert (
                    num_heads is not None
                ), "EmbeddingLayer requires num_heads when using relative positional embeddings"

                self.position_embed_helper = RelativePositionEmbeddingLayer(
                    num_heads,
                    relative_attention_bias=relative_attention_bias,
                    num_relative_attention_buckets=num_relative_attention_buckets,
                    bidirectional_relative_attention=bidirectional,
                    relative_attn_bias_initializer=initializer,
                )
            elif self.position_embedding_type == "alibi":
                assert (
                    num_heads is not None
                ), "AlibiPositionEmbeddingLayer requires num_heads when using alibi positional embeddings"

                self.position_embed_helper = AlibiPositionEmbeddingLayer(
                    num_heads,
                    slopes=alibi_slopes,
                    alibi_trainable_slopes=alibi_trainable_slopes,
                    slopes_initializer=initializer,
                    scaling_factor=pos_scaling_factor,
                )
            else:
                raise ValueError(
                    f"Unknown position embedding type: {position_embedding_type}"
                )

        self.segment_embeddings = None
        if num_segments:
            self.segment_embeddings = nn.Embedding(
                num_segments,
                segment_embedding_size,
                device=device,
            )

        # Initialize weights
        self.__reset_parameters()

    def reset_parameters(self):
        self.__reset_parameters()
        if self.position_embeddings:
            self.position_embeddings.reset_parameters()
        if self.position_embed_helper:
            self.position_embed_helper.reset_parameters()

    def __reset_parameters(self):
        create_initializer(self.embeddings_initializer)(
            self.word_embeddings.weight.data
        )

        if self.segment_embeddings:
            create_initializer(self.segment_embeddings_initializer)(
                self.segment_embeddings.weight.data
            )

        if self.pad_token_id:
            self.word_embeddings.weight.data[
                self.word_embeddings.padding_idx
            ].zero_()

            if self.segment_embeddings:
                self.segment_embeddings.weight.data[
                    self.segment_embeddings.padding_idx
                ].zero_()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.word_embeddings = new_embeddings

    def forward(
        self,
        input_ids,
        position_ids=None,
        segment_ids=None,
        past_length=0,
    ):
        """Convert input_ids to token embeddings according to the embedding type.
            Word embeddings (required), segment embeddings (optional) and position embeddings (optional).

        Args:
            input_ids (Tensor): input token ids with shape ``[batch_size, seq_length]``.
            position_ids (Tensor): position ids with shape ``[batch_size, seq_length]``.
            segment_ids (Tensor): input segment ids with shape ``[batch_size, seq_length]``.

        Returns:
            Token embedding output with shape ``[batch_size, seq_length, embedding_size]``.
        """
        embeddings = self.compute_token_embeddings(input_ids)
        if self.position_embeddings is not None:
            assert self.embedding_size == self.positional_embedding_size, (
                "Cannot use EmbeddingLayer's forward function since the word "
                "embedding size and positional embedding size are different. "
                "Manually apply the following functions with the appropriate"
                "projections in order to align the sizes: compute_token_embeddings, "
                "compute_positional_embeddings, and compute_segment_embeddings"
            )
            embeddings = embeddings + self.compute_positional_embeddings(
                input_ids, position_ids, past_length, embeddings.dtype
            )
        if segment_ids is not None and self.segment_embeddings is not None:
            assert self.embedding_size == self.segment_embedding_size, (
                "Cannot use EmbeddingLayer's forward function since the word "
                "embedding size and segment embedding size are different. "
                "Manually apply the following functions with the appropriate"
                "projections in order to align the sizes: compute_token_embeddings, "
                "compute_positional_embeddings, and compute_segment_embeddings"
            )
            embeddings = embeddings + self.compute_segment_embeddings(
                segment_ids
            )
        return embeddings

    def compute_token_embeddings(self, input_ids):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        embeddings = self.word_embeddings(input_ids)
        return embeddings

    def compute_positional_embeddings(
        self, input_ids, position_ids=None, past_length=0, dtype=None
    ):
        input_shape = input_ids.size()
        embed_dtype = (
            self.word_embeddings.weight.dtype if dtype is None else dtype
        )

        if position_ids is not None:
            assert (
                position_ids.size() == input_shape
            ), "position_ids must have shape [batch_size, seq_length]"

        position_embeddings = None
        if self.position_embedding_type == "learned":
            position_embeddings = self.position_embeddings(
                input_ids,
                position_ids=position_ids,
                past_length=past_length,
                dtype=dtype,
            )
        elif self.position_embedding_type == "fixed":
            position_embeddings = self.position_embeddings(
                input_shape[-1], position_ids=position_ids, dtype=embed_dtype
            )

        return position_embeddings

    def compute_segment_embeddings(self, segment_ids):
        segment_embeddings = None
        if segment_ids is not None and self.segment_embeddings is not None:
            segment_embeddings = self.segment_embeddings(segment_ids.int())
        return segment_embeddings

    def compute_position_bias(self, seq_length, key_length, past_kv=None):
        self_attn_position_bias = None
        if self.position_embed_helper and isinstance(
            self.position_embed_helper,
            (
                RelativePositionEmbeddingLayer,
                AlibiPositionEmbeddingLayer,
            ),
        ):
            if seq_length > 32768 or key_length > 32768:
                raise ValueError(
                    f"The tokenized data exceeded the 32768 MSL limit. Make "
                    f"sure that the tokenized data length and "
                    f"{self.position_embedding_type}'s max_position_embeddings "
                    f"<= 32768."
                )

            self_attn_position_bias = self.position_embed_helper(
                seq_length, key_length, past_kv=past_kv
            )
        return self_attn_position_bias

    def get_rope_helper(self):
        if self.position_embed_helper and isinstance(
            self.position_embed_helper, RotaryPositionEmbeddingHelper
        ):
            return self.position_embed_helper
        else:
            return None
