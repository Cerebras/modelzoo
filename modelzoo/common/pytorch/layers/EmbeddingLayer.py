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

import math

import torch
import torch.nn as nn

from modelzoo.common.pytorch.model_utils.create_initializer import (
    create_initializer,
)
from modelzoo.common.pytorch.model_utils.RotaryPositionEmbeddingHelper import (
    RotaryPositionEmbeddingHelper,
)

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
    :param Optional[int] min_timescale: The scale of the shortest sinusoid. Default to 1.0. (only need to be specified when position_embedding_type is fixed).
    :param Optional[int] max_timescale: The scale of the longest sinusoid. Default to 1.0e4. (only need to be specified when position_embedding_type is fixed).
    :param Optional[str,Callable] position_embeddings_initializer: Position
        embeddings initializer. Defaults to "uniform".
    :param Optional[int] num_segments: Number of segments for the segment
        embedding layer. Defaults to None, in which case the segment embedding
        layer is not created.
    :param Optional[str,Callable] segment_embeddings_initializer: Segment
        embeddings initializer. Defaults to "uniform".
    """

    def __init__(
        self,
        vocab_size,
        embedding_size,
        pad_token_id=None,
        positional_embedding_size=None,
        segment_embedding_size=None,
        embeddings_initializer='uniform',
        max_position_embeddings=None,
        position_embedding_type="learned",
        min_timescale=1.0,
        max_timescale=1.0e4,
        position_embeddings_initializer='uniform',
        num_segments=None,
        segment_embeddings_initializer='uniform',
        device=None,
    ):
        super(EmbeddingLayer, self).__init__()

        if segment_embedding_size is None:
            segment_embedding_size = embedding_size

        if positional_embedding_size is None:
            positional_embedding_size = embedding_size

        self.embeddings_initializer = embeddings_initializer
        self.position_embeddings_initializer = position_embeddings_initializer
        self.segment_embeddings_initializer = segment_embeddings_initializer

        self.max_position_embeddings = max_position_embeddings
        self.position_embedding_type = position_embedding_type
        self.pad_token_id = pad_token_id
        self.word_embeddings = nn.Embedding(
            vocab_size,
            embedding_size,
            padding_idx=self.pad_token_id,
            device=device,
        )

        if position_embedding_type is not None:
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings should be specified.")

            if position_embedding_type == "learned":
                self.position_embeddings = nn.Embedding(
                    max_position_embeddings,
                    positional_embedding_size,
                    device=device,
                )
            elif position_embedding_type == "fixed":
                self.position_embeddings = self.create_fix_pos_embedding(
                    max_position_embeddings,
                    positional_embedding_size,
                    min_timescale,
                    max_timescale,
                )
            elif (
                position_embedding_type == "relative"
                or position_embedding_type == "rotary"
            ):
                self.position_embeddings = None
            else:
                raise ValueError(
                    f"Unknown position embedding type: {position_embedding_type}"
                )
        else:
            self.position_embeddings = None

        if num_segments:
            self.segment_embeddings = nn.Embedding(
                num_segments, segment_embedding_size, device=device,
            )
        else:
            self.segment_embeddings = None

        # Initialize weights
        self.__reset_parameters()

    def create_fix_pos_embedding(
        self, seq_len, embed_len, min_timescale, max_timescale
    ):
        """
        adapted from: https://github.com/tensorflow/tensor2tensor/blob\
            /1843c72d1d5faf4c085bb198b5dde0908f4081d0/tensor2tensor/layers\
            /common_attention.py#L407
        """
        position = torch.arange(seq_len, dtype=torch.float32)
        num_timescales = embed_len // 2
        log_timescale_increment = math.log(
            float(max_timescale) / float(min_timescale)
        ) / (float(num_timescales) - 1)
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32)
            * -log_timescale_increment
        )
        scaled_time = torch.unsqueeze(position, 1) * torch.unsqueeze(
            inv_timescales, 0
        )
        signal = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], axis=1
        )
        signal = torch.reshape(signal, (seq_len, 2, num_timescales))
        signal = torch.transpose(signal, 1, 2)
        signal = torch.reshape(signal, (seq_len, 2 * num_timescales))
        signal = torch.nn.functional.pad(signal, (0, embed_len % 2, 0, 0),)

        return torch.nn.Parameter(signal, requires_grad=False)

    def position_embedding_helper(
        self,
        # relative
        num_heads=None,
        relative_attention_bias=None,
        num_relative_attention_buckets=32,
        bidirectional=False,
        initializer="xavier_uniform",
        # rotary
        rotary_dim=None,
    ):
        embedding_helper = None
        if self.position_embedding_type == "rotary":
            assert (
                rotary_dim is not None
            ), "RotaryPositionEmbeddingHelper requires rotary_dim"

            embedding_helper = RotaryPositionEmbeddingHelper(
                self.max_position_embeddings, rotary_dim
            )
        elif self.position_embedding_type == "relative":
            assert (
                num_heads is not None
            ), "RelativePositionEmbeddingLayer requires num_heads"

            embedding_helper = RelativePositionEmbeddingLayer(
                num_heads,
                relative_attention_bias=relative_attention_bias,
                num_relative_attention_buckets=num_relative_attention_buckets,
                bidirectional_relative_attention=bidirectional,
                relative_attn_bias_initializer=initializer,
            )

        return embedding_helper

    def reset_parameters(self):
        self.__reset_parameters()

    def __reset_parameters(self):
        create_initializer(self.embeddings_initializer)(
            self.word_embeddings.weight.data
        )

        if self.position_embedding_type == "learned":
            create_initializer(self.position_embeddings_initializer)(
                self.position_embeddings.weight.data
            )
        if self.segment_embeddings:
            create_initializer(self.segment_embeddings_initializer)(
                self.segment_embeddings.weight.data
            )

        if self.pad_token_id:
            self.word_embeddings.weight.data[
                self.word_embeddings.padding_idx
            ].zero_()

            if self.position_embedding_type == "learned":
                self.position_embeddings.weight.data[
                    self.position_embeddings.padding_idx
                ].zero_()

            if self.pad_token_id:
                self.segment_embeddings.weight.data[
                    self.segment_embeddings.padding_idx
                ].zero_()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.word_embeddings = new_embeddings

    def forward(
        self, input_ids, segment_ids=None, past_length=0,
    ):
        embeddings = self.compute_token_embeddings(input_ids)
        if self.position_embeddings is not None:
            embeddings += self.compute_positional_embeddings(
                input_ids, past_length, embeddings.dtype
            )
        if segment_ids is not None and self.segment_embeddings is not None:
            embeddings += self.compute_segment_embeddings(segment_ids)
        return embeddings

    def compute_token_embeddings(self, input_ids):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        embeddings = self.word_embeddings(input_ids)
        return embeddings

    def compute_positional_embeddings(
        self, input_ids, past_length=0, dtype=None
    ):
        input_shape = input_ids.size()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        embed_dtype = (
            self.word_embeddings.weight.dtype if dtype is None else dtype
        )

        position_embeddings = None
        if self.position_embedding_type == "learned":
            position_ids = torch.arange(
                past_length, input_shape[-1] + past_length, device=device,
            ).expand((batch_size, -1))
            position_embeddings = self.position_embeddings(position_ids)
        elif self.position_embedding_type == "fixed":
            position_embeddings = self.position_embeddings.to(dtype=embed_dtype)
            length = input_shape[-1]
            if length != position_embeddings.size(dim=0):
                position_embeddings = position_embeddings[:length]
        return position_embeddings

    def compute_segment_embeddings(self, segment_ids):
        segment_embeddings = None
        if segment_ids is not None and self.segment_embeddings is not None:
            segment_embeddings = self.segment_embeddings(segment_ids.int())
        return segment_embeddings
