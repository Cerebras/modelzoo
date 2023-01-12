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

from modelzoo.common.pytorch.layers import (
    EmbeddingLayer,
    FeedForwardNetwork,
    TransformerEncoder,
    TransformerEncoderLayer,
)


class BertPooler(nn.Module):
    def __init__(
        self,
        hidden_size,
        use_bias=True,
        activation="gelu",
        dropout=None,
        initializer="xavier_uniform",
    ):
        super().__init__()
        self.pooler = FeedForwardNetwork(
            input_unit=hidden_size,
            layers_units=[hidden_size],
            layers_activation=[activation],
            layers_dropout_rates=[dropout],
            use_bias=use_bias,
            kernel_initializer=initializer,
        )

    def reset_parameters(self):
        self.pooler.reset_parameters()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state
        # corresponding to the first token.
        # shape [batch_size, hidden_size]
        cls_hidden_states = hidden_states[:, 0]
        pooled_output = self.pooler(cls_hidden_states)
        return pooled_output


class BertModel(nn.Module):
    """
    The model behaves as a bidirectional encoder (with only self-attention), following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        position_embedding_type(:obj:`str`, `optional`, defaults to 'learned'):
            The type of position embeddings, should either be 'learned' or 'fixed'.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        embedding_dropout_rate (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the word embeddings.
        embedding_pad_token_id (:obj:`int`, `optional`, defaults to 0):
            The embedding vector at embedding_pad_token_id is not updated during training.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        layer_norm_epsilon (:obj:`float`, `optional`, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        num_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        attention_type (:obj:`str`, `optional`, defaults to 'scaled_dot_product'): 
            The attention variant to execute. Currently
            accepts ``dot_product`` and ``scaled_dot_product``.
        dropout_rate (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        nonlinearity: (:obj:`string`, `optional`, defaults to :obj:`gelu`):
            The non-linear activation function (function or string) in the encoder and pooler.
        attention_dropout_rate (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        use_projection_bias_in_attention (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If True, bias is used on the projection layers in attention.
        use_ffn_bias_in_attention (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If  True, bias is used in the dense layer in the attention.
        filter_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the feed-forward layer in the Transformer encoder.
        use_ffn_bias: (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If  True, bias is used in the dense layer in the encoder.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer as the default initializer.
        num_segments (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the segments (sentence types).
        embeddings_initializer (:obj:`dict`, `optional`, defaults to None):
            Initializer for word embeddings
        position_embeddings_initializer (:obj:`dict`, `optional`, defaults to None):
            Initializer for position embeddings (if learned position embeddings)
        segment_embeddings_initializer (:obj:`dict`, `optional`, defaults to None):
            Initializer for segment embeddings
        add_pooling_layer (:obj:`bool`, `optional`, defaults to True):
            Whether to add the pooling layer for sequence classification.
    """

    # TODO(SW-76063): We may need a general configuration class to avoid writing those params explicitly
    def __init__(
        self,
        # Embedding
        vocab_size=50257,
        max_position_embeddings=1024,
        position_embedding_type="learned",
        hidden_size=768,
        embedding_dropout_rate=0.1,  # need to be careful when testing
        embedding_pad_token_id=0,
        # Encoder
        num_hidden_layers=12,
        layer_norm_epsilon=1.0e-5,
        # Encoder Attn
        num_heads=12,
        attention_module_str="aiayn_attention",
        extra_attention_params={},
        attention_type="scaled_dot_product",
        dropout_rate=0.1,
        nonlinearity="gelu",
        attention_dropout_rate=0.1,
        use_projection_bias_in_attention=True,
        use_ffn_bias_in_attention=True,
        # Encoder ffn
        filter_size=3072,
        use_ffn_bias=True,
        # Task-specific
        initializer_range=0.02,
        num_segments=2,
        default_initializer=None,
        embeddings_initializer=None,
        position_embeddings_initializer=None,
        segment_embeddings_initializer=None,
        add_pooling_layer=True,
        **extra_args,
    ):
        super().__init__()

        self.initializer_range = initializer_range
        self.add_pooling_layer = add_pooling_layer

        if default_initializer is None:
            default_initializer = {
                "name": "truncated_normal",
                "std": self.initializer_range,
                "mean": 0.0,
                "a": self.initializer_range * -2.0,
                "b": self.initializer_range * 2.0,
            }

        if embeddings_initializer is None:
            embeddings_initializer = default_initializer

        if position_embeddings_initializer is None:
            position_embeddings_initializer = default_initializer

        if segment_embeddings_initializer is None:
            segment_embeddings_initializer = default_initializer

        self.embedding_layer = EmbeddingLayer(
            vocab_size=vocab_size,
            embedding_size=hidden_size,
            pad_token_id=embedding_pad_token_id,
            embeddings_initializer=embeddings_initializer,
            max_position_embeddings=max_position_embeddings,
            position_embedding_type=position_embedding_type,
            position_embeddings_initializer=position_embeddings_initializer,
            num_segments=num_segments,
            segment_embeddings_initializer=segment_embeddings_initializer,
        )

        self.dropout_embd = nn.Dropout(embedding_dropout_rate)

        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=filter_size,
            dropout=dropout_rate,
            activation=nonlinearity,
            layer_norm_eps=layer_norm_epsilon,
            attention_module_str=attention_module_str,
            extra_attention_params=extra_attention_params,
            attention_dropout_rate=attention_dropout_rate,
            attention_type=attention_type,
            use_projection_bias_in_attention=use_projection_bias_in_attention,
            use_ffn_bias_in_attention=use_ffn_bias_in_attention,
            use_ffn_bias=use_ffn_bias,
            attention_initializer=default_initializer,
            ffn_initializer=default_initializer,
        )

        self.embed_ln_f = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)

        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_layers=num_hidden_layers
        )

        self.pooler = (
            BertPooler(
                hidden_size,
                use_bias=use_ffn_bias,
                activation=nonlinearity,
                dropout=None,
                initializer=default_initializer,
            )
            if self.add_pooling_layer
            else None
        )

        self.__reset_parameters()

        # TODO: Add sparse attention

    def reset_parameters(self):
        self.embedding_layer.reset_parameters()
        self.transformer_encoder.reset_parameters()
        self.pooler.reset_parameters()
        self.__reset_parameters()

    def __reset_parameters(self):
        # Init norm layers
        self.embed_ln_f.bias.data.zero_()
        self.embed_ln_f.weight.data.fill_(1.0)

    def forward(self, input_ids=None, segment_ids=None, attention_mask=None):
        """
        Args:
            input_ids (Tensor): The id of input tokens
                Can be of shape ```[batch_size, seq_length]`
            segment_ids (Tensor): The segment id of input tokens, indicating which sequence the token belongs to
                Can be of shape ```[batch_size, seq_length]`
            attention_mask (Tensor):
                Can be 2D of shape ``[batch_size, seq_length]``,
                or 3D of shape ``[batch, query_length, seq_length]``,
                or 4D of shape ``[batch, num_heads, query_length, seq_length]``.
        """
        src_key_padding_mask = None
        if len(attention_mask.size()) == 2:
            src_key_padding_mask = attention_mask
            attention_mask = None

        hidden_states = self.embedding_layer(input_ids, segment_ids=segment_ids)
        hidden_states = self.embed_ln_f(hidden_states)
        hidden_states = self.dropout_embd(hidden_states)
        hidden_states = self.transformer_encoder(
            hidden_states,
            mask=attention_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        pooled_output = None
        if self.add_pooling_layer:
            pooled_output = self.pooler(hidden_states)

        return hidden_states, pooled_output
