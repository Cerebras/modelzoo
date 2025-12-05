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

from modelzoo.transformers.pytorch.huggingface_common.configuration_utils import (
    PretrainedConfig,
)


class GenomicBertConfig(PretrainedConfig):
    r"""
    Args:
        vocab_size_dna (:obj:`int`):
            Vocabulary size of the Genomic BERT model for dna sequences. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids_dna` passed when calling :class:`~.GenomicBertModel`. Defaults to 1027.
        vocab_size_ideas (:obj:`int`):
            Vocabulary size of the Genomic BERT model for ideas sequences. Defaults to 38.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        use_projection_bias_in_attention (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If True, bias is used on the projection layers in attention.
        use_ffn_bias_in_attention (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If  True, bias is used in the dense layer in the attention.
        use_output_bias_in_mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If True, bias is used in the output layer in the mlm head.
        use_ffn_bias_in_mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If True, bias is used in the dense layer in the mlm head.
        mlm_nonlinearity: (:obj:`string`, `optional`, defaults to :obj:`gelu`):
            Nonlinearity used in the mlm head.
        use_ffn_bias: (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If  True, bias is used in the dense layer in the encoder.
    """
    model_type = "bert"

    def __init__(
        self,
        vocab_size_dna=1028,
        vocab_size_ideas=39,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        gradient_checkpointing=False,
        use_projection_bias_in_attention=True,
        use_ffn_bias_in_attention=True,
        use_output_bias_in_mlm=True,
        use_ffn_bias_in_mlm=True,
        mlm_nonlinearity="gelu",
        use_ffn_bias=True,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size_dna = vocab_size_dna
        self.vocab_size_ideas = vocab_size_ideas
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.use_projection_bias_in_attention = use_projection_bias_in_attention
        self.use_ffn_bias_in_attention = use_ffn_bias_in_attention
        self.use_output_bias_in_mlm = use_output_bias_in_mlm
        self.use_ffn_bias_in_mlm = use_ffn_bias_in_mlm
        self.mlm_nonlinearity = mlm_nonlinearity
        self.use_ffn_bias = use_ffn_bias
