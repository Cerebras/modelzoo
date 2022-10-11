# This code is adapted from
# https://github.com/huggingface/transformers/blob/master/src/transformers/models/t5/configuration_t5.py
#
# Copyright 2022 Cerebras Systems.
#
# Copyright 2020 The T5 Authors and HuggingFace Inc. All rights reserved.
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

""" T5 model configuration """

from .configuration_utils import PretrainedConfig
from .utils import logging

logger = logging.get_logger(__name__)

T5_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "t5-small": "https://huggingface.co/t5-small/resolve/main/config.json",
    "t5-base": "https://huggingface.co/t5-base/resolve/main/config.json",
    "t5-large": "https://huggingface.co/t5-large/resolve/main/config.json",
    "t5-3b": "https://huggingface.co/t5-3b/resolve/main/config.json",
    "t5-11b": "https://huggingface.co/t5-11b/resolve/main/config.json",
}


class T5Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.T5Model` or a
    :class:`~transformers.TFT5Model`. It is used to instantiate a T5 model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the T5 `t5-small <https://huggingface.co/t5-small>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Arguments:
        src_vocab_size (:obj:`int`, `optional`, defaults to 32128):
            Source vocabulary size of the T5 model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.T5Model` or :class:`~transformers.TFT5Model`.
        tgt_vocab_size (:obj:`int`, `optional`, defaults to 32128):
            Target vocabulary size of the T5 model. Only useful if set for Transformer variant where source and target
            vocabularies can be different.
        d_model (:obj:`int`, `optional`, defaults to 512):
            Size of the encoder layers and the pooler layer.
        d_kv (:obj:`int`, `optional`, defaults to 64):
            Size of the key, query, value projections per attention head. :obj:`d_kv` has to be equal to :obj:`d_model
            // num_heads`.
        d_ff (:obj:`int`, `optional`, defaults to 2048):
            Size of the intermediate feed forward layer in each :obj:`T5Block`.
        encoder_num_hidden_layers (:obj:`int`, `optional`, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        decoder_num_hidden_layers (:obj:`int`, `optional`):
            Number of hidden layers in the Transformer decoder. Will use the same value as :obj:`num_layers` if not
            set.
        num_heads (:obj:`int`, `optional`, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        relative_attention_num_buckets (:obj:`int`, `optional`, defaults to 32):
            The number of buckets to use for each attention layer.
        use_t5_layer_norm (:obj:`bool`, `optional`, defaults to False):
            Whether to use T5 layer norm (with no mean subtraction and bias correction) or
            use the regular nn.LayerNorm module.
        dropout_rate (:obj:`float`, `optional`, defaults to 0.1):
            The ratio for all dropout layers.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        initializer_factor (:obj:`float`, `optional`, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        encoder_nonlinearity (:obj:`string`, `optional`, defaults to :obj:`"relu"`):
             Type of feed forward layer to be used in encoder. Should be one of :obj:`"relu"` or :obj:`"gated-gelu"` or
              :obj:`"gelu"`. T5v1.1 uses the :obj:`"gated-gelu"` feed forward projection. Original T5 uses :obj:`"relu"`.
        decoder_nonlinearity (:obj:`string`, `optional`, defaults to :obj:`"relu"`):
             Type of feed forward layer to be used in decoder. Should be one of :obj:`"relu"` or :obj:`"gated-gelu"` or
             :obj:`"gelu"`. T5v1.1 uses the :obj:`"gated-gelu"` feed forward projection. Original T5 uses :obj:`"relu"`.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        position_embedding_type (:obj: `string`, `optional`, defaults to :obj:`"relative"`):
            The type of position embedding to use. Should be one of :obj:`"fixed"`,
            :obj:`"learned_absolute"`, :obj:`"relative"`, or :obj:`None`. :obj:`"fixed"`
            uses a concatenation of sin curves to express relative position as used in
            the original Transformer paper. :obj:`"learned_absolute"` uses a learned
            vector for each position in the sequence. :obj:`"relative"` uses learned
            relative position embeddings as introduced in https://arxiv.org/abs/1803.02155,
            configured as done in the original T5 publication. :obj:`None` turns off
            position embedding altogether.
        src_max_position_embeddings (:obj:`int`, `optional`, defaults to :obj: 512):
            Maximum source sequence length to train using to train the model.
        tgt_max_position_embeddings (:obj:`int`, `optional`, defaults to :obj: 512):
            Maximum target sequence length to train using to train the model.
        use_dropout_outside_residual_path (:obj:`bool`, `optional`, defaults to :obj: True):
            Whether to set dropout calculations outside of the residual path.
            Set to `True` for T5, but `False` for Transformer.
        share_encoder_decoder_embedding (:obj:`bool`, `optional`, defaults to :obj: True):
            Whether to share encoder/decoder embedding layer.
            Set to `True` for both T5 and Transformer models.
        tie_word_embeddings (:obj:`bool`, `optional`, defaults to :obj: True):
            Whether to share embedding weights between encoder and decoder.
            T5 sets this to True, but Transformer can set to False in case source and target
            vocab files are of the different size.
        relu_dropout_rate (:obj:`int`, `optional`, defaults to :obj: 0.1):
            Dropout rate utilized in the FFN layer after applying relu activation function.
            This parameter is set to `0` for Transformer model, and set to `dropout_rate`
            for default T5 configuration.
            Transformer reference: https://github.com/tensorflow/tensor2tensor/blob/5623deb79cfcd28f8f8c5463b58b5bd76a81fd0d/tensor2tensor/models/transformer.py#L1811
            T5 reference: https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/models/t5/modeling_t5.py#L261
        use_pre_encoder_decoder_dropout (:obj:`bool`, `optional`, defaults to :obj: False):
            Whether to use dropout layer after positional embedding layer and encoder/decoder.
            This is set to `False` for T5 and `True` for Transformer.
        use_pre_encoder_decoder_layer_norm (:obj:`bool`, `optional`, defaults to :obj: False):
            Whether to use layer norm before passing input tensors into encoder/decoder.
            This is set to `False` for T5 and `True` for Transformer.
        use_ffn_bias (:obj:`bool`, `optional`, defaults to :obj: False):
            Whether to use bias in the hidden layer with relu activation.
            This is set to `False` for T5, and `True` for Transformer.
        lm_loss_weight (:obj:`float`, `optional`, default to :obj: 1.0):
            Value that scales loss by the mean number
            of predictions per sequence in the dataset.
        use_transformer_initialization (:obj:`bool`, `optional`, defaults to :obj:`False`):
            The Transformer model tends to converge best with a scaled variant on
            Xavier uniform initialization used for linear layers. This contrasts
            the initialization used for the original T5 paper, which uses He normal
            initialization for linear layers. Setting this flag to `True` switches
            the initialization to the Transformer specific scaled Xavier initialization.
    """
    model_type = "t5"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        src_vocab_size=32128,
        tgt_vocab_size=32128,
        d_model=512,
        d_kv=64,
        d_ff=2048,
        encoder_num_hidden_layers=6,
        decoder_num_hidden_layers=None,
        num_heads=8,
        relative_attention_num_buckets=32,
        use_t5_layer_norm=False,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        encoder_nonlinearity="relu",
        decoder_nonlinearity="relu",
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        sos_token_id=2,
        position_embedding_type="relative",
        src_max_position_embeddings=512,
        tgt_max_position_embeddings=512,
        use_dropout_outside_residual_path=True,
        share_encoder_decoder_embedding=True,
        tie_word_embeddings=True,
        relu_dropout_rate=0.1,
        use_pre_encoder_decoder_dropout=False,
        use_pre_encoder_decoder_layer_norm=False,
        use_ffn_bias=False,
        lm_loss_weight=1.0,
        use_transformer_initialization=False,
        **kwargs,
    ):
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.encoder_num_hidden_layers = encoder_num_hidden_layers
        self.decoder_num_hidden_layers = (
            decoder_num_hidden_layers
            if decoder_num_hidden_layers is not None
            else self.encoder_num_hidden_layers
        )  # default = symmetry
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.use_t5_layer_norm = use_t5_layer_norm
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.encoder_nonlinearity = encoder_nonlinearity
        self.decoder_nonlinearity = decoder_nonlinearity
        self.use_cache = use_cache
        self.position_embedding_type = position_embedding_type
        self.src_max_position_embeddings = src_max_position_embeddings
        self.tgt_max_position_embeddings = tgt_max_position_embeddings
        self.use_dropout_outside_residual_path = (
            use_dropout_outside_residual_path
        )
        self.share_encoder_decoder_embedding = share_encoder_decoder_embedding
        self.tie_word_embeddings = tie_word_embeddings
        self.relu_dropout_rate = relu_dropout_rate
        self.use_pre_encoder_decoder_dropout = use_pre_encoder_decoder_dropout
        self.use_pre_encoder_decoder_layer_norm = (
            use_pre_encoder_decoder_layer_norm
        )
        self.use_ffn_bias = use_ffn_bias
        self.lm_loss_weight = lm_loss_weight
        self.use_transformer_initialization = use_transformer_initialization
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
