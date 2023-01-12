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

# This code is adapted from
# https://github.com/huggingface/transformers/blob/master/src/transformers/models/t5/modeling_t5.py
#
# Copyright 2022 Cerebras Systems.
#
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
import torch.nn as nn

from modelzoo.common.pytorch.layers import (
    BiaslessLayerNorm,
    EmbeddingLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from modelzoo.transformers.pytorch.transformer_utils import (
    build_broadcastable_attention_mask,
    create_2D_autoregressive_mask,
    make_key_padding_mask_broadcastable,
)


class T5ForConditionalGeneration(nn.Module):
    r"""
    T5 Model with a `language modeling` head on top.

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
            Size of the key, query, value projections per attention head. :obj:`d_kv` does *not* have tobe equal to :obj:`d_model
            // num_heads`.
        d_ff (:obj:`int`, `optional`, defaults to 2048):
            Size of the intermediate feed forward layer in each :obj:`T5Block`.
        encoder_num_hidden_layers (:obj:`int`, `optional`, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        decoder_num_hidden_layers (:obj:`int`, `optional`):
            Number of hidden layers in the Transformer decoder. Will use the same value as :obj:`num_layers` if not
            set.
        num_heads (:obj:`int`, `optional`, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder and decoder.
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
            Whether to share embedding weights between decoder and language model head.
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
        relu_dropout_rate=None,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        encoder_nonlinearity="relu",
        decoder_nonlinearity="relu",
        use_projection_bias_in_attention=False,
        use_cache=False,
        decoder_start_token_id=None,
        pad_token_id=0,
        position_embedding_type="relative",
        src_max_position_embeddings=512,
        tgt_max_position_embeddings=512,
        use_dropout_outside_residual_path=True,
        share_encoder_decoder_embedding=True,
        tie_word_embeddings=True,
        tie_encoder_decoder=False,
        use_pre_encoder_decoder_dropout=False,
        use_pre_encoder_decoder_layer_norm=False,
        use_ffn_bias=False,
        lm_loss_weight=1.0,
        label_smoothing=0.0,
        mlm_loss_scaling="batch_size",
        use_transformer_initialization=False,
        **kwargs,
    ):
        super().__init__()

        # Copy only the subset of params that are referenced later
        self.d_model = d_model
        self.d_kv = d_kv
        self.attention_inner_dim = d_kv * num_heads
        self.initializer_factor = initializer_factor
        self.use_cache = use_cache
        self.share_encoder_decoder_embedding = share_encoder_decoder_embedding
        self.tie_word_embeddings = tie_word_embeddings
        self.tie_encoder_decoder = tie_encoder_decoder
        self.decoder_start_token_id = decoder_start_token_id
        self.pad_token_id = pad_token_id
        self.position_embedding_type = position_embedding_type
        self.label_smoothing = label_smoothing

        if relu_dropout_rate is None:
            relu_dropout_rate = dropout_rate

        assert position_embedding_type in (
            "fixed",
            "learned_absolute",
            "relative",
            None,
        ), (
            f"Position embedding must be one of `fixed`, `learned_absolute`, "
            f"`relative`, or None. Got {position_embedding_type}."
        )
        if position_embedding_type == "learned_absolute":
            position_embedding_type = "learned"

        self.encoder_embeddings = EmbeddingLayer(
            src_vocab_size,
            d_model,
            embeddings_initializer={
                "name": "truncated_normal",
                "mean": 0.0,
                "std": 1.0,
                "a": -2.0,
                "b": 2.0,
            }
            if use_transformer_initialization
            else {
                "name": "normal",
                "mean": 0.0,
                "std": initializer_factor * 1.0,
            },
            max_position_embeddings=src_max_position_embeddings,
            position_embedding_type=position_embedding_type,
        )

        self.decoder_embeddings = EmbeddingLayer(
            tgt_vocab_size,
            d_model,
            embeddings_initializer={
                "name": "truncated_normal",
                "mean": 0.0,
                "std": 1.0,
                "a": -2.0,
                "b": 2.0,
            }
            if use_transformer_initialization
            else {
                "name": "normal",
                "mean": 0.0,
                "std": initializer_factor * 1.0,
            },
            max_position_embeddings=tgt_max_position_embeddings,
            position_embedding_type=position_embedding_type,
        )

        if self.share_encoder_decoder_embedding:
            assert (
                src_vocab_size == tgt_vocab_size
            ), "Cannot share embeddings between encoder and decoder due to different vocab sizes"
            self.decoder_embeddings.set_input_embeddings(
                self.encoder_embeddings.get_input_embeddings()
            )

        self.relative_position_encoder = None
        self.relative_position_decoder = None
        if position_embedding_type == "relative":
            # we use bidirectional relative attention for encoder and unidirectional
            # for decoder
            self.relative_position_encoder = self.encoder_embeddings.position_embedding_helper(
                num_heads=num_heads,
                bidirectional=True,
                num_relative_attention_buckets=relative_attention_num_buckets,
            )

            self.relative_position_decoder = self.decoder_embeddings.position_embedding_helper(
                num_heads=num_heads,
                bidirectional=False,
                num_relative_attention_buckets=relative_attention_num_buckets,
            )

        self.pre_encoder_dropout = None
        self.pre_decoder_dropout = None
        # Transformer model uses dropout right after position embeddings
        # and before the encoder call, T5 does not use it.
        if use_pre_encoder_decoder_dropout:
            self.pre_encoder_dropout = nn.Dropout(dropout_rate)
            self.pre_decoder_dropout = nn.Dropout(dropout_rate)

        assert encoder_nonlinearity in [
            "relu",
            "gelu",
        ], "T5 doesn't support encoder_nonlinearity {}".format(
            encoder_nonlinearity
        )
        assert decoder_nonlinearity in [
            "relu",
            "gelu",
        ], "T5 doesn't support decoder_nonlinearity {}".format(
            decoder_nonlinearity
        )

        if (encoder_nonlinearity == "gelu" and use_ffn_bias) or (
            decoder_nonlinearity == "gelu" and use_ffn_bias
        ):
            print(
                "Warning. Overriding use_ffn_bias to false because using gelu"
            )
            use_ffn_bias = False

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout_rate,
            activation=encoder_nonlinearity,
            norm_layer=BiaslessLayerNorm if use_t5_layer_norm else nn.LayerNorm,
            layer_norm_eps=layer_norm_epsilon,
            norm_first=not use_pre_encoder_decoder_layer_norm,
            batch_first=True,
            attention_type="scaled_dot_product"
            if use_transformer_initialization
            else "dot_product",
            attention_inner_dim=self.attention_inner_dim,
            use_projection_bias_in_attention=use_projection_bias_in_attention,
            use_ffn_bias_in_attention=False,
            use_ffn_bias=use_ffn_bias,
            ffn_dropout_rate=relu_dropout_rate,
            use_ff_layer1_dropout=True,
            use_ff_layer2_dropout=True,
            attention_q_initializer={
                "name": "variance_scaling",
                "scale": 1.0 / (d_kv * 9.0),
                "mode": "fan_avg",
                "distribution": "uniform",
            }
            if use_transformer_initialization
            else {
                "name": "normal",
                "mean": 0.0,
                "std": initializer_factor * ((d_model * d_kv) ** -0.5),
            },
            attention_initializer={
                "name": "variance_scaling",
                "scale": 1.0 / 9.0,
                "mode": "fan_avg",
                "distribution": "uniform",
            }
            if use_transformer_initialization
            else {
                "name": "normal",
                "mean": 0.0,
                "std": initializer_factor * (d_model ** -0.5),
            },
            ffn_initializer={"name": "xavier_uniform", "gain": 1.0}
            if use_transformer_initialization
            else {
                "name": "normal",
                "mean": 0.0,
                "std": initializer_factor * (d_model ** -0.5),
            },
            ffn_output_layer_initializer={"name": "xavier_uniform", "gain": 1.0}
            if use_transformer_initialization
            else {
                "name": "normal",
                "mean": 0.0,
                "std": initializer_factor * (d_ff ** -0.5),
            },
        )

        if use_t5_layer_norm:
            encoder_final_layer_norm = BiaslessLayerNorm(
                d_model, eps=layer_norm_epsilon
            )
        else:
            encoder_final_layer_norm = torch.nn.LayerNorm(
                d_model, eps=layer_norm_epsilon
            )

        self.dropout_before_encoder = nn.Dropout(dropout_rate)
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_layers=encoder_num_hidden_layers,
            norm=encoder_final_layer_norm,
        )
        self.dropout_after_encoder = None
        if use_dropout_outside_residual_path:
            self.dropout_after_encoder = nn.Dropout(dropout_rate)

        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout_rate,
            activation=encoder_nonlinearity,
            norm_layer=BiaslessLayerNorm if use_t5_layer_norm else nn.LayerNorm,
            layer_norm_eps=layer_norm_epsilon,
            norm_first=not use_pre_encoder_decoder_layer_norm,
            batch_first=True,
            attention_type="scaled_dot_product"
            if use_transformer_initialization
            else "dot_product",
            attention_inner_dim=self.attention_inner_dim,
            use_projection_bias_in_attention=use_projection_bias_in_attention,
            use_ffn_bias_in_attention=False,
            use_ffn_bias=use_ffn_bias,
            use_ff_layer1_dropout=True,
            use_ff_layer2_dropout=True,
            attention_q_initializer={
                "name": "variance_scaling",
                "scale": 1.0 / (d_kv * 9.0),
                "mode": "fan_avg",
                "distribution": "uniform",
            }
            if use_transformer_initialization
            else {
                "name": "normal",
                "mean": 0.0,
                "std": initializer_factor * ((d_model * d_kv) ** -0.5),
            },
            attention_initializer={
                "name": "variance_scaling",
                "scale": 1.0 / 9.0,
                "mode": "fan_avg",
                "distribution": "uniform",
            }
            if use_transformer_initialization
            else {
                "name": "normal",
                "mean": 0.0,
                "std": initializer_factor * (d_model ** -0.5),
            },
            ffn_initializer={"name": "xavier_uniform", "gain": 1.0}
            if use_transformer_initialization
            else {
                "name": "normal",
                "mean": 0.0,
                "std": initializer_factor * (d_model ** -0.5),
            },
            ffn_output_layer_initializer={"name": "xavier_uniform", "gain": 1.0}
            if use_transformer_initialization
            else {
                "name": "normal",
                "mean": 0.0,
                "std": initializer_factor * (d_ff ** -0.5),
            },
        )

        if use_t5_layer_norm:
            decoder_final_layer_norm = BiaslessLayerNorm(
                d_model, eps=layer_norm_epsilon
            )
        else:
            decoder_final_layer_norm = torch.nn.LayerNorm(
                d_model, eps=layer_norm_epsilon
            )

        self.dropout_before_decoder = nn.Dropout(dropout_rate)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_layers=decoder_num_hidden_layers,
            norm=decoder_final_layer_norm,
        )
        self.dropout_after_decoder = None
        if use_dropout_outside_residual_path:
            self.dropout_after_decoder = nn.Dropout(dropout_rate)

        self.lm_head = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.__reset_parameters()
        assert (
            not tie_encoder_decoder
        ), "Implementation does not currently support tied Encoder/Decoder weights"
        self.tie_weights()

    def reset_parameters(self):
        self.encoder_embeddings.reset_parameters()
        self.decoder_embeddings.reset_parameters()
        if self.relative_position_encoder:
            self.relative_position_encoder.reset_parameters()
        if self.relative_position_decoder:
            self.relative_position_decoder.reset_parameters()
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        self.__reset_parameters()

    def __reset_parameters(self):
        # Initialize LM head
        if not self.tie_word_embeddings:
            self.lm_head.weight.data.normal_(
                mean=0.0, std=self.initializer_factor * 1.0
            )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        loss_weight=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

            >>> # training
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> # inference
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
            >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
            >>> # studies have shown that owning a dog is good for you.
        """

        assert (
            past_key_values is None
        ), "past_key_values should be None since inference is not supported yet"

        use_cache = use_cache if use_cache is not None else self.use_cache

        assert (
            not use_cache
        ), "cannot enable use_cache because inference is not supported yet"

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            src = self.encoder_embeddings(input_ids)
            # Transformer uses pre-encoder dropout
            if self.pre_encoder_dropout:
                src = self.pre_encoder_dropout(src)

            # Compute relative position bias for the encoder block if applicable
            encoder_self_attn_position_bias = None
            if self.relative_position_encoder:
                encoder_self_attn_position_bias = self.relative_position_encoder(
                    src.shape[1], src.shape[1]
                )
            src = self.dropout_before_encoder(src)
            if attention_mask is not None:
                attention_mask = make_key_padding_mask_broadcastable(
                    attention_mask, dtype=src.dtype
                )

            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                src,
                mask=attention_mask,
                self_attn_position_bias=encoder_self_attn_position_bias,
            )
            if self.dropout_after_encoder:
                encoder_outputs = self.dropout_after_encoder(
                    encoder_outputs
                )  # HF T5 Decoder also applies dropout at the end

        hidden_states = encoder_outputs

        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        decoder_inputs_embeds = self.decoder_embeddings(decoder_input_ids)

        # Transformer uses dropout before feeding to decoder module while
        # T5 does not use this layer
        if self.pre_decoder_dropout:
            decoder_inputs_embeds = self.pre_decoder_dropout(
                decoder_inputs_embeds
            )

        batch_size, decoder_seq_length = decoder_inputs_embeds.size()[:2]

        decoder_self_attn_position_bias = None
        if self.relative_position_decoder:
            decoder_self_attn_position_bias = self.relative_position_decoder(
                decoder_seq_length, decoder_seq_length
            )

        if decoder_attention_mask is None:
            extended_decoder_attention_mask = (
                create_2D_autoregressive_mask(
                    decoder_seq_length,
                    decoder_seq_length,
                    device=decoder_inputs_embeds.device,
                    dtype=hidden_states.dtype,
                )
                * -1e4
            )
        else:
            extended_decoder_attention_mask = build_broadcastable_attention_mask(
                decoder_attention_mask,
                build_causal=True,
                device=decoder_inputs_embeds.device,
                dtype=hidden_states.dtype,
            )

        decoder_inputs_embeds = self.dropout_before_decoder(
            decoder_inputs_embeds
        )
        decoder_outputs = self.decoder(
            decoder_inputs_embeds,
            memory=hidden_states,
            tgt_mask=extended_decoder_attention_mask,
            memory_mask=attention_mask,
            past_kv=past_key_values,
            cache_present_kv=use_cache,
            self_attn_position_bias=decoder_self_attn_position_bias,
        )
        if use_cache:
            sequence_output, present_kv = decoder_outputs
        else:
            sequence_output = decoder_outputs
        if self.dropout_after_decoder:
            sequence_output = self.dropout_after_decoder(sequence_output)

        if self.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.d_model ** -0.5)

        lm_logits = self.lm_head(sequence_output)
        return lm_logits

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings
        and (if enabled) tie encoder/decoder weights.
        """
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is not None and self.tie_word_embeddings:
            self._tie_or_clone_weights(
                output_embeddings, self.get_input_embeddings()
            )

        if self.tie_encoder_decoder:
            if hasattr(self, self.base_model_prefix):
                self = getattr(self, self.base_model_prefix)
            self._tie_encoder_decoder_weights(
                self.encoder, self.decoder, self.base_model_prefix
            )

    @staticmethod
    def _tie_encoder_decoder_weights(
        encoder: nn.Module, decoder: nn.Module, base_model_prefix: str
    ):
        uninitialized_encoder_weights: List[str] = []
        if decoder.__class__ != encoder.__class__:
            print(
                f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
            )

        def tie_encoder_to_decoder_recursively(
            decoder_pointer: nn.Module,
            encoder_pointer: nn.Module,
            module_name: str,
            uninitialized_encoder_weights: List[str],
            depth=0,
        ):
            assert isinstance(decoder_pointer, nn.Module) and isinstance(
                encoder_pointer, nn.Module
            ), f"{decoder_pointer} and {encoder_pointer} have to be of type nn.Module"
            if hasattr(decoder_pointer, "weight"):
                assert hasattr(encoder_pointer, "weight")
                encoder_pointer.weight = decoder_pointer.weight
                if hasattr(decoder_pointer, "bias"):
                    assert hasattr(encoder_pointer, "bias")
                    encoder_pointer.bias = decoder_pointer.bias
                return

            encoder_modules = encoder_pointer._modules
            decoder_modules = decoder_pointer._modules
            if len(decoder_modules) > 0:
                assert (
                    len(encoder_modules) > 0
                ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

                all_encoder_weights = set(
                    [
                        module_name + "/" + sub_name
                        for sub_name in encoder_modules.keys()
                    ]
                )
                encoder_layer_pos = 0
                for name, module in decoder_modules.items():
                    if name.isdigit():
                        encoder_name = str(int(name) + encoder_layer_pos)
                        decoder_name = name
                        if not isinstance(
                            decoder_modules[decoder_name],
                            type(encoder_modules[encoder_name]),
                        ) and len(encoder_modules) != len(decoder_modules):
                            # this can happen if the name corresponds to the position in a list module list of layers
                            # in this case the decoder has added a cross-attention that the encoder does not have
                            # thus skip this step and subtract one layer pos from encoder
                            encoder_layer_pos -= 1
                            continue
                    elif name not in encoder_modules:
                        continue
                    elif depth > 500:
                        raise ValueError(
                            "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                        )
                    else:
                        decoder_name = encoder_name = name
                    tie_encoder_to_decoder_recursively(
                        decoder_modules[decoder_name],
                        encoder_modules[encoder_name],
                        module_name + "/" + name,
                        uninitialized_encoder_weights,
                        depth=depth + 1,
                    )
                    all_encoder_weights.remove(module_name + "/" + encoder_name)

                uninitialized_encoder_weights += list(all_encoder_weights)

        # tie weights recursively
        tie_encoder_to_decoder_recursively(
            decoder, encoder, base_model_prefix, uninitialized_encoder_weights
        )
        if len(uninitialized_encoder_weights) > 0:
            print(
                f"The following encoder weights were not tied to the decoder {uninitialized_encoder_weights}"
            )

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone module weights depending of whether we are using TorchScript or not"""
        if not isinstance(output_embeddings, list):
            output_embeddings = [output_embeddings]
            input_embeddings = [input_embeddings]

        for output_embedding, input_embedding in zip(
            output_embeddings, input_embeddings
        ):
            output_embedding.weight = input_embedding.weight

            if getattr(output_embedding, "bias", None) is not None:
                output_embedding.bias.data = nn.functional.pad(
                    output_embedding.bias.data,
                    (
                        0,
                        output_embedding.weight.shape[0]
                        - output_embedding.bias.shape[0],
                    ),
                    "constant",
                    0,
                )
            if hasattr(output_embedding, "out_features") and hasattr(
                input_embedding, "num_embeddings"
            ):
                output_embedding.out_features = input_embedding.num_embeddings

    def get_input_embeddings(self):
        # This function returns decoder token embeddings
        # in order to properly tie embeddings between the decoder
        # input and decoder output.
        return self.decoder_embeddings.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.decoder_embeddings.set_input_embeddings(new_embeddings)
        if self.share_encoder_decoder_embedding:
            self.encoder_embeddings.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.decoder_start_token_id
        pad_token_id = self.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert (
            pad_token_id is not None
        ), "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(
            shifted_input_ids >= 0
        ).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids
