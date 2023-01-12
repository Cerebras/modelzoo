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

import tensorflow as tf

from modelzoo.common.tf.layers.AddLayer import AddLayer
from modelzoo.common.tf.layers.AttentionLayer import (
    AttentionLayer,
    SelfAttentionLayer,
)
from modelzoo.common.tf.layers.BaseLayer import BaseLayer
from modelzoo.common.tf.layers.DropoutLayer import DropoutLayer
from modelzoo.common.tf.layers.FeedForwardNetwork import FeedForwardNetwork
from modelzoo.common.tf.layers.LayerNormalizationLayer import (
    LayerNormalizationLayer,
)
from modelzoo.common.tf.layers.PrePostProcessWrapper import (
    PrePostProcessWrapper,
)


class Decoder(BaseLayer):
    """
    Transformer decoder.
    Made up of num_hidden_layers identical layers.
    Each layer is composed of the sublayers:
        1. Self-attention layer.
        2. Cross-attention (can be skipped if no encoder outputs provided).
        3. Feedforward network (which is 2 fully-connected layers).
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        num_hidden_layers,
        filter_size,
        use_projection_bias_in_attention=False,
        use_ffn_bias_in_attention=False,
        use_ffn_bias=False,
        attention_initializer="glorot_uniform",
        ffn_initializer="glorot_uniform",
        output_layer_initializer=None,
        weight_regularizer=None,
        attention_type="scaled_dot_product",
        attention_dropout_rate=0.0,
        nonlinearity="relu",
        dropout_rate=0.0,
        dropout_seed=None,
        layer_norm_epsilon=1e-8,
        use_pre_normalization=False,
        attention_softmax_fp32=True,
        boundary_casting=False,
        tf_summary=False,
        **kwargs,
    ):
        """
        Initialize the decoder object instance.

        :param int hidden_size: Hidden size
        :param int num_heads: Number of heads in the multi-headed attention layer
        :param int num_hidden_layers: Number of decoders in the decoder stack
        :param int filter_size: Output size of the first layer of the FFN sublayer
        :param bool use_projection_bias_in_attention: Add bias to Q,K,V projections
            in the Attention layer. Defaults to False.
        :param bool use_ffn_bias_in_attention: Add bias in the concluding FFN
            in the Attention layer. Defaults to False.
        :param bool use_ffn_bias: Add bias in all dense layers of the decoder's ffn sublayer
        :param string initializer: Kernel initializer. Defaults to "glorot_uniform".
        :param str or initializer output_layer_initializer: If not None, use this
            initializer for the output transform layers. Defaults to None.
        :param callable weight_regularizer: Weights regularizer.
        :param string attention_type: Type of attention.
            Currently support {"dot_product","scaled_dot_product"}.
            Defaults to "scaled_dot_product".
        :param float attention_dropout_rate: Attention dropout rate.
        :param string nonlinearity: Type of nonlinearity applied after the first dense layer
            of the FFN sublayer. Defaults to relu.
        :param float dropout_rate: Dropout rate. Same for all dense layers
            encountered in the Decoder. Defaults to 0 (i.e., no dropout)
        :param float layer_norm_epsilon: LayerNorm epsilon. Same for all layers
            encountered in the Decoder. Defaults to 1e-8
        :param bool use_pre_normalization: If False, uses original transformer decoder:
            "residual split -> transform -> dropout -> residual add -> layer norm".
            Otherwise, use residual block with pre-normaliztion:
            "residual split -> layer norm -> transform -> dropout -> residual add".
            Here transform is either a self-attention, cross-attention, or ffn sub-block.
            Defaults to False.
        :param bool attention_softmax_fp32: Perform softmax in attention
            layers in FP32. Defaults to True.
        """
        super(Decoder, self).__init__(boundary_casting, tf_summary, **kwargs)

        # Set up pre- and post-processing steps to be wrapped
        # around all decoder sublayers in the decoder stack
        pre_process_config = []

        if use_pre_normalization:
            pre_process_config.append(
                (
                    LayerNormalizationLayer,
                    {
                        'epsilon': layer_norm_epsilon,
                        'beta_regularizer': weight_regularizer,
                        'gamma_regularizer': weight_regularizer,
                        'boundary_casting': self.boundary_casting,
                        'tf_summary': self.tf_summary,
                        'dtype': self.dtype_policy,
                    },
                )
            )

        post_process_config = [
            (
                DropoutLayer,
                {
                    'rate': dropout_rate,
                    'seed': dropout_seed,
                    'boundary_casting': self.boundary_casting,
                    'tf_summary': tf_summary,
                    'dtype': self.dtype_policy,
                },
            ),
            (
                AddLayer,
                {
                    'boundary_casting': self.boundary_casting,
                    'tf_summary': self.tf_summary,
                    'dtype': self.dtype_policy,
                },
            ),
        ]

        if not use_pre_normalization:
            post_process_config.append(
                (
                    LayerNormalizationLayer,
                    {
                        'epsilon': layer_norm_epsilon,
                        'beta_regularizer': weight_regularizer,
                        'gamma_regularizer': weight_regularizer,
                        'boundary_casting': self.boundary_casting,
                        'tf_summary': self.tf_summary,
                        'dtype': self.dtype_policy,
                    },
                )
            )

        # Create sublayers for each layer.
        self.layers = []
        for _ in range(num_hidden_layers):
            self_attention_layer = SelfAttentionLayer(
                hidden_size,
                num_heads,
                use_projection_bias=use_projection_bias_in_attention,
                use_ffn_bias=use_ffn_bias_in_attention,
                initializer=attention_initializer,
                output_layer_initializer=output_layer_initializer,
                kernel_regularizer=weight_regularizer,
                bias_regularizer=weight_regularizer,
                attention_type=attention_type,
                dropout_rate=attention_dropout_rate,
                dropout_seed=dropout_seed,
                softmax_dtype_fp32=attention_softmax_fp32,
                boundary_casting=self.boundary_casting,
                tf_summary=self.tf_summary,
                dtype=self.dtype_policy,
                name="self_attention",
            )

            cross_attention_layer = AttentionLayer(
                hidden_size,
                num_heads,
                use_projection_bias=use_projection_bias_in_attention,
                use_ffn_bias=use_ffn_bias_in_attention,
                initializer=attention_initializer,
                output_layer_initializer=output_layer_initializer,
                kernel_regularizer=weight_regularizer,
                bias_regularizer=weight_regularizer,
                attention_type=attention_type,
                dropout_rate=attention_dropout_rate,
                dropout_seed=dropout_seed,
                boundary_casting=self.boundary_casting,
                tf_summary=self.tf_summary,
                dtype=self.dtype_policy,
                name="encoder_decoder_attention",
            )

            feed_forward_network = FeedForwardNetwork(
                layers_units=[filter_size, hidden_size],
                layers_activation=[nonlinearity, None],
                layers_dropout_rates=[0.0, 0.0],
                use_bias=use_ffn_bias,
                kernel_initializer=ffn_initializer,
                output_layer_initializer=output_layer_initializer,
                kernel_regularizer=weight_regularizer,
                bias_regularizer=weight_regularizer,
                boundary_casting=self.boundary_casting,
                tf_summary=self.tf_summary,
                dtype=self.dtype_policy,
                name="ffn",
            )

            self.layers.append(
                [
                    PrePostProcessWrapper(
                        self_attention_layer,
                        pre_process_config,
                        post_process_config,
                        dtype=self.dtype_policy,
                    ),
                    PrePostProcessWrapper(
                        cross_attention_layer,
                        pre_process_config,
                        post_process_config,
                        dtype=self.dtype_policy,
                    ),
                    PrePostProcessWrapper(
                        feed_forward_network,
                        pre_process_config,
                        post_process_config,
                        dtype=self.dtype_policy,
                    ),
                ]
            )

    def call(
        self,
        inputs,
        self_attention_mask=None,
        sparse_attention_mask=None,
        encoder_output=None,
        cross_attention_mask=None,
        past_keys_values=None,
        cache_present_keys_values=False,
        training=True,
    ):
        """
        :param Tensor inputs: Decoder input.
        :param Tensor self_attention_mask: Self attention mask.
        :param Tensor inputs: Encoder output. Used to
            generate keys and values for cross-attention.
            If None, no cross-attention is performed.
        :param Tensor cross_attention_mask: Padding mask for cross attention.
        :param Tensor past_keys_values: Tensor of past keys and values of shape
            ``[num_hidden_layers, 2, batch_size, num_heads, None, hidden_size//num_heads]``,
            where that [i,0,:,:,:,:] and [i,1,:,:,:,:] contain past keys and values
            for layer i, respectively. The shape is partially defined because
            past_keys_values sequence length can increase within autoregressive
            loop. This information allows speeding up decoder
            computations within autoregressive loop, e.g., when decoding
            tokens one-by-one at prediction. Defaults to None.
        :param bool cache_present_keys_values: Specifies if present keys and values
            must be cached and returned. Needed to speed up computations
            when the decoder is called within an autoregressive loop.
            Defaults to False.
        :param bool training: Whether run in training mode.

        :returns: Returns a tuple, where the 0th entry contains decoder
            output and the 1st entry contains a tensor of keys and values
            computed and stored at the current application of the decoder.
            The shape of the tensor is
            ``[num_hidden_layers, 2, batch_size, num_heads, inputs.shape[1], hidden_size//num_heads]``,
            If cache_present_keys_values is False, returns ``(decoder_output, None)``.
        """

        output = inputs
        present_keys_values = []

        if past_keys_values is not None:
            past_keys_values = tf.unstack(past_keys_values)
            assert len(past_keys_values) == len(
                self.layers
            ), "Length of past_keys_values must equal the number of decoder blocks"

        for i, layer in enumerate(self.layers):
            self_attention_layer = layer[0]
            output = self_attention_layer(
                output,
                mask=sparse_attention_mask
                if sparse_attention_mask is not None and i % 2 != 0
                else self_attention_mask,
                past_kv=past_keys_values[i] if past_keys_values else None,
                cache_present_kv=cache_present_keys_values,
                training=training,
            )

            if cache_present_keys_values:
                present_keys_values.append(output[1])
                output = output[0]

            if encoder_output is not None:
                cross_attention_layer = layer[1]
                output = cross_attention_layer(
                    output,
                    v=encoder_output,
                    mask=cross_attention_mask,
                    training=training,
                )

            feed_forward_network = layer[2]
            output = feed_forward_network(output, training=training)

        return (
            output,
            tf.stack(present_keys_values)
            if cache_present_keys_values
            else None,
        )
