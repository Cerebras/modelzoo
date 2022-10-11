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

from modelzoo.common.tf.layers.AddLayer import AddLayer
from modelzoo.common.tf.layers.AttentionLayer import SelfAttentionLayer
from modelzoo.common.tf.layers.BaseLayer import BaseLayer
from modelzoo.common.tf.layers.DropoutLayer import DropoutLayer
from modelzoo.common.tf.layers.FeedForwardNetworkV2 import FeedForwardNetworkV2
from modelzoo.common.tf.layers.LayerNormalizationLayer import (
    LayerNormalizationLayer,
)
from modelzoo.common.tf.layers.PrePostProcessWrapper import (
    PrePostProcessWrapper,
)


class T5Encoder(BaseLayer):
    """
    T5 model encoder: https://arxiv.org/pdf/1910.10683.pdf.
    Made up of num_hidden_layers identical layers.
    Each layer is composed of the sublayers.
        1. Self-attention layer.
        3. Feedforward network (which is 2 fully-connected layers).
        4. LayerNorm layer.
        5. Dropout layer.
    """

    def __init__(
        self,
        d_kv,
        d_model,
        num_heads,
        num_hidden_layers,
        d_ff,
        use_projection_bias_in_attention=False,
        use_ffn_bias_in_attention=False,
        use_ffn_bias=False,
        query_layer_initializer=None,
        key_layer_initializer=None,
        value_layer_initializer=None,
        relative_attention_bias_weight_initializer=None,
        output_layer_initializer=None,
        feed_forward_input_layer_initializer=None,
        feed_forward_output_layer_initializer=None,
        weight_regularizer=None,
        attention_type="scaled_dot_product",
        nonlinearity="relu",
        dropout_rate=0.1,
        dropout_seed=None,
        use_relative_attention_bias=True,
        num_relative_attention_buckets=32,
        layer_norm_epsilon=1e-6,
        use_pre_normalization=True,
        boundary_casting=False,
        tf_summary=False,
        **kwargs,
    ):
        """
        Initialize the encoder object instance.

        :param int d_kv: Size of the key, query, value projections per attention head.
        :param int d_model: Size of the encoder layers and the pooler layer.
        :param int num_heads: d_kv * num_heads = hidden size.
        :param int num_hidden_layers: Number of encoders in the encoder stack.
        :param int d_ff: Size of the intermediate feed forward layer in t5 blocks.
        :param bool use_projection_bias_in_attention: Add bias to Q,K,V projections
            in the Attention layer. Defaults to False.
        :param bool use_ffn_bias_in_attention: Add bias in the concluding FFN
            in the Attention layer. Defaults to False.
        :param bool use_ffn_bias: Add bias in all dense layers of the encoder's ffn sublayer.
        :param initializer query_layer_initializer: Query kernel initializer.
        :param initializer key_layer_initializer: Key kernel initializer.
        :param initializer value_layer_initializer: Value kernel initializer.
        :param initializer relative_attention_bias_weight_initializer: Relative Attention Bias weight.
        :param initializer output_layer_initializer: Initializer for the output transform layer.
        :param initializer feed_forward_input_layer_initializer: Initializer for the input FFN layer.
        :param initializer feed_forward_output_layer_initializer: Initializer for the output FFN layer.
        :param callable weight_regularizer: Weights regularizer.
        :param string attention_type: Type of attention.
            Currently support {"dot_product","scaled_dot_product"}.
            Defaults to "scaled_dot_product".
        :param string nonlinearity: Type of nonlinearity applied after the first dense layer
            of the FFN sublayer. Defaults to gelu.
        :param float dropout_rate: Dropout rate. Same for all dense layers
            encountered in the Encoder. Defaults to 0 (i.e., no dropout).
        :param int dropout_seed: seed for initializing dropout layers.
        :param use_relative_attention_bias: Whether to use relative position bias
            when calculating attention.
        :param num_relative_attention_buckets : Used to calculate relative position
            bias when use_relative_attention_bias set to True.
        :param float layer_norm_epsilon: LayerNorm epsilon. Same for all layers
            encountered in the Encoder. Defaults to 1e-8.
        :param bool use_pre_normalization: If False, uses original transformer encoder:
            "residual split -> transform -> dropout -> residual add -> layer norm".
            Otherwise, use residual block with pre-normaliztion:
            "residual split -> layer norm -> transform -> dropout -> residual add".
            Here transform is either a self-attention or ffn sub-block.
            Defaults to False.
        """
        super(T5Encoder, self).__init__(boundary_casting, tf_summary, **kwargs)

        layer_norm_config = {
            "epsilon": layer_norm_epsilon,
            "beta_regularizer": weight_regularizer,
            "gamma_regularizer": weight_regularizer,
            "boundary_casting": boundary_casting,
            "tf_summary": tf_summary,
            "dtype": self.dtype_policy,
        }

        # Set up pre- and post-processing steps to be wrapped
        # around all encoder sublayers in the encoder stack.
        pre_process_config = []
        if use_pre_normalization:
            pre_process_config.append(
                (LayerNormalizationLayer, layer_norm_config)
            )

        post_process_config = [
            (
                DropoutLayer,
                {
                    "rate": dropout_rate,
                    "seed": dropout_seed,
                    "boundary_casting": boundary_casting,
                    "tf_summary": tf_summary,
                    "dtype": self.dtype_policy,
                },
            ),
            (
                AddLayer,
                {
                    "boundary_casting": boundary_casting,
                    "tf_summary": tf_summary,
                    "dtype": self.dtype_policy,
                },
            ),
        ]

        # Shared position embedding parameters.
        relative_attention_bias = (
            self.add_weight(
                name="encoder/relative_attention_bias",
                shape=[num_relative_attention_buckets, num_heads],
                dtype=self.variable_dtype,
            )
            if use_relative_attention_bias
            else None
        )

        # Create sublayers for each layer.
        self.layers = []
        for _ in range(num_hidden_layers):
            self_attention_layer = SelfAttentionLayer(
                d_kv * num_heads,  # hidden_size
                num_heads,
                output_projection_size=d_model,
                use_projection_bias=use_projection_bias_in_attention,
                use_ffn_bias=use_ffn_bias_in_attention,
                query_layer_initializer=query_layer_initializer,
                key_layer_initializer=key_layer_initializer,
                value_layer_initializer=value_layer_initializer,
                relative_attention_bias_weight_initializer=relative_attention_bias_weight_initializer,
                output_layer_initializer=output_layer_initializer,
                kernel_regularizer=weight_regularizer,
                bias_regularizer=weight_regularizer,
                attention_type=attention_type,
                dropout_rate=dropout_rate,
                dropout_seed=dropout_seed,
                use_relative_attention_bias=use_relative_attention_bias,
                relative_attention_bias=relative_attention_bias,
                num_relative_attention_buckets=num_relative_attention_buckets,
                bidirectional_relative_attention=True,
                boundary_casting=boundary_casting,
                tf_summary=tf_summary,
                dtype=self.dtype_policy,
                name="self_attention",
            )

            feed_forward_network = FeedForwardNetworkV2(
                d_ff,
                d_model,
                activation=nonlinearity,
                dropout_rate=dropout_rate,
                use_bias=use_ffn_bias,
                input_layer_initializer=feed_forward_input_layer_initializer,
                output_layer_initializer=feed_forward_output_layer_initializer,
                dropout_seed=dropout_seed,
                boundary_casting=boundary_casting,
                tf_summary=tf_summary,
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
                        feed_forward_network,
                        pre_process_config,
                        post_process_config,
                        dtype=self.dtype_policy,
                    ),
                ]
            )

        self.layer_norm_layer = LayerNormalizationLayer(**layer_norm_config)
        self.dropout_layer = DropoutLayer(
            rate=dropout_rate,
            seed=dropout_seed,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.dtype_policy,
        )

    def call(
        self,
        inputs,
        self_attention_mask=None,
        is_training=True,
        cache_position_bias=False,
    ):
        """
         :param Tensor inputs: Encoder input.
         :param Tensor self_attention_mask: Self attention mask.
         :param bool is_training: Whether run in training mode.
        :param bool cache_position_bias: Specifies if position bias
            must be cached and returned. Needed to speed up computations
            when the decoder is called within an autoregressive loop.
            Defaults to False.

         :returns: Returns encoder outputs.
        """
        outputs = inputs
        position_bias = None

        for i, layer in enumerate(self.layers):
            self_attention_layer = layer[0]
            self_attention_outputs = self_attention_layer(
                outputs,
                mask=self_attention_mask,
                training=is_training,
                position_bias=position_bias,
                cache_position_bias=cache_position_bias,
            )

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, (past_keys_values), (position bias).
            # The last two items in the tuple are optional and present if
            # cache_present_kv set to True.
            outputs = self_attention_outputs

            if cache_position_bias:
                outputs = self_attention_outputs[0]
                position_bias = self_attention_outputs[-1]

            feed_forward_network = layer[1]
            outputs = feed_forward_network(outputs, training=is_training)

        outputs = self.layer_norm_layer(outputs)
        outputs = self.dropout_layer(outputs, training=is_training)

        return outputs
