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

from modelzoo.common.tf.layers.BaseLayer import BaseLayer
from modelzoo.common.tf.layers.FeedForwardNetwork import FeedForwardNetwork
from modelzoo.common.tf.layers.LayerNormalizationLayer import (
    LayerNormalizationLayer,
)
from modelzoo.transformers.tf.gptj.layers.GptJAttentionLayer import (
    GptJAttentionLayer,
)


class GptJBlock(BaseLayer):
    def __init__(
        self,
        hidden_size,
        num_heads,
        filter_size,
        max_position_embeddings,
        use_untied_layer_norm=False,
        use_projection_bias_in_attention=False,
        use_ffn_bias_in_attention=False,
        use_ffn_bias=True,
        rotary_dim=None,
        attention_initializer="variance_scaling",
        ffn_initializer="variance_scaling",
        output_initializer=None,
        attention_dropout_rate=0.0,
        attention_residual_dropout_rate=0.0,
        ffn_residual_dropout_rate=0.0,
        dropout_seed=None,
        nonlinearity="gelu",
        layer_norm_epsilon=1e-8,
        attention_softmax_fp32=True,
        boundary_casting=False,
        tf_summary=False,
        **kwargs,
    ):
        super(GptJBlock, self).__init__(boundary_casting, tf_summary, **kwargs)

        self.ln_1 = LayerNormalizationLayer(
            epsilon=layer_norm_epsilon,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.dtype_policy,
            name="ln_1",
        )
        if use_untied_layer_norm:
            self.ln_2 = LayerNormalizationLayer(
                epsilon=layer_norm_epsilon,
                boundary_casting=boundary_casting,
                tf_summary=tf_summary,
                dtype=self.dtype_policy,
                name="ln_2",
            )
        else:
            self.ln_2 = None

        self.attn = GptJAttentionLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            output_projection_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            use_projection_bias=use_projection_bias_in_attention,
            use_ffn_bias=use_ffn_bias_in_attention,
            rotary_dim=rotary_dim,
            initializer=attention_initializer,
            output_layer_initializer=output_initializer,
            attn_dropout_rate=attention_dropout_rate,
            residual_dropout_rate=attention_residual_dropout_rate,
            dropout_seed=dropout_seed,
            softmax_dtype_fp32=attention_softmax_fp32,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.dtype_policy,
            name="attn",
        )
        self.mlp = FeedForwardNetwork(
            layers_units=[filter_size, hidden_size],
            layers_activation=[nonlinearity, None],
            layers_dropout_rates=[0.0, ffn_residual_dropout_rate],
            use_bias=use_ffn_bias,
            kernel_initializer=ffn_initializer,
            output_layer_initializer=output_initializer,
            dropout_seed=dropout_seed,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.dtype_policy,
            name="mlp",
        )

    def call(
        self,
        inputs,
        layer_past=None,
        attention_mask=None,
        use_cache=False,
        training=True,
        **kwargs,
    ):
        residual = inputs
        hidden_states = self.ln_1(inputs)
        attention_outputs, present = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            use_cache=use_cache,
            training=training,
        )
        # used for Gpt-NeoX
        if self.ln_2 is not None:
            hidden_states = self.ln_2(inputs)
        feed_forward_outputs = self.mlp(hidden_states, training=training)

        output = residual + attention_outputs + feed_forward_outputs
        return (output, present if use_cache else None)
