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

from modelzoo.common.tf.layers.BaseLayer import BaseLayer
from modelzoo.transformers.tf.gptj.layers.GptJBlock import GptJBlock


class GptJDecoder(BaseLayer):
    def __init__(
        self,
        hidden_size,
        num_heads,
        filter_size,
        num_hidden_layers,
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
        super(GptJDecoder, self).__init__(
            boundary_casting, tf_summary, **kwargs
        )

        layers = []
        for i in range(num_hidden_layers):
            layers.append(
                GptJBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    filter_size=filter_size,
                    max_position_embeddings=max_position_embeddings,
                    use_untied_layer_norm=use_untied_layer_norm,
                    use_projection_bias_in_attention=use_projection_bias_in_attention,
                    use_ffn_bias_in_attention=use_ffn_bias_in_attention,
                    use_ffn_bias=use_ffn_bias,
                    rotary_dim=rotary_dim,
                    attention_initializer=attention_initializer,
                    ffn_initializer=ffn_initializer,
                    output_initializer=output_initializer,
                    attention_dropout_rate=attention_dropout_rate,
                    attention_residual_dropout_rate=attention_residual_dropout_rate,
                    ffn_residual_dropout_rate=ffn_residual_dropout_rate,
                    dropout_seed=dropout_seed,
                    nonlinearity=nonlinearity,
                    layer_norm_epsilon=layer_norm_epsilon,
                    attention_softmax_fp32=attention_softmax_fp32,
                    boundary_casting=boundary_casting,
                    tf_summary=tf_summary,
                    dtype=self.dtype_policy,
                    name=f"{str(i)}",
                )
            )
        self.layers = layers

    def call(
        self,
        inputs,
        attention_mask=None,
        past_keys_values=None,
        use_cache=False,
        training=True,
        **kwargs,
    ):
        present_key_values = [] if use_cache else None

        if past_keys_values is not None:
            past_keys_values = tf.unstack(past_keys_values)
            assert len(past_keys_values) == len(
                self.layers
            ), "Length of past_keys_values must equal the number of decoder blocks"

        hidden_states = inputs

        for i, block in enumerate(self.layers):
            hidden_states, present = block(
                hidden_states,
                layer_past=past_keys_values[i] if past_keys_values else None,
                attention_mask=attention_mask,
                use_cache=use_cache,
                training=training,
                **kwargs,
            )

            if use_cache:
                present_key_values.append(present)

        return (
            hidden_states,
            tf.stack(present_key_values) if use_cache else None,
        )
