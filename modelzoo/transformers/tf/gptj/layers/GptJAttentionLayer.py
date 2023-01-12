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
from modelzoo.common.tf.layers.DenseLayer import DenseLayer
from modelzoo.common.tf.layers.DropoutLayer import DropoutLayer
from modelzoo.transformers.tf.gptj.layers.attention_utils import (
    apply_rotary_position_embedding,
    create_causal_bias,
    fixed_position_embeddings,
)


class GptJAttentionLayer(BaseLayer):
    """Multi-head attention layer with rotatory position embeddings. Based on
    `<https://github.com/huggingface/transformers/blob/master/src/transformers/models/gptj/modeling_gptj.py>`_

    Args:
        hidden_size (int): Number of units in each projection output.
        num_heads (int): Number of attention heads.
        max_position_embeddings (int): Maximum number of position embeddings for
            model training.
        output_projection_size (int)": Number of units for output projection
        use_projection_bias (bool): Whether to use bias in the key, query, and
            value projections.
        use_ffn_bias (bool): Whether to use bias in the output projection.
        rotary_dim (int): Number of positions to apply rotations on
        initializer (str): Projection kernel intializer. Defaults to
            ``variance_scaling``.
        output_layer_initializer (str or initializer): If not None, use this
            initializer for the output transform layer. Defaults to None.
        kernel_regularizer (Optional[Callable]): Projection kernel regularizer.
            Defaults to ``None``.
        bias_regularizer (Optional[Callable]): Projection bias regularizer.
            Defaults to ``None``.
        attn_dropout_rate (float): Dropout rate for key-query weights. Defaults to
            0.0.
        residual_dropout_rate (float): Dropout rate for output weights. Defaults to
            0.0.
        dropout_seed (int): Seed with which to initialize the dropout layer.
            Defaults to ``None``.
        softmax_dtype_fp32 (bool): If ``True``, cast query-key logits to FP32
            before sending into softmax calculation in FP32.
        boundary_casting (bool): If ``True``, then outputs the values in half
            precision and casts the  input values up to full precision.
        tf_summary (bool): If ``True``, then saves the activations with
            ``summary_layer``.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        max_position_embeddings,
        output_projection_size=None,
        use_projection_bias=False,
        use_ffn_bias=False,
        rotary_dim=None,
        initializer="variance_scaling",
        output_layer_initializer=None,
        kernel_regularizer=None,
        bias_regularizer=None,
        attn_dropout_rate=0.0,
        residual_dropout_rate=0.0,
        dropout_seed=None,
        softmax_dtype_fp32=True,
        boundary_casting=False,
        tf_summary=False,
        **kwargs,
    ):
        assert (
            hidden_size % num_heads == 0
        ), "hidden_size must be divisible by num_heads."

        super(GptJAttentionLayer, self).__init__(
            boundary_casting, tf_summary, **kwargs
        )

        self.hidden_size = hidden_size
        self.output_projection_size = (
            output_projection_size
            if output_projection_size is not None
            else hidden_size
        )
        self.num_heads = num_heads

        self.masked_bias = tf.constant(-1e9, dtype=self.variable_dtype)
        self.bias = create_causal_bias(
            max_position_embeddings, dtype=self.variable_dtype
        )

        proj_params = {
            "units": hidden_size,
            "use_bias": use_projection_bias,
            "kernel_initializer": initializer,
            "kernel_regularizer": kernel_regularizer,
            "bias_regularizer": bias_regularizer,
            "boundary_casting": boundary_casting,
            "tf_summary": tf_summary,
            "dtype": self.dtype_policy,
        }

        self.q_proj = DenseLayer(**proj_params, name="q_proj")
        self.k_proj = DenseLayer(**proj_params, name="k_proj")
        self.v_proj = DenseLayer(**proj_params, name="v_proj")

        if attn_dropout_rate > 0.0:
            self.attn_dropout = DropoutLayer(
                rate=attn_dropout_rate,
                seed=dropout_seed,
                boundary_casting=boundary_casting,
                tf_summary=tf_summary,
                dtype=self.dtype_policy,
                name="attn_dropout",
            )
        else:
            self.attn_dropout = None

        if residual_dropout_rate > 0.0:
            self.resid_dropout = DropoutLayer(
                rate=residual_dropout_rate,
                seed=dropout_seed,
                boundary_casting=boundary_casting,
                tf_summary=tf_summary,
                dtype=self.dtype_policy,
                name="resid_dropout",
            )
        else:
            self.resid_dropout = None

        output_initializer = initializer
        if output_layer_initializer is not None:
            output_initializer = output_layer_initializer

        self.out_proj = DenseLayer(
            self.output_projection_size,
            use_bias=use_ffn_bias,
            kernel_initializer=output_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.dtype_policy,
            name="out_proj",
        )

        self.rotary_dim = rotary_dim
        self.softmax_dtype_fp32 = softmax_dtype_fp32

    def call(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        use_cache=False,
        training=True,
        **kwargs,
    ):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, rotary=True)
        key = self._split_heads(key, rotary=True)
        value = self._split_heads(value, rotary=False)

        seq_len = tf.shape(key)[1]
        offset = 0

        if layer_past is not None:
            k_past, _ = tf.unstack(layer_past)
            offset = tf.shape(input=k_past)[-2]
            seq_len += offset

        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            sincos = fixed_position_embeddings(k_rot, 1, seq_len=seq_len)
            k_rot = apply_rotary_position_embedding(
                k_rot, sincos, offset=offset
            )
            q_rot = apply_rotary_position_embedding(
                q_rot, sincos, offset=offset
            )

            key = tf.concat([k_rot, k_pass], axis=-1)
            query = tf.concat([q_rot, q_pass], axis=-1)
        else:
            sincos = fixed_position_embeddings(key, 1, seq_len=seq_len)
            key = apply_rotary_position_embedding(key, sincos, offset=offset)
            query = apply_rotary_position_embedding(
                query, sincos, offset=offset
            )

        key = tf.transpose(a=key, perm=[0, 2, 1, 3])
        query = tf.transpose(a=query, perm=[0, 2, 1, 3])

        present = None
        if use_cache:
            present = tf.stack([key, value])

        if layer_past is not None:
            k_past, v_past = tf.unstack(layer_past)
            key = tf.concat([k_past, key], axis=-2)
            value = tf.concat([v_past, value], axis=-2)

        attn_output = self._attn(
            query, key, value, attention_mask, layer_past, training=training,
        )

        attn_output = self._combine_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        if self.resid_dropout is not None:
            attn_output = self.resid_dropout(attn_output, training=training)

        return (attn_output, present if use_cache else None)

    def _attn(
        self, query, key, value, mask=None, layer_past=None, training=True,
    ):
        query_length, key_length = (
            tf.shape(input=query)[-2],
            tf.shape(input=key)[-2],
        )
        causal_mask = tf.cast(
            self.bias[
                :, :, key_length - query_length : key_length, :key_length
            ],
            dtype=tf.bool,
        )

        depth = self.hidden_size // self.num_heads
        query *= depth ** -0.5

        attn_weights = tf.matmul(query, key, transpose_b=True)
        attn_weights = tf.where(
            causal_mask,
            attn_weights,
            tf.cast(self.masked_bias, dtype=attn_weights.dtype),
        )

        # Apply attention mask.
        if mask is not None:
            # this attention mask is more simple than the triangular masking
            # of causal attention used in OpenAI GPT, we just need to
            # prepare the broadcast dimension here.
            neg_inf = -1e4
            assert len(mask.shape) in [
                2,
                3,
            ], "Only 2D and 3D masks are supported"

            if len(mask.shape) == 2:
                if layer_past is not None:
                    # Augment mask to attend to all past keys
                    # Used for decoding during inference
                    past_mask = tf.zeros(
                        [tf.shape(query)[0], tf.shape(layer_past)[-2]],
                        dtype=mask.dtype,
                    )
                    mask = tf.concat([past_mask, mask], axis=-1)

                batch_size, seq_length, query_length = (
                    tf.shape(mask)[0],
                    tf.shape(mask)[1],
                    1,
                )
            else:
                if layer_past is not None:
                    # Augment mask to attend to all past keys
                    # Used for decoding during inference
                    past_mask = tf.zeros(
                        [
                            tf.shape(query)[0],
                            tf.shape(query)[-2],
                            tf.shape(layer_past)[-2],
                        ],
                        dtype=mask.dtype,
                    )
                    mask = tf.concat([past_mask, mask], axis=-1)

                batch_size, query_length, seq_length = (
                    tf.shape(mask)[0],
                    tf.shape(mask)[1],
                    tf.shape(mask)[2],
                )

            # Compute the attention_bias based on the mask.
            # shape: (batch_size, 1, 1, seq_length).
            attention_bias = (
                tf.reshape(
                    tf.cast(mask, attn_weights.dtype),
                    shape=[batch_size, 1, query_length, seq_length],
                )
                * neg_inf
            )
            attn_weights += attention_bias

        if self.compute_dtype != 'float32' and self.softmax_dtype_fp32:
            attn_weights = tf.cast(attn_weights, tf.float32)
        attn_weights = tf.cast(tf.nn.softmax(attn_weights), self.compute_dtype)
        if self.attn_dropout is not None:
            attn_weights = self.attn_dropout(attn_weights, training=training)

        attn_output = tf.matmul(attn_weights, value)
        return attn_output

    def _split_heads(self, x, rotary):
        """Split x into different heads, and transpose the resulting value. The
        tensor is transposed to insure the inner dimensions hold the correct
        values during the matrix multiplication.

        Args:
            x: A tensor with shape ``[batch_size, seq_length, hidden_size]``

        Returns:
            If rotary is true, a tensor with shape
            ``[batch_size, seq_length, num_heads, hidden_size/num_heads]``
            else, a tensor with shape
            ``[batch_size, num_heads, seq_length, hidden_size/num_heads]``

        """

        assert (
            len(x.shape) == 3
        ), f"Input tensor rank should be one of 3, but is: {len(x.shape)}"

        # Calculate depth of last dimension after it has been split.
        depth = self.hidden_size // self.num_heads
        batch_size = tf.shape(input=x)[0]
        seq_length = tf.shape(input=x)[1]
        x = tf.reshape(x, [batch_size, seq_length, self.num_heads, depth])

        # Transpose the result if not rotary
        if rotary:
            return x
        return tf.transpose(a=x, perm=[0, 2, 1, 3])

    def _combine_heads(self, x):
        """Combine tensor that has been split.

        Args:
            x: A tensor ``[batch_size, num_heads, seq_length, idden_size/num_heads]``

        Returns:
            A tensor with shape ``[batch_size, seq_length, hidden_size]``
        """

        assert (
            len(x.shape) == 4
        ), f"Input tensor rank should be one of 4, but is: {len(x.shape)}"

        # Transpose and reshape tensors
        x = tf.transpose(a=x, perm=[0, 2, 1, 3])
        batch_size = tf.shape(input=x)[0]
        seq_length = tf.shape(input=x)[1]
        return tf.reshape(x, [batch_size, seq_length, self.hidden_size])
