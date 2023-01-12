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

import tensorflow as tf

from modelzoo.common.tf.layers.BaseLayer import BaseLayer
from modelzoo.common.tf.layers.DenseLayer import DenseLayer
from modelzoo.common.tf.layers.DropoutLayer import DropoutLayer


class AttentionLayer(BaseLayer):
    """Multi-head attention layer. Based on `MLCommons model\
    <https://github.com/mlperf/training/blob/master/translation/tensorflow\
    /transformer/model/attention_layer.py>`_.

    Args:
        hidden_size (int): Number of units in each projection output.
        num_heads (int): Number of attention heads.
        use_projection_bias (bool): Whether to use bias in the key, query, and
            value projections.
        use_ffn_bias (bool): Whether to use bias in the output projection.
        initializer (str): Projection kernel intializer. Defaults to
            ``glorot_uniform``.
        query_layer_initializer (initializer): Query kernel initializer. Defaults to
             None in which case ``initializer`` will be used.
        key_layer_initializer (initializer): Key kernel initializer. Defaults to
             `None in which case ``initializer`` will be used.
        value_layer_initializer (initializer): Value kernel initializer. Defaults to
             None in which case ``initializer`` will be used.
        relative_attention_bias_weight_initializer (initializer): Relative Attention Bias weight
              None in which case ``initializer`` will be used.
        output_layer_initializer (str or initializer): If not None, use this
            initializer for the output transform layer. Defaults to None.
        kernel_regularizer (Optional[Callable]): Projection kernel regularizer.
            Defaults to ``None``.
        bias_regularizer (Optional[Callable]): Projection bias regularizer.
            Defaults to ``None``.
        attention_type (str): The attention variant to execute. Currently
            accepts ``dot_product`` and ``scaled_dot_product``. Defaults to
            ``scaled_dot_product``.
        dropout_rate (float): Dropout rate for key-query weights. Defaults to
            0.0.
        dropout_seed (int): Seed with which to initialize the dropout layer.
            Defaults to ``None``.
        use_relative_attention_bias (bool): Whether to use relative position bias
            when calculating attention.
        relative_attention_bias (Tensor): Tensor with relative attention weights.
            Shape: [`num_relative_attention_buckets`, `num_heads`]. Defaults set to None.
        num_relative_attention_buckets (int): Used to calculate relative position
            bias when use_relative_attention_bias set to True.
        bidirectional_relative_attention (bool): Whether attention is bidirectional.
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
        output_projection_size=None,
        use_projection_bias=False,
        use_ffn_bias=False,
        initializer="glorot_uniform",
        query_layer_initializer=None,
        key_layer_initializer=None,
        value_layer_initializer=None,
        relative_attention_bias_weight_initializer=None,
        output_layer_initializer=None,
        kernel_regularizer=None,
        bias_regularizer=None,
        attention_type="scaled_dot_product",
        dropout_rate=0.0,
        dropout_seed=None,
        use_relative_attention_bias=False,
        relative_attention_bias=None,
        num_relative_attention_buckets=32,
        bidirectional_relative_attention=False,
        softmax_dtype_fp32=True,
        boundary_casting=False,
        tf_summary=False,
        **kwargs,
    ):

        _SUPPORTED_ATTENTION_TYPES = ["dot_product", "scaled_dot_product"]
        assert (
            attention_type in _SUPPORTED_ATTENTION_TYPES
        ), f"Attention type {attention_type} is not supported."
        assert (
            hidden_size % num_heads == 0
        ), "hidden_size must be divisible by num_heads."

        super(AttentionLayer, self).__init__(
            boundary_casting, tf_summary, **kwargs
        )

        self.hidden_size = hidden_size
        self.output_projection_size = (
            output_projection_size
            if output_projection_size is not None
            else hidden_size
        )

        self.num_heads = num_heads
        self.scale_dot_product = (
            True if attention_type == "scaled_dot_product" else False
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

        proj_params["kernel_initializer"] = (
            query_layer_initializer or initializer
        )
        self.q_dense_layer = DenseLayer(**proj_params, name="q_projection")

        proj_params["kernel_initializer"] = key_layer_initializer or initializer
        self.k_dense_layer = DenseLayer(**proj_params, name="k_projection")

        proj_params["kernel_initializer"] = (
            value_layer_initializer or initializer
        )
        self.v_dense_layer = DenseLayer(**proj_params, name="v_projection")

        self.dropout_layer = DropoutLayer(
            dropout_rate,
            seed=dropout_seed,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.dtype_policy,
            name="dropout_layer",
        )

        output_initializer = initializer
        if output_layer_initializer is not None:
            output_initializer = output_layer_initializer
        self.output_dense_layer = DenseLayer(
            self.output_projection_size,
            use_bias=use_ffn_bias,
            kernel_initializer=output_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.dtype_policy,
            name="output_transform",
        )

        self.num_heads = num_heads
        self.initializer = initializer
        self.use_relative_attention_bias = use_relative_attention_bias
        self.relative_attention_bias = relative_attention_bias
        self.num_relative_attention_buckets = num_relative_attention_buckets
        self.bidirectional_relative_attention = bidirectional_relative_attention
        self.relative_attention_bias_weight_initializer = (
            relative_attention_bias_weight_initializer
        )
        self.softmax_dtype_fp32 = softmax_dtype_fp32

    def build(self, input_shape):
        # If no relative attention bias weights are provided when
        # `use_relative_attention_bias` set to True, we are creating them.
        if (
            self.use_relative_attention_bias
            and self.relative_attention_bias is None
        ):
            self.relative_attention_bias = self.add_weight(
                name="relative_attention_bias",
                shape=[self.num_relative_attention_buckets, self.num_heads],
                dtype=self.variable_dtype,
                experimental_autocast=False,
                initializer=self.relative_attention_bias_weight_initializer
                if self.relative_attention_bias_weight_initializer is not None
                else self.initializer,
                trainable=True,
            )

        self.built = True

    def call(
        self,
        q,
        v,
        mask=None,
        past_kv=None,
        cache_present_kv=False,
        training=True,
        position_bias=None,
        cache_position_bias=False,
    ):
        """Applies the attention mechanism to queries ``q`` and values ``v``.
        Keys will be set to be same as ``v``.

        Args:
            q (Tensor): Queries, shape ``[batch_size, seq_length, hidden_size]``.
            v (Tensor): Values, shape ``[batch_size, seq_length, hidden_size]``.
            mask (Tensor): Attention mask. Can be 2D of shape
                ``[batch_size, seq_length]``, or 3D of shape
                ``[batch, query_length, seq_length]``.
            past_kv (Tensor): Past keys and values. Has shape
                ``[2, batch_size, num_heads, seq_length, hidden_size / num_heads]``.
                The tensors in ``[0,:,:,:,:]`` and ``[1,:,:,:,:]`` contain the
                past keys and values, respectively. Defaults to ``None``.
            cache_present_kv (bool): Specifies if the present keys and values
                must be cached and returned. Needed to speed up the
                computations when the decoder is called within an
                autoregressive loop. Defaults to ``False``.
            training (bool): Training the model if ``True``. Needed to call the
                ``dropout`` (after softmax) in the appropriate mode.
            position_bias (Tensor): Tensor containing position bias to apply in attention.
            cache_position_bias (bool): Specifies if position bias
                must be cached and returned. Needed to speed up the
                computations when the decoder is called within an
                autoregressive loop. Defaults to ``False``.

        Returns:
            when ``cache_present_kv`` is ``True`` and ``cache_position_bias`` is
            ``True``, returns a tuple, where the 0th entry contains the attention output,
            1st entry contains a tensor of keys and values computed at the current
            application of the attention layer, and the 3rd entry contains a tensor
            of position bias computed at the current application of the attention layer.

            If ``cache_present_kv`` is ``False``, no entry for present keys and values
            is provided.

            If ``cache_position_bias`` is ``False``, no entry for position bias
            is provided.

            if both ``cache_present_kv`` ``cache_position_bias`` are set to False,
            return a tensor of shape equal to shape of ``past_kv`` (see above).
        """
        batch_size, seq_length = tf.shape(q)[0], tf.shape(q)[1]
        # `real_seq_length` equal to `seq_length` if `present_kv` is None, otherwise
        # it is equal to 2 * `seq_length`.
        real_seq_length = seq_length

        # Linearly project the query (q), key (k) and value (v) using different
        # learned projections.
        q = self.q_dense_layer(q)
        k = self.k_dense_layer(v)
        v = self.v_dense_layer(v)

        # Split q, k, v into heads.
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        present_kv = None
        if cache_present_kv:
            present_kv = tf.stack([k, v])

        if past_kv is not None:
            k_past, v_past = tf.unstack(past_kv)
            k = tf.concat([k_past, k], axis=-2)
            v = tf.concat([v_past, v], axis=-2)
            real_seq_length += seq_length

        key_length = real_seq_length if present_kv is None else seq_length

        if self.scale_dot_product:
            depth = self.hidden_size // self.num_heads
            q *= depth ** -0.5

        # Calculate dot product attention.
        logits = tf.matmul(q, k, transpose_b=True)

        # Apply attention mask.
        if mask is not None:
            neg_inf = -1e4
            assert len(mask.shape) in [2, 3, 4], (
                "Support 2D, 3D and 4D masks for now. 4D masks can be passed"
                + " in when we are defining unique masks per head."
            )

            num_heads = 1
            if len(mask.shape) == 2:
                if past_kv is not None:
                    # Augment mask to attend
                    # to all past keys.
                    past_mask = tf.zeros(
                        [tf.shape(q)[0], tf.shape(past_kv)[-2]],
                        dtype=mask.dtype,
                    )
                    mask = tf.concat([past_mask, mask], axis=-1)

                batch_size, seq_length = tf.shape(mask)[0], tf.shape(mask)[1]
                query_length = 1
            elif len(mask.shape) == 3:
                if past_kv is not None:
                    # Augment mask to attend
                    # to all past keys.
                    past_mask = tf.zeros(
                        [
                            tf.shape(q)[0],
                            tf.shape(q)[-2],
                            tf.shape(past_kv)[-2],
                        ],
                        dtype=mask.dtype,
                    )
                    mask = tf.concat([past_mask, mask], axis=-1)

                batch_size, query_length, seq_length = (
                    tf.shape(mask)[0],
                    tf.shape(mask)[1],
                    tf.shape(mask)[2],
                )
            else:
                batch_size, num_heads, query_length, seq_length = (
                    tf.shape(mask)[0],
                    tf.shape(mask)[1],
                    tf.shape(mask)[2],
                    tf.shape(mask)[3],
                )

            # Compute the attention_bias based on the mask.
            # Most frequently the shape is [batch_size, 1, 1, seq_length]
            attention_bias = (
                tf.reshape(
                    tf.cast(mask, logits.dtype),
                    shape=[batch_size, num_heads, query_length, seq_length],
                )
                * neg_inf
            )
            logits += attention_bias

        if self.use_relative_attention_bias:
            if position_bias is None:
                position_bias = self._compute_bias(real_seq_length, key_length)

                # if key and values are already calculated
                # we want only the last query position bias.
                if past_kv is not None:
                    position_bias = position_bias[:, -seq_length:, :]

            logits += position_bias

        # Softmax.
        if self.compute_dtype != 'float32' and self.softmax_dtype_fp32:
            logits = tf.cast(logits, tf.float32)
        weights = tf.cast(
            tf.nn.softmax(logits, name="attention_weights"), self.compute_dtype,
        )

        # Dropout.
        weights = self.dropout_layer(weights, training)

        # Shape: (batch_size, num_heads, query_length, hidden_size / num_heads).
        attention_output = tf.matmul(weights, v)

        # Recombine heads --> [batch_size, seq_length, hidden_size].
        attention_output = self._combine_heads(attention_output)

        # Run the combined outputs through another linear projection layer.
        attention_output = self.output_dense_layer(attention_output)

        if cache_position_bias and cache_present_kv:
            return attention_output, present_kv, position_bias

        if cache_position_bias:
            return attention_output, position_bias

        if cache_present_kv:
            return attention_output, present_kv

        return attention_output

    def _compute_bias(self, query_length, key_length):
        """Compute binned relative position bias.

            Args:
                query_length (int): length of the query tensor.
                key_length (int): length of the key tensor.

            Returns:
                values (Tensor): computed values for position bias.
        """
        context_position = tf.range(query_length)[:, None]
        memory_position = tf.range(key_length)[None, :]

        # Shape: (query_length, key_length).
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position
        )

        # Shape: (query_length * key_length, num_heads).
        values = tf.gather(
            # Multiply by 1.0 to avoid TF sparse resource ops
            1.0 * self.relative_attention_bias,
            tf.reshape(relative_position_bucket, [-1]),
        )

        # Shape: (query_length, key_length, num_heads).
        values = tf.reshape(values, [query_length, key_length, -1])
        # Shape: (num_heads, query_length, key_length).
        values = tf.transpose(values, [2, 0, 1])
        return values

    def _relative_position_bucket(self, relative_position, max_distance=128):
        """
        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position,
        i.e. the distance in tokens from the attending position to the attended-to position.

        If ``self.bidirectional_relative_attention`` = False, then positive relative
        positions are invalid. We use smaller buckets for small absolute
        relative positions and larger buckets for larger absolute relative positions.
        All relative positions >= max_distance map to the same bucket.
        All relative positions <= -max_distance map to the same bucket.

        This should allow for more graceful generalization to longer sequences
        than the model has been trained on.

        Args:
            relative_position (Tensor): Tensor with relative positions.
            max_distance (int): Used in order to calculate relative position buckets.

        Returns:
            a Tensor with the same shape as ``relative_position``,
            containing int32 values in the range [0, self.num_relative_attention_buckets).
        """
        relative_buckets = 0
        num_buckets = self.num_relative_attention_buckets
        if self.bidirectional_relative_attention:
            num_buckets //= 2
            relative_buckets += (
                tf.cast(
                    tf.math.greater(relative_position, 0),
                    dtype=relative_position.dtype,
                )
                * num_buckets
            )
            relative_position = tf.math.abs(relative_position)
        else:
            relative_position = -tf.math.minimum(relative_position, 0)

        max_exact = num_buckets // 2

        is_small = tf.math.less(relative_position, max_exact)
        relative_position_if_large = max_exact + tf.cast(
            tf.math.log(relative_position / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact),
            dtype=relative_position.dtype,
        )
        relative_position_if_large = tf.math.minimum(
            relative_position_if_large, num_buckets - 1
        )
        relative_buckets += tf.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def _split_heads(self, x):
        """Split x into different heads, and transpose the resulting value. The
        tensor is transposed to insure the inner dimensions hold the correct
        values during the matrix multiplication.

        Args:
          x: A tensor with shape ``[batch_size, seq_length, hidden_size]``.

        Returns:
          A tensor with shape
          ``[batch_size, num_heads, seq_length, hidden_size/num_heads]``.
        """

        with tf.compat.v1.name_scope("split_heads"):
            batch_size = tf.shape(input=x)[0]
            seq_length = tf.shape(input=x)[1]

        # Calculate depth of last dimension after it has been split.
        depth = self.hidden_size // self.num_heads

        # Split the last dimension.
        x = tf.reshape(x, [batch_size, seq_length, self.num_heads, depth])

        # Transpose the result.
        return tf.transpose(a=x, perm=[0, 2, 1, 3])

    def _combine_heads(self, x):
        """Combine tensor that has been split.

        Args:
            x: A tensor ``[batch_size, num_heads, seq_length,
                hidden_size/num_heads]``.
        Returns:
            A tensor with shape ``[batch_size, seq_length, hidden_size]``.
        """

        with tf.compat.v1.name_scope("combine_heads"):
            batch_size = tf.shape(input=x)[0]
            seq_length = tf.shape(input=x)[2]
            x = tf.transpose(
                a=x, perm=[0, 2, 1, 3]
            )  # --> [batch, seq_length, num_heads, depth]
            return tf.reshape(x, [batch_size, seq_length, self.hidden_size])


class SelfAttentionLayer(AttentionLayer):
    """Multiheaded self-attention layer."""

    def call(
        self,
        x,
        mask=None,
        past_kv=None,
        cache_present_kv=False,
        training=True,
        position_bias=None,
        cache_position_bias=False,
    ):
        return super(SelfAttentionLayer, self).call(
            x,
            x,
            mask=mask,
            past_kv=past_kv,
            cache_present_kv=cache_present_kv,
            training=training,
            position_bias=position_bias,
            cache_position_bias=cache_position_bias,
        )
