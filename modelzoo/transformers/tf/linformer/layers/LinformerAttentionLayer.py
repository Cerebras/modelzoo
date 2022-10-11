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


class LinformerAttentionLayer(BaseLayer):
    """Multi-head Linformer attention layer. Based on `MLCommons model\
    <https://github.com/mlperf/training/blob/master/translation/tensorflow\
    /transformer/model/attention_layer.py>`_.
    and
    https://arxiv.org/pdf/2006.04768.pdf

    Args:
        hidden_size (int): Number of units in each projection output.
        num_heads (int): Number of attention heads.
        use_projection_bias (bool): Whether to use bias in the key, query, and
            value projections.
        K_projection_kernel(tf.Variable): Projection kernel used for "Keys". Default=None.
            If None, the Variable is initialized  based on `attention_style`.
        V_projection_kernel(tf.Variable): Projection kernel used for "Values". Default=None.
            If None, the Variable is initialized  based on `attention_style`.
        use_ffn_bias (bool): Whether to use bias in the output projection.
        initializer (str): Projection kernel intializer. Defaults to
            ``glorot_uniform``.
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
        boundary_casting (bool): If ``True``, then outputs the values in half
            precision and casts the  input values up to full precision.
        tf_summary (bool): If ``True``, then saves the activations with
            ``summary_layer``.
        :param int projected_dims: the projected dimension size in Linformer (`k`)
        :param str attention_style: Linformer attention style. One of {"linformer-shared-heads", "linformer-shared-kv"}.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        projected_dims,
        attention_style=None,
        K_projection_kernel=None,
        V_projection_kernel=None,
        output_projection_size=None,
        use_projection_bias=False,
        use_ffn_bias=False,
        initializer="glorot_uniform",
        kernel_regularizer=None,
        bias_regularizer=None,
        attention_type="scaled_dot_product",
        dropout_rate=0.0,
        dropout_seed=None,
        use_relative_attention_bias=False,
        relative_attention_bias=None,
        num_relative_attention_buckets=32,
        bidirectional_relative_attention=False,
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

        super(LinformerAttentionLayer, self).__init__(
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

        # Linformer params
        self.projected_dims = projected_dims
        self.attention_style = attention_style
        self.K_projection_kernel = K_projection_kernel
        self.V_projection_kernel = V_projection_kernel

        if any([self.K_projection_kernel, self.V_projection_kernel]):
            raise ValueError(
                f"For attention style - {self.attention_style}, "
                f"the projection weights should not be passed as input."
                f"These weights will be created inside build() function"
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

        self.q_dense_layer = DenseLayer(**proj_params, name="q_projection")
        self.k_dense_layer = DenseLayer(**proj_params, name="k_projection")
        self.v_dense_layer = DenseLayer(**proj_params, name="v_projection")

        self.dropout_layer = DropoutLayer(
            dropout_rate,
            seed=dropout_seed,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.dtype_policy,
            name="dropout_layer",
        )

        self.output_dense_layer = DenseLayer(
            self.output_projection_size,
            use_bias=use_ffn_bias,
            kernel_initializer=initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.dtype_policy,
            name="output_transform",
        )

        # If no relative attention bias weights are provided when
        # `use_relative_attention_bias` set to True, we are creating them.
        if use_relative_attention_bias and relative_attention_bias is None:
            relative_attention_bias = self.add_weight(
                name="relative_attention_bias",
                shape=[num_relative_attention_buckets, num_heads],
            )

        self.use_relative_attention_bias = use_relative_attention_bias
        self.relative_attention_bias = relative_attention_bias
        self.num_relative_attention_buckets = num_relative_attention_buckets
        self.bidirectional_relative_attention = bidirectional_relative_attention

        self.initializer = initializer
        self.use_bias = use_projection_bias
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

    def _build_projection_kernel(self, name):
        kernel = DenseLayer(
            self.projected_dims,
            use_bias=self.use_bias,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            dtype=self.dtype_policy,
            boundary_casting=self.boundary_casting,
            tf_summary=self.tf_summary,
            name=f"{name}",
        )

        return kernel

    def _build_msl_projection_layers(self):
        self.K_projection_kernel = self._build_projection_kernel("E_projection")

        if self.attention_style == "linformer-shared-kv":
            self.V_projection_kernel = self.K_projection_kernel
        else:
            self.V_projection_kernel = self._build_projection_kernel(
                "F_projection"
            )

    def build(self, input_shape):
        self.max_sequence_length = input_shape[1]
        self._build_msl_projection_layers()
        self.built = True

    def call(
        self,
        q,
        k,
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
        k = self.k_dense_layer(k)
        v = self.v_dense_layer(v)

        if self.attention_style in [
            "linformer-shared-heads",
            "linformer-shared-kv",
        ]:
            # outshape: [bsz, hidden_size, msl]
            k = tf.transpose(k, perm=[0, 2, 1])
            # outshape = [bsz, hidden_size, projected_dims]
            k = self.K_projection_kernel(k)
            # outshape = [bsz, projected_dims, hidden_size]
            k = tf.transpose(k, perm=[0, 2, 1])

            # outshape: [bsz, hidden_size, msl]
            v = tf.transpose(v, perm=[0, 2, 1])
            # outshape = [bsz, hidden_size, projected_dims]
            v = self.V_projection_kernel(v)
            # outshape = [bsz, projected_dims, hidden_size]
            v = tf.transpose(v, perm=[0, 2, 1])

        # Split q, k, v into heads.
        # outshape = [bsz, num_heads, msl (or) projected_dims, hidden_size/num_heads]
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
            assert len(mask.shape) in [
                2,
                3,
            ], "Only 2D and 3D masks are supported"

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
            else:
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

            # Compute the attention_bias based on the mask.
            # shape: (batch_size, 1, 1, seq_length).
            attention_bias = (
                tf.reshape(
                    tf.cast(mask, logits.dtype),
                    shape=[batch_size, 1, query_length, seq_length],
                )
                * neg_inf
            )

            logits += attention_bias

        if self.use_relative_attention_bias:
            if position_bias is None:
                position_bias = self._compute_bias(
                    real_seq_length, key_length, dtype=logits.dtype
                )

                # if key and values are already calculated
                # we want only the last query position bias.
                if past_kv is not None:
                    position_bias = position_bias[:, :, -seq_length:, :]

            logits += position_bias

        # Softmax.
        weights = tf.cast(
            tf.nn.softmax(
                tf.cast(logits, tf.float32), name="attention_weights"
            ),
            self.compute_dtype,
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

    def _compute_bias(self, query_length, key_length, dtype):
        """Compute binned relative position bias.

            Args:
                query_length (int): length of the query tensor.
                key_length (int): length of the key tensor.
                dtype (tf.dtype): type of the output tensor.

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

        # Shape: (1, num_heads, query_length, key_length).
        values = tf.expand_dims(tf.transpose(values, [2, 0, 1]), axis=0)
        values = tf.cast(values, dtype=dtype)
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


class LinformerSelfAttentionLayer(LinformerAttentionLayer):
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
        return super(LinformerSelfAttentionLayer, self).call(
            x,
            x,
            x,
            mask=mask,
            past_kv=past_kv,
            cache_present_kv=cache_present_kv,
            training=training,
            position_bias=position_bias,
            cache_position_bias=cache_position_bias,
        )
