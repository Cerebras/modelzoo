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


def fixed_position_embeddings(x, seq_dim=1, seq_len=None):
    """Generate fixed position embeddings

    Args:
        x (Tensor): input to generate position embeddings for
        seq_dim (int): dimension for sequence length in x
        seq_len (int): sequence length to generate position embeddings for

    Returns
        Tuple of fixed position embeddings (sin, cos) of inverse frequency domain
    """
    dim = tf.shape(input=x)[-1]
    if seq_len is None:
        seq_len = tf.shape(input=x)[seq_dim]

    # get frequency
    positions = tf.range(start=0, limit=dim, delta=2, dtype=tf.int32)
    positions = tf.cast(tf.math.divide(x=positions, y=dim), tf.float32)
    inv_freq = tf.math.divide(x=1.0, y=tf.math.pow(x=10000.0, y=positions))
    inv_freq = tf.expand_dims(inv_freq, axis=-1)

    # get positional numbers
    seq_positions = tf.range(seq_len, dtype=tf.float32)
    seq_positions = tf.expand_dims(seq_positions, axis=-1)

    # final positional embeddings
    sinusoid_op = tf.matmul(seq_positions, inv_freq, transpose_b=True)
    sinusoid_op = tf.cast(sinusoid_op, dtype=x.dtype)
    return tf.math.sin(sinusoid_op), tf.math.cos(sinusoid_op)


def _rotate_every_two(x):
    """Rotate every alternate inputs

    Args:
        x (Tensor): input to rotate

    Returns:
        Rotated tensor
    """
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = tf.stack([-x2, x1], axis=-1)

    batch_size, seq_len, num_heads = (
        tf.shape(input=x)[0],
        tf.shape(input=x)[1],
        tf.shape(input=x)[2],
    )
    return tf.reshape(x, [batch_size, seq_len, num_heads, -1])


def apply_rotary_position_embedding(x, sincos, offset=0):
    """Apply position embeddings after rotations to enrich tensors

    Args:
        x (Tensor): tensor to apply position embeddings to
        sincos (sequence): tuple containing the position embeddings to apply
        offset (int): offset in the position embeddings to be applied

    Returns:
        Tensor with rotary position embeddings applied
    """
    sin_pos, cos_pos = sincos
    length = tf.shape(input=x)[1]
    sin_pos = tf.repeat(
        sin_pos[None, offset : length + offset, None, :], repeats=2, axis=3
    )
    cos_pos = tf.repeat(
        cos_pos[None, offset : length + offset, None, :], repeats=2, axis=3
    )
    return (x * cos_pos) + (_rotate_every_two(x) * sin_pos)


def create_causal_bias(max_position_embeddings, dtype=None):
    """
    Create autoregressive (triangular) mask for applying bias.

    Args:
        max_position_embeddings (int): Max position embeddings.
        dtype: Dtype of the resulting mask.

    Returns:
        The autoregressive mask of shape
        [1, 1, max_position_embeddings, max_position_embeddings].
    """

    # Triangular mask
    # The first dimension here is the query sequence length, and the
    # second dimension is the key sequence length. An autoregressive
    # model permits each query to attend to all keys up to and
    # including the position of the query, so each row, `i`, should
    # mask all positions after position `i`.
    diag_vals = tf.ones(
        [max_position_embeddings, max_position_embeddings], dtype=dtype
    )
    tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
    auto_attn_mask = tf.expand_dims(tf.expand_dims(tril, axis=0), axis=0)
    return auto_attn_mask
