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


import numpy as np
import tensorflow as tf

from modelzoo.common.tf.layers.EmbeddingLayer import EmbeddingLayer
from modelzoo.common.tf.layers.PositionEmbeddingLayer import (
    PositionEmbeddingLayer,
)

from modelzoo.common.tf.layers.SegmentEmbeddingLayer import (  # noqa
    SegmentEmbeddingLayer,
)


def create_embedding_layers(
    vocab_size,
    embedding_size,
    segment_embedding_size=None,
    embeddings_initializer='uniform',
    bias_initializer='zeros',
    embeddings_regularizer=None,
    activity_regularizer=None,
    embeddings_constraint=None,
    mask_zero=False,
    use_bias=False,
    max_position_embeddings=None,
    position_embeddings_type=None,
    position_embeddings_initializer='uniform',
    position_embeddings_regularizer=None,
    num_segments=None,
    segment_embeddings_initializer='uniform',
    segment_embeddings_regularizer=None,
    boundary_casting=False,
    tf_summary=False,
    dtype=None,
):
    """
    Creates token and, optionally,position and segment embeddings.

    :param int vocab_size: Size of input vocabulary.
    :param int embedding_size: Dimension of the embedding space.
    :param int segment_embedding_size: Dimension of the embedding space for segment
        embeddings. Useful when factorized embeddings are used for tokens and
        so the size of the embedding space for segments differs from that for
        tokens. Defaults to the same value as embedding_size.
    :param Optional[str,Callable] embeddings_initializer: Token embeddings
        initializer. Defaults to 'uniform'.
    :param Optional[string,Callable] bias_initializer: Token embeddings
        bias initializer. Defaults to 'zeros'.
    :param Optional[Callable] embeddings_regularizer: Tokens
        embeddings regularizer. Defaults to None.
    :param Optional[Callable] activity_regularizer: Token embeddings
        activation regularizer. Defaults to None.
    :param Optional embeddings_constraint: Token embeddings constraint.
        Defaults to None.
    :param Optional[bool] mask_zero: Whether or not the input value 0 is a
        special "padding" value that should be masked out. Defaults to False.
    :param Optional[bool] use_bias: Whether to use bias for token embeddings.
        Defaults to False.
    :param Optional[int] max_position_embeddings: Maximum sequence length to train
        using model. If None (default), set to input sequence length.
    :param str position_embeddings_type: 'learned' or 'fixed'. Defaults to None,
        in which case position embeddings are not created.
    :param Optional[str,Callable] position_embeddings_initializer: Position
        embeddings initializer. Defaults to "uniform".
    :param Optional[Callable] position_embeddings_regularizer: Position
        embeddings regularizer. Defaults to None.
    :param Optional[int] num_segments: Number of segments for the segment
        embedding layer. Defaults to None, in which case the segment embedding
        layer is not created.
    :param Optional[str,Callable] segment_embeddings_initializer: Segment
        embeddings initializer. Defaults to "uniform".
    :param Optional[Callable] segment_embeddings_regularizer: Segment
        embeddings regularizer. Defaults to None.
    :param bool boundary_casting: Flag to enable boundary casting.
        Defaults to False.
    :param tf_summary: Flag to enable debug summaries. Defaults to False.
    :param dtype: Dtype or policy. Defaults to None.

    Returns:
       Token, position, and embedding layers
    """

    if segment_embedding_size is None:
        segment_embedding_size = embedding_size

    token_embedding = EmbeddingLayer(
        input_dim=vocab_size,
        output_dim=embedding_size,
        embeddings_initializer=embeddings_initializer,
        embeddings_regularizer=embeddings_regularizer,
        boundary_casting=boundary_casting,
        tf_summary=tf_summary,
        dtype=dtype,
        name="input_embedding",
    )

    position_embedding = (
        PositionEmbeddingLayer(
            max_position_embeddings=max_position_embeddings,
            embedding_type=position_embeddings_type,
            embeddings_initializer=position_embeddings_initializer,
            embeddings_regularizer=position_embeddings_regularizer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=dtype,
            name="position_embedding",
        )
        if position_embeddings_type
        else None
    )

    segment_embedding = (
        EmbeddingLayer(
            input_dim=num_segments,
            output_dim=segment_embedding_size,
            embeddings_initializer=segment_embeddings_initializer,
            embeddings_regularizer=segment_embeddings_regularizer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=dtype,
            name="segment_embedding",
            weight_name="segment_embedding_weights",
        )
        if num_segments
        else None
    )

    return token_embedding, position_embedding, segment_embedding


def create_autoregressive_attention_mask(
    max_sequence_length, batch_size=1, dtype=None
):
    """
    Create autoregressive (triangular) mask.

    :param int batch_size: Batch size.
    :param int max_sequence_length: Max sequence length.
    :param dtype: Dtype of the resulting mask.

    Returns:
        The autoregressive mask of shape
        [batch_size, max_sequence_length, max_sequence_length].
    """

    # Triangular mask
    with tf.compat.v1.variable_scope('autoregressive_mask'):
        # The first dimension here is the query sequence length, and the
        # second dimension is the key sequence length. An autoregressive
        # model permits each query to attend to all keys up to and
        # including the position of the query, so each row, `i`, should
        # mask all positions after position `i`.
        diag_vals = tf.ones(
            [max_sequence_length, max_sequence_length], dtype=dtype
        )
        # The tril looks like:
        # [ 1, 0, 0,
        #   1, 1, 0,
        #   1, 1, 1 ]
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        # Swap 0s and 1s since we use 1 to indicate masked positions
        tril = 1 - tril
        # Expand the batch dimension
        auto_attn_mask = tf.tile(
            tf.expand_dims(tril, axis=0), [batch_size, 1, 1]
        )

    return auto_attn_mask


def get_bits_per_x_dataset(params):
    """Get the dataset to get the associated bits_per_byte constant

    Args:
        params (dict): Parameters for the current model training

    Returns:
        Parameters dictionary with the correct dataset set
    """
    eparams = params.get("eval_input", None)
    if not eparams:
        eparams = params.get("train_input", None)
        assert (
            eparams
        ), "Neither eval_input nor train_input are specified. Aborting run!!"

    # get the correct dataset for bits_per_byte metric
    # defaults to empty data directory
    dataset = None
    data_dir = eparams.get("data_dir", "")
    if "pile" in data_dir:
        dataset = "pile"
    elif "owt" in data_dir:
        dataset = "openwebtext2"

    params["model"]["bits_per_x_dataset"] = dataset
    return params


def create_fixed_sparse_attention_mask(
    max_sequence_length,
    n_heads,
    batch_size=1,
    dtype=None,
    local_attn_ctx=16,
    num_verts=64,
    vert_size=16,
    different_layout_per_head=False,
):
    """
    Create GPT-3 Fixed Sparse mask.
    Adapted from https://github.com/openai/sparse_attention/blob/master/attention.py#L135

    :param int batch_size: Batch size.
    :param int max_sequence_length: Max sequence length.
    :param dtype: Dtype of the resulting mask.

    Returns:
        The autoregressive fixed sparse mask of shape
        [batch_size, n_heads, max_sequence_length, max_sequence_length].
    """

    with tf.compat.v1.variable_scope('fixed_sparse_attention_mask'):
        n_ctx = max_sequence_length
        stride = local_attn_ctx
        # checks for correct mask creation
        assert n_heads % num_verts == 0
        assert vert_size <= stride
        assert stride % vert_size == 0
        indices = [i for i in range(stride - 1, -1, -1)]
        indices = np.array(indices).reshape([-1, vert_size])
        if num_verts == 1:
            layout = np.zeros([n_ctx, n_ctx])
            for idx in indices[0]:
                layout[:, idx::stride] = 1
            for q_idx in range(n_ctx):
                # Each thing can attend to its local block
                row = q_idx // stride
                layout[q_idx, row * stride : (row + 1) * stride] = 1
                # Any query cannot attend to keys above it
                layout[q_idx, q_idx + 1 :] = 0
        else:
            layouts = []
            indices = indices[:num_verts]
            for h in range(n_heads):
                layout = np.zeros([n_ctx, n_ctx])
                subindices = indices[h % num_verts]
                for idx in subindices:
                    layout[:, idx::stride] = 1
                for q_idx in range(n_ctx):
                    # Each position can attend to its local block
                    row = q_idx // stride
                    layout[q_idx, row * stride : (row + 1) * stride] = 1
                    # Any query cannot attend to keys above it
                    layout[q_idx, q_idx + 1 :] = 0
                layouts.append(layout)
            layout = np.array(layouts)

            if not different_layout_per_head:
                layout = layout[0, :, :]

        mask = tf.constant(layout, dtype=dtype)

        # Swap 0s and 1s since we use 1 to indicate masked positions
        mask = 1 - mask

        # Expand the batch dimension
        fixed_sparse_attn_mask = tf.tile(
            tf.expand_dims(mask, axis=0),
            [batch_size, 1, 1, 1]
            if different_layout_per_head
            else [batch_size, 1, 1],
        )

    return fixed_sparse_attn_mask


def print_fixed_sparse_mask(
    max_sequence_length: int,
    num_heads: int,
    batch_size: int,
    local_attn_ctx: int,
    num_verts: int,
    vert_size: int,
    different_layout_per_head: int,
):
    from matplotlib import pyplot as plt

    tf.compat.v1.enable_eager_execution()
    np.set_printoptions(threshold=np.inf)

    name = f""
    name += f"msl_{max_sequence_length}"
    name += f"_nheads_{num_heads}"
    name += f"_bs_{batch_size}"
    name += f"_ctx_{local_attn_ctx}"
    name += f"_nverts_{num_verts}"
    name += f"_vsize_{vert_size}"
    name += f"_layout_{different_layout_per_head}.jpg"

    mask = create_fixed_sparse_attention_mask(
        max_sequence_length=max_sequence_length,
        n_heads=num_heads,
        batch_size=batch_size,
        local_attn_ctx=local_attn_ctx,
        num_verts=num_verts,
        vert_size=vert_size,
        different_layout_per_head=different_layout_per_head,
    )

    if not different_layout_per_head:
        mask = tf.squeeze(mask)
        np_mask = mask.numpy()
        binary = np_mask > 0

        plt.clf()
        plt.imshow(binary)
        plt.savefig(name, dpi=300)
        plt.show()

    else:
        for i in range(num_heads):
            mask = tf.squeeze(mask)
            np_mask = mask.numpy()
            binary = np_mask > 0
            f_binary = binary[i, :, :]

            f_name = f"head_{i}_{name}"

            plt.clf()
            plt.imshow(f_binary)
            plt.savefig(f_name, dpi=300)
            plt.show()
