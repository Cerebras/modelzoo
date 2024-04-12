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
import torch


def create_fixed_sparse_attention_mask(
    max_sequence_length,
    n_heads,
    dtype=None,
    local_attn_ctx=16,
    num_verts=64,
    vert_size=16,
    different_layout_per_head=False,
):
    """
    Create GPT-3 Fixed Sparse mask.
    Adapted from https://github.com/openai/sparse_attention/blob/master/attention.py#L135

    :param int max_sequence_length: Max sequence length.
    :param dtype: Dtype of the resulting mask.

    Returns:
        The autoregressive fixed sparse mask of shape
        [n_heads, max_sequence_length, max_sequence_length].
    """

    n_ctx = max_sequence_length
    assert n_heads % num_verts == 0
    stride = local_attn_ctx
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

    # Swap 0s and 1s since we use 1 to indicate masked positions
    mask = 1 - layout

    fixed_sparse_attn_mask = torch.Tensor(mask).to(dtype)

    return fixed_sparse_attn_mask
