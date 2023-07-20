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

import torch


def _duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m


class RotaryPositionEmbeddingHelper:
    def __init__(self, max_position_embeddings, rotary_dim):
        super(RotaryPositionEmbeddingHelper, self).__init__()
        self.max_position_embeddings = max_position_embeddings
        self.rotary_dim = rotary_dim
        self.sin_cached = None
        self.cos_cached = None
        self.offset = 0

    def create_fixed_pos_emb(self, x, offset):
        if self.sin_cached is not None and self.cos_cached is not None:
            if self.offset == offset:
                return self.sin_cached, self.cos_cached
        self.offset = offset

        from modelzoo.common.pytorch import cb_model as cm

        device = "cpu" if cm.use_cs() else x.device

        inv_freq = 1.0 / (
            10000
            ** (
                torch.arange(0, self.rotary_dim, 2, device=device)
                / self.rotary_dim
            )
        )
        sinusoid_inp = torch.einsum(
            "i , j -> i j",
            torch.arange(self.max_position_embeddings, device=device),
            inv_freq,
        )
        sin, cos = (
            torch.sin(sinusoid_inp).to(x.dtype),
            torch.cos(sinusoid_inp).to(x.dtype),
        )

        sin, cos = map(_duplicate_interleave, (sin, cos))

        def slice_at_offset(t):
            return t[None, offset : x.shape[1] + offset, None, :]

        assert (
            self.max_position_embeddings >= x.shape[1] + offset
        ), "RoPE requires max position embeddings ({}) >= sequence length ({}) + offset ({})".format(
            self.max_position_embeddings, x.shape[1], offset,
        )
        sin, cos = map(slice_at_offset, (sin, cos))

        # For cs runs, wrap the sin and cos matrices in xla_literal so that
        # constant folding is performed.
        self.sin_cached = cm.make_constant(sin)
        self.cos_cached = cm.make_constant(cos)
        return self.sin_cached, self.cos_cached

    def _apply_rotary_pos_emb(self, x, real_seq_length, offset=0):
        def rotate_every_two(x):
            x1 = x[:, :, :, ::2]
            x2 = x[:, :, :, 1::2]
            x = torch.stack((-x2, x1), dim=-1)
            # in einsum notation: rearrange(x, '... d j -> ... (d j)')
            return x.flatten(-2)

        sin, cos = self.create_fixed_pos_emb(x, offset)

        # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
        return (x * cos) + (rotate_every_two(x) * sin)

    def rotate_tensor(self, x, real_seq_length, offset=0):
        assert (
            len(x.shape) == 4
        ), "Tensor should be of shape [batch_size, seq_length, num_heads, head_dim] !"
        x_rotary = x[:, :, :, : self.rotary_dim]
        x_pass = x[:, :, :, self.rotary_dim :]
        x_rotated = self._apply_rotary_pos_emb(
            x_rotary, real_seq_length, offset=offset
        )
        x = torch.cat([x_rotated, x_pass], dim=-1)
        return x
