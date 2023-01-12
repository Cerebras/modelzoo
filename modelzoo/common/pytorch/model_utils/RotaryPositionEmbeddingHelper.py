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


class RotaryPositionEmbeddingHelper:
    def __init__(self, max_position_embeddings, rotary_dim):
        super(RotaryPositionEmbeddingHelper, self).__init__()
        self.max_position_embeddings = max_position_embeddings
        self.rotary_dim = rotary_dim

    def create_fixed_pos_emb(self, device, dtype):
        from modelzoo.common.pytorch import cb_model as cm

        use_cs = cm.use_cs()
        inv_freq = 1.0 / (
            10000
            ** (
                torch.arange(
                    0, self.rotary_dim, 2, device="cpu" if use_cs else device
                )
                / self.rotary_dim
            )
        )
        # TODO: We may not support einsum notiations. Leave here for now
        # need to follow up if things don't work out
        sinusoid_inp = torch.einsum(
            "i , j -> i j",
            torch.arange(
                self.max_position_embeddings, device="cpu" if use_cs else device
            ),
            inv_freq,
        ).float()
        sin, cos = (
            torch.sin(sinusoid_inp).to(dtype),
            torch.cos(sinusoid_inp).to(dtype),
        )
        # For cs runs, wrap the sin and cos matrices in xla_literal so that
        # constant folding is performed.
        sin_literal = cm.make_constant(sin)
        cos_literal = cm.make_constant(cos)
        return sin_literal, cos_literal

    def _apply_rotary_pos_emb(self, x, real_seq_length, offset=0):
        def rotate_every_two(x):
            x1 = x[:, :, :, ::2]
            x2 = x[:, :, :, 1::2]
            x = torch.stack((-x2, x1), dim=-1)
            # in einsum notation: rearrange(x, '... d j -> ... (d j)')
            return x.flatten(-2)

        def duplicate_interleave(m):
            """
            A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
            """
            dim0 = m.shape[0]
            m = m.view(-1, 1)  # flatten the matrix
            m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
            m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
            return m

        fpe_sin, fpe_cos = self.create_fixed_pos_emb(x.device, x.dtype)

        sin, cos = map(
            lambda t: duplicate_interleave(t)[
                None, offset : x.shape[1] + offset, None, :
            ],
            (fpe_sin[:real_seq_length, :], fpe_cos[:real_seq_length, :]),
        )
        # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
        return (x * cos) + (rotate_every_two(x) * sin)

    def rotate_tensor(self, x, real_seq_length, offset=0):
        assert (
            len(x.shape) == 4
        ), "Tensor should be of shape [batch_size, num_heads, seq_length, head_dim] !"
        x_rotary = x[:, :, :, : self.rotary_dim]
        x_pass = x[:, :, :, self.rotary_dim :]
        x_rotated = self._apply_rotary_pos_emb(
            x_rotary, real_seq_length, offset=offset
        )
        x = torch.cat([x_rotated, x_pass], dim=-1)
        return x
