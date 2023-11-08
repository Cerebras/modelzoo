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


class SparseAttentionBuilder:
    def __init__(
        self, num_heads, max_sequence_length, num_different_head_attn_configs=1
    ):
        self.num_heads = num_heads
        self.max_sequence_length = max_sequence_length
        self.num_different_head_attn_configs = num_different_head_attn_configs
        self.attention_mask = np.zeros(
            (self.num_heads, self.max_sequence_length, self.max_sequence_length)
        )
        self.mask_is_built = False

    def build_mask(self):
        raise NotImplementedError(
            "Make sure that a `build_mask` function "
            f"is correctly implemented in {self.__class__.__module__}"
        )

    def get_pytorch_mask(self, dtype=None):
        import torch

        # Avoid rebuilding mask if already exists
        if not self.mask_is_built:
            self.build_mask()
        attn_matrix = self.attention_mask
        if self.num_different_head_attn_configs == 1:
            attn_matrix = self.attention_mask[0, :, :]
        mask = 1 - attn_matrix
        sparse_attn_mask = torch.Tensor(mask).to(dtype)

        return sparse_attn_mask


class FixedSparseAttentionBuilder(SparseAttentionBuilder):
    def __init__(
        self,
        num_heads,
        max_sequence_length,
        num_different_head_attn_configs=1,
        local_attn_ctx=4,
        global_attn_ctx=1,
        attention_type="unidirectional",
        horizontal_global_attention=False,
    ):
        super().__init__(
            num_heads, max_sequence_length, num_different_head_attn_configs
        )
        self.local_attn_ctx = local_attn_ctx
        self.global_attn_ctx = global_attn_ctx
        assert self.local_attn_ctx % self.global_attn_ctx == 0, (
            f"Number of blocks in a local window, {self.local_attn_ctx}, "
            f"must be dividable by number of global blocks, {self.global_attn_ctx}!"
        )

        self.attention_type = attention_type.lower()
        assert (
            self.attention_type == "unidirectional"
            or self.attention_type == "bidirectional"
        ), "Only 'unidirectional' or 'bidirectional' attention types are supported"

        self.horizontal_global_attention = horizontal_global_attention
        if self.horizontal_global_attention:
            assert (
                self.attention_type == "bidirectional"
            ), "Horizontal global attention is only supported by the 'bidirectional' attention type"

    def set_local_attention(self, head_id):
        for i in range(0, self.max_sequence_length, self.local_attn_ctx):
            end = min(i + self.local_attn_ctx, self.max_sequence_length)
            for row in range(i, end):
                col_end = (
                    row + 1 if self.attention_type == "unidirectional" else end
                )
                self.attention_mask[head_id, row, i:col_end] = 1

    def set_global_attention(self, head_id):
        first_global_attn_col_idx = (
            self.local_attn_ctx
            - (1 + head_id % self.num_different_head_attn_configs)
            * self.global_attn_ctx
        )

        # set all global attention in all self.local_attn_ctx windows except the last one if (in last local window)
        end = self.max_sequence_length - (
            self.max_sequence_length % self.local_attn_ctx
        )
        for i in range(first_global_attn_col_idx, end, self.local_attn_ctx):
            # veritcal attn
            first_global_attn_row_idx = (
                0 if self.attention_type == "bidirectional" else i
            )
            self.attention_mask[
                head_id,
                first_global_attn_row_idx:,
                i : i + self.global_attn_ctx,
            ] = 1

            if self.horizontal_global_attention:
                self.attention_mask[
                    head_id, i : i + self.global_attn_ctx, :
                ] = 1

        # set last global attention in the last self.local_attn_ctx window
        if end < self.max_sequence_length:
            start = min(
                end + first_global_attn_col_idx,
                self.max_sequence_length - self.global_attn_ctx,
            )
            end = start + self.global_attn_ctx
            first_global_attn_row_idx = (
                0 if self.attention_type == "bidirectional" else start
            )
            self.attention_mask[
                head_id, first_global_attn_row_idx:, start:end
            ] = 1
            if self.horizontal_global_attention:
                self.attention_mask[head_id, start:end, :] = 1

    def trim_to_autoregressive_format(self, head_id):
        # to ensure in the unidirectional case, the mask is in an autoregressive format
        autoregressive_mask = np.ones(
            (self.max_sequence_length, self.max_sequence_length)
        )
        autoregressive_mask = np.tril(autoregressive_mask)
        self.attention_mask[head_id] = np.multiply(
            autoregressive_mask, self.attention_mask[head_id]
        )

    def build_mask(self):
        num_head_attn_configs_to_build = (
            self.num_heads if self.num_different_head_attn_configs > 1 else 1
        )
        for head_id in range(0, num_head_attn_configs_to_build):
            self.set_local_attention(head_id)
            self.set_global_attention(head_id)
            if self.attention_type == "unidirectional":
                self.trim_to_autoregressive_format(head_id)
        self.mask_is_built = True


class StridedSparseAttentionBuilder(SparseAttentionBuilder):
    def __init__(
        self,
        num_heads,
        max_sequence_length,
        local_attn_ctx=4,
        stride=2,
        attention_type="unidirectional",
    ):
        super().__init__(num_heads, max_sequence_length, 1)
        self.local_attn_ctx = local_attn_ctx
        self.stride = stride
        self.attention_type = attention_type
        assert (
            self.attention_type == "unidirectional"
            or self.attention_type == "bidirectional"
        ), "Only 'unidirectional' or 'bidirectional' attention types are supported"

    def set_local_attention(self):
        for row_idx in range(0, self.max_sequence_length):
            col_start = max(0, row_idx - self.local_attn_ctx + 1)
            col_end = (
                row_idx + 1
                if self.attention_type == "unidirectional"
                else row_idx + self.local_attn_ctx
            )
            self.attention_mask[0, row_idx, col_start:col_end] = 1

    def set_global_attention(self):
        for row_idx in range(0, self.max_sequence_length):
            col_end = (
                row_idx - self.local_attn_ctx
            )  # if self.attention_type == 'unidirectional' else self.max_sequence_length - 1
            col_start = -1
            if col_end < 0:
                continue
            global_attention_indices = [
                *range(col_end, col_start, -self.stride)
            ]
            self.attention_mask[0, row_idx, global_attention_indices] = 1
        if self.attention_type == "bidirectional":
            for row_idx in range(0, self.max_sequence_length):
                col_start = row_idx + self.local_attn_ctx
                col_end = self.max_sequence_length
                global_attention_indices = [
                    *range(col_start, col_end, self.stride)
                ]
                self.attention_mask[0, row_idx, global_attention_indices] = 1

    def build_mask(self):
        self.set_local_attention()
        self.set_global_attention()
        self.mask_is_built = True
