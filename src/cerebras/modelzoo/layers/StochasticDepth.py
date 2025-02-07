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
import torch.nn as nn


class StochasticDepth(nn.Module):
    def __init__(self, p, mode="batch"):
        """
        Args:
        p (float) : probability of the input to be zeroed.
        mode (str):  "batch" or "row".
        "batch" randomly zeroes the entire input,
        "row" zeroes randomly selected rows from the batch.
        """
        super().__init__()
        if p < 0.0 or p > 1.0:
            raise ValueError(
                f"drop probability has to be between 0 and 1, but got {p}"
            )
        if mode not in ["batch", "row"]:
            raise ValueError(
                f"mode has to be either 'batch' or 'row', but got {mode}"
            )
        self.p = p
        self.survival_p = 1.0 - self.p
        self.mode = mode
        self.dp = nn.Dropout(p, inplace=True)

    def forward(self, input_tensor):
        """
        Args:
            input_tensor (Tensor[N, ...]): The input tensor or arbitrary dimensions with the first one
                being its batch i.e. a batch with ``N`` rows.
        Returns:
            Tensor[N, ...]: The randomly zeroed tensor based on `mode` input
        """
        if not self.training or self.p == 0.0:
            return input_tensor
        if self.mode == "row":
            dim_out = [input_tensor.shape[0]] + [1] * (
                len(input_tensor.shape) - 1
            )
            end_arange = input_tensor.shape[0] + 1
        elif self.mode == "batch":
            dim_out = [1] * input_tensor.ndim
            end_arange = 2  # Only need one value to represent whether to drop entire batch or not
        const_size = end_arange - 1
        input_tensor_batch = torch.full(
            [const_size],
            0.5,
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        input_tensor_batch = input_tensor_batch.reshape(*dim_out)
        out = self.dp(input_tensor_batch).mul_(2)
        if len(input_tensor.shape) == 3:
            out = out.reshape(const_size)
            out = out[:, None].broadcast_to(
                input_tensor.shape[0], input_tensor.shape[1]
            )
            out = out[:, :, None].broadcast_to(
                input_tensor.shape[0],
                input_tensor.shape[1],
                input_tensor.shape[2],
            )
        out = out * input_tensor
        return out
