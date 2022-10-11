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
from torch.nn.functional import log_softmax


def create_autoregressive_mask(
    max_sequence_length, dtype=torch.float16, device=None
):
    """Create autoregressive (triangular) mask.

    Args:
        batch_size (int): Batch size.
        max_sequence_length (int): Max sequence length.
        dtype (torch.dtype): Dtype of the resulting mask.

    Returns:
        The autoregressive mask of shape
        [batch_size, max_sequence_length, max_sequence_length].
    """

    attention_mask = torch.triu(
        torch.ones(
            (max_sequence_length, max_sequence_length),
            device=device,
            dtype=dtype,
        ),
        diagonal=1,
    )

    return attention_mask


def smooth_loss(prediction_scores, loss, label_smoothing, classes):
    """
    Add label smoothing to loss function,
    this is a workaround method of label smoothing in our system
    """
    logits = prediction_scores.view(-1, classes)
    logprobs = log_softmax(logits, dim=-1)
    smooth_loss = -1.0 * logprobs.mean(dim=-1)
    loss = (1.0 - label_smoothing) * loss + label_smoothing * smooth_loss

    return loss
