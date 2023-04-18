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

from modelzoo.common.pytorch import cb_model as cm


def tiebreak_for_topk(score, method, eps):
    if method == "random":
        return score + score.new_empty(score.shape).uniform_(to=eps)

    elif method == "first":
        # add a small, linearly increasing epsilon to the score. Thus, later
        # logical entries will be chosen before earlier entries with equal
        # score.
        iota = torch.arange(score.numel(), device=score.device)
        eps = torch.tensor(eps, dtype=score.dtype, device=score.device)
        return score + (iota * eps).view(score.shape)
    else:
        return score


def make_mask_topk_sparsity(score, sparsity):
    density = 1 - sparsity
    numel = torch.tensor(score.numel(), dtype=torch.float)
    num_dense_elem = (density * numel).int()
    return make_mask_topk_k(score, num_dense_elem)


def make_mask_top_atleast_k(score, num_dense_elem):
    flat_score = score.view(-1)
    sorted_values = torch.sort(flat_score).values
    cutoff = sorted_values.index_select(0, num_dense_elem)
    mask = score > cutoff
    return mask


def make_mask_topk_k(score, num_dense_elem):
    flat_score = score.view(-1)
    if cm.use_cs() and score.device == cm.device():
        # Temporarily, use cutoff mechanism to produce at-least k mask
        # positions rather than precisely k.
        return make_mask_top_atleast_k(score, num_dense_elem)

        # WSE
        # `torch.topk` uses a python integer for the `k` operand, which will
        # change throughout training. Even though this integer is computed from
        # tensors (the sparsity schedule), calling .item() on it breaks the
        # ability to trace the dataflow.
        # Since we only trace the program once, this prevents us from using
        # `torch.topk. Although even if it somehow did accept a traceable
        # tensor for `k`, the result would not be statically shaped, causing
        # other issues.

        # Instead, sort the whole tensor...
        indices = torch.sort(flat_score).indices
        # .. and mask off all but the first k indices, replacing them with the
        # top-magnitude value. This works even if num_dense_elem == numel.
        indices[num_dense_elem:] = indices[0]
    else:
        # CPU/GPU
        _, indices = torch.topk(flat_score, num_dense_elem.item())

    flat_mask = torch.zeros_like(flat_score, dtype=torch.bool)
    flat_mask = flat_mask.scatter(0, indices, True)
    mask = flat_mask.view(score.shape).contiguous()

    return mask
