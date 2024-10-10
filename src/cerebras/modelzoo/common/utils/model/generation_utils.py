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

from warnings import warn

import torch

import cerebras.pytorch as cstorch


def sample_tokens(token_logits, temperature=None, top_k=None, top_p=None):
    """Function used to sample tokens, if needed. Sampling supports one
    of temperature, top_k, top_p or a mix of these. If all sampling arguments
    are None, we run greedy decoding of the logits.

    Args:
        token_logits (torch.Tensor): Tensor with logits for tokens
        temperature (float): Parameter to control the randomness of the predicted tokens
        top_k (int): Sample tokens by restricting to the `k` highest probability elements
        top_p (float): Sample tokens by restricting to top tokens summing to
            prob_cut_off <= prob_cut_off

    Returns:
        Greedy or sampled token from the logits based on sampling parameters
    """

    if top_k or top_p or temperature:
        # token_logits is shape [B][1][V] due to index being [B][1]
        # [B][1][V] -> [B][V]
        token_logits = token_logits.squeeze(1)
        if temperature:
            token_logits = token_logits / temperature
        token_prob = torch.softmax(token_logits, dim=-1)

        # If k isn't specified, set it to 100 by default as a performance optimization
        # instead of looking at all vocab positions (i.e. token_prob.shape[-1]), unless
        # the vocab size is less than 100.
        if top_k is None:
            k = min(100, token_prob.shape[-1])
            warn(
                f"TopK sampling is not specified. Setting K to {k} "
                f"as a performance optimization for nongreedy sampling."
            )
        else:
            k = top_k

        # each are [B][K]
        sorted_probs, token_ids = token_prob.topk(k=k)
        token_ids = token_ids.int()  # must be integers
        if top_p:
            # Only consider the probs that sum to top_p
            with cstorch.amp.disable_casts():
                cum_probs = sorted_probs.cumsum(dim=-1)
            mask = (cum_probs < top_p).to(token_logits.dtype)
            # shift mask 1 position right to include the first token that
            # crosses the threshold as well.
            mask = torch.nn.functional.pad(mask, [1, 0], value=1)[:, :k]
            sorted_probs *= mask
        # [B][K] -> [B][1]
        chosen = sorted_probs.multinomial(num_samples=1)
        # Lookup chosen [B][1] from the sorted token_ids
        token_pred = token_ids.gather(dim=-1, index=chosen)
    else:
        token_pred = torch.argmax(token_logits, dim=-1).int()

    return token_pred
