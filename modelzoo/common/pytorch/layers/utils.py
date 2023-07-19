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

import copy
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch import cbtorch

from .AlibiPositionEmbeddingLayer import AlibiPositionEmbeddingLayer
from .RelativePositionEmbeddingLayer import RelativePositionEmbeddingLayer

LOSS_SCOPE = "loss"


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(
        "activation should be relu/gelu, not {}".format(activation)
    )


def apply_loss_reduction(loss, reduction):
    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    else:
        return loss


def apply_position_bias(embedding_helper, seq_length, key_length, past_kv=None):
    self_attn_position_bias = None
    if isinstance(
        embedding_helper,
        (RelativePositionEmbeddingLayer, AlibiPositionEmbeddingLayer,),
    ):
        self_attn_position_bias = embedding_helper(
            seq_length, key_length, past_kv=past_kv
        )
    return self_attn_position_bias


# Since we do not have automatic loss scope detection yet, some changes in user model would
# be required to tag the ops belong to loss.
# A wrapper to wrap loss function for Autogen support, so that loss function
# will be handled by Autogen.
# To use this Autogen loss function wrapper, when initializing the loss function,
# using our loss in Layer API, and adding a keyword arguement 'use_autogen=True' (the default is False).
# For example:
#   loss = MSELoss(..., use_autogen=True)
#
# To apply the Autogen loss wrapper to any custom loss function, just add this wrapper as the decorator
# of the custom loss class.
# For example:
#   @autogen_loss
#   class CustomLoss(nn.Module):
#       def __init__(...):
#
# In the future, we may remove this temporary wrapper after we developed better technics to support Autogen.
def autogen_loss(loss_cls):
    loss_cls._old_init = loss_cls.__init__
    loss_cls._old_forward = loss_cls.forward

    def autogen_init(self, *args, **kwargs):
        self.autogen_enabled = kwargs.pop("use_autogen", False)
        self._old_init(*args, **kwargs)
        if self.autogen_enabled and cm.use_cs():
            self.mark_with_autogen = cbtorch.nn.Scope(scope_name=LOSS_SCOPE)

    def autogen_forward(self, *args, **kwargs):
        if self.autogen_enabled and cm.use_cs():
            args = [self.mark_with_autogen(arg) for arg in args]
            kwargs = {k: self.mark_with_autogen(v) for k, v in kwargs.items()}
            loss = self._old_forward(*args, **kwargs)
            return self.mark_with_autogen.exit(loss)
        else:
            return self._old_forward(*args, **kwargs)

    loss_cls.__init__ = autogen_init
    loss_cls.forward = autogen_forward
    return loss_cls
