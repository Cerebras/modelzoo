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

# This code is adapted from
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
#
# Copyright 2022 Cerebras Systems.
#
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Set, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoraConfig:
    r"""
    r: Rank of LoRA matrix projections
    alpha: Scaling factor (see paper for additional details)
    dropout: Dropout to apply to LoRA updates
    fan_in_fan_out:
    merge_weights: Determines whether lora weights should be merged/folded
        into underlying layers
    target_modules: A list of module names that must all exist in layers
        that will be converted to LoRA. For example, setting target_modules
        to ["TransformerDecoderLayer", "Linear"] would mean that all linear
        layers that were children of a TransformerDecoderLayer would be
        converted to LoRA.
    """

    r: int = 0
    alpha: int = 1
    dropout: float = 0.0
    fan_in_fan_out: bool = False
    merge_weights: bool = False
    target_modules: Optional[list] = None


def disable_lora_merge_weights(lora_params_dict: Union[dict, List[dict]]):
    r"""Sets merge_weights=False in LoRA parameters. This is helpful during
    eval mode to ensure that the weights don't get folded prior to checkpoint
    loading.
    """

    def _disable_merge_weights(params, printed_already=False):
        if params["merge_weights"] and not printed_already:
            logging.warning(
                "Automatically switching LoRA merge_weights to False in order "
                "to run evals."
            )
            printed_already = True

        params["merge_weights"] = False

        return printed_already

    if isinstance(lora_params_dict, list):
        printed = True
        for params in lora_params_dict:
            printed = _disable_merge_weights(params, printed)
    else:
        _disable_merge_weights(lora_params_dict)


class LoRALayer:
    r"""
    Base LoRA layer
    From https://github.com/microsoft/LoRA/blob/main/loralib/layers.py.
    """

    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class LoRA_Embedding(nn.Embedding, LoRALayer):
    r"""
    LoRA embedding layer
    From https://github.com/microsoft/LoRA/blob/main/loralib/layers.py.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0,
            merge_weights=merge_weights,
        )
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r, num_embeddings))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((embedding_dim, r))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(
                        0, 1
                    ) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(
                        0, 1
                    ) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r > 0:
                after_A = F.embedding(
                    x,
                    self.lora_A.transpose(0, 1),
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )
                result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)


class LoRA_Linear(nn.Linear, LoRALayer):
    r"""
    LoRA linear layer
    From https://github.com/microsoft/LoRA/blob/main/loralib/layers.py.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (
                        T(self.lora_B @ self.lora_A) * self.scaling
                    )
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (
                        T(self.lora_B @ self.lora_A) * self.scaling
                    )
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (
                    self.lora_dropout(x)
                    @ self.lora_A.transpose(0, 1)
                    @ self.lora_B.transpose(0, 1)
                ) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


def get_lora_config_for_module(
    lora_params: Union[LoraConfig, List[LoraConfig]], module_names: List[str]
) -> Optional[LoraConfig]:
    r"""
    Gets lora parameters for a particular module.

    Args:
        lora_params: LoRA top-level config.
        module_names: Hierarchical list of module names.

    Returns:
        lora parameters (LoraConfig) for the given module if applicable or None
        if the module is not targeted.
    """
    lora_params_list = (
        lora_params if isinstance(lora_params, list) else [lora_params]
    )

    for group_params in lora_params_list:
        target_modules = group_params.target_modules
        if target_modules is None or all(
            [e in module_names for e in target_modules]
        ):
            return group_params
    return None


def make_model_lora(
    model: nn.Module,
    lora_params: Union[dict, List[dict], LoraConfig, List[LoraConfig]],
):
    r"""
    Create a Low Rank Adaptation (LoRA) model from a non-LoRA model. Note that
    the original non-LoRA model may be modified through this process.

    Args:
        model: Initial model to make LoRA
        lora_params: LoRA parameters (in the form of a dict or list of
            dicts) which dictate how the supplied model will be converted into
            a LoRA model. The parameters should align with LoraConfig.

    Returns:
        LoRA model
    """
    if isinstance(lora_params, LoraConfig):
        lora_params = lora_params
    if isinstance(lora_params, list):
        lora_params = [
            e if isinstance(e, LoraConfig) else LoraConfig(**e)
            for e in lora_params
        ]
    else:
        lora_params = LoraConfig(**lora_params)

    loraified_modules = set()
    lora_model = make_model_lora_helper(
        model, lora_params, [], loraified_modules
    )

    if len(loraified_modules) == 0:
        raise RuntimeError(
            f"No modules were converted to LoRA. Please ensure that the "
            f"target_modules listed in the lora_params are valid."
        )

    logging.info(
        f"All layers matching the following module names were converted to LoRA"
        f": {loraified_modules}"
    )

    for n, p in lora_model.named_parameters():
        if not n.endswith(".lora_A") and not n.endswith(".lora_B"):
            p.requires_grad = False

    return lora_model


def make_model_lora_helper(
    model: nn.Module,
    lora_params: Union[LoraConfig, List[LoraConfig]],
    module_names: List[str],
    loraified_modules: Set[str],
):
    module_names = module_names + [type(model).__name__]
    for name, child in model.named_children():
        model.add_module(
            name,
            make_model_lora_helper(
                child, lora_params, module_names, loraified_modules
            ),
        )
    module_lora_params = get_lora_config_for_module(lora_params, module_names)
    if module_lora_params is not None and isinstance(model, nn.Embedding):
        loraified_modules.add(".".join(module_names))
        lora_embedding = LoRA_Embedding(
            # Embedding Args:
            model.num_embeddings,
            model.embedding_dim,
            padding_idx=model.padding_idx,
            max_norm=model.max_norm,
            norm_type=model.norm_type,
            scale_grad_by_freq=model.scale_grad_by_freq,
            sparse=model.sparse,
            device=model.weight.device,
            dtype=model.weight.dtype,
            # LoRA Args:
            r=module_lora_params.r,
            lora_alpha=module_lora_params.alpha,
            merge_weights=module_lora_params.merge_weights,
        )
        with torch.no_grad():
            lora_embedding.weight.copy_(model.weight)
        del model
        return lora_embedding
    elif module_lora_params is not None and isinstance(model, nn.Linear):
        loraified_modules.add(".".join(module_names))
        lora_linear = LoRA_Linear(
            # Linear Args:
            model.in_features,
            model.out_features,
            bias=model.bias is not None,
            device=model.weight.device,
            dtype=model.weight.dtype,
            # LoRA Args:
            r=module_lora_params.r,
            lora_alpha=module_lora_params.alpha,
            lora_dropout=module_lora_params.dropout,
            fan_in_fan_out=module_lora_params.fan_in_fan_out,
            merge_weights=module_lora_params.merge_weights,
        )
        with torch.no_grad():
            lora_linear.weight.copy_(model.weight)

        if model.bias is not None:
            with torch.no_grad():
                lora_linear.bias.copy_(model.bias)
        del model
        return lora_linear
    else:
        return model
