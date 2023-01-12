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

from typing import Dict

import torch
import torch.optim

from modelzoo.common.model_utils.sparsity.sparsifiers import SPARSIFIER_MAP
from modelzoo.common.model_utils.sparsity.utils import extract_mask_from_weight
from modelzoo.common.pytorch.PyTorchBaseModel import PyTorchBaseModel


def sparsify_pytorch_model(
    model: PyTorchBaseModel, optimizer: torch.optim.Optimizer, params: Dict,
):
    torch_params = {
        name: param for name, param in model.trainable_named_parameters()[1]
    }
    dense_weights_dict = {
        name: param.cpu().data.numpy() for name, param in torch_params.items()
    }
    # Configure and construct sparsifier.
    sparsity_level = float(params.get("sparsity_level", 0))
    sparsity_distribution = params.get("sparsity_distribution", "uniform")
    mask_type = params.get("mask_type", "topk")
    sparse_val = params.get("sparse_val", float("nan"))  # hardcoded
    n_iter = int(params.get("n_iter", 0))
    epsilon = params.get("epsilon")
    zeta = params.get("zeta")
    seed = params.get("seed")
    mask_file = params.get("mask_file")
    erk_power_scale = params.get("erk_power_scale", 1.0)
    sparsifier = SPARSIFIER_MAP[mask_type](
        n_iter=n_iter,
        sparsity_level=sparsity_level,
        sparsity_distribution=sparsity_distribution,
        erk_power_scale=erk_power_scale,
        epsilon=epsilon,
        zeta=zeta,
        seed=seed,
        mask_file=mask_file,
    )
    # Go compute new .
    sparse_weight_dict = sparsifier.get_masked_weights(
        0, dense_weights_dict, sparse_val
    )
    param_masks = {}  # map param-> numpy mask
    for key, tensor in sparse_weight_dict.items():
        mask = extract_mask_from_weight(tensor, sparse_val)
        param = torch_params[key]
        param.data *= mask
        # Save mask for use by optimizer state.
        param_masks[param] = mask

    # Now go sparsify optimizer state.
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            for state in optimizer.state[param]:
                if (
                    optimizer.state[param][state].shape
                    == param_masks[param].shape
                ):
                    optimizer.state[param][state] *= param_masks[param]
