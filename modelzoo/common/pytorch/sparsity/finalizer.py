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

#!/usr/bin/env python
"""
Sideband sparse training on CS2 uses an inband representation for pruned weights
which needs to be finalized before running training or inference on another
device. This module exposes a helper script for finalizing the sparsity.
"""

from typing import Optional

import torch


@torch.no_grad()
def finalize_cs2_sparsity(
    model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    """
    Given a module loaded from a checkpoint trained on CS2 with sideband
    sparsity, finalize the sparsity into the model's parameters as zeros and
    return the mask representing the sparsity pattern for each sparse parameter.

    Args:
        model: The model whose parameters should be updated to freeze sparsity
        optimizer: If given, the corresponding optimizer states sparsity pattern
        is also frozen

    Returns:
        Dict mapping the parameter names to the sparsity pattern as a bool torch
        tensor, where True values indicates present weights and False represents
        pruned weights.
    """

    masks = {}
    for name, param in model.named_parameters():
        mask = param.isfinite()
        if mask.any():
            masks[name] = mask
        param[~mask] = 0
        if optimizer:
            for state in optimizer.state[param].values():
                if (
                    isinstance(state, torch.Tensor)
                    and state.shape == param.shape
                ):
                    state[~mask] = 0
    return masks


def finalize_cs2_sparsity_checkpoint(state_dict: dict):
    """
    Given a state_dict trained on CS2 with sideband sparsity, finalize the
    sparsity of all tensors, both weights and optimizer state by replacing the
    inband pruned weight representation with zeros for use in dense training or
    evaluation.

    Args:
        state_dict: state_dict to finalize the sparsity pattern in. Modified.
    """
    from collections.abc import Iterable

    def apply(obj):
        if isinstance(obj, torch.Tensor):
            obj[obj.isnan()] = 0
        elif isinstance(obj, dict):
            for v in obj.values():
                apply(v)
        elif isinstance(obj, Iterable):
            for v in obj:
                apply(v)

    apply(state_dict)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="""
        Given a checkpoint trained on CS2 with sideband sparsity, finalize the
        sparsity of all tensors, both weights and optimizer state by replacing
        the inband pruned weight representation with zeros for use in dense
        training or evaluation.
        """
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to input checkpoint"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to output checkpoint"
    )

    args = parser.parse_args()
    from modelzoo.common.pytorch import cbtorch

    state_dict = cbtorch.load(args.input)
    finalize_cs2_sparsity_checkpoint(state_dict)
    cbtorch.save(state_dict, args.output)
