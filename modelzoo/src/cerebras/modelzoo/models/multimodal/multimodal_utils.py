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

import logging
import re
from typing import List

import torch


def freeze_modules(
    model: torch.nn.Module,
    module_name_patterns: List[str],
):
    """
    Freeze modules if their name matches a regex pattern.

    Args:
        model: The model to be frozen.
        module_name_patterns: Filter to select which parameters are frozen
            Note that regex patterns should be specified as single quotes
            in the yaml for escape codes.
    """
    leaf_modules = [
        (n, m)
        for n, m in model.named_modules()
        if not next(m.children(), False)
    ]
    patterns = list(map(re.compile, module_name_patterns))

    for pattern in patterns:
        module_list = [
            (name, param)
            for name, param in leaf_modules
            if pattern.search(name)
        ]

        if len(module_list) == 0:
            raise ValueError(f"{pattern} did not match any module names!")

        for _, m in module_list:
            m.eval()
            m.requires_grad_(False)

        logging.debug(
            f"The following modules are frozen due to pattern: {pattern.pattern}: "
            f"{[n for n, _ in module_list]}"
        )

        # Additional pass through parameters since a module may have child modules
        # but also child parameters. For example, the classification token in an embedding
        # layer.
        for n, p in model.named_parameters():
            if pattern.search(n):
                p.requires_grad_(False)

    logging.debug(
        f"The following parameters are being trained: "
        f"{[n for n, p in model.named_parameters() if p.requires_grad]}"
    )
