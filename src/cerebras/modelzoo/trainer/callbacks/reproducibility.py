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

"""This module contains a callback that handles reproducibility."""

from typing import Optional
from warnings import warn

import torch

from cerebras.modelzoo.trainer.callbacks import CoreCallback


class Reproducibility(CoreCallback):
    """A callback that facilitates reproducibility."""

    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: If provided, sets the torch seed.
        """
        self._seed = seed

    def pre_setup(self, trainer):
        if self._seed is not None:
            # Ensure we set seed before any model initialization
            torch.manual_seed(self._seed)
        else:
            warn(
                "RNG seed was not set. This affects determinism of the model and "
                "can cause missed cached compile when running the same model again. "
            )
