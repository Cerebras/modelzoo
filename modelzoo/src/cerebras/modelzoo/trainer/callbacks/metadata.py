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

"""
Contains the ModelZooParamsMetadata class that stores the model zoo
parameters in the checkpoint metadata.
"""

from typing import Optional

import cerebras.pytorch as cstorch
from cerebras.modelzoo.trainer.callbacks import Callback


class ModelZooParamsMetadata(Callback):
    """
    Callback class that stores the model zoo parameters in the checkpoint
    metadata.
    """

    def __init__(self, params: Optional[dict] = None):
        """
        Args:
            params: Model zoo parameters.
        """
        self.params = params or {}

    def on_save_checkpoint(self, trainer, state_dict):
        state_dict["__metadata__"] = [
            {
                "version": cstorch.__version__,
                "model_name": trainer.model.__class__.__name__,
                "params": self.params,
            }
        ]

    def on_load_checkpoint(self, trainer, state_dict):
        trainer.metadata = state_dict.pop("__metadata__", None)

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass
