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

"""LoRA Callback class."""

from functools import wraps
from typing import List, Union

import torch

from cerebras.modelzoo.common.utils.model.lora import (
    LoraConfig,
    make_model_lora,
)
from cerebras.modelzoo.trainer.callbacks import Callback


class Lora(Callback):
    """Callback class that handles lorafying the model."""

    def __init__(
        self, lora_params: Union[dict, List[dict], LoraConfig, List[LoraConfig]]
    ):
        """
        Args:
            lora_params: The parameters to configure LoRA.
        """
        self.lora_params = lora_params

    def pre_setup(self, trainer):
        model = trainer.callbacks["model"].model

        if isinstance(model, torch.nn.Module):

            def model_fn():
                return make_model_lora(model, self.lora_params)

        else:

            @wraps(model)
            def model_fn():
                return make_model_lora(model(), self.lora_params)

        trainer.callbacks["model"].model = model_fn

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass
