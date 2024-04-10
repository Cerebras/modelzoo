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

from dataclasses import dataclass
from typing import List, Optional

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.config_manager.config_classes.base.base_config import (
    BaseConfig,
    required,
)
from cerebras.modelzoo.config_manager.config_classes.base.data_config import (
    DataConfig,
)
from cerebras.modelzoo.config_manager.config_classes.base.model_config import (
    ModelConfig,
)
from cerebras.modelzoo.config_manager.config_classes.base.optimizer_config import (
    OptimizerConfig,
)
from cerebras.modelzoo.config_manager.config_classes.base.run_config import (
    RunConfig,
)
from cerebras.modelzoo.config_manager.config_classes.base.sparsity_config import (
    SparsityConfig,
)


@dataclass
class LlavaModelConfig(ModelConfig):
    freeze: Optional[List[str]] = None
    image_feature_select_layer_idx: Optional[int] = None
    image_start_idx: int = required
    image_feature_select_mode: str = "patch"
    loss_scaling: str = required
    loss_weight: float = required
    image_model: dict = required
    "The underlying image model being used"
    text_model: dict = required
    "The underlying text model being used"
    projector: dict = required


@registry.register_config("llava")
@dataclass
class LlavaConfig(BaseConfig):
    train_input: Optional[DataConfig] = None
    "Input params class for train mode"

    eval_input: Optional[DataConfig] = None
    "Input params class for eval mode"

    model: LlavaModelConfig = required
    "Model level params class. Supported params differ for each model."

    optimizer: OptimizerConfig = required
    "Optimizer specific prameters captured in this class."

    runconfig: RunConfig = required
    "Params class to define params for controlling runs."

    sparsity: Optional[SparsityConfig] = None
    "Params class for sparsity related cofigurations"
