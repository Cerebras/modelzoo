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
from typing import Optional

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.config_manager.config_classes.base.base_config import (
    BaseConfig,
    required,
)
from cerebras.modelzoo.config_manager.config_classes.base.data_config import (
    DataConfig,
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
from cerebras.modelzoo.models.nlp.bert.config import BertModelConfig


@dataclass
class ESM2ModelConfig(BertModelConfig):
    token_dropout: bool = False
    mask_token_id: Optional[int] = None
    use_final_layer_norm: bool = False
    embedding_layer_norm: bool = True


@registry.register_config("esm2")
@dataclass
class ESMConfig(BaseConfig):
    train_input: Optional[DataConfig] = None
    "Dataloader configuration for train mode"

    eval_input: Optional[DataConfig] = None
    "Dataloader configuration for eval mode"

    model: ESM2ModelConfig = required
    "Model architecture configuration"

    sparsity: Optional[SparsityConfig] = None
    optimizer: OptimizerConfig = required
    runconfig: RunConfig = required
