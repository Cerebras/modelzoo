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

from dataclasses import dataclass, is_dataclass
from typing import Optional, Union, get_args, get_origin, get_type_hints

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


@dataclass()
class DPOParameters(BaseConfig):
    beta: float = 0.1
    reference_free: bool = False
    loss_type: str = "sigmoid"


@dataclass
class DPOModelConfig(ModelConfig):
    model_name: str = required
    "Name of the model to use with DPO"

    dpo: DPOParameters = required
    "Parameters for DPO configuration"

    compute_eval_metrics: bool = True


def get_class_type(config_class, parameter):
    annotations = get_type_hints(config_class)
    field_type = annotations[parameter]
    if get_origin(field_type) is Union:
        for union_type in get_args(field_type):
            if is_dataclass(union_type):
                return union_type
    elif is_dataclass(field_type):
        return field_type
    return None


@registry.register_config("dpo")
@dataclass
class DPOConfig(BaseConfig):
    eval_input: Optional[DataConfig] = None
    "Input params class for eval mode"

    model: DPOModelConfig = required
    "Model level params class. Supported params differ for each model."

    optimizer: OptimizerConfig = required
    "Optimizer specific prameters captured in this class."

    runconfig: RunConfig = required
    "Params class to define params for controlling runs."

    train_input: Optional[DataConfig] = None
    "Input params class for train mode"

    sparsity: Optional[SparsityConfig] = None
    "Params class for sparsity related cofigurations"

    def __post_init__(self):
        name = self.model.get("model_name")
        config_class_to_use = None
        config_class_to_use = registry.get_config_class(name)

        if config_class_to_use is not None:
            hints = get_type_hints(config_class_to_use)
            model_type = get_class_type(config_class_to_use, 'model')
            if model_type:

                @dataclass
                class DPOBoundModelConfig(DPOModelConfig, model_type):
                    pass

                self.model = DPOBoundModelConfig(**self.model)
            else:
                raise ValueError(
                    f"No known DPO backend model found for key {model_type}"
                )
        else:
            raise ValueError(f"No known DPO backend model found for key {name}")
        super().__post_init__()
