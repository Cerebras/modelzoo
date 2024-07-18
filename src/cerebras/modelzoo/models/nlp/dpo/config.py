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
    get_class_type,
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
from cerebras.modelzoo.models.nlp.dpo.model import MODEL_MAPPING


@dataclass
class DPOParameters(BaseConfig):
    beta: float = 0.1
    reference_free: bool = False
    loss_type: str = "sigmoid"
    disable_dropout: bool = True


@dataclass
class DPOModelConfig(ModelConfig):
    model_name: str = required
    "Name of the model to use with DPO"

    dpo: DPOParameters = required
    "Parameters for DPO configuration"

    compute_eval_metrics: bool = True

    def __new__(cls, **kwargs):
        # Avoid infinite recursion
        if cls is not DPOModelConfig:
            return super().__new__(cls)

        if "model_name" not in kwargs:
            raise ValueError(f"DPO config requires a \"model_name\" key.")

        name = kwargs["model_name"]

        config_class_to_use = registry.get_config_class(MODEL_MAPPING[name])
        if config_class_to_use is None:
            raise ValueError(f"No known DPO backend model found for key {name}")

        model_type = get_class_type(config_class_to_use, 'model')
        if not model_type:
            raise ValueError(
                f"No known DPO backend model found for key {model_type}"
            )

        @dataclass
        class DPOBoundModelConfig(DPOModelConfig, model_type):
            pass

        instance = DPOBoundModelConfig(**kwargs)
        model_type.__post_init__(instance)
        return instance


@dataclass
class DPODataConfig(DataConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __post_init__(self):
        super().__post_init__()
        self.params["data_processor"] = self.data_processor


@registry.register_config("dpo")
@dataclass
class DPOConfig(BaseConfig):
    eval_input: Optional[DPODataConfig] = None
    "Input params class for eval mode"

    model: DPOModelConfig = required
    "Model level params class. Supported params differ for each model."

    optimizer: OptimizerConfig = required
    "Optimizer specific prameters captured in this class."

    runconfig: RunConfig = required
    "Params class to define params for controlling runs."

    train_input: Optional[DPODataConfig] = None
    "Input params class for train mode"

    sparsity: Optional[SparsityConfig] = None
    "Params class for sparsity related cofigurations"

    def __post_init__(self):
        super().__post_init__()
        self.set_config_defaults(
            self.model,
            self.train_input.params if self.train_input else None,
            [self.eval_input.params if self.eval_input else None],
        )

    @staticmethod
    def set_config_defaults(mparams, tparams, eparams_list):
        config_class_to_use = registry.get_config_class(
            MODEL_MAPPING[mparams.model_name]
        )
        if config_class_to_use is None:
            raise ValueError(
                f"No known DPO backend model found for key {mparams.model_name}"
            )

        if fn := getattr(config_class_to_use, "set_config_defaults", None):
            fn(mparams, tparams, eparams_list)
