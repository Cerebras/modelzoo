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

from cerebras.modelzoo.config_manager.config_classes.base.base_config import *
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
from cerebras.modelzoo.config_manager.config_validators import *


@dataclass
class BertForSequenceClassificationModelConfig(ModelConfig):
    num_labels: int = required
    problem_type: str = required
    task_dropout: float = required

    embedding_dropout_rate: float = 0.1
    is_mnli_dataset: bool = False
    dropout_rate: float = 0.1

    vocab_size: int = 30522
    "The size of the vocabulary used in the model. Max supported value - `512000`."

    attention_dropout_rate: Optional[float] = 0.1
    "Dropout rate for attention layer. Default - same as `dropout`"

    num_heads: Optional[int] = 12
    "The number of attention heads in the multi-head attention layer."

    max_position_embeddings: int = 1024
    "The maximum sequence length that the model can handle."

    hidden_size: int = 768
    "The size of the transformer hidden layers."

    num_hidden_layers: int = 12
    "Number of hidden layers in the Transformer encoder/decoder."

    layer_norm_epsilon: float = 1e-5
    "The epsilon value used in layer normalization layers."

    filter_size: int = 3072
    "Dimensionality of the feed-forward layer in the Transformer block."

    encoder_nonlinearity: Literal["gelu", "relu", "silu", "gelu_new"] = "gelu"
    pooler_nonlinearity: Optional[str] = None
    compute_eval_metrics: Optional[bool] = None


@registry.register_config("classifier")
@dataclass
class BertForSequenceClassificationConfig(BaseConfig):
    train_input: Optional[DataConfig] = None
    "Input params class for train mode"

    eval_input: Optional[DataConfig] = None
    "Input params class for eval mode"

    model: BertForSequenceClassificationModelConfig = required
    "Model level params class. Supported params differ for each model."

    optimizer: OptimizerConfig = required
    "Optimizer specific prameters captured in this class."

    runconfig: RunConfig = required
    "Params class to define params for controlling runs."

    sparsity: Optional[SparsityConfig] = None
    "Params class for sparsity related cofigurations"
