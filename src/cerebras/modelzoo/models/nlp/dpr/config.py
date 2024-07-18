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
from typing import Literal, Optional

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
from cerebras.modelzoo.models.nlp.bert.config import BertModelConfig


@dataclass
class DPREncoderConfig(BertModelConfig):
    # Includes the same Bert model params + the following:
    add_pooling_layer: bool = False


@dataclass
class DPRModelConfig(ModelConfig):
    q_encoder: Optional[DPREncoderConfig] = None
    "Encoder for question in biencoder model (e.g., DPR)"
    ctx_encoder: Optional[DPREncoderConfig] = None
    "Encoder for context in biencoder model (e.g., DPR)"
    encoder: Optional[DPREncoderConfig] = None
    """
    Encoder for both question and context model.
    - If `encoder` is already provided, users should not provide 
    `q_encoder` and `ctx_encoder` in the same config file. 
    - Simply providing `encoder` doesn't automatically make the architecture a uni-encoder model;
    instead, the users should explicitly set `use_biencoder` to be False. Otherwise, a bi-encoder
    model will be instantiated with question & context encoders have the same config.
    """
    softmax_temperature: float = 1.0
    "Divide the score matrix by temperature before softmax computation"
    mutual_information: bool = False
    "Whether to add context-to-question loss in addition to question-to-context loss"
    use_biencoder: bool = True
    "Use uniencoder or biencoder architecture"
    pooler_type: Literal["mean", "cls", "ffn_pooler"] = "cls"
    """Pooler method for generating sequence embedding out of output token embeddings.
    Can be one of - 
    `mean` -  average all token embeddings as the final sequence embedding, 
    `fixed` - use the token embedding of the [CLS] token as the final sequence embedding", 
    `ffn_pooler` -  apply an additional linear layer on top of the token embedding of the [CLS] token as the final sequence embedding
    """
    compute_eval_metrics: bool = False
    "Computes accuracy metrics in addition to loss"
    selected_encoder: Optional[
        Literal["q_encoder", "ctx_encoder", "encoder"]
    ] = None
    "Select which encoder to use in embedding_generation. This field is only used in embedding_generation."
    fp16_type: Optional[Literal["bfloat16", "float16", "cbfloat16"]] = (
        "bfloat16"
    )
    "Type of 16bit precision used"

    def __post_init__(self):
        super().__post_init__()
        valid_biencoder_config = (
            self.q_encoder and self.ctx_encoder and not self.encoder
        )
        valid_uniencoder_config = (
            not self.q_encoder and not self.ctx_encoder and self.encoder
        )
        assert (
            valid_uniencoder_config or valid_biencoder_config
        ), "Either provide both q_encoder and ctx_encoder, or only encoder in config"
        if not self.use_biencoder:
            assert (
                valid_uniencoder_config
            ), "If uniencoder is used, only provide encoder attribute in config"


@registry.register_config("dpr")
@dataclass
class DPRConfig(BaseConfig):
    train_input: Optional[DataConfig] = None
    "Dataloader configuration for train mode"

    eval_input: Optional[DataConfig] = None
    "Dataloader configuration for eval mode"

    model: DPRModelConfig = required
    "Model architecture configuration"

    sparsity: Optional[SparsityConfig] = None
    optimizer: OptimizerConfig = required
    runconfig: RunConfig = required
