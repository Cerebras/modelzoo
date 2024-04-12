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
class T5ModelConfig(ModelConfig):
    # Embedding:
    src_vocab_size: int = 32128
    "The size of the source vocabulary. Max supported value - `512000`."

    tgt_vocab_size: int = None
    """The size of the target vocabulary. Max supported value - `512000`. 
    When not provided, same as src_vocab_size"""

    share_embedding_weights: bool = True
    "Whether to share the embedding weights between the input and out put embedding."

    position_embedding_type: Optional[
        Literal["learned_absolute", "fixed", "relative", "rotary", "alibi"]
    ] = "relative"
    """The type of position embedding to use in the model. Can be one of -
    `fixed` - Sinusoidal from original [Transformer](https://arxiv.org/abs/1706.03762),
    `relative` - Relative position embedding, [to exploit pairwise, relative positional 
                 information](https://arxiv.org/abs/1803.02155).,
    `rotary` - a.k.a [RoPE](https://arxiv.org/pdf/2104.09864v4.pdf) ,
    `learned_absolute` - Learned embedding matrix,
    `None` 
    """

    src_max_position_embeddings: Optional[int] = None
    """Maximum source sequence length that the model's position embeddings can handle.
    When not specified, the value from src_max_sequence_length (specified in the
    dataloader config) is used"""

    tgt_max_position_embeddings: Optional[int] = None
    """Maximum target sequence length that the model's position embeddings can handle.
    When not specified, the value from tgt_max_position_embeddings (specified in the
    dataloader config) is used"""

    relative_attention_num_buckets: int = 32
    "The number of buckets to use for each attention layer."

    share_encoder_decoder_embedding: bool = True
    "Whether to share the embedding weights between the encoder and decoder."

    extra_ids: Optional[int] = 0
    "Number of sentinel tokens for T5 objective"

    # Transformer:
    d_model: int = 512
    "The number of features (hidden dimensionality) of the transformer."

    d_kv: int = 64
    """Size of the query/key/value projections per attention head. `d_kv` does
    *not* have to be equal to `d_model//num_heads`.
    """

    d_ff: int = 2048
    "Size of the intermediate feed forward layer in each `T5Block`."

    encoder_num_hidden_layers: int = 6
    "Number of hidden layers in the encoder."

    decoder_num_hidden_layers: Optional[int] = None
    """Number of hidden layers in the Transformer decoder. Will use the same
    value as `encoder_num_hidden_layers` if not set.
    """

    norm_type: str = "rmsnorm"
    "Determines the type of normalization. See modelzoo/layers/norms.py"

    dropout_rate: float = 0.1
    "The dropout probability for all fully connected layers."

    layer_norm_epsilon: float = 1.0e-5
    "The epsilon value used in layer normalization layers."

    encoder_nonlinearity: str = "relu"
    "Type of nonlinearity to be used in encoder."

    decoder_nonlinearity: Optional[str] = "relu"
    """Type of nonlinearity to be used in decoder. If decoder_nonlinearity isn't 
    provided, it will be the same as encoder_nonlinearity"""

    relu_dropout_rate: Optional[float] = None
    "The dropout rate for ReLU activation function."

    use_pre_encoder_decoder_dropout: bool = False
    "Whether to use dropout layer after positional embedding layer and encoder/decoder."

    use_pre_encoder_decoder_layer_norm: bool = True
    "Whether to use layer norm before passing input tensors into encoder/decoder."

    use_ffn_bias: bool = False
    "Whether to use bias in the feedforward network (FFN)."

    # Transformer Attention:
    num_heads: int = 8
    "The number of attention heads in the multi-head attention layer."

    use_projection_bias_in_attention: bool = False
    "Whether to include bias in the attention layer for projection."

    attention_softmax_fp32: bool = True
    "Whether to use fp32 precision for attention softmax."

    # Loss:
    mlm_loss_scaling: str = "batch_size"
    """A string specifying the scaling factor type used for the language modeling loss.
    Accepts one of - `"num_masked"` - uses the off-the shelf loss scaling by number
    of valid (non-padding) tokens the cross entropy loss function,
    `"precomputed_num_masked"` - uses loss scaling from the computed num valid
    masks in the data loader, when enabling
    """

    lm_loss_weight: float = 1.0
    """Value that scales loss by the mean number of predictions per sequence in the dataset.
    This number varies per dataset and can be calculated by getting the reciprocal of
    average number of tokens per sequence in the training dataset. This is only needed
    when setting loss scaling to `"batch_size"`."""

    label_smoothing: float = 0.0
    "The label smoothing factor used during training."

    use_dropout_outside_residual_path: bool = True
    "Whether to set dropout calculations outside of the residual path."

    # Initialization:
    use_transformer_initialization: bool = False
    """The Transformer model tends to converge best with a scaled variant on Xavier uniform
    initialization used for linear layers. This contrasts the initialization used for the
    original T5 paper, which uses He normal initialization for linear layers. Setting this
    flag to `True` switches the initialization to the Transformer specific scaled Xavier
    initialization."""

    # Misc:
    compute_eval_metrics: Optional[bool] = True
    "Computes perplexity & accuracy metrics in addition to loss"

    fp16_type: Optional[Literal["bfloat16", "float16", "cbfloat16"]] = "float16"
    "Type of 16bit precision used"

    def __post_init__(self):
        if self.tgt_vocab_size == None:
            self.tgt_vocab_size = self.src_vocab_size


@registry.register_config("t5")
@dataclass
class T5Config(BaseConfig):
    train_input: Optional[DataConfig] = None
    "Dataloader configuration for train mode"

    eval_input: Optional[DataConfig] = None
    "Dataloader configuration for eval mode"

    model: T5ModelConfig = required
    "Model architecture configuration"

    sparsity: Optional[SparsityConfig] = None
    optimizer: OptimizerConfig = required
    runconfig: RunConfig = required

    def __post_init__(self):
        super().__post_init__()
        if self.model.src_max_position_embeddings is None:
            self.model.src_max_position_embeddings = self.train_input.get(
                "src_max_sequence_length"
            )
        if self.model.tgt_max_position_embeddings is None:
            self.model.tgt_max_position_embeddings = self.train_input.get(
                "tgt_max_sequence_length"
            )
