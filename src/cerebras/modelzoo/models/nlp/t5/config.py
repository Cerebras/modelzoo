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

import logging
from dataclasses import dataclass, field
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

DEFAULT_EMBEDDINGS_ALPHA = 1.0
DEFAULT_OUTPUT_LOGITS_ALPHA = None
DEFAULT_SCALE_OUTPUT_LOGITS_BY_D = False
DEFAULT_ENCODER_ATTENTION_LOGITS_ALPHA = 1.0
DEFAULT_DECODER_ATTENTION_LOGITS_ALPHA = 1.0


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

    attention_kernel: Optional[str] = None
    "attention kernel type."

    # Loss:
    mlm_loss_scaling: Optional[str] = "batch_size"
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

    # muP:
    mup_base_d_model: Optional[int] = None
    """The d_model of the base model in muP transfer used to calculate the
    necessary multipliers. Required in muP training when scaling the encoder/decoder
    attention module"""

    mup_base_d_ff: Optional[int] = None
    """The d_ff of the base model in muP transfer used to calculate the
    necessary multipliers. Required in muP training when scaling the encoder/decoder
    ffn"""

    mup_base_d_kv: Optional[int] = None
    """The d_kv of the base model in muP transfer used to calculate the
    necessary multipliers. Required in muP training when varying d_kv alongside
    d_model"""

    embeddings_alpha: Optional[float] = DEFAULT_EMBEDDINGS_ALPHA
    """Scales the embedding hidden states (i.e. the tensor after embeddings &
    embedding layer norm are applied). The embeddings are scaled by
    embeddings_alpha * d_model**0.5. Required in muP training"""

    output_logits_alpha: Optional[float] = DEFAULT_OUTPUT_LOGITS_ALPHA
    """Constant applied to the output logits scalar in muP training. The output
    logits are scaled by output_logits_alpha / hidden_size_width_mult"""

    scale_encoder_qk_dot_by_d: Optional[bool] = None
    """Scales encoder attention QK dot product by d instead of sqrt(d). Must
    be enabled for muP training. Note that this flag only has effect if
    muP params are specified or use_transformer_initialization==`True`"""

    scale_decoder_qk_dot_by_d: Optional[bool] = None
    """Scales decoder attention QK dot product by d instead of sqrt(d). Must
    be enabled for muP training. Note that this flag only has effect if
    muP params are specified or use_transformer_initialization==`True`"""

    encoder_attention_logits_alpha: Optional[float] = (
        DEFAULT_ENCODER_ATTENTION_LOGITS_ALPHA
    )
    """Additionally scales the encoder attention QK dot product by the specified value.
    Recommended to tune for stabilizing attention logits in muP training."""

    decoder_attention_logits_alpha: Optional[float] = (
        DEFAULT_DECODER_ATTENTION_LOGITS_ALPHA
    )
    """Additionally scales the decoder attention QK dot product by the specified value.
    Recommended to tune for stabilizing attention logits in muP training."""

    scale_output_logits_by_d: bool = DEFAULT_SCALE_OUTPUT_LOGITS_BY_D
    """Scales the output logits in muP by mup_base_d_model/d_model if
    True and sqrt(mup_base_d_model/d_model) if False. Only applies to
    muP training when scaling d_model"""

    # Misc:
    compute_eval_metrics: Optional[bool] = True
    "Computes perplexity & accuracy metrics in addition to loss"

    fp16_type: Optional[Literal["bfloat16", "float16", "cbfloat16"]] = "float16"
    "Type of 16bit precision used"

    def __post_init__(self):
        if self.tgt_vocab_size == None:
            self.tgt_vocab_size = self.src_vocab_size

        supported_mup_dimensions = [
            'mup_base_d_model',
            'mup_base_d_ff',
            'mup_base_d_kv',
        ]
        required_mup_dimensions = [
            'mup_base_d_model',
            'mup_base_d_ff',
        ]
        detected_mup_dimensions = [
            dimension
            for dimension in supported_mup_dimensions
            if getattr(self, dimension)
        ]
        if detected_mup_dimensions:
            if self.use_transformer_initialization:
                raise RuntimeError(
                    f"Detected mup base dimensions {detected_mup_dimensions}, but "
                    f"T5 only supports muP when use_transformer_initialization=`False`"
                )
            required_mup_dimensions_found = all(
                dimension in detected_mup_dimensions
                for dimension in required_mup_dimensions
            )
            if not required_mup_dimensions_found:
                raise RuntimeError(
                    f"Our muP formulation requires that you specify both "
                    f"a 'mup_base_d_model' and a 'mup_base_d_ff' "
                    f"but only the following dimensions were found: "
                    f"{detected_mup_dimensions}"
                )
        else:
            mup_tunable_params = [
                ("embeddings_alpha", DEFAULT_EMBEDDINGS_ALPHA),
                ("output_logits_alpha", DEFAULT_OUTPUT_LOGITS_ALPHA),
                ("scale_output_logits_by_d", DEFAULT_SCALE_OUTPUT_LOGITS_BY_D),
                (
                    "encoder_attention_logits_alpha",
                    DEFAULT_ENCODER_ATTENTION_LOGITS_ALPHA,
                ),
                (
                    "decoder_attention_logits_alpha",
                    DEFAULT_DECODER_ATTENTION_LOGITS_ALPHA,
                ),
            ]
            detected_mup_tunable_params = [
                param
                for param, default in mup_tunable_params
                if getattr(self, param) != default
            ]
            if detected_mup_tunable_params:
                logging.warning(
                    f"The following muP parameters were changed from their default "
                    f"value outside of a muP run: {detected_mup_tunable_params}. "
                    f"As a result, they may have an undesired effect. Please "
                    f"specify the muP base dimensions {supported_mup_dimensions} "
                    f"to trigger a muP run."
                )


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
        self.set_config_defaults(
            self.model,
            self.train_input.params if self.train_input else None,
            [self.eval_input.params if self.eval_input else None],
        )

    @staticmethod
    def set_config_defaults(mparams, tparams, eparams_list):
        if tparams is not None:
            if mparams.src_max_position_embeddings is None:
                mparams.src_max_position_embeddings = tparams.get(
                    "src_max_sequence_length"
                )
            if mparams.tgt_max_position_embeddings is None:
                mparams.tgt_max_position_embeddings = tparams.get(
                    "tgt_max_sequence_length"
                )
            tparams["dynamic_loss_weight"] = (
                mparams.mlm_loss_scaling or "batch_size"
            ) == "precomputed_num_masked"


@registry.register_submodel_config("t5forconditionalgeneration")
@dataclass
class T5ForConditionalGenerationModelConfig(ModelConfig):
    name: Optional[str] = None

    # Embedding:
    src_vocab_size: int = 32128
    "The size of the source vocabulary. Max supported value - `512000`."

    tgt_vocab_size: int = 32128
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

    src_max_position_embeddings: Optional[int] = 512
    """Maximum source sequence length that the model's position embeddings can handle.
    When not specified, the value from src_max_sequence_length (specified in the
    dataloader config) is used"""

    tgt_max_position_embeddings: Optional[int] = 512
    """Maximum target sequence length that the model's position embeddings can handle.
    When not specified, the value from tgt_max_position_embeddings (specified in the
    dataloader config) is used"""

    relative_attention_num_buckets: int = 32
    "The number of buckets to use for each attention layer."

    share_encoder_decoder_embedding: bool = True
    "Whether to share the embedding weights between the encoder and decoder."

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

    layer_norm_epsilon: float = 1.0e-6
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

    attention_kernel: Optional[str] = None

    # Loss:
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

    initializer_factor: float = 1.0

    use_cache: bool = False

    decoder_start_token_id: Optional[int] = None

    pad_token_id: int = 0

    tie_encoder_decoder: bool = False

    attention_module: Literal["aiayn_attention", "multiquery_attention"] = (
        "aiayn_attention"
    )

    extra_attention_params: dict = field(default_factory=dict)

    extra_ids: Optional[int] = 0
    "Number of sentinel tokens for T5 objective"

    def __post_init__(self):
        super().__post_init__()

        if self.tgt_vocab_size == None:
            self.tgt_vocab_size = self.src_vocab_size
