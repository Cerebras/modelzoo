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
import os
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

DEFAULT_EMBEDDINGS_SCALE = 1.0
DEFAULT_OUTPUT_LOGITS_ALPHA = None
DEFAULT_SCALE_OUTPUT_LOGITS_BY_D = True
DEFAULT_ATTENTION_LOGITS_ALPHA = 1.0


@dataclass
class BertModelConfig(ModelConfig):
    # Embedding:
    vocab_size: int = required
    "The size of the vocabulary used in the model. Max supported value - `512000`."

    share_embedding_weights: bool = True
    "Whether to share the embedding weights between the input and output embedding."

    num_segments: Optional[int] = None
    """Number of segments (token types) in embedding. When not specified
    (and NSP objective is enabled), num_segments will default to 2"""

    position_embedding_type: Optional[
        Literal["learned", "fixed", "relative", "rotary", "alibi"]
    ] = "learned"
    """The type of position embedding to use in the model.
    Can be one of - `learned` - Learned embedding matrix,
    `fixed` - Sinusoidal from original [Transformer](https://arxiv.org/abs/1706.03762),
    `relative` - Relative position embedding
    [to exploit pairwise, relative positional information](https://arxiv.org/abs/1803.02155).,
    `rotary` - a.k.a [RoPE](https://arxiv.org/pdf/2104.09864v4.pdf),
    `alibi` [Attention With Linear Biases](https://arxiv.org/pdf/2108.12409.pdf),
    or `None` for no position embeddings.
    """

    max_position_embeddings: Optional[int] = None
    "The maximum sequence length that the model can handle."

    pad_token_id: int = 0
    "The embedding vector at pad_token_id is not updated during training."

    mask_padding_in_positional_embed: bool = False
    """Whether to mask padding in positional embeddings.
    Only supported with `position_embedding_type` set to `learned`."""

    num_relative_attention_buckets: int = 32
    "Number of buckets to use in relative position embedding"

    rotary_dim: Optional[int] = None
    "The number of dimensions used for the rotary position embedding."

    rope_theta: float = 10000
    """Frequency (theta) used in rotary position embedding. This value is
    typically adjusted in long MSL runs as described in
    [CodeLlama](https://arxiv.org/pdf/2308.12950.pdf)"""

    pad_rope: bool = False

    alibi_trainable_slopes: bool = False
    "Replaces alibi's fixed slopes with trainable slopes."

    pos_scaling_factor: float = 1.0
    """Position interpolation scaling factor for rotary & alibi. See
    https://arxiv.org/pdf/2306.15595.pdf for details"""

    pos_scaling_type: str = "linear"
    """Can be either `linear` or `YaRN`,
    For YaRN see https://arxiv.org/pdf/2309.00071"""

    pos_scaling_extra_args: Optional[dict] = None
    """A dict including parameters for YaRN RoPE scaling"""

    # Transformer:
    hidden_size: int = 768
    "The size of the transformer hidden layers."

    num_hidden_layers: int = 12
    "Number of hidden layers in the Transformer decoder."

    dropout_rate: float = 0.1
    "The dropout probability for all fully connected layers."

    layer_norm_epsilon: float = 1e-5
    "The epsilon value used in layer normalization layers."

    # Transformer Attention:
    num_heads: int = 12
    "The number of attention heads."

    attention_module: Literal["aiayn_attention", "multiquery_attention"] = (
        "aiayn_attention"
    )
    """Determines whether to use multiheaded attention (from the Attention is
    All You Need paper) or multi-query/grouped-query attention. When using the
    latter, you must specify extra_attention_params (see below).
    """

    extra_attention_params: dict = field(default_factory=dict)
    """When enabling multi-query/grouped-query attention, you must specify the
    the number of key-value groups. Within the extra attention params dict, you
    can set `num_kv_groups = 1` to enable MQA or `num_kv_groups = <groups>` for
    GQA. The number of groups should be divisible by `num_heads`.
    """

    attention_type: Literal["dot_product", "scaled_dot_product"] = (
        "scaled_dot_product"
    )
    """Determines whether the QK dot product should be scaled -
    dot_product -> QK^T
    scaled_dot_product -> QK^T / sqrt(d)
    """

    attention_dropout_rate: Optional[float] = None
    "Dropout rate for attention layer. When None, defaults to same as `dropout_rate`"

    use_projection_bias_in_attention: bool = True
    "Whether to include bias in the attention layer for Q/K/V projections."

    use_ffn_bias_in_attention: bool = True
    """Whether to include bias in the attention layer for output projection
    after values have been combined (W_O in original Transformer paper).
    """

    attention_softmax_fp32: bool = True
    "Whether to use fp32 precision for attention softmax."

    # Transformer FFN:
    filter_size: int = 3072
    "Dimensionality of the feed-forward layer in the Transformer block."

    encoder_nonlinearity: str = "gelu"
    """The non-linear activation function used in the feed forward network
    in each transformer block.
    See list of non-linearity functions [here](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity).
    """

    mlm_nonlinearity: Optional[str] = None
    """The non-linear activation function used in the MLM head. If not
    specified, defaults to encoder_nonlinearity."""

    pooler_nonlinearity: Optional[str] = None
    """The non-linear activation function used in the pooler layer. If not
    specified, defaults to encoder_nonlinearity."""

    attention_kernel: Optional[str] = None
    """The type of attention kernel"""

    use_ffn_bias: bool = True
    "Whether to use bias in the feedforward network (FFN)."

    use_ffn_bias_in_mlm: bool = True
    "Whether to use bias in MLM head's FFN layer"

    use_output_bias_in_mlm: bool = True
    "Whether to use bias in MLM head's output (classifier) layer"

    # Loss:
    mlm_loss_weight: float = 1.0
    """Value that scales the Masked Language Modelling (MLM) loss
    """

    label_smoothing: float = 0.0
    "The label smoothing factor used during training."

    # Task-specific:
    disable_nsp: Optional[bool] = False
    """Disables Next Sentence Prediction (NSP) objective"""

    num_classes: int = 2
    """Number of classes used by the classifier head (NSP)"""

    # Initializers:
    initializer_range: float = 0.02
    "The standard deviation of the truncated_normal_initializer as the default initializer"

    # Misc:
    compute_eval_metrics: bool = True
    "Computes perplexity & accuracy metrics in addition to loss"

    freeze_ffn_bias_in_glu: bool = False
    "Prevents gradients from being computed for FFN biases for GLU activation layers"
    # muP:
    mup_base_hidden_size: Optional[float] = None
    """The hidden size of the base model in muP transfer used to calculate the
    necessary multipliers. Required in muP training when scaling the encoder
    attention module"""

    mup_base_filter_size: Optional[float] = None
    """The filter size of the base model in muP transfer used to calculate the
    necessary multipliers. Required in muP training when scaling the encoder
    ffn"""

    embeddings_scale: Optional[float] = DEFAULT_EMBEDDINGS_SCALE
    """Scales the embedding hidden states (i.e. the tensor after embeddings &
    embedding layer norm are applied). Required in muP training"""

    output_logits_alpha: Optional[float] = DEFAULT_OUTPUT_LOGITS_ALPHA
    """Constant applied to the output logits scalar in muP training. The msm
    and nsp logits are scaled by output_logits_alpha * mup_base_hidden_size/hidden_size"""

    scale_qk_dot_by_d: Optional[bool] = None
    """Scales attention QK dot product by d instead of sqrt(d). Must be enabled
    for muP training."""

    attention_logits_alpha: Optional[float] = DEFAULT_ATTENTION_LOGITS_ALPHA
    """Additionally scales the attention QK dot product by the specified value.
    Recommended to tune for stabilizing attention logits in muP training."""

    scale_output_logits_by_d: Optional[bool] = DEFAULT_SCALE_OUTPUT_LOGITS_BY_D
    """Scales the output logits in muP by mup_base_hidden_size/hidden_size if
    True and sqrt(mup_base_hidden_size/hidden_size) if False. Only applies to
    muP training when scaling the hidden_size"""

    fp16_type: Literal["bfloat16", "float16", "cbfloat16"] = "float16"
    "Type of 16bit precision used"

    def __post_init__(self):
        super().__post_init__()
        if self.position_embedding_type == "rotary":
            if self.rotary_dim == None:
                self.rotary_dim = int(self.hidden_size // self.num_heads * 0.25)
            # https://github.com/huggingface/transformers/blob/f0577df6de36e7e7f28e90fa76da0657de038a39/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L84-L85
            # https://arxiv.org/pdf/2104.09864.pdf Section 3.3
            assert (
                self.rotary_dim <= self.hidden_size / self.num_heads
            ), "Rotary dimensions should be <= hidden size divided by number of attention heads."
            assert (
                self.rotary_dim % 2 == 0
            ), "Rotary dimension must be an even number."
        if self.num_segments is None:
            self.num_segments = None if self.disable_nsp else 2

        supported_mup_dimensions = [
            'mup_base_hidden_size',
            'mup_base_filter_size',
        ]

        detected_mup_dimensions = [
            dimension
            for dimension in supported_mup_dimensions
            if getattr(self, dimension)
        ]

        if detected_mup_dimensions:
            if detected_mup_dimensions != supported_mup_dimensions:
                raise RuntimeError(
                    f"Our muP formulation requires that you specify all "
                    f"of the following base dimensions: {supported_mup_dimensions} "
                    f"but only the following dimensions were found: "
                    f"{detected_mup_dimensions}"
                )
        else:
            mup_tunable_params = [
                ("embeddings_scale", DEFAULT_EMBEDDINGS_SCALE),
                ("output_logits_alpha", DEFAULT_OUTPUT_LOGITS_ALPHA),
                ("scale_output_logits_by_d", DEFAULT_SCALE_OUTPUT_LOGITS_BY_D),
                ("attention_logits_alpha", DEFAULT_ATTENTION_LOGITS_ALPHA),
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


@dataclass
class BertDataConfig(DataConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __post_init__(self):
        if self.params.get("vocab_file", None):
            self.params["vocab_file"] = os.path.abspath(
                self.params["vocab_file"]
            )

        super().__post_init__()


@registry.register_config("bert")
@dataclass
class BertConfig(BaseConfig):
    train_input: Optional[BertDataConfig] = None
    "Dataloader configuration for train mode"

    eval_input: Optional[BertDataConfig] = None
    "Dataloader configuration for eval mode"

    model: BertModelConfig = required
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
        # Pass settings into data loader sections directly
        for model_key in ("disable_nsp", "vocab_size", "mixed_precision"):
            for input_key in [tparams, *eparams_list]:
                if input_key is not None:
                    input_key[model_key] = getattr(mparams, model_key, None)

        if tparams is not None:
            mparams.max_position_embeddings = (
                mparams.max_position_embeddings
                or tparams["max_sequence_length"]
            )
