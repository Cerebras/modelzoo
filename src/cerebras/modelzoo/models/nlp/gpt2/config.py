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

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.config_manager.config_classes.base.base_config import (
    BaseConfig,
    required,
)
from cerebras.modelzoo.config_manager.config_classes.base.data_config import (
    DataConfig,
)
from cerebras.modelzoo.config_manager.config_classes.base.model_config import (
    InitializerConfig,
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
class GPT2ModelConfig(ModelConfig):
    # Embedding:
    vocab_size: int = required
    "The size of the vocabulary used in the model. Max supported value - 512000."

    embedding_layer_norm: bool = False
    "Apply normalization to embeddings"

    embedding_dropout_rate: Optional[float] = None
    "Dropout rate for embeddings. When none, `dropout_rate` is used."

    share_embedding_weights: bool = True
    "Whether to share the embedding weights between the input and out put embedding."

    position_embedding_type: Optional[
        Literal["learned", "fixed", "relative", "rotary", "alibi"]
    ] = "learned"
    """The type of position embedding to use in the model.
    Can be one of - `learned` - Learned embedding matrix
    `fixed` - Sinusoidal from original
    [Transformer](https://arxiv.org/abs/1706.03762),
    `relative` - Relative position embedding
    [to exploit pairwise, relative positional information](https://arxiv.org/abs/1803.02155).,
    `rotary` - a.k.a [RoPE](https://arxiv.org/pdf/2104.09864v4.pdf),
    `alibi` [Attention With Linear Biases](https://arxiv.org/pdf/2108.12409.pdf),
    or `None` for no position embeddings.
    """

    max_position_embeddings: int = 1024
    "The maximum sequence length that the model can handle."

    position_embedding_offset: int = 0
    "Position offset for learned embeddings"

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

    # Transformer:
    hidden_size: int = 768
    "The size of the transformer hidden layers."

    num_hidden_layers: int = 12
    "Number of hidden layers in the Transformer decoder."

    dropout_rate: float = 0.1
    "The dropout probability for all fully connected layers."

    norm_type: str = "layernorm"
    "Determines the type of normalization. See modelzoo/layers/norms.py"

    layer_norm_epsilon: float = 1e-5
    "The epsilon value used in layer normalization layers."

    # Transformer Attention:
    num_heads: int = 12
    "The number of attention heads."

    attention_module: Literal[
        "aiayn_attention", "multiquery_attention"
    ] = "aiayn_attention"
    """Determines whether to use multiheaded attention (from the Attention is
    All You Need paper) or multi-query/grouped-query attention. When using the
    latter, you must specify extra_attention_params (see below).
    """

    extra_attention_params: dict = field(default_factory=dict)
    """When enabling multi-query/grouped-query attention, you must specify the
    the number of key-value groups. Within the extra attention params dict, you
    can set `num_kv_groups: 1` to enable MQA or `num_kv_groups: <groups>` for
    GQA. The number of groups should be divisible by `num_heads`.
    """

    attention_type: Literal[
        "dot_product", "scaled_dot_product"
    ] = "scaled_dot_product"
    """Determines whether the QK dot product should be scaled -
    dot_product -> QK^T 
    scaled_dot_product -> QK^T / sqrt(d) 

    Note that setting either scale_qk_dot_by_d or scale_qk_dot_by_layer_idx will
    result in different behavior.
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

    attention_kernel: Optional[str] = None
    """The specific attention kernel implementation to use. All implementations are functionally
     the same but may be compiled differently.
    """

    attention_sliding_window_length: Optional[int] = None
    "If specified, sliding window attention is used (as seen in Mistral)."

    scale_qk_dot_by_layer_idx: bool = False
    """Scales the attention QK dot product by the layer index (as seen in Santacoder)
    Note that using this flag in conjunction with attention_type=scaled_dot_product
    will result in scaling by both: QK^T / (sqrt(d) * (layer idx + 1))"""

    fixed_sparse_attention: Optional[dict] = None
    "Applies a fixed sparse attention mask in attention. See GPT-3 configs for examples."

    # Transformer FFN:
    filter_size: int = 3072
    "Dimensionality of the feed-forward layer in the Transformer block."

    nonlinearity: str = "gelu"
    """The non-linear activation function used in the feed forward network
    in each transformer block.
    See list of non-linearity functions [here](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity).
    """

    use_ffn_bias: bool = True
    "Whether to use bias in the feedforward network (FFN)."

    use_bias_in_output: bool = False
    "Whether to use bias in the final output layer."

    # Loss:
    loss_scaling: Literal["batch_size", "num_tokens"] = "num_tokens"
    """The scaling type used to calculate the loss. Accepts - `batch_size`, `num_tokens`.
    See [more](https://docs.cerebras.net/en/latest/wsc/general/num-tokens-loss-scaling.html).
    **Note:** It is recommended to set this to `num_tokens` for convenience."""

    loss_weight: float = 1.0
    """The weight for the loss scaling when `loss_scaling = 'batch_size'`, generally set to
    '1/max_sequence_length`.
    """

    # muP:
    embeddings_scale: Optional[float] = None
    """Scales the embedding hidden states (i.e. the tensor after embeddings &
    embedding layer norm are applied). Required in muP training"""

    scale_qk_dot_by_d: bool = False
    """Scales attention QK dot product by d instead of sqrt(d). Must be enabled
    for muP training. Note that this flag only has effect if
    attention_type=scaled_dot_product"""

    output_logits_scale: Optional[float] = None
    "Scales model's output logits (required for muP training)"

    # Initializers:
    initializer: Optional[InitializerConfig] = None
    """The initializer to be used for all the initializers used in the model.
    See [supported initializers]"
    "(./common/pytorch/model_utils/create_initializer.py). Default - varies based on model"""

    initializer_range: float = 0.02
    "The standard deviation of the truncated_normal_initializer as the default initializer"

    embedding_initializer: Optional[InitializerConfig] = None
    """Initializer to use for embeddings. See [supported initializers]
    (./common/pytorch/model_utils/create_initializer.py). Default - 'normal'
    """

    output_layer_initializer: Optional[InitializerConfig] = None
    """The name of the initializer for the weights of the output layer.
    See [supported initializers](./common/pytorch/model_utils/create_initializer.py)."""

    # Misc:
    compute_eval_metrics: bool = True
    "Computes perplexity & accuracy metrics in addition to loss"

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

        if self.embedding_dropout_rate is None:
            self.embedding_dropout_rate = self.dropout_rate


@registry.register_config("gpt2")
@dataclass
class GPT2Config(BaseConfig):
    train_input: Optional[DataConfig] = None
    "Dataloader configuration for train mode"

    eval_input: Optional[DataConfig] = None
    "Dataloader configuration for eval mode"

    model: GPT2ModelConfig = required
    "Model architecture configuration"

    sparsity: Optional[SparsityConfig] = None
    optimizer: OptimizerConfig = required
    runconfig: RunConfig = required


@dataclass
class GptInferenceModelConfig(GPT2ModelConfig):
    start_token: Union[int, List[int]] = required
    loop_dim: int = 1
    stop_sequences: Union[int, List[List[int]]] = required
    max_tokens: Optional[int] = None


@registry.register_config("gpt_inference")
@dataclass
class GPT2InferenceConfig(BaseConfig):
    train_input: Optional[DataConfig] = None
    "Dataloader configuration for train mode"

    eval_input: Optional[DataConfig] = None
    "Dataloader configuration for eval mode"

    inference_input: Optional[DataConfig] = None
    "Dataloader configuration for inference mode"

    model: GptInferenceModelConfig = required
    "Model architecture configuration"

    sparsity: Optional[SparsityConfig] = None
    optimizer: OptimizerConfig = required
    runconfig: RunConfig = required
