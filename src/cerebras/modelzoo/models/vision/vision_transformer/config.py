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
from typing import List, Literal, Optional

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
class VisionTransformerModelConfig(ModelConfig):
    # Embedding
    embedding_dropout_rate: float = 0.0
    "Dropout rate for embeddings."

    patch_size: List[int] = field(default_factory=lambda: [16, 16])
    "Size of patches to use when converting input image to patches, [Height, Width]."

    pooler_nonlinearity: Optional[str] = None
    """The non-linear activation function used in the pooler layer. When left as None, uses
    `nonlinearity`."""

    position_embedding_initializer: Optional[dict] = None
    """Initializer for position embedding layer. Either a string indicating the name of
    the initializer or a dict that includes the name + other params if relevant. If left
    unspecified will apply truncated normal initialization."""

    position_embedding_type: str = "learned"
    """The type of position embedding to use in the model. Can be one of:
    `fixed` - Sinusoidal from original [Transformer](https://arxiv.org/abs/1706.03762),
    `relative` - Relative position embedding, [to exploit pairwise, relative positional
                 information](https://arxiv.org/abs/1803.02155).,
    `rotary` - a.k.a [RoPE](https://arxiv.org/pdf/2104.09864v4.pdf) ,
    `learned` - Learned embedding matrix,
    `None` """

    use_conv_patchified_embedding: bool = False
    "If True, use conv2D to convert image to patches."

    use_embed_proj_bias: bool = True
    "If True, adds bias to position embeddings."

    image_size: List[int] = field(default_factory=lambda: [224, 224])
    "Input image size [Height, Width]."

    prepend_cls_token: bool = True
    "If True, prepends cls token to input."

    # Transformer Attention
    attention_dropout_rate: float = 0.0
    "Dropout rate for attention layer."

    attention_module: str = "aiayn_attention"
    """Determines whether to use multiheaded attention (from the Attention is
    All You Need paper) or multi-query attention (MQA) or grouped-query
    attention (GQA). Note that when using MQA/GQA, you must specify
    extra_attention_params (see below). MQA/GQA are differentiated through
    that parameter. Can be one of: ["aiayn_attention", "multiquery_attention"]
    """

    attention_softmax_fp32: bool = True
    "Whether to use fp32 precision for attention softmax."

    attention_type: str = "scaled_dot_product"
    "Type of attention. Accepted values: [dot_product, scaled_dot_product]."

    use_projection_bias_in_attention: bool = True
    "Whether to include bias in the attention layer for projection."

    use_ffn_bias_in_attention: bool = True
    "Whether to include bias in the attention layer for feed-forward network (FFN)."

    extra_attention_params: dict = field(default_factory=dict)
    """When enabling MQA/GQA, you must specify the the number of key-value
    groups. Within the extra attention params dict, you can set
    `num_kv_groups - 1` to enable MQA or `num_kv_groups - <groups>` for
    GQA. The number of groups should be divisible by `num_heads`.
    """

    # Transformer
    dropout_rate: float = 0.0
    "The dropout probability for all fully connected layers."

    filter_size: Optional[int] = 3072
    "Dimensionality of the feed-forward layer in the Transformer block."

    hidden_size: Optional[int] = 768
    "The size of the transformer hidden layers."

    layer_norm_epsilon: float = 1.00e-05
    "The epsilon value used in layer normalization layers."

    mixed_precision: bool = True
    "Whether to use mixed precision training or not."

    nonlinearity: str = "gelu"
    """The non-linear activation function used in the feed forward network
    in each transformer block. Some may have to use autogen_policy: `medium`."""

    num_channels: int = 2
    "Number of input channels"

    num_classes: int = 2
    "Number of possible classes."

    num_heads: int = 12
    "The number of attention heads in the multi-head attention layer."

    num_hidden_layers: int = 12
    "Number of hidden layers in the Transformer encoder."

    norm_first: bool = True
    """Enables normalization before the Attention & FFN blocks (i.e Pre-LN as
    described in https://arxiv.org/pdf/2002.04745.pdf. When disabled,
    normalization is applied *after* the residual (Post-LN)"""

    use_encoder_pooler_layer: bool = False
    """If True, applies a linear transformation to the element in the encoder output
    corresponding to the first input token"""

    use_ffn_bias: Optional[bool] = True
    "Whether to use bias in the feedforward network (FFN)."

    use_post_embed_layer_norm: Optional[bool] = False
    "If True, applies layer norm to the embeddings."

    # Initialization
    attention_initializer: Optional[str] = None
    "Attention layer initializer. Defaults to `xavier_uniform`."

    initializer_range: float = 0.02
    """The standard deviation of the truncated_normal_initializer as the
    default initializer"""

    projection_initializer: Optional[dict] = None
    """Initializer for embedding linear layer. Either a string indicating the name of the
    initializer or a dict that includes the name + other params if relevant. If left
    unspecified will apply truncated normal initialization."""
    ffn_initializer: Optional[dict] = None
    pooler_initializer: Optional[dict] = None
    use_final_layer_norm: Optional[bool] = True
    use_bias_in_output: Optional[bool] = True
    compute_eval_metrics: bool = True
    fp16_type: Literal["bfloat16", "float16", "cbfloat16"] = "bfloat16"
    "Type of 16bit precision used"


@registry.register_submodel_config("vit")
@dataclass
class ViTModelConfig(BaseConfig):
    name: Optional[str] = None
    attention_dropout_rate: float = 0.0
    attention_initializer: Optional[str] = None
    attention_module: str = "aiayn_attention"
    attention_softmax_fp32: bool = True
    attention_type: str = "scaled_dot_product"
    dropout_rate: float = 0.0
    embedding_dropout_rate: float = 0.0
    extra_attention_params: Optional[dict] = field(default_factory=dict)
    filter_size: Optional[int] = 3072
    hidden_size: Optional[int] = 768
    image_size: List[int] = field(default_factory=lambda: [224, 224])
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1.00e-05
    nonlinearity: str = "gelu"
    num_channels: Optional[int] = 3
    num_heads: int = 12
    num_hidden_layers: int = 12
    norm_first: bool = True
    patch_size: Optional[List[int]] = field(default_factory=lambda: [16, 16])
    pooler_nonlinearity: Optional[str] = None
    position_embedding_initializer: Optional[dict] = None
    position_embedding_type: str = "learned"
    projection_initializer: Optional[dict] = None
    prepend_cls_token: bool = True
    use_conv_patchified_embedding: Optional[bool] = False
    use_embed_proj_bias: bool = True
    use_encoder_pooler_layer: bool = False
    use_ffn_bias: Optional[bool] = True
    use_ffn_bias_in_attention: Optional[bool] = True
    use_post_embed_layer_norm: Optional[bool] = False
    use_projection_bias_in_attention: Optional[bool] = True
    use_final_layer_norm: bool = True
    default_initializer: Optional[InitializerConfig] = None
    ffn_initializer: Optional[dict] = None
    pooler_initializer: Optional[dict] = None
    image_layer_idx: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()

        if self.default_initializer is None:
            self.default_initializer = InitializerConfig(
                name="truncated_normal",
                std=self.initializer_range,
                mean=0.0,
                a=self.initializer_range * -2.0,
                b=self.initializer_range * 2.0,
            )


@registry.register_config("vision_transformer")
@dataclass
class VisionTransformerConfig(BaseConfig):
    eval_input: Optional[DataConfig] = None
    "Input params class for eval mode."

    model: VisionTransformerModelConfig = required
    "Model level params class. Supported params differ for each model."

    optimizer: OptimizerConfig = required
    "Optimizer specific parameters captured in this class."

    runconfig: RunConfig = required
    "Params class to define params for controlling runs."

    train_input: Optional[DataConfig] = None
    "Input params class for train mode."

    sparsity: Optional[SparsityConfig] = None
    "Params class for sparsity related configurations."

    def __post_init__(self):
        super().__post_init__()
        self.set_config_defaults(
            self.model,
            self.train_input.params if self.train_input else None,
            [self.eval_input.params if self.eval_input else None],
        )

    @staticmethod
    def set_config_defaults(mparams, tparams, eparams_list):
        if hasattr(mparams, "mixed_precision"):
            for section in [tparams, *eparams_list]:
                if section is None:
                    continue

                section["mixed_precision"] = mparams.mixed_precision
