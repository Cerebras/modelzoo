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

from typing import List, Literal, Optional
from warnings import warn

import numpy as np
from annotated_types import Ge, Le
from pydantic import PositiveInt, field_validator, model_validator
from torch import nn
from typing_extensions import Annotated

from cerebras.modelzoo.config import ModelConfig
from cerebras.modelzoo.layers import (
    TransformerEncoder,
    TransformerEncoderLayer,
    ViTEmbeddingLayer,
)
from cerebras.modelzoo.layers.activations import ActivationType
from cerebras.modelzoo.layers.init import (
    InitializerConfig,
    TruncatedNormalInitializer,
)
from cerebras.modelzoo.layers.ViTEmbeddingLayer import (
    InterpolatePositionEmbeddingConfig,
)
from cerebras.modelzoo.models.nlp.bert.bert_model import BertPooler


class ViTEncoder(nn.Module):
    def __init__(
        self,
        # Embedding
        hidden_size=768,
        # Encoder
        num_hidden_layers=12,
        layer_norm_epsilon=1.0e-5,
        # Encoder Attn
        num_heads=12,
        attention_module="aiayn_attention",
        extra_attention_params={},
        attention_type="scaled_dot_product",
        attention_softmax_fp32=True,
        dropout_rate=0.0,
        nonlinearity="gelu",
        pooler_nonlinearity=None,
        attention_dropout_rate=0.0,
        use_projection_bias_in_attention=True,
        use_ffn_bias_in_attention=True,
        # Encoder ffn
        filter_size=3072,
        use_ffn_bias=True,
        # Task-specific
        use_final_layer_norm=True,
        initializer_range=0.02,
        default_initializer=None,
        attention_initializer=None,
        ffn_initializer=None,
        pooler_initializer=None,
        norm_first=True,
        use_encoder_pooler_layer=False,
        layerscale_value=None,
        stochastic_depth_drop_prob=0.0,
        stochastic_depth_drop_prob_schedule="linear",
        stochastic_depth_mode="batch",
        **extra_args,
    ):
        super(ViTEncoder, self).__init__()

        self.initializer_range = initializer_range

        if default_initializer is None:
            default_initializer = {
                "name": "truncated_normal",
                "std": self.initializer_range,
                "mean": 0.0,
                "a": self.initializer_range * -2.0,
                "b": self.initializer_range * 2.0,
            }

        if attention_initializer is None:
            attention_initializer = default_initializer
        if ffn_initializer is None:
            ffn_initializer = default_initializer
        if pooler_initializer is None:
            pooler_initializer = default_initializer

        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=filter_size,
            dropout=dropout_rate,
            activation=nonlinearity,
            layer_norm_eps=layer_norm_epsilon,
            norm_first=norm_first,
            attention_module=attention_module,
            attention_dropout_rate=attention_dropout_rate,
            attention_type=attention_type,
            attention_softmax_fp32=attention_softmax_fp32,
            use_projection_bias_in_attention=use_projection_bias_in_attention,
            use_ffn_bias_in_attention=use_ffn_bias_in_attention,
            use_ffn_bias=use_ffn_bias,
            attention_initializer=attention_initializer,
            ffn_initializer=ffn_initializer,
            use_ff_layer1_dropout=False,
            use_ff_layer2_dropout=True,
            layerscale_value=layerscale_value,
            stochastic_depth_drop_prob=stochastic_depth_drop_prob,
            stochastic_depth_mode=stochastic_depth_mode,
        )

        final_ln_f = None
        if use_final_layer_norm:
            final_ln_f = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)

        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_layers=num_hidden_layers, norm=final_ln_f
        )
        # Update the StochasticDepth probabilty of
        # individual layers based on scheduler.
        if stochastic_depth_drop_prob > 0.0:
            if stochastic_depth_drop_prob_schedule == "linear":
                stochastic_depth_sch = np.linspace(
                    0.0, stochastic_depth_drop_prob, num_hidden_layers
                )
            elif stochastic_depth_drop_prob_schedule == "constant":
                stochastic_depth_sch = (
                    1.0
                    * np.ones(num_hidden_layers)
                    * stochastic_depth_drop_prob
                )

            for i, layer in enumerate(self.transformer_encoder.layers):
                layer.drop_path.p = stochastic_depth_sch[i]

        if pooler_nonlinearity is None:
            pooler_nonlinearity = nonlinearity

        self.pooler = (
            BertPooler(
                hidden_size,
                pooler_norm=False,
                layer_norm_epsilon=layer_norm_epsilon,
                use_bias=use_ffn_bias,
                activation=pooler_nonlinearity,
                dropout=None,
                initializer=pooler_initializer,
            )
            if use_encoder_pooler_layer
            else None
        )

    def reset_parameters(self):
        self.transformer_encoder.reset_parameters()
        if self.pooler is not None:
            self.pooler.reset_parameters()

    def forward(self, input_embeddings, extract_layer_idx=None, src_mask=None):
        # no mask required for now
        hidden_states = self.transformer_encoder(
            input_embeddings, extract_layer_idx=extract_layer_idx, mask=src_mask
        )

        pooled_states = None
        if self.pooler is not None:
            pooled_states = self.pooler(hidden_states)
        else:
            pooled_states = hidden_states[:, 0]

        return hidden_states, pooled_states


class ViTModelConfig(ModelConfig):
    name: Literal["ViTModel", "vit"]

    attention_dropout_rate: Annotated[float, Ge(0), Le(1)] = 0.0
    "Dropout rate for attention layer."

    attention_initializer: Optional[InitializerConfig] = None
    "Attention layer initializer. Defaults to `xavier_uniform`."

    attention_module: Literal["aiayn_attention", "multiquery_attention"] = (
        "aiayn_attention"
    )
    """Determines whether to use multiheaded attention (from the Attention is
    All You Need paper) or multi-query attention (MQA) or grouped-query
    attention (GQA). Note that when using MQA/GQA, you must specify
    extra_attention_params (see below). MQA/GQA are differentiated through
    that parameter. Can be one of: ["aiayn_attention", "multiquery_attention"]
    """

    attention_softmax_fp32: bool = True
    "Whether to use fp32 precision for attention softmax."

    attention_type: Literal["dot_product", "scaled_dot_product"] = (
        "scaled_dot_product"
    )
    "Type of attention. Accepted values: [dot_product, scaled_dot_product]."

    dropout_rate: Annotated[float, Ge(0), Le(1)] = 0.1
    "The dropout probability for all fully connected layers."

    embedding_dropout_rate: Annotated[float, Ge(0), Le(1)] = 0.0
    "Dropout rate for embeddings."

    extra_attention_params: Optional[dict] = {}
    """When enabling MQA/GQA, you must specify the the number of key-value
    groups. Within the extra attention params dict, you can set
    `num_kv_groups - 1` to enable MQA or `num_kv_groups - <groups>` for
    GQA. The number of groups should be divisible by `num_heads`.
    """

    filter_size: Optional[PositiveInt] = 3072
    "Dimensionality of the feed-forward layer in the Transformer block."

    hidden_size: Optional[PositiveInt] = 768
    "The size of the transformer hidden layers."

    image_size: List[int] = [224, 224]
    "Input image size [Height, Width]."

    initializer_range: float = 0.02
    """The standard deviation of the truncated_normal_initializer as the
    default initializer"""

    layer_norm_epsilon: float = 1.00e-05
    "The epsilon value used in layer normalization layers."

    nonlinearity: ActivationType = "gelu"
    """The non-linear activation function used in the feed forward network
    in each transformer block. Some may have to use autogen_policy: `medium`."""

    num_channels: Optional[int] = 3
    "Number of input channels"

    num_heads: int = 12
    "The number of attention heads in the multi-head attention layer."

    num_hidden_layers: int = 12
    "Number of hidden layers in the Transformer encoder."

    norm_first: bool = True
    """Enables normalization before the Attention & FFN blocks (i.e Pre-LN as
    described in https://arxiv.org/pdf/2002.04745.pdf. When disabled,
    normalization is applied *after* the residual (Post-LN)"""

    patch_size: Optional[List[int]] = [16, 16]
    "Size of patches to use when converting input image to patches, [Height, Width]."

    pooler_nonlinearity: Optional[ActivationType] = None
    """The non-linear activation function used in the pooler layer. When left as None, uses
    `nonlinearity`."""

    position_embedding_initializer: Optional[InitializerConfig] = None
    """Initializer for position embedding layer. Either a string indicating the name of
    the initializer or a dict that includes the name + other params if relevant. If left
    unspecified will apply truncated normal initialization."""

    position_embedding_type: Optional[
        Literal["learned", "fixed", "relative", "rotary", "alibi"]
    ] = "learned"
    """
    The type of position embedding to use in the model. Can be one of:
    `fixed` - Sinusoidal from original [Transformer](https://arxiv.org/abs/1706.03762),
    `relative` - Relative position embedding, [to exploit pairwise, relative positional
                 information](https://arxiv.org/abs/1803.02155).,
    `rotary` - a.k.a [RoPE](https://arxiv.org/pdf/2104.09864v4.pdf) ,
    `learned` - Learned embedding matrix,
    `None`
    """

    interpolate_position_embedding: Optional[
        InterpolatePositionEmbeddingConfig
    ] = None
    """Use interpolation instead of clamp when input sequence is less than position_embedding sequence length.
    Can only be used with `learned` position_embedding_type.
    The default `None` would use clamp without interpolation.
    """

    projection_initializer: Optional[InitializerConfig] = None
    """Initializer for embedding linear layer. Either a string indicating the name of the
    initializer or a dict that includes the name + other params if relevant. If left
    unspecified will apply truncated normal initialization."""

    prepend_cls_token: bool = True
    "If True, prepends cls token to input."

    cls_token_initializer: Optional[InitializerConfig] = None

    use_conv_patchified_embedding: Optional[bool] = False
    "If True, use conv2D to convert image to patches."

    use_embed_proj_bias: bool = True
    "If True, adds bias to position embeddings."

    use_encoder_pooler_layer: bool = False
    """If True, applies a linear transformation to the element in the encoder output
    corresponding to the first input token"""

    use_ffn_bias: Optional[bool] = True
    "Whether to use bias in the feedforward network (FFN)."

    use_ffn_bias_in_attention: Optional[bool] = True
    "Whether to include bias in the attention layer for feed-forward network (FFN)."

    use_post_embed_layer_norm: Optional[bool] = False
    "If True, applies layer norm to the embeddings."

    use_projection_bias_in_attention: Optional[bool] = True
    "Whether to include bias in the attention layer for projection."

    use_final_layer_norm: bool = True

    default_initializer: Optional[InitializerConfig] = None

    ffn_initializer: Optional[InitializerConfig] = None

    pooler_initializer: Optional[InitializerConfig] = None

    image_layer_idx: Optional[int] = None

    layerscale_value: Optional[float] = None

    stochastic_depth_drop_prob: float = 0.0

    stochastic_depth_drop_prob_schedule: Literal["linear", "constant"] = (
        "linear"
    )

    stochastic_depth_mode: str = "batch"

    use_masked_patches: Optional[bool] = False

    @field_validator("name", mode="after")
    def validate_name(cls, name):
        if name == "ViTModel":
            warn(
                "Passing 'ViTModel' as the model name is deprecated. "
                "Please use 'vit' instead.",
                category=FutureWarning,
            )
            return "vit"
        return name

    @model_validator(mode="before")
    @classmethod
    def validate_image_layer_idx(cls, data):
        image_layer_idx = data.get("image_layer_idx", None)
        num_hidden_layers = data["num_hidden_layers"]
        if image_layer_idx is not None and image_layer_idx >= num_hidden_layers:
            raise ValueError(
                f"`image_layer_idx`(={image_layer_idx}) should be less than `num_hidden_layers` (={num_hidden_layers})"
            )
        return data

    def post_init(self, context):
        if self.default_initializer is None:
            self.default_initializer = TruncatedNormalInitializer(
                std=self.initializer_range,
                mean=0.0,
                a=self.initializer_range * -2.0,
                b=self.initializer_range * 2.0,
            )

        if self.image_layer_idx is not None:
            # convert negative index and positive index representing layer_id of
            # encoder to positive index. All indices are zero-based.
            image_layer_idx = self.image_layer_idx % self.num_hidden_layers
            self.image_layer_idx = image_layer_idx

    @property
    def __model_cls__(self):
        return ViTModel


class ViTModel(nn.Module):
    def __init__(self, config: ViTModelConfig):
        if isinstance(config, dict):
            config = ViTModelConfig(**config)

        super().__init__()

        self.embedding_layer = ViTEmbeddingLayer(
            image_size=config.image_size,
            num_channels=config.num_channels,
            patch_size=config.patch_size,
            hidden_size=config.hidden_size,
            initializer_range=config.initializer_range,
            embedding_dropout_rate=config.embedding_dropout_rate,
            projection_initializer=config.projection_initializer,
            position_embedding_type=config.position_embedding_type,
            position_embedding_initializer=config.position_embedding_initializer,
            cls_token_initializer=config.cls_token_initializer,
            use_conv_patchified_embedding=config.use_conv_patchified_embedding,
            prepend_cls_token=config.prepend_cls_token,
            use_post_embed_layer_norm=config.use_post_embed_layer_norm,
            use_embed_proj_bias=config.use_embed_proj_bias,
            interpolate_position_embedding=config.interpolate_position_embedding,
            use_masked_patches=config.use_masked_patches,
        )

        self.encoder = ViTEncoder(
            # Embedding
            hidden_size=config.hidden_size,
            # Encoder
            num_hidden_layers=config.num_hidden_layers,
            layer_norm_epsilon=config.layer_norm_epsilon,
            # Encoder Attn
            num_heads=config.num_heads,
            attention_module=config.attention_module,
            extra_attention_params=config.extra_attention_params,
            attention_type=config.attention_type,
            attention_softmax_fp32=config.attention_softmax_fp32,
            dropout_rate=config.dropout_rate,
            nonlinearity=config.nonlinearity,
            pooler_nonlinearity=config.pooler_nonlinearity,
            attention_dropout_rate=config.attention_dropout_rate,
            use_projection_bias_in_attention=config.use_projection_bias_in_attention,
            use_ffn_bias_in_attention=config.use_ffn_bias_in_attention,
            # Encoder ffn
            filter_size=config.filter_size,
            use_ffn_bias=config.use_ffn_bias,
            # Task-specific
            use_final_layer_norm=config.use_final_layer_norm,
            initializer_range=config.initializer_range,
            default_initializer=config.default_initializer,
            attention_initializer=config.attention_initializer,
            ffn_initializer=config.ffn_initializer,
            pooler_initializer=config.pooler_initializer,
            norm_first=config.norm_first,
            use_encoder_pooler_layer=config.use_encoder_pooler_layer,
            layerscale_value=config.layerscale_value,
            stochastic_depth_drop_prob=config.stochastic_depth_drop_prob,
            stochastic_depth_drop_prob_schedule=config.stochastic_depth_drop_prob_schedule,
            stochastic_depth_mode=config.stochastic_depth_mode,
        )

    def reset_parameters(self):
        self.embedding_layer.reset_parameters()
        self.encoder.reset_parameters()

    def forward(self, input_image=None, input_image_embeddings=None):
        if input_image is not None and input_image_embeddings is not None:
            raise ValueError(
                f"Only one of `input_image` or `input_image_embeddings` should be passed to model.forward"
            )

        if input_image_embeddings is None:
            input_image_embeddings = self.embedding_layer(input_image)

        hidden_states, pooled_states = self.encoder(input_image_embeddings)

        return hidden_states, pooled_states

    def compute_input_embeddings(self, input_image, masks=None):
        input_image_embeddings = self.embedding_layer(input_image, masks=masks)
        return input_image_embeddings

    def tie_weights(self):
        # weights not tied
        pass

    def extract_features(self, input_embeddings, extract_layer_idx):
        """
        Extract features from `extract_layer_idx` of encoder
        by passing input_tensor through encoder
        input_embeddings: Tensor with output from embeddings layer
        extract_layer_idx: (inclusive)layer index in range [0, self.num_layers) (zero-indexed)
                Applies encoder layers up to (and including) `extract_layer_idx`
                instead of all encoder layers.
                For ex: extract_layer_idx=3 would run fwd pass from encoder_block_0 to encoder_block_3
                and return outputs from encoder_block_3.
                If `extract_layer_idx` = None and `norm` != None, then
                the output returned would be encoder_block_{self.num_layers-1} -> norm -> output (return).
        """
        hidden_states, _ = self.encoder(
            input_embeddings, extract_layer_idx=extract_layer_idx
        )
        return hidden_states
