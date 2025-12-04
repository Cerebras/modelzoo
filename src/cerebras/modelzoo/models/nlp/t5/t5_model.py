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

# This code is adapted from
# https://github.com/huggingface/transformers/blob/master/src/transformers/models/t5/modeling_t5.py
#
# Copyright 2022 Cerebras Systems.
#
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
from typing import List, Literal, Optional, Union
from warnings import warn

import torch
import torch.nn as nn
from annotated_types import Ge, Le
from pydantic import NonNegativeInt, field_validator
from typing_extensions import Annotated

import cerebras.pytorch as cstorch
from cerebras.modelzoo.common.utils.model.mup_utils import (
    scale_initializers_by_dimension,
)
from cerebras.modelzoo.common.utils.model.transformer_utils import (
    create_broadcasted_autoregressive_mask,
    make_key_padding_mask_broadcastable,
)
from cerebras.modelzoo.config import ModelConfig
from cerebras.modelzoo.layers import (
    EmbeddingLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from cerebras.modelzoo.layers.norms import NormType, get_norm

DEFAULT_EMBEDDINGS_ALPHA = 1.0
DEFAULT_OUTPUT_LOGITS_ALPHA = None
DEFAULT_SCALE_OUTPUT_LOGITS_BY_D = False
DEFAULT_ENCODER_ATTENTION_LOGITS_ALPHA = 1.0
DEFAULT_DECODER_ATTENTION_LOGITS_ALPHA = 1.0


class T5ForConditionalGenerationModelConfig(ModelConfig):
    name: Literal["t5", "T5ForConditionalGeneration"]

    # Embedding:
    src_vocab_size: Annotated[int, Ge(1), Le(512000)] = 32128
    "The size of the source vocabulary. Max supported value - `512000`."

    tgt_vocab_size: Optional[Annotated[int, Ge(1), Le(512000)]] = None
    """The size of the target vocabulary. Max supported value - `512000`.
    When not provided, same as src_vocab_size"""

    extra_ids: int = 0
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

    encoder_num_hidden_layers: NonNegativeInt = 6
    "Number of hidden layers in the encoder."

    decoder_num_hidden_layers: Optional[NonNegativeInt] = None
    """Number of hidden layers in the Transformer decoder. Will use the same
    value as `encoder_num_hidden_layers` if not set.
    """

    # Transformer Attention:
    num_heads: int = 8
    "The number of attention heads in the multi-head attention layer."

    relative_attention_num_buckets: int = 32
    "The number of buckets to use for each attention layer."

    norm_type: NormType = "rmsnorm"
    "Determines the type of normalization. See modelzoo/layers/norms.py"

    dropout_rate: Annotated[float, Ge(0), Le(1)] = 0.1
    "The dropout probability for all fully connected layers."

    relu_dropout_rate: Optional[Annotated[float, Ge(0), Le(1)]] = None
    "The dropout rate for ReLU activation function."

    layer_norm_epsilon: Union[float, List[float]] = 1e-5
    "The epsilon value used in layer normalization layers."

    initializer_factor: float = 1.0
    """
    A factor for initializing all weight matrices (should be kept to 1, used
    internally for initialization testing).
    """

    encoder_nonlinearity: Literal[
        "relu", "gelu", "reglu", "geglu", "swiglu"
    ] = "relu"
    "Type of nonlinearity to be used in encoder."

    decoder_nonlinearity: Optional[
        Literal["relu", "gelu", "reglu", "geglu", "swiglu"]
    ] = None
    """Type of nonlinearity to be used in decoder. If decoder_nonlinearity isn't
    provided, it will be the same as encoder_nonlinearity"""

    use_projection_bias_in_attention: bool = False
    "Whether to include bias in the attention layer for projection."

    attention_softmax_fp32: bool = True
    "Whether to use fp32 precision for attention softmax."

    attention_kernel: Optional[str] = None

    use_cache: bool = False
    """
    Whether or not the model should return the last key/values attentions (not
    used by all models).
    """

    decoder_start_token_id: Optional[int] = None

    pad_token_id: int = 0

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

    src_max_position_embeddings: Optional[int] = None  # 512
    "Maximum source sequence length that the model's position embeddings can handle."

    tgt_max_position_embeddings: Optional[int] = None  # 512
    "Maximum target sequence length that the model's position embeddings can handle."

    use_dropout_outside_residual_path: bool = True
    "Whether to set dropout calculations outside of the residual path."

    share_encoder_decoder_embedding: bool = True
    "Whether to share the embedding weights between the encoder and decoder."

    share_embedding_weights: bool = True
    "Whether to share the embedding weights between the input and out put embedding."

    tie_encoder_decoder: bool = False

    use_pre_encoder_decoder_dropout: bool = False
    "Whether to use dropout layer after positional embedding layer and encoder/decoder."

    use_pre_encoder_decoder_layer_norm: bool = True
    "Whether to use layer norm before passing input tensors into encoder/decoder."

    use_ffn_bias: bool = False
    "Whether to use bias in the feedforward network (FFN)."

    label_smoothing: float = 0.0
    "The label smoothing factor used during training."

    use_transformer_initialization: bool = False
    """The Transformer model tends to converge best with a scaled variant on Xavier uniform
    initialization used for linear layers. This contrasts the initialization used for the
    original T5 paper, which uses He normal initialization for linear layers. Setting this
    flag to `True` switches the initialization to the Transformer specific scaled Xavier
    initialization."""

    # muP (maximal update parameterization)  parameters
    lr_adjustment_groups: Optional[dict] = None

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

    encoder_attention_logits_alpha: float = (
        DEFAULT_ENCODER_ATTENTION_LOGITS_ALPHA
    )
    """Additionally scales the encoder attention QK dot product by the specified value.
    Recommended to tune for stabilizing attention logits in muP training."""

    decoder_attention_logits_alpha: float = (
        DEFAULT_DECODER_ATTENTION_LOGITS_ALPHA
    )
    """Additionally scales the decoder attention QK dot product by the specified value.
    Recommended to tune for stabilizing attention logits in muP training."""

    scale_encoder_qk_dot_by_d: Optional[bool] = None
    """Scales encoder attention QK dot product by d instead of sqrt(d). Must
    be enabled for muP training. Note that this flag only has effect if
    muP params are specified or use_transformer_initialization==`True`"""

    scale_decoder_qk_dot_by_d: Optional[bool] = None
    """Scales decoder attention QK dot product by d instead of sqrt(d). Must
    be enabled for muP training. Note that this flag only has effect if
    muP params are specified or use_transformer_initialization==`True`"""

    scale_output_logits_by_d: bool = DEFAULT_SCALE_OUTPUT_LOGITS_BY_D
    """Scales the output logits in muP by mup_base_d_model/d_model if
    True and sqrt(mup_base_d_model/d_model) if False. Only applies to
    muP training when scaling d_model"""

    output_logits_alpha: Optional[float] = DEFAULT_OUTPUT_LOGITS_ALPHA
    """Constant applied to the output logits scalar in muP training. The output
    logits are scaled by output_logits_alpha / hidden_size_width_mult"""

    attention_module: Literal["aiayn_attention", "multiquery_attention"] = (
        "aiayn_attention"
    )

    extra_attention_params: dict = {}

    dtype: Optional[torch.dtype] = None

    @field_validator("name", mode="after")
    def validate_name(cls, name):
        if name == "T5ForConditionalGeneration":
            warn(
                "Passing 'T5ForConditionalGeneration' as the model name is deprecated. "
                "Please use 't5' instead.",
                category=FutureWarning,
            )
            return "t5"
        return name

    def post_init(self, context):
        if self.tgt_vocab_size is None:
            self.tgt_vocab_size = self.src_vocab_size

        if self.decoder_num_hidden_layers is None:
            self.decoder_num_hidden_layers = self.encoder_num_hidden_layers

        if self.relu_dropout_rate is None:
            self.relu_dropout_rate = self.dropout_rate

        if self.decoder_nonlinearity is None:
            self.decoder_nonlinearity = self.encoder_nonlinearity

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
            if detected_mup_tunable_params := (
                self.model_fields_set
                & {
                    "embeddings_scale",
                    "output_logits_alpha",
                    "scale_qk_dot_by_d",
                    "scale_output_logits_by_d",
                    "attention_logits_alpha",
                }
            ):
                logging.warning(
                    f"The following muP parameters were changed from their default "
                    f"value outside of a muP run: {sorted(detected_mup_tunable_params)}. "
                    f"As a result, they may have an undesired effect. Please "
                    f"specify the muP base dimensions {supported_mup_dimensions} "
                    f"to trigger a muP run."
                )

    @property
    def __model_cls__(self):
        return T5ForConditionalGeneration


class T5ForConditionalGeneration(nn.Module):
    r"""
    T5 Model with a `language modeling` head on top.

    Args:
        config: T5ForConditionalGenerationModelConfig
    """

    def __init__(self, config: T5ForConditionalGenerationModelConfig):
        if isinstance(config, dict):
            config = T5ForConditionalGenerationModelConfig(**config)

        # Unpack config
        src_vocab_size = config.src_vocab_size
        tgt_vocab_size = config.tgt_vocab_size
        d_model = config.d_model
        d_kv = config.d_kv
        d_ff = config.d_ff
        encoder_num_hidden_layers = config.encoder_num_hidden_layers
        decoder_num_hidden_layers = config.decoder_num_hidden_layers
        num_heads = config.num_heads
        relative_attention_num_buckets = config.relative_attention_num_buckets
        norm_type = config.norm_type
        dropout_rate = config.dropout_rate
        relu_dropout_rate = config.relu_dropout_rate
        layer_norm_epsilon = config.layer_norm_epsilon
        initializer_factor = config.initializer_factor
        encoder_nonlinearity = config.encoder_nonlinearity
        decoder_nonlinearity = config.decoder_nonlinearity
        use_projection_bias_in_attention = (
            config.use_projection_bias_in_attention
        )
        attention_softmax_fp32 = config.attention_softmax_fp32
        attention_kernel = config.attention_kernel
        use_cache = config.use_cache
        decoder_start_token_id = config.decoder_start_token_id
        pad_token_id = config.pad_token_id
        position_embedding_type = config.position_embedding_type
        src_max_position_embeddings = config.src_max_position_embeddings
        tgt_max_position_embeddings = config.tgt_max_position_embeddings
        use_dropout_outside_residual_path = (
            config.use_dropout_outside_residual_path
        )
        share_encoder_decoder_embedding = config.share_encoder_decoder_embedding
        share_embedding_weights = config.share_embedding_weights
        tie_encoder_decoder = config.tie_encoder_decoder
        use_pre_encoder_decoder_dropout = config.use_pre_encoder_decoder_dropout
        use_pre_encoder_decoder_layer_norm = (
            config.use_pre_encoder_decoder_layer_norm
        )
        use_ffn_bias = config.use_ffn_bias
        label_smoothing = config.label_smoothing
        use_transformer_initialization = config.use_transformer_initialization
        lr_adjustment_groups = config.lr_adjustment_groups
        mup_base_d_model = config.mup_base_d_model
        mup_base_d_ff = config.mup_base_d_ff
        mup_base_d_kv = config.mup_base_d_kv
        embeddings_alpha = config.embeddings_alpha
        encoder_attention_logits_alpha = config.encoder_attention_logits_alpha
        decoder_attention_logits_alpha = config.decoder_attention_logits_alpha
        scale_encoder_qk_dot_by_d = config.scale_encoder_qk_dot_by_d
        scale_decoder_qk_dot_by_d = config.scale_decoder_qk_dot_by_d
        scale_output_logits_by_d = config.scale_output_logits_by_d
        output_logits_alpha = config.output_logits_alpha
        attention_module = config.attention_module
        extra_attention_params = config.extra_attention_params
        dtype = config.dtype

        super().__init__()

        # Copy only the subset of params that are referenced later
        self.d_model = d_model
        self.d_kv = d_kv
        self.attention_inner_dim = d_kv * num_heads
        self.initializer_factor = initializer_factor
        self.use_cache = use_cache
        self.share_encoder_decoder_embedding = share_encoder_decoder_embedding
        self.share_embedding_weights = share_embedding_weights
        self.tie_encoder_decoder = tie_encoder_decoder
        self.decoder_start_token_id = decoder_start_token_id
        self.pad_token_id = pad_token_id
        self.position_embedding_type = position_embedding_type
        self.label_smoothing = label_smoothing
        self.decoder_num_heads = num_heads

        self.dtype = dtype

        if decoder_num_hidden_layers is None:
            decoder_num_hidden_layers = encoder_num_hidden_layers

        if relu_dropout_rate is None:
            relu_dropout_rate = dropout_rate

        assert position_embedding_type in (
            "fixed",
            "learned_absolute",
            "relative",
            "rotary",
            "alibi",
            None,
        ), (
            f"Position embedding must be one of `fixed`, `learned_absolute`, "
            f"`relative`, or None. Got {position_embedding_type}."
        )
        if position_embedding_type == "learned_absolute":
            position_embedding_type = "learned"

        # Initialization
        if use_transformer_initialization:
            embeddings_initializer = {
                "name": "truncated_normal",
                "mean": 0.0,
                "std": 1.0,
                "a": -2.0,
                "b": 2.0,
            }
            attention_q_initializer = {
                "name": "variance_scaling",
                "scale": 1.0 / (d_kv * 9.0),
                "mode": "fan_avg",
                "distribution": "uniform",
            }
            attention_initializer = {
                "name": "variance_scaling",
                "scale": 1.0 / 9.0,
                "mode": "fan_avg",
                "distribution": "uniform",
            }
            ffn_initializer = {"name": "xavier_uniform", "gain": 1.0}
            ffn_output_layer_initializer = {
                "name": "xavier_uniform",
                "gain": 1.0,
            }
        else:
            embeddings_initializer = {
                "name": "normal",
                "mean": 0.0,
                "std": initializer_factor * 1.0,
            }
            attention_q_initializer = {
                "name": "normal",
                "mean": 0.0,
                "std": initializer_factor * ((d_model * d_kv) ** -0.5),
            }
            attention_initializer = {
                "name": "normal",
                "mean": 0.0,
                "std": initializer_factor * (d_model**-0.5),
            }
            ffn_initializer = {
                "name": "normal",
                "mean": 0.0,
                "std": initializer_factor * (d_model**-0.5),
            }
            ffn_output_layer_initializer = {
                "name": "normal",
                "mean": 0.0,
                "std": initializer_factor * (d_ff**-0.5),
            }
        encoder_output_layer_initializer = attention_initializer.copy()
        decoder_output_layer_initializer = attention_initializer.copy()
        encoder_ffn_output_layer_initializer = (
            ffn_output_layer_initializer.copy()
        )
        decoder_ffn_output_layer_initializer = (
            ffn_output_layer_initializer.copy()
        )

        scale_qk_dot = (
            use_transformer_initialization or mup_base_d_model or mup_base_d_ff
        )
        # Handle muP scaling
        q_projection_scale = 1.0
        k_projection_scale = 1.0
        v_projection_scale = 1.0
        output_projection_scale = 1.0
        self.embeddings_scale = embeddings_alpha
        self.output_logits_scale = None
        self.mup_base_d_model = mup_base_d_model
        if mup_base_d_model and not use_transformer_initialization:
            d_model_width_mult = d_model / mup_base_d_model
            self.embeddings_scale *= d_model**0.5
            scale_initializers_by_dimension(
                embeddings_initializer,
                width_scale=d_model**-0.5,
            )
            for lr_adjustment_group in [
                "embedding",
                "lm_head",
            ]:
                lr_adjustment_groups[lr_adjustment_group].set_scale(
                    1 / d_model_width_mult**0.5
                )
            for lr_adjustment_group in [
                "decoder_input_ffn",
                "encoder_input_ffn",
            ]:
                lr_adjustment_groups[lr_adjustment_group].set_scale(
                    1 / d_model_width_mult
                )
            if not output_logits_alpha:
                output_logits_alpha = 1.0
            if scale_output_logits_by_d:
                self.output_logits_scale = (
                    output_logits_alpha / d_model_width_mult
                )
            else:
                self.output_logits_scale = (
                    output_logits_alpha / d_model_width_mult**0.5
                )
            if mup_base_d_kv:
                # If varying d_kv, the q, k, v projections are considered
                # hidden weights
                scale_initializers_by_dimension(
                    encoder_output_layer_initializer,
                    width_scale=(d_model**0.5)
                    / (self.attention_inner_dim**0.5),
                    depth_scale=2 * encoder_num_hidden_layers,
                )
                scale_initializers_by_dimension(
                    decoder_output_layer_initializer,
                    width_scale=(d_model**0.5)
                    / (self.attention_inner_dim**0.5),
                    depth_scale=2 * decoder_num_hidden_layers,
                )
                for lr_adjustment_group in [
                    "decoder_qkv_projection",
                    "encoder_qkv_projection",
                ]:
                    lr_adjustment_groups[lr_adjustment_group].set_scale(
                        1 / d_model_width_mult
                    )
                for lr_adjustment_group in [
                    "decoder_output_projection",
                    "encoder_output_projection",
                ]:
                    lr_adjustment_groups[lr_adjustment_group].set_scale(
                        mup_base_d_kv / d_kv
                    )
            else:
                q_projection_scale = d_model_width_mult**-0.5
                k_projection_scale = d_model_width_mult**-0.5
                v_projection_scale = d_model_width_mult**-0.5
                output_projection_scale = d_model_width_mult**0.5
                scale_initializers_by_dimension(
                    encoder_output_layer_initializer,
                    depth_scale=2 * encoder_num_hidden_layers,
                )
                scale_initializers_by_dimension(
                    decoder_output_layer_initializer,
                    depth_scale=2 * decoder_num_hidden_layers,
                )
                for lr_adjustment_group in [
                    "decoder_qkv_projection",
                    "encoder_qkv_projection",
                    "decoder_output_projection",
                    "encoder_output_projection",
                ]:
                    lr_adjustment_groups[lr_adjustment_group].set_scale(
                        1 / d_model_width_mult**0.5
                    )

        if mup_base_d_ff and not use_transformer_initialization:
            scale_initializers_by_dimension(
                encoder_ffn_output_layer_initializer,
                depth_scale=2 * encoder_num_hidden_layers,
            )
            scale_initializers_by_dimension(
                decoder_ffn_output_layer_initializer,
                depth_scale=2 * decoder_num_hidden_layers,
            )
            for lr_adjustment_group in [
                "encoder_output_ffn",
                "decoder_output_ffn",
            ]:
                lr_adjustment_groups[lr_adjustment_group].set_scale(
                    mup_base_d_ff / d_ff
                )

        self.encoder_embeddings = EmbeddingLayer(
            src_vocab_size,
            d_model,
            embeddings_initializer=embeddings_initializer,
            max_position_embeddings=src_max_position_embeddings,
            position_embedding_type=position_embedding_type,
            # RPE:
            num_heads=num_heads,
            bidirectional=True,
            num_relative_attention_buckets=relative_attention_num_buckets,
            dtype=dtype,
        )

        self.decoder_embeddings = EmbeddingLayer(
            tgt_vocab_size,
            d_model,
            embeddings_initializer=embeddings_initializer,
            max_position_embeddings=tgt_max_position_embeddings,
            position_embedding_type=position_embedding_type,
            # RPE:
            num_heads=num_heads,
            bidirectional=False,
            num_relative_attention_buckets=relative_attention_num_buckets,
            dtype=dtype,
        )

        if self.share_encoder_decoder_embedding:
            assert (
                src_vocab_size == tgt_vocab_size
            ), "Cannot share embeddings between encoder and decoder due to different vocab sizes"
            self.decoder_embeddings.set_input_embeddings(
                self.encoder_embeddings.get_input_embeddings()
            )

        self.pre_encoder_dropout = None
        self.pre_decoder_dropout = None
        # Transformer model uses dropout right after position embeddings
        # and before the encoder call, T5 does not use it.
        if use_pre_encoder_decoder_dropout:
            self.pre_encoder_dropout = nn.Dropout(dropout_rate)
            self.pre_decoder_dropout = nn.Dropout(dropout_rate)

        assert encoder_nonlinearity in [
            "relu",
            "gelu",
            "reglu",
            "geglu",
            "swiglu",
        ], "T5/Transformer doesn't support encoder_nonlinearity {}".format(
            encoder_nonlinearity
        )
        assert decoder_nonlinearity in [
            "relu",
            "gelu",
            "reglu",
            "geglu",
            "swiglu",
        ], "T5/Transformer doesn't support decoder_nonlinearity {}".format(
            decoder_nonlinearity
        )

        if (encoder_nonlinearity == "gelu" and use_ffn_bias) or (
            decoder_nonlinearity == "gelu" and use_ffn_bias
        ):
            logging.warning(
                "Overriding use_ffn_bias to false because using gelu"
            )
            use_ffn_bias = False

        norm_class = get_norm(norm_type)

        extra_attention_params["attention_kernel"] = attention_kernel
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout_rate,
            activation=encoder_nonlinearity,
            norm_layer=norm_class,
            layer_norm_eps=layer_norm_epsilon,
            norm_first=use_pre_encoder_decoder_layer_norm,
            batch_first=True,
            extra_attention_params=extra_attention_params,
            attention_type=(
                "scaled_dot_product" if scale_qk_dot else "dot_product"
            ),
            scale_qk_dot_by_d=scale_encoder_qk_dot_by_d,
            attention_logits_alpha=encoder_attention_logits_alpha,
            q_projection_scale=q_projection_scale,
            k_projection_scale=k_projection_scale,
            v_projection_scale=v_projection_scale,
            output_projection_scale=output_projection_scale,
            attention_module=attention_module,
            attention_inner_dim=self.attention_inner_dim,
            attention_softmax_fp32=attention_softmax_fp32,
            use_projection_bias_in_attention=use_projection_bias_in_attention,
            use_ffn_bias_in_attention=False,
            use_ffn_bias=use_ffn_bias,
            ffn_dropout_rate=relu_dropout_rate,
            use_ff_layer1_dropout=True,
            use_ff_layer2_dropout=True,
            attention_q_initializer=attention_q_initializer,
            attention_initializer=attention_initializer,
            attention_output_layer_initializer=encoder_output_layer_initializer,
            ffn_initializer=ffn_initializer,
            ffn_output_layer_initializer=encoder_ffn_output_layer_initializer,
        )

        encoder_final_layer_norm = norm_class(d_model, eps=layer_norm_epsilon)

        self.dropout_before_encoder = nn.Dropout(dropout_rate)
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_layers=encoder_num_hidden_layers,
            norm=encoder_final_layer_norm,
        )
        self.dropout_after_encoder = None
        if use_dropout_outside_residual_path:
            self.dropout_after_encoder = nn.Dropout(dropout_rate)

        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout_rate,
            activation=encoder_nonlinearity,
            norm_layer=norm_class,
            layer_norm_eps=layer_norm_epsilon,
            norm_first=use_pre_encoder_decoder_layer_norm,
            batch_first=True,
            extra_attention_params=extra_attention_params,
            attention_type=(
                "scaled_dot_product" if scale_qk_dot else "dot_product"
            ),
            scale_qk_dot_by_d=scale_decoder_qk_dot_by_d,
            attention_logits_alpha=decoder_attention_logits_alpha,
            q_projection_scale=q_projection_scale,
            k_projection_scale=k_projection_scale,
            v_projection_scale=v_projection_scale,
            output_projection_scale=output_projection_scale,
            attention_module=attention_module,
            attention_inner_dim=self.attention_inner_dim,
            use_projection_bias_in_attention=use_projection_bias_in_attention,
            attention_softmax_fp32=attention_softmax_fp32,
            use_ffn_bias_in_attention=False,
            use_ffn_bias=use_ffn_bias,
            use_ff_layer1_dropout=True,
            use_ff_layer2_dropout=True,
            attention_q_initializer=attention_q_initializer,
            attention_initializer=attention_initializer,
            attention_output_layer_initializer=decoder_output_layer_initializer,
            ffn_initializer=ffn_initializer,
            ffn_output_layer_initializer=decoder_ffn_output_layer_initializer,
        )

        decoder_final_layer_norm = norm_class(d_model, eps=layer_norm_epsilon)

        self.dropout_before_decoder = nn.Dropout(dropout_rate)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_layers=decoder_num_hidden_layers,
            norm=decoder_final_layer_norm,
        )
        self.dropout_after_decoder = None
        if use_dropout_outside_residual_path:
            self.dropout_after_decoder = nn.Dropout(dropout_rate)

        self.lm_head = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.__reset_parameters()
        assert (
            not tie_encoder_decoder
        ), "Implementation does not currently support tied Encoder/Decoder weights"
        self.tie_weights()

    def reset_parameters(self):
        self.encoder_embeddings.reset_parameters()
        self.decoder_embeddings.reset_parameters()
        if self.relative_position_encoder:
            self.relative_position_encoder.reset_parameters()
        if self.relative_position_decoder:
            self.relative_position_decoder.reset_parameters()
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        self.__reset_parameters()

    def __reset_parameters(self):
        # Initialize LM head
        if not self.share_embedding_weights:
            if self.mup_base_d_model:
                self.lm_head.weight.data.normal_(
                    mean=0.0,
                    std=self.initializer_factor / self.d_model**0.5,
                )
            else:
                self.lm_head.weight.data.normal_(
                    mean=0.0, std=self.initializer_factor
                )

    # Helper function `forward` for computing everything up to (but not
    # including) the model head. This is helpful for models that inherit from
    # T5 that apply a different head at the end of the model
    def compute_hidden_states(
        self,
        input_ids=None,
        attention_mask=None,
        prepend_embeddings=None,
    ):

        src = self.encoder_embeddings(input_ids)

        if prepend_embeddings is not None:
            src = torch.cat([prepend_embeddings, src], dim=1)

        src = src * torch.tensor(float(self.embeddings_scale), dtype=src.dtype)

        # Transformer uses pre-encoder dropout
        if self.pre_encoder_dropout:
            src = self.pre_encoder_dropout(src)

        # Compute relative position bias for the encoder block if applicable
        encoder_self_attn_position_bias = (
            self.encoder_embeddings.compute_position_bias(
                src.shape[1], src.shape[1]
            )
        )
        src = self.dropout_before_encoder(src)
        if attention_mask is not None:
            attention_mask = make_key_padding_mask_broadcastable(
                attention_mask, dtype=src.dtype
            )

        # Convert encoder inputs in embeddings if needed
        hidden_states = self.encoder(
            src,
            mask=attention_mask,
            self_attn_position_bias=encoder_self_attn_position_bias,
        )
        if self.dropout_after_encoder:
            hidden_states = self.dropout_after_encoder(
                hidden_states
            )  # HF T5 Decoder also applies dropout at the end

        return hidden_states

    def compute_decoder_states(
        self,
        hidden_states=None,
        memory_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
    ):
        assert (
            past_key_values is None
        ), "past_key_values should be None since inference is not supported yet"

        use_cache = use_cache if use_cache is not None else self.use_cache

        assert (
            not use_cache
        ), "cannot enable use_cache because inference is not supported yet"

        if labels is not None and decoder_input_ids is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        decoder_inputs_embeds = self.decoder_embeddings(decoder_input_ids)

        decoder_inputs_embeds = decoder_inputs_embeds * torch.tensor(
            float(self.embeddings_scale), dtype=decoder_inputs_embeds.dtype
        )

        # Transformer uses dropout before feeding to decoder module while
        # T5 does not use this layer
        if self.pre_decoder_dropout:
            decoder_inputs_embeds = self.pre_decoder_dropout(
                decoder_inputs_embeds
            )

        batch_size, decoder_seq_length = decoder_inputs_embeds.size()[:2]

        decoder_self_attn_position_bias = (
            self.decoder_embeddings.compute_position_bias(
                decoder_seq_length, decoder_seq_length
            )
        )

        if memory_mask is not None:
            memory_mask = make_key_padding_mask_broadcastable(
                memory_mask, dtype=hidden_states.dtype
            )

        causal_mask = create_broadcasted_autoregressive_mask(
            batch_size=decoder_input_ids.shape[0],
            num_heads=self.decoder_num_heads,
            tgt_seq_length=decoder_input_ids.shape[1],
            device=decoder_inputs_embeds.device,
            dtype=hidden_states.dtype,
        )

        decoder_inputs_embeds = self.dropout_before_decoder(
            decoder_inputs_embeds
        )

        decoder_outputs = self.decoder(
            decoder_inputs_embeds,
            memory=hidden_states,
            tgt_mask=causal_mask,
            memory_mask=memory_mask,
            past_kv=past_key_values,
            cache_present_kv=use_cache,
            self_attn_position_bias=decoder_self_attn_position_bias,
        )
        if use_cache:
            sequence_output, present_kv = decoder_outputs
        else:
            sequence_output = decoder_outputs
        if self.dropout_after_decoder:
            sequence_output = self.dropout_after_decoder(sequence_output)

        return sequence_output

    def compute_sequence_output(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        prepend_embeddings=None,
    ):

        if encoder_outputs is None:
            hidden_states = self.compute_hidden_states(
                input_ids,
                attention_mask,
                prepend_embeddings=prepend_embeddings,
            )
        else:
            hidden_states = encoder_outputs

        sequence_output = self.compute_decoder_states(
            hidden_states=hidden_states,
            memory_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
        )

        return sequence_output

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        prepend_embeddings=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``.

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

            >>> # training
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> # inference
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
            >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
            >>> # studies have shown that owning a dog is good for you.
        """

        sequence_output = self.compute_sequence_output(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            prepend_embeddings=prepend_embeddings,
        )

        if self.share_embedding_weights and not self.output_logits_scale:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.d_model**-0.5)

        if (
            cstorch.use_cs()
            and cstorch.backends.csx.precision.optimization_level == 1
        ):
            lm_logits = cstorch.pol(bwd_level=0)(self.lm_head)(sequence_output)
        else:
            lm_logits = self.lm_head(sequence_output)

        # scale lm_logits for muP transfer
        if self.output_logits_scale:
            lm_logits = lm_logits * torch.tensor(
                float(self.output_logits_scale),
                dtype=lm_logits.dtype,
            )

        return lm_logits

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings
        and (if enabled) tie encoder/decoder weights.
        """
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is not None and self.share_embedding_weights:
            self._tie_or_clone_weights(
                output_embeddings, self.get_input_embeddings()
            )

        if self.tie_encoder_decoder:
            if hasattr(self, self.base_model_prefix):
                self = getattr(self, self.base_model_prefix)
            self._tie_encoder_decoder_weights(
                self.encoder, self.decoder, self.base_model_prefix
            )

    @staticmethod
    def _tie_encoder_decoder_weights(
        encoder: nn.Module, decoder: nn.Module, base_model_prefix: str
    ):
        uninitialized_encoder_weights: List[str] = []
        if decoder.__class__ != encoder.__class__:
            print(
                f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
            )

        def tie_encoder_to_decoder_recursively(
            decoder_pointer: nn.Module,
            encoder_pointer: nn.Module,
            module_name: str,
            uninitialized_encoder_weights: List[str],
            depth=0,
        ):
            assert isinstance(decoder_pointer, nn.Module) and isinstance(
                encoder_pointer, nn.Module
            ), f"{decoder_pointer} and {encoder_pointer} have to be of type nn.Module"
            if hasattr(decoder_pointer, "weight"):
                assert hasattr(encoder_pointer, "weight")
                encoder_pointer.weight = decoder_pointer.weight
                if hasattr(decoder_pointer, "bias"):
                    assert hasattr(encoder_pointer, "bias")
                    encoder_pointer.bias = decoder_pointer.bias
                return

            encoder_modules = encoder_pointer._modules
            decoder_modules = decoder_pointer._modules
            if len(decoder_modules) > 0:
                assert (
                    len(encoder_modules) > 0
                ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

                all_encoder_weights = set(
                    [
                        module_name + "/" + sub_name
                        for sub_name in encoder_modules.keys()
                    ]
                )
                encoder_layer_pos = 0
                for name, module in decoder_modules.items():
                    if name.isdigit():
                        encoder_name = str(int(name) + encoder_layer_pos)
                        decoder_name = name
                        if not isinstance(
                            decoder_modules[decoder_name],
                            type(encoder_modules[encoder_name]),
                        ) and len(encoder_modules) != len(decoder_modules):
                            # this can happen if the name corresponds to the position in a list module list of layers
                            # in this case the decoder has added a cross-attention that the encoder does not have
                            # thus skip this step and subtract one layer pos from encoder
                            encoder_layer_pos -= 1
                            continue
                    elif name not in encoder_modules:
                        continue
                    elif depth > 500:
                        raise ValueError(
                            "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                        )
                    else:
                        decoder_name = encoder_name = name
                    tie_encoder_to_decoder_recursively(
                        decoder_modules[decoder_name],
                        encoder_modules[encoder_name],
                        module_name + "/" + name,
                        uninitialized_encoder_weights,
                        depth=depth + 1,
                    )
                    all_encoder_weights.remove(module_name + "/" + encoder_name)

                uninitialized_encoder_weights += list(all_encoder_weights)

        # tie weights recursively
        tie_encoder_to_decoder_recursively(
            decoder, encoder, base_model_prefix, uninitialized_encoder_weights
        )
        if len(uninitialized_encoder_weights) > 0:
            print(
                f"The following encoder weights were not tied to the decoder {uninitialized_encoder_weights}"
            )

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone module weights depending of whether we are using TorchScript or not."""
        if not isinstance(output_embeddings, list):
            output_embeddings = [output_embeddings]
            input_embeddings = [input_embeddings]

        for output_embedding, input_embedding in zip(
            output_embeddings, input_embeddings
        ):
            output_embedding.weight = input_embedding.weight

            if getattr(output_embedding, "bias", None) is not None:
                output_embedding.bias.data = nn.functional.pad(
                    output_embedding.bias.data,
                    (
                        0,
                        output_embedding.weight.shape[0]
                        - output_embedding.bias.shape[0],
                    ),
                    "constant",
                    0,
                )
            if hasattr(output_embedding, "out_features") and hasattr(
                input_embedding, "num_embeddings"
            ):
                output_embedding.out_features = input_embedding.num_embeddings

    def get_input_embeddings(self):
        # This function returns decoder token embeddings
        # in order to properly tie embeddings between the decoder
        # input and decoder output.
        return self.decoder_embeddings.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.decoder_embeddings.set_input_embeddings(new_embeddings)
        if self.share_embedding_weights:
            self.encoder_embeddings.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.decoder_start_token_id
        pad_token_id = self.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert (
            pad_token_id is not None
        ), "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(
            shifted_input_ids >= 0
        ).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids
