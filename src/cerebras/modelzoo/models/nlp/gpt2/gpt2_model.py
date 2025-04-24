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

import copy
import logging
from typing import Dict, List, Literal, Optional, Union

import torch
import torch.nn as nn
from annotated_types import Ge, Le
from pydantic import Field, NonNegativeInt, PositiveInt, model_validator
from typing_extensions import Annotated

import cerebras.pytorch as cstorch
from cerebras.modelzoo.common.utils.model.mup_utils import (
    LRAdjustmentGroup,
    scale_initializers_by_dimension,
)
from cerebras.modelzoo.common.utils.model.transformer_utils import (
    create_broadcasted_autoregressive_mask,
    make_key_padding_mask_broadcastable,
    make_sparse_mask_broadcastable,
)
from cerebras.modelzoo.config import ModelConfig
from cerebras.modelzoo.layers import (
    EmbeddingLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
)
from cerebras.modelzoo.layers.activations import ActivationType
from cerebras.modelzoo.layers.FeedForwardNetwork import MoEConfig
from cerebras.modelzoo.layers.init import (
    InitializerConfig,
    TruncatedNormalInitializer,
)
from cerebras.modelzoo.layers.MemoryTokenHelpers import MemoryTokensConfig
from cerebras.modelzoo.layers.norms import NormType, get_norm
from cerebras.modelzoo.models.nlp.gpt2.sparse_mask import (
    create_fixed_sparse_attention_mask,
)

DEFAULT_EMBEDDINGS_SCALE = 1.0
DEFAULT_OUTPUT_LOGITS_ALPHA = None
DEFAULT_SCALE_OUTPUT_LOGITS_BY_D = True
DEFAULT_ATTENTION_LOGITS_ALPHA = 1.0


class GPT2LMHeadModelConfig(ModelConfig):
    name: Literal["GPT2LMHeadModel"]

    # Embedding
    vocab_size: Annotated[int, Ge(1), Le(512000)] = 50257
    "The size of the vocabulary used in the model. Max supported value - 512000."

    num_extra_input_vocab_tokens: int = 0
    "The number of extra input vocab tokens, used for Llama 3.2 Vision model"

    max_position_embeddings: PositiveInt = 1024
    "The maximum sequence length that the model can handle."

    embd_pdrop: Optional[Annotated[float, Ge(0), Le(1)]] = Field(
        None, alias="embedding_dropout_rate"
    )
    "The dropout probability for the embeddings."

    position_embedding_type: Optional[
        Literal["learned", "fixed", "relative", "rotary", "alibi"]
    ] = "learned"
    """
    The type of position embedding to use in the model.
    Can be one of:
    - `learned`: Learned embedding matrix
    - `fixed`:  Sinusoidal from original [Transformer](https://arxiv.org/abs/1706.03762)
    - `relative`:  Relative position embedding [to exploit pairwise, relative positional information](https://arxiv.org/abs/1803.02155).
    - `rotary`: a.k.a [RoPE](https://arxiv.org/pdf/2104.09864v4.pdf)
    - `alibi`: [Attention With Linear Biases](https://arxiv.org/pdf/2108.12409.pdf)
    - `None`: No position embeddings
    """

    constant_pos_embedding: Optional[int] = None

    position_embedding_offset: int = 0
    "Position offset for learned embeddings."

    hidden_size: NonNegativeInt = 768
    "The size of the transformer hidden layers."

    share_embedding_weights: bool = True
    "Whether to share the embedding weights between the input and out put embedding."

    embedding_layer_norm: bool = False
    "Whether to apply layer norm to the embeddings."

    num_relative_attention_buckets: NonNegativeInt = 32
    "Number of buckets to use in relative position embedding."

    rotary_dim: Optional[int] = None
    "The number of dimensions used for the rotary position embedding."

    rope_theta: float = 10000
    """
    Frequency (theta) used in rotary position embedding. This value is
    typically adjusted in long MSL runs as described in
    [codellama](https://arxiv.org/pdf/2308.12950.pdf)
    """

    fold_rope_consts: bool = False
    """If True, folds the rotary position embedding constants compile time.

    For very large models consider generating them on the fly by setting this to
    False. It avoids working with large constant tensors.
    """

    #
    # Encoder
    #

    num_hidden_layers: int = 12
    "Number of hidden layers in the Transformer decoder."

    dropout_rate: Annotated[float, Ge(0), Le(1)] = 0.1
    "The dropout probability for all fully connected layers."

    norm_type: NormType = "layernorm"
    "Determines the type of normalization. See modelzoo/layers/norms.py"

    layer_norm_epsilon: float = 1e-5
    "The epsilon value used in layer normalization layers."

    #
    # Encoder - Attention
    #

    num_heads: PositiveInt = 12
    "The number of attention heads in the model."

    attention_type: Literal["dot_product", "scaled_dot_product"] = (
        "scaled_dot_product"
    )
    """
    Determines whether the QK dot product should be scaled.
    - dot_product -> QK^T
    - scaled_dot_product -> QK^T / sqrt(d)

    Note that setting either scale_qk_dot_by_d or scale_qk_dot_by_layer_idx will
    result in different behavior.
    """

    attention_module: Literal["aiayn_attention", "multiquery_attention"] = (
        "aiayn_attention"
    )
    """
    Determines whether to use multiheaded attention (from
    the Attention is All You Need paper) or multi-query/grouped-query
    attention. When using the latter, you must specify
    extra_attention_params (see below).
    """

    attention_sliding_window_length: Optional[int] = None
    "If specified, sliding window attention is used (as seen in Mistral)."

    sliding_window_every_other_decoder_layer: bool = False
    """
    When enabled, sliding window attention is only applied every other
    decoder layer. Note that attention_sliding_window_length must be specified
    when using this feature. Cannot be used in conjunction with
    fixed_sparse_attention.
    """

    attention_sink_tokens: Optional[int] = None
    "Number of sink tokens to use in the attention module."

    attention_vertical_column_spacing: Optional[int] = None
    "The spacing between vertical columns in the attention module."

    attention_vertical_column_width: Optional[int] = None
    "The width of the vertical columns in the attention module."

    attention_chunk_size: Optional[int] = None
    "Chunk size for locally banded attention (as seen in GPT-3)."

    attention_qk_norm_layer: Optional[NormType] = None
    "The normalization layer to use for the QK dot product."

    attention_qk_norm_eps: float = 1.0e-5
    "The epsilon value used in the QK normalization layer."

    extra_attention_params: dict = {}
    """
    When enabling multi-query/grouped-query
    attention, you must specify the the number of key-value groups.
    Within the extra attention params dict, you can set `num_kv_groups:
    1` to enable MQA or `num_kv_groups: <groups>` for GQA. The number of
    groups should be divisible by `num_heads`.
    """

    extra_ffn_params: dict = {}
    "When setting ffn-specific variants such as sparsity+spec decoding or multimodality."

    attention_inner_dim: Optional[int] = None
    """
    The dimensionality after QKV projection within the attention module.
    When set to None, hidden_size will be used.
    """

    use_projection_bias_in_attention: bool = True
    "Whether to include bias in the attention layer for Q/K/V projections."

    use_ffn_bias_in_attention: bool = True
    """
    Whether to include bias in the attention layer for output projection
    after values have been combined (W_O in original Transformer paper).
    """

    attention_dropout_rate: Optional[Annotated[float, Ge(0), Le(1)]] = 0.1
    "Dropout rate for attention layer. When None, defaults to same as `dropout_rate`."

    attention_softmax_fp32: bool = True
    "Whether to use fp32 precision for attention softmax."

    attention_kernel: Optional[str] = None
    """
    The specific attention kernel implementation to use.
    All implementations are functionally the same but may be compiled differently.
    """

    attention_logit_softcapping: Optional[float] = None
    "Scaling factor when applying tanh softcapping on the attention scores (as seen in Gemma2)"

    fixed_sparse_attention: Optional[dict] = None
    "Applies a fixed sparse attention mask in attention. See GPT-3 configs for examples."

    memory_tokens_config: MemoryTokensConfig = Field(
        default_factory=MemoryTokensConfig, alias="memory_tokens"
    )
    "Memory tokens configuration"

    use_experimental_flex_api: bool = False
    """If true, causes attention mask construction to use the new FlexAtttention-like API and
    generate the kernel sparsity hint by iteratively analyzing the attention decoder mask on
    CPU during the compilation."""

    #
    # Encoder - ffn
    #

    filter_size: PositiveInt = 3072
    "Dimensionality of the feed-forward layer in the Transformer block."

    nonlinearity: ActivationType = "gelu"
    """
    The non-linear activation function used in the feed forward network
    in each transformer block.
    See list of non-linearity functions [here](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity).
    """

    use_ffn_bias: bool = True
    """
    Whether to use bias in the feedforward network (FFN).
    """

    #
    # Task-specific
    #

    use_bias_in_output: bool = False
    "Whether to use bias in the final output layer."

    initializer_range: float = 0.02
    "The standard deviation of the truncated_normal_initializer as the default initializer."

    embedding_initializer: Optional[InitializerConfig] = None
    "Initializer to use for embeddings."

    initializer: Optional[InitializerConfig] = None
    "The initializer to be used for all the initializers used in the model."

    output_layer_initializer: Optional[InitializerConfig] = None
    "The name of the initializer for the weights of the output layer."

    ffn_initializer: Optional[InitializerConfig] = None
    "The name of the initializer for the weights of the ffn kernel."

    ffn_output_layer_initializer: Optional[InitializerConfig] = None
    "The name of the initializer for the weights of the ffn output layer."

    #
    # muP (maximal update parameterization)  parameters
    #

    lr_adjustment_groups: Optional[Dict[str, LRAdjustmentGroup]] = None
    "A dictionary of groups to adjust the learning rate for."

    mup_base_hidden_size: Optional[float] = None
    """
    The hidden size of the base model in muP transfer used to calculate the
    necessary multipliers. Required in muP training when scaling the decoder
    attention module
    """

    mup_base_filter_size: Optional[float] = None
    """
    The filter size of the base model in muP transfer used to calculate the
    necessary multipliers. Required in muP training when scaling the decoder ffn
    """

    embeddings_scale: Optional[float] = 1.0
    """
    Scales the embedding hidden states (i.e. the tensor after embeddings &
    embedding layer norm are applied). Required in muP training
    """

    scale_qk_dot_by_d: Optional[bool] = None
    """
    Scales attention QK dot product by d instead of sqrt(d). Must be enabled
    for muP training. Note that this flag only has effect if
    attention_type=scaled_dot_product
    """

    attention_logits_alpha: Optional[float] = 1.0
    """
    Additionally scales the attention QK dot product by the specified value.
    Recommended to tune for stabilizing attention logits in muP training.
    """

    scale_output_logits_by_d: Optional[bool] = True
    """
    Scales the output logits in muP by mup_base_hidden_size/hidden_size if
    True and sqrt(mup_base_hidden_size/hidden_size) if False. Only applies to
    muP training when scaling the hidden_size.
    """

    output_logits_alpha: Optional[float] = None
    """
    Constant applied to the output logits scalar in muP training. The output
    logits are scaled by output_logits_alpha * mup_base_hidden_size/hidden_size.
    """

    alibi_trainable_slopes: bool = False
    "Replaces alibi's fixed slopes with trainable slopes."

    pos_scaling_factor: float = 1.0
    """
    Position interpolation scaling factor for rotary & alibi.
    See https://arxiv.org/pdf/2306.15595.pdf for details.
    """

    pos_scaling_type: Literal[
        "linear", "YaRN", "yarn", "llama3", "longrope"
    ] = "linear"
    """Can be either `linear` or `YaRN` or 'longrope' or 'llama3',
    For YaRN see https://arxiv.org/pdf/2309.00071
    For LongRope see https://arxiv.org/pdf/2402.13753"""

    pos_scaling_extra_args: Optional[dict] = None
    "A dict including parameters for YaRN RoPE scaling"

    cross_attention_layers: List[int] = []
    "Indices of the cross attention layers."

    rel_distance_mode: Literal["default", "capped", "grouped"] = "default"
    """Mode of relative distance computation in RoPE
    rel_distance_mode=`default` corresponds to vanilla RoPE;
    rel_distance_mode=`capped` corresponds to capped relative distances
    (see LM-Infinite paper https://arxiv.org/abs/2308.16137);
    rel_distance_mode=`grouped` corresponds to grouped relative distances
    (see Self-Extend paper https://arxiv.org/abs/2401.01325);
    """

    rel_distance_extra_args: Optional[dict] = None
    """A dict including parameters for relative distance calculation,
    must contain `rope_local_window_size` if rel_distance_mode=`capped`;
    `rope_local_window_size` and `rope_group_size`
    if rel_distance_mode=`grouped`
    """

    scale_qk_dot_by_layer_idx: bool = False
    """
    Scales the attention QK dot product by the layer index (as seen in Santacoder)
    Note that using this flag in conjunction with attention_type=scaled_dot_product
    will result in scaling by both: QK^T / (sqrt(d) * (layer idx + 1))
    """

    # muP backwards compatibility
    norm_first_sandwich: bool = False
    """
    Normally pre-LN (norm_first=True) performs the following computation:
    y = f(norm(x)) + x (where f could be either the attention block or the
    the FFN). Notice how the norm is applied in parallel to the residual branch,
    and before the input to f. Architectures like Gemma2 "sandwich" f with
    layernorms rather than only applying them before f. In other words, the
    computation is: y = norm(f(norm(x))) + x. Notice how both normalization's
    are still parallel to the residual, and so we're still applying pre-LN
    (norm_first=True). The difference is that we're applying LN before and after
    f.
    """

    output_logits_scale: Optional[float] = None
    """
    Scales the embedding hidden states (i.e. the tensor after embeddings &
    embedding layer norm are applied). Required in muP training
    """

    final_logit_softcapping: Optional[float] = None
    "Scaling factor when applying tanh softcapping on the LM head's logits."

    moe_params: MoEConfig = Field(default_factory=MoEConfig, alias="moe")
    "A dict of MoE params including num_experts, top_k and load_balancing_loss_coef."

    dtype: Union[torch.dtype, str, None] = None
    "The embedding dtype"

    @model_validator(mode="after")
    def validate_rotary_dim(self):
        if (
            self.position_embedding_type == "rotary"
            and self.rotary_dim is not None
        ):
            # https://github.com/huggingface/transformers/blob/f0577df6de36e7e7f28e90fa76da0657de038a39/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L84-L85
            # https://arxiv.org/pdf/2104.09864.pdf Section 3.3
            inner_dim = (
                self.attention_inner_dim
                if self.attention_inner_dim is not None
                else self.hidden_size
            )
            if self.rotary_dim > inner_dim / self.num_heads:
                raise ValueError(
                    "Rotary dimensions should be <= head_dim of the attention layer, "
                    "where head_dim = attention_inner_dim // num_heads."
                )
            if self.rotary_dim % 2 != 0:
                raise ValueError("Rotary dimension must be an even number.")

        return self

    @model_validator(mode="after")
    def validate_mup(self):
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
            if self.output_logits_scale:
                raise RuntimeError(
                    f"Detected mup base dimensions {detected_mup_dimensions}, but "
                    f"the deprecated muP param 'output_logits_scale' was also "
                    f"found. Please convert the config from 2.2 to 2.3 using our "
                    f"config converter"
                )
        else:
            if self.output_logits_scale:
                logging.warning(
                    "The detected configuration of muP has been deprecated and will "
                    "be removed in future versions. Please convert the config from "
                    "2.2 to 2.3 using our config converter."
                )
                mup_tunable_params = [
                    ("output_logits_alpha", DEFAULT_OUTPUT_LOGITS_ALPHA),
                    (
                        "scale_output_logits_by_d",
                        DEFAULT_SCALE_OUTPUT_LOGITS_BY_D,
                    ),
                    ("attention_logits_alpha", DEFAULT_ATTENTION_LOGITS_ALPHA),
                ]
            else:
                mup_tunable_params = [
                    ("embeddings_scale", DEFAULT_EMBEDDINGS_SCALE),
                    ("output_logits_alpha", DEFAULT_OUTPUT_LOGITS_ALPHA),
                    (
                        "scale_output_logits_by_d",
                        DEFAULT_SCALE_OUTPUT_LOGITS_BY_D,
                    ),
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

        return self

    @model_validator(mode="after")
    def validate_sliding_window_every_other_decoder_layer(self):
        if (
            self.sliding_window_every_other_decoder_layer
            and self.fixed_sparse_attention
        ):
            raise ValueError(
                "Cannot use sliding_window_every_other_decoder_layer and "
                "fixed_sparse_attention modes simultaneously."
            )

        if (
            self.sliding_window_every_other_decoder_layer
            and self.attention_sliding_window_length is None
        ):
            raise ValueError(
                "When sliding_window_every_other_decoder_layer is enabled, the"
                "attention_sliding_window_length must be specified."
            )

        return self

    def post_init(self, context):
        if self.embd_pdrop is None:
            self.embd_pdrop = self.dropout_rate

        if self.attention_dropout_rate is None:
            self.attention_dropout_rate = self.dropout_rate

        if self.position_embedding_type == "rotary" and self.rotary_dim is None:
            self.rotary_dim = int(self.hidden_size // self.num_heads * 0.25)

        if self.lr_adjustment_groups is None:
            self.lr_adjustment_groups = (
                self.create_default_lr_adjustment_groups()
            )

        if (
            self.memory_tokens_config
            and self.memory_tokens_config.memory_tokens_enabled
        ):
            updated_params = self.memory_tokens_config.update_model_config_params(
                max_position_embeddings=self.max_position_embeddings,
                position_embedding_type=self.position_embedding_type,
                moe_enabled=self.moe_params.num_experts > 1,
                fixed_sparse_attention=self.fixed_sparse_attention,
                attention_sliding_window_length=self.attention_sliding_window_length,
                attention_vertical_column_spacing=self.attention_vertical_column_spacing,
                attention_vertical_column_width=self.attention_vertical_column_width,
                attention_chunk_size=self.attention_chunk_size,
            )
            for attr_name in updated_params:
                setattr(self, attr_name, updated_params[attr_name])

    def create_default_lr_adjustment_groups(self):
        return {
            "embedding": LRAdjustmentGroup("*embedding*weight"),
            # decoder_kernel for muP backward compatibility
            "decoder_kernel": LRAdjustmentGroup(
                [
                    "*decoder*dense*weight",
                    "*decoder*linear*weight",
                    "*decoder*linear*expert_weights",  # moe
                ]
            ),
            "decoder_attention": LRAdjustmentGroup(
                "*decoder*attn*dense*weight"
            ),
            "decoder_input_ffn": LRAdjustmentGroup(
                [
                    "*decoder*ffn.ffn.[!1]*weight",
                    "*decoder*ffn.ffn.[!1]*expert_weights",  # moe
                ]
            ),
            "decoder_output_ffn": LRAdjustmentGroup(
                [
                    "*decoder*ffn.ffn.[1]*weight",
                    "*decoder*ffn.ffn.[1]*expert_weights",  # moe
                ]
            ),
        }


class GPT2LMHeadModel(nn.Module):
    """
    GPT-2 model with LM head.

    Args:
        config: The configuration object for the model.
    """

    def __init__(self, config: GPT2LMHeadModelConfig):
        if isinstance(config, dict):
            config = GPT2LMHeadModelConfig(**config)

        super().__init__()

        # Unpack all the fields in the config
        vocab_size = config.vocab_size
        num_extra_input_vocab_tokens = config.num_extra_input_vocab_tokens
        max_position_embeddings = config.max_position_embeddings
        embd_pdrop = config.embd_pdrop
        position_embedding_type = config.position_embedding_type
        constant_pos_embedding = config.constant_pos_embedding
        position_embedding_offset = config.position_embedding_offset
        hidden_size = config.hidden_size
        share_embedding_weights = config.share_embedding_weights
        embedding_layer_norm = config.embedding_layer_norm
        num_relative_attention_buckets = config.num_relative_attention_buckets
        rotary_dim = config.rotary_dim
        rope_theta = config.rope_theta
        fold_rope_consts = config.fold_rope_consts
        num_hidden_layers = config.num_hidden_layers
        dropout_rate = config.dropout_rate
        norm_type = config.norm_type
        layer_norm_epsilon = config.layer_norm_epsilon
        num_heads = config.num_heads
        attention_type = config.attention_type
        attention_module = config.attention_module
        attention_sliding_window_length = config.attention_sliding_window_length
        sliding_window_every_other_decoder_layer = (
            config.sliding_window_every_other_decoder_layer
        )
        attention_sink_tokens = config.attention_sink_tokens
        attention_vertical_column_spacing = (
            config.attention_vertical_column_spacing
        )
        attention_vertical_column_width = config.attention_vertical_column_width
        attention_chunk_size = config.attention_chunk_size
        attention_qk_norm_layer = config.attention_qk_norm_layer
        attention_qk_norm_eps = config.attention_qk_norm_eps
        extra_attention_params = config.extra_attention_params
        extra_ffn_params = config.extra_ffn_params
        attention_inner_dim = config.attention_inner_dim
        use_projection_bias_in_attention = (
            config.use_projection_bias_in_attention
        )
        use_ffn_bias_in_attention = config.use_ffn_bias_in_attention
        attention_dropout_rate = config.attention_dropout_rate
        attention_softmax_fp32 = config.attention_softmax_fp32
        attention_kernel = config.attention_kernel
        attention_logit_softcapping = config.attention_logit_softcapping
        fixed_sparse_attention = config.fixed_sparse_attention
        memory_tokens_config = config.memory_tokens_config
        filter_size = config.filter_size
        nonlinearity = config.nonlinearity
        use_ffn_bias = config.use_ffn_bias
        use_bias_in_output = config.use_bias_in_output
        initializer_range = config.initializer_range
        embedding_initializer = config.embedding_initializer
        initializer = config.initializer
        output_layer_initializer = config.output_layer_initializer
        ffn_initializer = config.ffn_initializer
        ffn_output_layer_initializer = config.ffn_output_layer_initializer
        lr_adjustment_groups = config.lr_adjustment_groups
        mup_base_hidden_size = config.mup_base_hidden_size
        mup_base_filter_size = config.mup_base_filter_size
        embeddings_scale = config.embeddings_scale
        scale_qk_dot_by_d = config.scale_qk_dot_by_d
        attention_logits_alpha = config.attention_logits_alpha
        scale_output_logits_by_d = config.scale_output_logits_by_d
        output_logits_alpha = config.output_logits_alpha
        alibi_trainable_slopes = config.alibi_trainable_slopes
        pos_scaling_factor = config.pos_scaling_factor
        pos_scaling_type = config.pos_scaling_type
        pos_scaling_extra_args = config.pos_scaling_extra_args
        cross_attention_layers = config.cross_attention_layers
        rel_distance_mode = config.rel_distance_mode
        rel_distance_extra_args = config.rel_distance_extra_args
        scale_qk_dot_by_layer_idx = config.scale_qk_dot_by_layer_idx
        norm_first_sandwich = config.norm_first_sandwich
        output_logits_scale = config.output_logits_scale
        final_logit_softcapping = config.final_logit_softcapping
        moe_params = config.moe_params
        dtype = config.dtype
        use_experimental_flex_api = config.use_experimental_flex_api

        # std deviation for weight initialization
        self.initializer_range = initializer_range
        self.num_hidden_layers = num_hidden_layers
        self.share_embedding_weights = share_embedding_weights
        self.embedding_layer_norm = embedding_layer_norm
        self.max_position_embeddings = max_position_embeddings
        self.position_embedding_type = position_embedding_type
        self.constant_pos_embedding = constant_pos_embedding

        self.num_heads = num_heads
        total_experts = moe_params.num_experts
        if moe_params.num_shared_experts:
            total_experts += moe_params.num_shared_experts
        self.moe_enabled = total_experts > 1
        self.moe_num_experts = moe_params.num_experts
        self.moe_routing_algorithm = moe_params.routing_algorithm

        default_initializer = TruncatedNormalInitializer(
            mean=0.0,
            std=self.initializer_range,
        )
        if initializer is None:
            attention_initializer = default_initializer
            if ffn_initializer is None:
                ffn_initializer = default_initializer
            if moe_params.gate_initializer is None:
                gate_initializer = default_initializer
            else:
                gate_initializer = moe_params.gate_initializer
        else:
            attention_initializer = initializer
            if ffn_initializer is None:
                ffn_initializer = initializer
            gate_initializer = initializer

        # Update the config with initializer selected
        moe_params = moe_params.copy(
            update=dict(gate_initializer=gate_initializer)
        )

        if embedding_initializer is None:
            embedding_initializer = default_initializer

        # Handle muP scaling
        self.embeddings_scale = embeddings_scale
        self.output_logits_scale = output_logits_scale
        if mup_base_hidden_size:
            hidden_size_width_mult = hidden_size / mup_base_hidden_size
            attention_initializer, ffn_initializer = (
                scale_initializers_by_dimension(
                    [attention_initializer, ffn_initializer],
                    width_scale=hidden_size_width_mult**-0.5,
                )
            )
            if output_layer_initializer is None:
                output_layer_initializer = default_initializer

            (output_layer_initializer,) = scale_initializers_by_dimension(
                output_layer_initializer,
                width_scale=hidden_size_width_mult**-0.5,
                depth_scale=(2 * num_hidden_layers) ** -0.5,
            )
            if not output_logits_alpha:
                output_logits_alpha = 1.0
            if scale_output_logits_by_d:
                self.output_logits_scale = (
                    output_logits_alpha / hidden_size_width_mult
                )
            else:
                self.output_logits_scale = (
                    output_logits_alpha / hidden_size_width_mult**0.5
                )
            for lr_adjustment_group in [
                "decoder_attention",
                "decoder_input_ffn",
            ]:
                lr_adjustment_groups[lr_adjustment_group].set_scale(
                    1 / hidden_size_width_mult
                )

        if mup_base_filter_size:
            filter_size_width_mult = filter_size / mup_base_filter_size
            if ffn_output_layer_initializer is None:
                ffn_output_layer_initializer = default_initializer

            (ffn_output_layer_initializer,) = scale_initializers_by_dimension(
                ffn_output_layer_initializer,
                width_scale=filter_size_width_mult**-0.5,
                depth_scale=(2 * num_hidden_layers) ** -0.5,
            )
            lr_adjustment_groups["decoder_output_ffn"].set_scale(
                1 / filter_size_width_mult
            )

        self.lr_adjustment_groups = lr_adjustment_groups

        if output_layer_initializer is None and initializer is None:
            (output_layer_initializer,) = scale_initializers_by_dimension(
                default_initializer,
                depth_scale=(2 * num_hidden_layers) ** -0.5,
            )

        if ffn_output_layer_initializer is None:
            ffn_output_layer_initializer = output_layer_initializer

        norm_class = get_norm(norm_type)

        qk_norm_class = get_norm(attention_qk_norm_layer)

        if position_embedding_type == "rotary":
            if rotary_dim is None:
                rotary_dim = hidden_size // num_heads
            # https://github.com/huggingface/transformers/blob/f0577df6de36e7e7f28e90fa76da0657de038a39/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L84-L85
            # https://arxiv.org/pdf/2104.09864.pdf Section 3.3

            inner_dim = (
                attention_inner_dim
                if attention_inner_dim is not None
                else hidden_size
            )

            assert (
                rotary_dim <= inner_dim / num_heads
            ), "Rotary dimensions should be <= head_dim of the attention layer, where head_dim = attention_inner_dim // num_heads"
            assert (
                rotary_dim % 2 == 0
            ), "Rotary dimension must be an even number."

        self.use_experimental_flex_api = use_experimental_flex_api

        self.embedding_layer = EmbeddingLayer(
            vocab_size=vocab_size + num_extra_input_vocab_tokens,
            embedding_size=hidden_size,
            embeddings_initializer=embedding_initializer,
            position_embedding_type=position_embedding_type,
            constant_pos_embedding=constant_pos_embedding,
            position_embeddings_initializer=embedding_initializer,
            max_position_embeddings=max_position_embeddings,
            position_embedding_offset=position_embedding_offset,
            num_heads=num_heads,
            num_relative_attention_buckets=num_relative_attention_buckets,
            rotary_dim=rotary_dim,
            rope_theta=rope_theta,
            fold_rope_consts=fold_rope_consts,
            alibi_trainable_slopes=alibi_trainable_slopes,
            pos_scaling_factor=pos_scaling_factor,
            pos_scaling_type=pos_scaling_type,
            pos_scaling_extra_args=pos_scaling_extra_args,
            memory_tokens_config=memory_tokens_config,
            rel_distance_mode=rel_distance_mode,
            rel_distance_extra_args=rel_distance_extra_args,
            dtype=dtype,
        )

        if self.embedding_layer_norm:
            self.embedding_ln_f = norm_class(
                hidden_size, eps=layer_norm_epsilon
            )

        self.drop_embd = nn.Dropout(embd_pdrop)

        extra_attention_params["attention_kernel"] = attention_kernel
        decoder_layer = TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=filter_size,
            dropout=dropout_rate,
            activation=nonlinearity,
            layer_norm_eps=layer_norm_epsilon,
            norm_layer=norm_class,
            norm_first=True,
            extra_attention_params=extra_attention_params,
            extra_ffn_params=extra_ffn_params,
            add_cross_attention=False,
            attention_type=attention_type,
            scale_qk_dot_by_d=scale_qk_dot_by_d,
            attention_logits_alpha=attention_logits_alpha,
            scale_qk_dot_by_layer_idx=scale_qk_dot_by_layer_idx,
            attention_module=attention_module,
            attention_qk_norm_layer=qk_norm_class,
            attention_qk_norm_eps=attention_qk_norm_eps,
            attention_inner_dim=attention_inner_dim,
            attention_dropout_rate=attention_dropout_rate,
            attention_softmax_fp32=attention_softmax_fp32,
            use_projection_bias_in_attention=use_projection_bias_in_attention,
            use_ffn_bias_in_attention=use_ffn_bias_in_attention,
            use_ffn_bias=use_ffn_bias,
            attention_initializer=attention_initializer,
            attention_output_layer_initializer=output_layer_initializer,
            attention_logit_softcapping=attention_logit_softcapping,
            ffn_initializer=ffn_initializer,
            ffn_output_layer_initializer=ffn_output_layer_initializer,
            use_ff_layer1_dropout=False,
            norm_first_sandwich=norm_first_sandwich,
            moe_params=moe_params,
            memory_tokens_config=memory_tokens_config,
        )

        # Final LayerNorm
        self.ln_f = norm_class(hidden_size, eps=layer_norm_epsilon)

        if len(cross_attention_layers) > 0:
            cross_attention_decoder_layer = TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=filter_size,
                dropout=dropout_rate,
                activation=nonlinearity,
                layer_norm_eps=layer_norm_epsilon,
                norm_layer=norm_class,
                norm_first=True,
                extra_attention_params=extra_attention_params,
                extra_ffn_params=extra_ffn_params,
                add_cross_attention=True,
                disable_self_attention=True,
                cross_attention_gate_attention=True,
                cross_attention_gate_mlp=True,
                attention_type=attention_type,
                scale_qk_dot_by_d=scale_qk_dot_by_d,
                attention_logits_alpha=attention_logits_alpha,
                scale_qk_dot_by_layer_idx=scale_qk_dot_by_layer_idx,
                attention_module=attention_module,
                attention_qk_norm_layer=get_norm("rmsnorm"),
                attention_qk_norm_eps=attention_qk_norm_eps,
                attention_inner_dim=attention_inner_dim,
                attention_dropout_rate=attention_dropout_rate,
                attention_softmax_fp32=attention_softmax_fp32,
                use_projection_bias_in_attention=use_projection_bias_in_attention,
                use_ffn_bias_in_attention=use_ffn_bias_in_attention,
                use_ffn_bias=use_ffn_bias,
                attention_initializer=attention_initializer,
                attention_output_layer_initializer=output_layer_initializer,
                attention_logit_softcapping=attention_logit_softcapping,
                ffn_initializer=ffn_initializer,
                ffn_output_layer_initializer=ffn_output_layer_initializer,
                use_ff_layer1_dropout=False,
                norm_first_sandwich=norm_first_sandwich,
                moe_params=moe_params,
                memory_tokens_config=memory_tokens_config,
            )

            decoder_layers = []
            for layer_idx in range(num_hidden_layers):
                if layer_idx in cross_attention_layers:
                    decoder_layers.append(
                        copy.deepcopy(cross_attention_decoder_layer)
                    )
                else:
                    decoder_layers.append(copy.deepcopy(decoder_layer))

            self.transformer_decoder = TransformerDecoder(
                decoder_layers,
                num_layers=num_hidden_layers,
                norm=self.ln_f,
            )
        else:
            self.transformer_decoder = TransformerDecoder(
                decoder_layer,
                num_layers=num_hidden_layers,
                norm=self.ln_f,
            )

        self.attention_sliding_window_length = attention_sliding_window_length
        self.attention_sink_tokens = attention_sink_tokens
        self.attention_vertical_column_spacing = (
            attention_vertical_column_spacing
        )
        self.attention_vertical_column_width = attention_vertical_column_width
        self.attention_chunk_size = attention_chunk_size
        assert not (
            sliding_window_every_other_decoder_layer and fixed_sparse_attention
        ), (
            "Cannot use sliding_window_every_other_decoder_layer and "
            "fixed_sparse_attention modes simultaneously."
        )
        if sliding_window_every_other_decoder_layer:
            assert attention_sliding_window_length is not None, (
                "When sliding_window_every_other_decoder_layer is enabled, the "
                "attention_sliding_window_length must be specified."
            )

        self.sliding_window_every_other_decoder_layer = (
            sliding_window_every_other_decoder_layer
        )
        if fixed_sparse_attention is not None:
            self.fixed_sparsity_mask = create_fixed_sparse_attention_mask(
                max_sequence_length=max_position_embeddings,
                n_heads=num_heads,
                **fixed_sparse_attention,
            )
        else:
            self.fixed_sparsity_mask = None

        self.lm_head = nn.Linear(
            hidden_size, vocab_size, bias=use_bias_in_output
        )
        self.final_logit_softcapping = final_logit_softcapping

        self.tie_weights()

        self.__reset_parameters()

    def reset_parameters(self):
        self.embedding_layer.reset_parameters()
        self.transformer_decoder.reset_parameters()
        self.__reset_parameters()

    def __reset_parameters(self):
        # Init final norm layer
        if hasattr(self.ln_f, "bias") and hasattr(self.ln_f.bias, 'data'):
            self.ln_f.bias.data.zero_()
        if hasattr(self.ln_f, "weight") and hasattr(self.ln_f.weight, 'data'):
            self.ln_f.weight.data.fill_(1.0)

        # Initialize LM head
        if not self.share_embedding_weights:
            self.lm_head.weight.data.normal_(
                mean=0.0, std=self.initializer_range
            )
        if self.lm_head.bias is not None:
            self.lm_head.bias.data.zero_()

    def tie_weights(self):
        if not self.share_embedding_weights:
            return

        output_embedding = self.get_output_embeddings()
        input_embedding = self.get_input_embeddings()
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

    def get_output_embeddings(self):
        """
        Extract the final layer that produces logits.
        """
        return self.lm_head

    def get_input_embeddings(self):
        return self.embedding_layer.get_input_embeddings()

    def compute_input_embeddings(
        self, input_ids, position_ids=None, special_token_meta=None
    ):
        hidden_states = self.embedding_layer(
            input_ids,
            position_ids=position_ids,
            special_token_meta=special_token_meta,
        )

        if self.embedding_layer_norm:
            hidden_states = self.embedding_ln_f(hidden_states)
        hidden_states = hidden_states * torch.tensor(
            float(self.embeddings_scale), dtype=hidden_states.dtype
        )
        hidden_states = self.drop_embd(hidden_states)
        return hidden_states

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        attention_span=None,
        cross_attention_states=None,
        cross_attention_mask=None,
        tgt_key_padding_mask=None,
        position_ids=None,
        input_embeddings=None,
        inference_loop_index=None,
        token_modality_idx=None,
        constant_pos_mask=None,
        special_token_meta=None,
        full_text_row_masked_out_mask=None,
    ):
        if input_ids is not None and input_embeddings is not None:
            raise ValueError(
                f"Only one of `input_ids` or `input_embeddings` "
                f"should be passed to model.forward"
            )
        elif input_ids is None and input_embeddings is None:
            raise ValueError(
                f"Both `input_ids` and `input_embeddings` are None, "
                f"either one of them should be passed to model.forward"
            )

        if special_token_meta is not None:
            if input_embeddings is not None:
                raise ValueError(
                    "Adding special tokens to the input is only supported if "
                    "`input_ids` are passed to the model, but got `input_embeddings`"
                )
            if not isinstance(special_token_meta, dict):
                raise ValueError("Expected `special_token_meta` to be a dict")
            for key in ("memory_token_mask",):
                if not isinstance(special_token_meta.get(key), torch.Tensor):
                    raise ValueError(
                        f"Expected `special_token_meta['{key}']` to be a tensor, "
                        f"but got {type(special_token_meta.get(key))}"
                    )

        if input_embeddings is None:
            hidden_states = self.compute_input_embeddings(
                input_ids, position_ids, special_token_meta
            )
        else:
            hidden_states = input_embeddings

        expert_hash_idx = None
        if self.moe_enabled and self.moe_routing_algorithm == "hash":
            expert_hash_idx = input_ids.to(torch.float) % self.moe_num_experts
            expert_hash_idx = expert_hash_idx.to(input_ids.dtype)

        decoder_outputs = self.apply_decoder(
            hidden_states,
            attention_mask=attention_mask,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            attention_span=attention_span,
            position_ids=position_ids,
            tgt_key_padding_mask=tgt_key_padding_mask,
            expert_hash_idx=expert_hash_idx,
            token_modality_idx=token_modality_idx,
            constant_pos_mask=constant_pos_mask,
            special_token_meta=special_token_meta,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
        )

        if self.moe_enabled:
            hidden_states, routing_weights, expert_masks = decoder_outputs
        else:
            hidden_states = decoder_outputs

        if inference_loop_index is not None:
            # When running an implicit autoregressive loop for generation, this
            # tensor holds the "current token" index. We can pull out only the
            # hidden states from that token to avoid unnecessary work in the
            # lm_head matmul.
            hidden_states = cstorch.experimental.get_loop_iteration_slice(
                hidden_states, inference_loop_index
            )

        if (
            cstorch.use_cs()
            and cstorch.backends.csx.precision.optimization_level == 1
        ):
            lm_logits = cstorch.pol(bwd_level=0)(self.lm_head)(hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        # scale lm_logits for muP transfer
        if self.output_logits_scale:
            lm_logits = lm_logits * torch.tensor(
                float(self.output_logits_scale),
                dtype=lm_logits.dtype,
            )

        if self.final_logit_softcapping:
            lm_logits = (
                torch.tanh(lm_logits / self.final_logit_softcapping)
                * self.final_logit_softcapping
            )

        if self.moe_enabled:
            return lm_logits, routing_weights, expert_masks
        else:
            return lm_logits

    def apply_decoder(
        self,
        input_embeddings,
        attention_mask=None,
        attention_span=None,
        cross_attention_states=None,
        cross_attention_mask=None,
        tgt_key_padding_mask=None,
        position_ids=None,
        extract_layer_idx=None,
        expert_hash_idx=None,
        token_modality_idx=None,
        constant_pos_mask=None,
        special_token_meta=None,
        full_text_row_masked_out_mask=None,
    ):
        # `extract_layer_idx` is used only multimodal use case
        # input_embeddings : shape (bsz, MSL, H)
        sparse_attention_mask = None

        sparse_mask_args = {}
        for sparse_mask_arg_key in ("attention_sliding_window_length",):
            sparse_mask_args[sparse_mask_arg_key] = (
                getattr(self, sparse_mask_arg_key)
                if not self.sliding_window_every_other_decoder_layer
                else None
            )

        causal_attention_mask = create_broadcasted_autoregressive_mask(
            batch_size=input_embeddings.shape[0],
            num_heads=self.num_heads,
            tgt_seq_length=input_embeddings.shape[1],
            attention_span=attention_span,
            device=input_embeddings.device,
            dtype=input_embeddings.dtype,
            use_experimental_flex_api=self.use_experimental_flex_api,
            **sparse_mask_args,
        )

        if self.sliding_window_every_other_decoder_layer:
            # Models like Gemma2 apply SWA every other decoder layer. This is
            # implemented by creating a non-SWA mask named causal_attention_mask
            # and a SWA mask named sparse_attention_mask. The TransformerDecoder
            # layer automatically switches between these two masks based on
            # layer index.
            sparse_attention_mask = create_broadcasted_autoregressive_mask(
                batch_size=input_embeddings.shape[0],
                num_heads=self.num_heads,
                tgt_seq_length=input_embeddings.shape[1],
                attention_span=attention_span,
                attention_sliding_window_length=self.attention_sliding_window_length,
                device=input_embeddings.device,
                dtype=input_embeddings.dtype,
            )
        elif self.use_experimental_flex_api:
            # If the flag is active, the mask range annotation is going to be
            # constructed based on the mask sparsity directly, and inserted
            # via model annotation (instead of analyzing and inserting during
            # compile).
            sparse_attn_mask = causal_attention_mask
            for layer in self.transformer_decoder.layers:
                layer.self_attn.sparse_attn_mask_ranges = (
                    sparse_attn_mask.sparsity_annotation
                )
            causal_attention_mask = sparse_attn_mask.mask_tensor

        # Fixed sparse attention, used in GPT-3 model
        if self.fixed_sparsity_mask is not None:
            sparse_attention_mask = make_sparse_mask_broadcastable(
                self.fixed_sparsity_mask,
                attention_mask,
                dtype=input_embeddings.dtype,
                device=input_embeddings.device,
                revert_mask=False,
            )

        # Helpers on alibi/relative position embeddings bias
        length = input_embeddings.shape[1]
        batch_size = input_embeddings.shape[0]
        self_attn_position_bias = self.embedding_layer.compute_position_bias(
            length,
            length,
            constant_pos_mask=constant_pos_mask,
            batch_size=batch_size,
        )

        if cross_attention_mask is not None:
            memory_mask = make_key_padding_mask_broadcastable(
                cross_attention_mask, dtype=input_embeddings.dtype
            )
        else:
            memory_mask = None

        return self.transformer_decoder(
            input_embeddings,
            tgt_mask=causal_attention_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            sparse_mask=sparse_attention_mask,
            memory=cross_attention_states,
            memory_mask=memory_mask,
            rotary_position_embedding_helper=self.embedding_layer.get_rope_helper(),
            self_attn_position_bias=self_attn_position_bias,
            extract_layer_idx=extract_layer_idx,
            expert_hash_idx=expert_hash_idx,
            token_modality_idx=token_modality_idx,
            position_ids=position_ids,
            constant_pos_mask=constant_pos_mask,
            special_token_meta=special_token_meta,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
        )

    def extract_features(
        self,
        input_embeddings,
        extract_layer_idx,
        attention_mask=None,
        cross_attention_states=None,
        cross_attention_mask=None,
        attention_span=None,
        tgt_key_padding_mask=None,
        position_ids=None,
        full_text_row_masked_out_mask=None,
    ):
        """
        Extract features of input_embeddings from `extract_layer_idx` of decoder
        extract_layer_idx: (inclusive)layer index in range [0, self.num_layers) (zero-indexed)
            Applies decoder layers up to (and including) `extract_layer_idx`
            instead of all decoder layers.
            For ex: extract_layer_idx=3 would run fwd pass from decoder_block_0 to decoder_block_3
            and return outputs from decoder_block_3.
            If `extract_layer_idx` = None and `norm` != None, then
            the output returned would be decoder_block_{self.num_layers-1} -> norm -> output (return).

        This function is added for multimodal use case.
        """
        hidden_states = self.apply_decoder(
            input_embeddings,
            extract_layer_idx=extract_layer_idx,
            attention_mask=attention_mask,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            attention_span=attention_span,
            tgt_key_padding_mask=tgt_key_padding_mask,
            position_ids=position_ids,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
        )

        if isinstance(hidden_states, tuple):
            return hidden_states[0]
        else:
            return hidden_states
