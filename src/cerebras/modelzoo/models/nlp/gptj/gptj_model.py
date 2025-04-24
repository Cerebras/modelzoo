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
from typing import List, Literal, Optional, Union

import torch
import torch.nn as nn
from annotated_types import Ge, Le
from pydantic import Field, PositiveInt, model_validator
from typing_extensions import Annotated

import cerebras.pytorch as cstorch
from cerebras.modelzoo.common.utils.model.mup_utils import (
    scale_initializers_by_dimension,
)
from cerebras.modelzoo.common.utils.model.transformer_utils import (
    create_broadcasted_autoregressive_mask,
)
from cerebras.modelzoo.config import ModelConfig
from cerebras.modelzoo.layers import (
    EmbeddingLayer,
    GPTJDecoderLayer,
    TransformerDecoder,
)
from cerebras.modelzoo.layers.activations import ActivationType
from cerebras.modelzoo.layers.FeedForwardNetwork import MoEConfig
from cerebras.modelzoo.layers.init import InitializerConfig
from cerebras.modelzoo.layers.norms import NormType, get_norm

DEFAULT_EMBEDDINGS_SCALE = 1.0
DEFAULT_OUTPUT_LOGITS_ALPHA = None
DEFAULT_SCALE_OUTPUT_LOGITS_BY_D = True
DEFAULT_ATTENTION_LOGITS_ALPHA = 1.0


class GPTJSubModelConfig(ModelConfig):
    name: Literal["gptj"]

    hidden_size: int = 768
    "The size of the transformer hidden layers"

    # Embedding params
    vocab_size: Annotated[int, Ge(1), Le(512000)] = 50257
    "The size of the vocabulary used in the model. Max supported value - `512000`."

    max_position_embeddings: PositiveInt = 1024
    "The maximum sequence length that the model can handle."

    embd_pdrop: Optional[Annotated[float, Ge(0), Le(1)]] = Field(
        0.1, alias="embedding_dropout_rate"
    )
    "Dropout rate for attention layer."

    share_embedding_weights: bool = True
    "Whether to share the embedding weights between the input and out put embedding."

    position_embedding_type: Optional[
        Literal["learned", "fixed", "relative", "rotary", "alibi"]
    ] = "rotary"
    """The type of position embedding to use in the model.
    Can be one of - `learned` - Learned embedding matrix,
    `fixed` - Sinusoidal from original [Transformer](https://arxiv.org/abs/1706.03762),
    `relative` - Relative position embedding
    [to exploit pairwise, relative positional information](https://arxiv.org/abs/1803.02155).,
    `rotary` - a.k.a [RoPE](https://arxiv.org/pdf/2104.09864v4.pdf),
    `alibi` [Attention With Linear Biases](https://arxiv.org/pdf/2108.12409.pdf),
    or `None` for no position embeddings.
    """

    rotary_dim: Optional[int] = None
    "The number of dimensions used for the rotary position encoding."

    rope_theta: float = 10000
    """Frequency (theta) used in rotary position embedding. This value is
    typically adjusted in long MSL runs as described in
    [CodeLlama](https://arxiv.org/pdf/2308.12950.pdf)"""

    fold_rope_consts: bool = False
    """If True, folds the rotary position embedding constants compile time.

    For very large models consider generating them on the fly by setting this to
    False. It avoids working with large constant tensors.
    """

    num_relative_attention_buckets: Optional[int] = None
    "Number of buckets to use in relative position embedding"

    # Decoder params
    num_hidden_layers: int = 12
    "Number of hidden layers in the Transformer decoder"

    filter_size: int = 3072
    """Dimensionality of the feed-forward layer in the Transformer block. Commonly
    set to 4*hidden_size.
    """

    dropout_rate: float = Field(0.1, alias="residual_dropout_rate")
    "Default dropout for the model if specified."

    nonlinearity: ActivationType = "gelu"
    """The non-linear activation function used in the feed forward network
    in each transformer block.
    See list of non-linearity functions [here](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity).
    """

    norm_type: NormType = "layernorm"
    "Determines the type of normalization. See modelzoo/layers/norms.py"

    layer_norm_epsilon: Union[float, List[float]] = 1e-5
    "The epsilon value used in layer normalization layers."

    use_ffn_bias: bool = False
    "Whether to use bias in the FFN."

    use_untied_layer_norm: bool = False
    """When using parallel decoder architecture, tied layer norm means that the
    inputs to FFN and attention use normalization-layers with the same parameters,
    i.e. x + Attn(LN_1(x)) + FFN(LN_1(x)) vs x + Attn(LN_1(x)) + FFN(LN_2(x)).
    GPT-NeoX uses untied whereas GPT-J uses tied."""

    # Attention params
    num_heads: Optional[int] = 12
    "The number of attention heads."

    attention_module: Literal["aiayn_attention", "multiquery_attention"] = (
        "aiayn_attention"
    )
    """Determines whether to use multiheaded attention (from the Attention is
    All You Need paper) or multi-query attention (MQA) or grouped-query
    attention (GQA). Note that when using MQA/GQA, you must specify
    extra_attention_params (see below). MQA/GQA are differentiated through
    that parameter.
    """

    attention_sliding_window_length: Optional[int] = None
    "If specified, sliding window attention is used (as seen in Mistral)."

    attention_sink_tokens: Optional[int] = None
    "Number of sink tokens to use in the attention module."

    extra_attention_params: dict = {}
    """
    When enabling multi-query/grouped-query
    attention, you must specify the the number of key-value groups.
    Within the extra attention params dict, you can set `num_kv_groups:
    1` to enable MQA or `num_kv_groups: <groups>` for GQA. The number of
    groups should be divisible by `num_heads`.
    """

    attention_type: Literal["dot_product", "scaled_dot_product"] = (
        "scaled_dot_product"
    )
    """Determines whether the QK dot product should be scaled -
    dot_product -> QK^T
    scaled_dot_product -> QK^T / sqrt(d)
    """

    attention_dropout_rate: Optional[float] = 0.1
    "Dropout rate for attention layer. When None, defaults to same as `residual_dropout_rate`"

    attention_softmax_fp32: bool = True
    "Whether to use fp32 precision for attention softmax."

    attention_kernel: Optional[str] = None

    use_projection_bias_in_attention: bool = True
    "Whether to include bias in the attention layer for Q/K/V projections."

    use_ffn_bias_in_attention: bool = True
    """Whether to include bias in the attention layer for output projection
    after values have been combined (W_O in original Transformer paper).
    """

    # Task-specific
    initializer_range: float = 0.02
    "The standard deviation of the truncated_normal_initializer as the default initializer"

    use_bias_in_output: bool = False
    "Whether to use bias in the final output layer."

    embedding_initializer: Optional[InitializerConfig] = None
    """Initializer to use for embeddings. See [supported initializers]
    (./common/pytorch/model_utils/create_initializer.py). Default: 'normal'
    """

    attention_initializer: Optional[InitializerConfig] = Field(
        None, alias="initializer"
    )
    """The initializer to be used for all the initializers used in the model.
    See [supported initializers]"
    "(./common/pytorch/model_utils/create_initializer.py). Default: varies based on model"""

    output_layer_initializer: Optional[InitializerConfig] = None
    """The name of the initializer for the weights of the output layer.
    See [supported initializers](./common/pytorch/model_utils/create_initializer.py)."""

    ffn_initializer: Optional[InitializerConfig] = None
    """The name of the initializer for the weights of the ffn kernel.
    See [supported initializers](./common/pytorch/model_utils/create_initializer.py)."""

    ffn_output_layer_initializer: Optional[InitializerConfig] = None
    """The name of the initializer for the weights of the ffn output layer.
    See [supported initializers](./common/pytorch/model_utils/create_initializer.py)."""

    # muP (maximal update parameterization)  parameters
    lr_adjustment_groups: Optional[dict] = None
    "A dictionary of groups to adjust the learning rate for."

    mup_base_hidden_size: Optional[float] = None
    """The hidden size of the base model in muP transfer used to calculate the
    necessary multipliers. Required in muP training when scaling the decoder
    attention module"""

    mup_base_filter_size: Optional[float] = None
    """The filter size of the base model in muP transfer used to calculate the
    necessary multipliers. Required in muP training when scaling the decoder
    ffn"""

    embeddings_scale: Optional[float] = DEFAULT_EMBEDDINGS_SCALE
    """Scales the embedding hidden states (i.e. the tensor after embeddings &
    embedding layer norm are applied). Required in muP training"""

    scale_qk_dot_by_d: bool = False
    """Scales attention QK dot product by d instead of sqrt(d). Must be enabled
    for muP training. Note that this flag only has effect if
    attention_type=scaled_dot_product"""

    attention_logits_alpha: Optional[float] = DEFAULT_ATTENTION_LOGITS_ALPHA
    """Additionally scales the attention QK dot product by the specified value.
    Recommended to tune for stabilizing attention logits in muP training."""

    scale_output_logits_by_d: Optional[bool] = DEFAULT_SCALE_OUTPUT_LOGITS_BY_D
    """Scales the output logits in muP by mup_base_hidden_size/hidden_size if
    True and sqrt(mup_base_hidden_size/hidden_size) if False. Only applies to
    muP training when scaling the hidden_size"""

    output_logits_alpha: Optional[float] = DEFAULT_OUTPUT_LOGITS_ALPHA
    """Constant applied to the output logits scalar in muP training. The output
    logits are scaled by output_logits_alpha * mup_base_hidden_size/hidden_size"""

    alibi_trainable_slopes: bool = False
    "Replaces alibi's fixed slopes with trainable slopes."

    pos_scaling_factor: float = 1.0
    """Position interpolation scaling factor for rotary & alibi. See
    https://arxiv.org/pdf/2306.15595.pdf for details"""

    pos_scaling_type: Literal["linear", "YaRN", "yarn", "longrope"] = "linear"
    """Can be either `linear` or `YaRN` or 'longrope',
    For YaRN see https://arxiv.org/pdf/2309.00071
    For LongRope see https://arxiv.org/pdf/2402.13753"""

    pos_scaling_extra_args: Optional[dict] = None
    """A dict including parameters for YaRN/longrope RoPE scaling"""

    moe_params: MoEConfig = Field(default_factory=MoEConfig, alias="moe")

    dtype: Optional[torch.dtype] = None

    @model_validator(mode="after")
    def validate_rotary_dim(self):
        if self.position_embedding_type == "rotary":
            # https://github.com/huggingface/transformers/blob/f0577df6de36e7e7f28e90fa76da0657de038a39/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L84-L85
            # https://arxiv.org/pdf/2104.09864.pdf Section 3.3
            if self.rotary_dim > self.hidden_size / self.num_heads:
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

        return self

    def post_init(self, context):
        super().post_init(context)

        if self.position_embedding_type == "rotary":
            if self.rotary_dim is None:
                self.rotary_dim = int(self.hidden_size // self.num_heads * 0.25)

            self.num_relative_attention_buckets = None
        else:
            self.rotary_dim = None

            if self.num_relative_attention_buckets == None:
                self.num_relative_attention_buckets = 32


class GPTJModel(nn.Module):
    def __init__(self, config: GPTJSubModelConfig):
        if isinstance(config, dict):
            config = GPTJSubModelConfig(**config)

        super().__init__()

        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        max_position_embeddings = config.max_position_embeddings
        embd_pdrop = config.embd_pdrop
        share_embedding_weights = config.share_embedding_weights
        position_embedding_type = config.position_embedding_type
        rotary_dim = config.rotary_dim
        rope_theta = config.rope_theta
        fold_rope_consts = config.fold_rope_consts
        num_relative_attention_buckets = config.num_relative_attention_buckets
        num_hidden_layers = config.num_hidden_layers
        filter_size = config.filter_size
        dropout_rate = config.dropout_rate
        nonlinearity = config.nonlinearity
        norm_type = config.norm_type
        layer_norm_epsilon = config.layer_norm_epsilon
        use_ffn_bias = config.use_ffn_bias
        use_untied_layer_norm = config.use_untied_layer_norm
        num_heads = config.num_heads
        attention_module = config.attention_module
        attention_sliding_window_length = config.attention_sliding_window_length
        attention_sink_tokens = config.attention_sink_tokens
        extra_attention_params = config.extra_attention_params
        attention_type = config.attention_type
        attention_dropout_rate = config.attention_dropout_rate
        attention_softmax_fp32 = config.attention_softmax_fp32
        attention_kernel = config.attention_kernel
        use_projection_bias_in_attention = (
            config.use_projection_bias_in_attention
        )
        use_ffn_bias_in_attention = config.use_ffn_bias_in_attention
        initializer_range = config.initializer_range
        use_bias_in_output = config.use_bias_in_output
        embedding_initializer = config.embedding_initializer
        attention_initializer = config.attention_initializer
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
        moe_params = config.moe_params
        dtype = config.dtype

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.share_embedding_weights = share_embedding_weights
        self.initializer_range = initializer_range
        self.num_heads = num_heads
        total_experts = moe_params.num_experts
        if moe_params.num_shared_experts:
            total_experts += moe_params.num_shared_experts
        self.moe_enabled = total_experts > 1
        self.moe_num_experts = moe_params.num_experts
        self.moe_routing_algorithm = moe_params.routing_algorithm
        default_initializer = {
            "name": "truncated_normal",
            "std": self.initializer_range,
            "mean": 0.0,
            "a": self.initializer_range * -2.0,
            "b": self.initializer_range * 2.0,
        }
        if embedding_initializer is None:
            embedding_initializer = default_initializer.copy()
        if attention_initializer is None:
            attention_initializer = default_initializer.copy()
        if output_layer_initializer is None:
            output_layer_initializer = default_initializer.copy()
        if moe_params.gate_initializer is None:
            # Update the config with initializer selected
            moe_params = moe_params.copy(
                update=dict(gate_initializer=default_initializer)
            )
        if ffn_initializer is None:
            ffn_initializer = output_layer_initializer.copy()
        if ffn_output_layer_initializer is None:
            ffn_output_layer_initializer = output_layer_initializer.copy()

        # Handle muP scaling
        self.embeddings_scale = embeddings_scale
        self.output_logits_scale = None
        if mup_base_hidden_size:
            hidden_size_width_mult = hidden_size / mup_base_hidden_size
            scale_initializers_by_dimension(
                [attention_initializer, ffn_initializer],
                width_scale=hidden_size_width_mult**-0.5,
            )
            scale_initializers_by_dimension(
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
            scale_initializers_by_dimension(
                ffn_output_layer_initializer,
                width_scale=filter_size_width_mult**-0.5,
                depth_scale=(2 * num_hidden_layers) ** -0.5,
            )
            lr_adjustment_groups["decoder_output_ffn"].set_scale(
                1 / filter_size_width_mult
            )

        # embedding layer that only contains token embeddings
        self.embedding_layer = EmbeddingLayer(
            vocab_size=vocab_size,
            embedding_size=hidden_size,
            embeddings_initializer=embedding_initializer,
            position_embedding_type=position_embedding_type,
            position_embeddings_initializer=embedding_initializer,
            max_position_embeddings=max_position_embeddings,
            num_heads=num_heads,
            num_relative_attention_buckets=num_relative_attention_buckets,
            rotary_dim=rotary_dim,
            rope_theta=rope_theta,
            fold_rope_consts=fold_rope_consts,
            pos_scaling_factor=pos_scaling_factor,
            pos_scaling_type=pos_scaling_type,
            pos_scaling_extra_args=pos_scaling_extra_args,
            dtype=dtype,
        )

        self.drop_embd = nn.Dropout(embd_pdrop)

        norm_class = get_norm(norm_type)

        extra_attention_params["attention_kernel"] = attention_kernel
        decoder_layer = GPTJDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            use_untied_layer_norm=use_untied_layer_norm,
            dim_feedforward=filter_size,
            dropout=dropout_rate,
            activation=nonlinearity,
            layer_norm_eps=layer_norm_epsilon,
            norm_layer=norm_class,
            attention_module=attention_module,
            extra_attention_params=extra_attention_params,
            add_cross_attention=False,
            attention_type=attention_type,
            scale_qk_dot_by_d=scale_qk_dot_by_d,
            attention_logits_alpha=attention_logits_alpha,
            attention_dropout_rate=attention_dropout_rate,
            attention_softmax_fp32=attention_softmax_fp32,
            use_projection_bias_in_attention=use_projection_bias_in_attention,
            use_ffn_bias_in_attention=use_ffn_bias_in_attention,
            use_ffn_bias=use_ffn_bias,
            attention_initializer=attention_initializer,
            attention_output_layer_initializer=output_layer_initializer,
            ffn_initializer=ffn_initializer,
            ffn_output_layer_initializer=ffn_output_layer_initializer,
            use_ff_layer1_dropout=False,
            norm_first=True,
            moe_params=moe_params,
        )

        self.ln_f = norm_class(hidden_size, eps=layer_norm_epsilon)

        self.transformer_decoder = TransformerDecoder(
            decoder_layer, num_layers=num_hidden_layers, norm=self.ln_f
        )

        self.attention_sliding_window_length = attention_sliding_window_length
        self.attention_sink_tokens = attention_sink_tokens

        self.lm_head = nn.Linear(
            hidden_size, vocab_size, bias=use_bias_in_output
        )

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

    def get_input_embeddings(self):
        return self.embedding_layer.get_input_embeddings()

    def get_output_embeddings(self):
        return self.lm_head

    def compute_input_embeddings(self, input_ids, position_ids=None):
        hidden_states = self.embedding_layer(
            input_ids, position_ids=position_ids
        )
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
        tgt_key_padding_mask=None,
        position_ids=None,
        input_embeddings=None,
        inference_loop_index=None,
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

        if input_embeddings is None:
            hidden_states = self.compute_input_embeddings(
                input_ids, position_ids
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
            attention_span=attention_span,
            position_ids=position_ids,
            tgt_key_padding_mask=tgt_key_padding_mask,
            expert_hash_idx=expert_hash_idx,
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

        if self.moe_enabled:
            return lm_logits, routing_weights, expert_masks
        else:
            return lm_logits

    def apply_decoder(
        self,
        input_embeddings,
        attention_mask=None,
        attention_span=None,
        tgt_key_padding_mask=None,
        position_ids=None,
        extract_layer_idx=None,
        expert_hash_idx=None,
    ):
        # `extract_layer_idx` is used only multimodal use case
        # input_embeddings : shape (bsz, MSL, H)
        causal_attention_mask = create_broadcasted_autoregressive_mask(
            batch_size=input_embeddings.shape[0],
            num_heads=self.num_heads,
            tgt_seq_length=input_embeddings.shape[1],
            attention_span=attention_span,
            attention_sliding_window_length=self.attention_sliding_window_length,
            attention_sink_tokens=self.attention_sink_tokens,
            device=input_embeddings.device,
            dtype=input_embeddings.dtype,
        )

        # Helpers on alibi/relative position embeddings bias
        length = input_embeddings.shape[1]
        self_attn_position_bias = self.embedding_layer.compute_position_bias(
            length, length
        )

        return self.transformer_decoder(
            input_embeddings,
            tgt_mask=causal_attention_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            rotary_position_embedding_helper=self.embedding_layer.get_rope_helper(),
            self_attn_position_bias=self_attn_position_bias,
            extract_layer_idx=extract_layer_idx,
            expert_hash_idx=expert_hash_idx,
        )

    def extract_features(
        self,
        input_embeddings,
        extract_layer_idx,
        attention_mask=None,
        attention_span=None,
        tgt_key_padding_mask=None,
        position_ids=None,
    ):
        """
        Extract features of input_embeddings from `extract_layer_idx` of decoder
        extract_layer_idx: (inclusive)layer index in range [0, self.num_layers) (zero-indexed)
            Applies decoder layers up to (and including) `extract_layer_idx`
            instead of all decoder layers.
            For ex: extract_layer_idx=3 would run fwd pass from decoder_block_0 to decoder_block_3
            and return outputs from decoder_block_3.
            If `extract_layer_idx` = None and `norm` != None, then
            the output returned would be decoder_block_{self.num_layers-1} -> norm -> output (return)

        This function is added for multimodal use case.
        """
        hidden_states = self.apply_decoder(
            input_embeddings,
            extract_layer_idx=extract_layer_idx,
            attention_mask=attention_mask,
            attention_span=attention_span,
            tgt_key_padding_mask=tgt_key_padding_mask,
            position_ids=position_ids,
        )

        if isinstance(hidden_states, tuple):
            return hidden_states[0]
        else:
            return hidden_states
