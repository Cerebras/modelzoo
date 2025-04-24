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
import warnings
from typing import Dict, List, Literal, Optional, Union
from warnings import warn

import torch
import torch.nn as nn
from annotated_types import Ge, Le
from pydantic import field_validator, model_validator
from typing_extensions import Annotated

from cerebras.modelzoo.common.utils.model.mup_utils import (
    LRAdjustmentGroup,
    scale_initializers_by_dimension,
)
from cerebras.modelzoo.common.utils.model.transformer_utils import (
    create_vsl_mask,
    make_key_padding_mask_broadcastable,
)
from cerebras.modelzoo.config import ModelConfig
from cerebras.modelzoo.layers import (
    EmbeddingLayer,
    FeedForwardNetwork,
    FeedForwardNetworkConfig,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from cerebras.modelzoo.layers.activations import ActivationType
from cerebras.modelzoo.layers.init import (
    InitializerConfig,
    TruncatedNormalInitializer,
)


class BertModelConfig(ModelConfig):
    name: Literal["BertModel", "bert_model"]

    ### Embedding
    vocab_size: Annotated[int, Ge(1), Le(512000)] = 50257
    "The size of the vocabulary used in the model. Max supported value is `512000`."

    max_position_embeddings: Optional[int] = None
    "The maximum sequence length that the model can handle."

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

    hidden_size: int = 768
    "Dimensionality of the encoder layers and the pooler layer."

    embedding_dropout_rate: Annotated[float, Ge(0.0), Le(1.0)] = 0.1
    "The dropout ratio for the word embeddings."

    embedding_pad_token_id: Optional[int] = 0
    "The embedding vector at embedding_pad_token_id is not updated during training."

    mask_padding_in_positional_embed: bool = False
    """Whether to mask padding in positional embeddings.
    Only supported with `position_embedding_type` set to `learned`."""

    rotary_dim: Optional[int] = None
    "The number of dimensions used for the rotary position embedding."

    rope_theta: float = 10000
    """Frequency (theta) used in rotary position embedding. This value is
    typically adjusted in long MSL runs as described in
    [CodeLlama](https://arxiv.org/pdf/2308.12950.pdf)"""

    fold_rope_consts: bool = False
    """If True, folds the rotary position embedding constants compile time.

    For very large models consider generating them on the fly by setting this to
    False. It avoids working with large constant tensors.
    """

    num_relative_attention_buckets: int = 32
    "Number of buckets to use in relative position embedding"

    alibi_trainable_slopes: bool = False
    "Replaces alibi's fixed slopes with trainable slopes."

    pos_scaling_factor: float = 1.0
    """Position interpolation scaling factor for rotary & alibi. See
    https://arxiv.org/pdf/2306.15595.pdf for details"""

    pos_scaling_type: Literal["linear", "yarn"] = "linear"
    """Can be either `linear` or `YaRN`,
    For YaRN see https://arxiv.org/pdf/2309.00071"""

    pos_scaling_extra_args: Optional[dict] = None
    """A dict including parameters for YaRN RoPE scaling"""

    ### Encoder
    num_hidden_layers: int = 12
    "Number of hidden layers in the Transformer decoder."

    layer_norm_epsilon: Union[float, List[float]] = 1e-5
    "The epsilon value used in layer normalization layers."

    norm_first: bool = False

    embedding_layer_norm: bool = True

    ### Encoder Attn
    num_heads: int = 12
    "The number of attention heads."

    attention_module: Literal["aiayn_attention", "multiquery_attention"] = (
        "aiayn_attention"
    )
    """Determines whether to use multiheaded attention (from the Attention is
    All You Need paper) or multi-query/grouped-query attention. When using the
    latter, you must specify extra_attention_params (see below).
    """

    extra_attention_params: dict = {}
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

    attention_softmax_fp32: bool = True
    "Whether to use fp32 precision for attention softmax."

    attention_kernel: Optional[str] = None
    """The type of attention kernel"""

    dropout_rate: float = 0.1
    "The dropout probability for all fully connected layers."

    nonlinearity: ActivationType = "gelu"
    "The non-linear activation function (function or string) in the encoder and pooler."

    pooler_nonlinearity: Optional[str] = None
    """The non-linear activation function used in the pooler layer. If not
    specified, defaults to encoder_nonlinearity."""

    attention_dropout_rate: Optional[float] = None
    "Dropout rate for attention layer. When None, defaults to same as `dropout_rate`."

    use_projection_bias_in_attention: bool = True
    "Whether to include bias in the attention layer for Q/K/V projections."

    use_ffn_bias_in_attention: bool = True
    """Whether to include bias in the attention layer for output projection
    after values have been combined (W_O in original Transformer paper).
    """

    ### Encoder ffn
    filter_size: int = 3072
    "Dimensionality of the feed-forward layer in the Transformer block."

    use_ffn_bias: bool = True
    "Whether to use bias in the feedforward network (FFN)."

    ### Task-specific
    use_final_layer_norm: bool = False

    initializer_range: float = 0.02
    "The standard deviation of the truncated_normal_initializer as the default initializer"

    default_initializer: Optional[InitializerConfig] = None

    embeddings_initializer: Optional[InitializerConfig] = None
    "Initializer for word embeddings."

    position_embeddings_initializer: Optional[InitializerConfig] = None
    "Initializer for position embeddings (if learned position embeddings)."

    segment_embeddings_initializer: Optional[InitializerConfig] = None
    "Initializer for segment embeddings."

    num_segments: Optional[int] = None
    """Number of segments (token types) in embedding. When not specified
    (and NSP objective is enabled), num_segments will default to 2"""

    add_pooling_layer: bool = True
    "Whether to add the pooling layer for sequence classification."

    freeze_ffn_bias_in_glu: bool = False
    "Prevents gradients from being computed for FFN biases for GLU activation layers"

    ### muP (Maximal Update Parametrization)
    lr_adjustment_groups: Dict[str, Union[str, List[str]]] = {
        "embedding": "*embedding*weight",
        "encoder_attention": "*transformer_encoder*attn*dense*weight",
        "encoder_input_ffn": "*transformer_encoder*ffn.ffn.[!1]*weight",
        "encoder_output_ffn": "*transformer_encoder*ffn.ffn.[1]*weight",
        "pooler": "*pooler*weight",
    }
    "A dictionary of groups to adjust the learning rate for."

    mup_base_hidden_size: Optional[float] = None
    """The hidden size of the base model in muP transfer used to calculate the
    necessary multipliers. Required in muP training when scaling the encoder
    attention module"""

    mup_base_filter_size: Optional[float] = None
    """The filter size of the base model in muP transfer used to calculate the
    necessary multipliers. Required in muP training when scaling the encoder
    ffn"""

    embeddings_scale: float = 1.0
    """Scales the embedding hidden states (i.e. the tensor after embeddings &
    embedding layer norm are applied). Required in muP training"""

    scale_qk_dot_by_d: Optional[bool] = None
    """Scales attention QK dot product by d instead of sqrt(d). Must be enabled
    for muP training."""

    attention_logits_alpha: float = 1.0
    """Additionally scales the attention QK dot product by the specified value.
    Recommended to tune for stabilizing attention logits in muP training."""

    output_logits_alpha: float = 1.0
    """Constant applied to the output logits scalar in muP training. The msm
    and nsp logits are scaled by output_logits_alpha * mup_base_hidden_size/hidden_size"""

    scale_output_logits_by_d: bool = True
    """Scales the output logits in muP by mup_base_hidden_size/hidden_size if
    True and sqrt(mup_base_hidden_size/hidden_size) if False. Only applies to
    muP training when scaling the hidden_size"""

    dtype: Optional[torch.dtype] = None

    @field_validator("name", mode="after")
    def validate_name(cls, name):
        if name == "BertModel":
            warn(
                "Passing 'BertModel' as the model name is deprecated. "
                "Please use 'bert_model' instead.",
                category=FutureWarning,
            )
            return "bert_model"
        return name

    @field_validator("pos_scaling_type", mode="before")
    @classmethod
    def validate_pos_scaling_type(cls, value: str) -> str:
        if isinstance(value, str):
            return value.lower()
        return value

    @model_validator(mode="after")
    def validate_position_embeddings(self):
        if (
            self.position_embedding_type is not None
            and self.max_position_embeddings is None
        ):
            raise ValueError(
                "max_position_embeddings should be specified "
                "if position_embedding_type is specified."
            )
        return self

    def post_init(self, context):
        super().post_init(context)

        if self.position_embedding_type == "rotary":
            if self.rotary_dim is None:
                self.rotary_dim = int(self.hidden_size // self.num_heads * 0.25)
            # https://github.com/huggingface/transformers/blob/f0577df6de36e7e7f28e90fa76da0657de038a39/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L84-L85
            # https://arxiv.org/pdf/2104.09864.pdf Section 3.3
            if self.rotary_dim > (self.hidden_size / self.num_heads):
                raise ValueError(
                    f"Rotary dimension ({self.rotary_dim}) must be <= "
                    f"hidden size ({self.hidden_size}) divided by number of "
                    f"attention heads ({self.num_heads})."
                )
            if self.rotary_dim % 2 != 0:
                raise ValueError(
                    f"Rotary dimension ({self.rotary_dim}) must be an even number."
                )

        if self.default_initializer is None:
            self.default_initializer = TruncatedNormalInitializer(
                std=self.initializer_range,
                mean=0.0,
                a=self.initializer_range * -2.0,
                b=self.initializer_range * 2.0,
            )

        if self.embeddings_initializer is None:
            self.embeddings_initializer = self.default_initializer.copy()

        if self.position_embeddings_initializer is None:
            self.position_embeddings_initializer = (
                self.default_initializer.copy()
            )

        if self.segment_embeddings_initializer is None:
            self.segment_embeddings_initializer = (
                self.default_initializer.copy()
            )

        supported_mup_dimensions = [
            'mup_base_hidden_size',
            'mup_base_filter_size',
        ]

        if detected_mup_dimensions := [
            dimension
            for dimension in supported_mup_dimensions
            if getattr(self, dimension)
        ]:
            if detected_mup_dimensions != supported_mup_dimensions:
                raise ValueError(
                    f"Our muP formulation requires that you specify all "
                    f"of the following base dimensions: {supported_mup_dimensions} "
                    f"but only the following dimensions were found: "
                    f"{detected_mup_dimensions}"
                )

            if self.scale_qk_dot_by_d is None:
                self.scale_qk_dot_by_d = True

        else:
            if detected_mup_tunable_params := [
                name
                for name in [
                    "embeddings_scale",
                    "output_logits_alpha",
                    "scale_qk_dot_by_d",
                    "scale_output_logits_by_d",
                    "attention_logits_alpha",
                ]
                if getattr(self, name)
                != self.__class__.model_fields[name].default
            ]:
                warnings.warn(
                    f"The following muP parameters were changed from their default "
                    f"value outside of a muP run: {detected_mup_tunable_params}. "
                    f"As a result, they may have an undesired effect. Please "
                    f"specify the muP base dimensions {supported_mup_dimensions} "
                    f"to trigger a muP run."
                )

            if self.scale_qk_dot_by_d is None:
                self.scale_qk_dot_by_d = False


class BertModel(nn.Module):
    """
    The model behaves as a bidirectional encoder (with only self-attention), following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    """

    def __init__(self, config: BertModelConfig):
        super().__init__()

        self.num_heads = config.num_heads
        self.add_pooling_layer = config.add_pooling_layer
        self.freeze_ffn_bias_in_glu = config.freeze_ffn_bias_in_glu

        attention_initializer = config.default_initializer.copy()
        ffn_initializer = config.default_initializer.copy()
        output_layer_initializer = config.default_initializer.copy()
        ffn_output_layer_initializer = config.default_initializer.copy()

        self.lr_adjustment_groups = {
            key: LRAdjustmentGroup(value)
            for key, value in config.lr_adjustment_groups.items()
        }

        if config.mup_base_hidden_size or config.mup_base_filter_size:
            logging.info("This is a muP configured run")

        # Handle muP scaling
        self.embeddings_scale = config.embeddings_scale
        if config.mup_base_hidden_size:
            hidden_size_width_mult = (
                config.hidden_size / config.mup_base_hidden_size
            )
            scale_initializers_by_dimension(
                [attention_initializer, ffn_initializer],
                width_scale=hidden_size_width_mult**-0.5,
            )
            scale_initializers_by_dimension(
                output_layer_initializer,
                width_scale=hidden_size_width_mult**-0.5,
                depth_scale=(2 * config.num_hidden_layers) ** -0.5,
            )
            for lr_adjustment_group in [
                "encoder_attention",
                "encoder_input_ffn",
                "pooler",
            ]:
                self.lr_adjustment_groups[lr_adjustment_group].set_scale(
                    1 / hidden_size_width_mult
                )

        if config.mup_base_filter_size:
            filter_size_width_mult = (
                config.filter_size / config.mup_base_filter_size
            )
            scale_initializers_by_dimension(
                ffn_output_layer_initializer,
                width_scale=filter_size_width_mult**-0.5,
                depth_scale=(2 * config.num_hidden_layers) ** -0.5,
            )
            self.lr_adjustment_groups["encoder_output_ffn"].set_scale(
                1 / filter_size_width_mult
            )

        self.embedding_layer = EmbeddingLayer(
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            pad_token_id=config.embedding_pad_token_id,
            embeddings_initializer=config.embeddings_initializer,
            max_position_embeddings=config.max_position_embeddings,
            position_embedding_type=config.position_embedding_type,
            position_embedding_offset=(
                # We only need to add position embedding offset when we're using
                # masked padding in positional embed
                config.embedding_pad_token_id
                if config.mask_padding_in_positional_embed
                else 0
            ),
            mask_padding_in_positional_embed=config.mask_padding_in_positional_embed,
            position_embeddings_initializer=config.position_embeddings_initializer,
            num_segments=config.num_segments,
            segment_embeddings_initializer=config.segment_embeddings_initializer,
            num_heads=self.num_heads,
            num_relative_attention_buckets=config.num_relative_attention_buckets,
            rotary_dim=config.rotary_dim,
            rope_theta=config.rope_theta,
            fold_rope_consts=config.fold_rope_consts,
            alibi_trainable_slopes=config.alibi_trainable_slopes,
            pos_scaling_factor=config.pos_scaling_factor,
            pos_scaling_type=config.pos_scaling_type,
            pos_scaling_extra_args=config.pos_scaling_extra_args,
            dtype=config.dtype,
        )

        self.dropout_embd = nn.Dropout(config.embedding_dropout_rate)

        config.extra_attention_params["attention_kernel"] = (
            config.attention_kernel
        )
        encoder_layer = TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=config.filter_size,
            dropout=config.dropout_rate,
            activation=config.nonlinearity,
            layer_norm_eps=config.layer_norm_epsilon,
            scale_qk_dot_by_d=config.scale_qk_dot_by_d,
            attention_logits_alpha=config.attention_logits_alpha,
            attention_module=config.attention_module,
            extra_attention_params=config.extra_attention_params,
            attention_dropout_rate=config.attention_dropout_rate,
            attention_type=config.attention_type,
            attention_softmax_fp32=config.attention_softmax_fp32,
            use_projection_bias_in_attention=config.use_projection_bias_in_attention,
            use_ffn_bias_in_attention=config.use_ffn_bias_in_attention,
            use_ffn_bias=config.use_ffn_bias,
            attention_initializer=attention_initializer,
            ffn_initializer=ffn_initializer,
            attention_output_layer_initializer=output_layer_initializer,
            ffn_output_layer_initializer=ffn_output_layer_initializer,
            norm_first=config.norm_first,
        )

        if config.embedding_layer_norm:
            self.embed_ln_f = nn.LayerNorm(
                config.hidden_size, eps=config.layer_norm_epsilon
            )
        else:
            self.embed_ln_f = None

        final_ln_f = None
        if config.use_final_layer_norm:
            final_ln_f = nn.LayerNorm(
                config.hidden_size, eps=config.layer_norm_epsilon
            )

        self.transformer_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=config.num_hidden_layers,
            norm=final_ln_f,
        )

        self.pooler = (
            BertPooler(
                config.hidden_size,
                use_bias=config.use_ffn_bias,
                activation=config.pooler_nonlinearity or config.nonlinearity,
                dropout=None,
                initializer=attention_initializer,
            )
            if self.add_pooling_layer
            else None
        )

        self.__reset_parameters()

        # TODO: Add sparse attention

    def reset_parameters(self):
        self.embedding_layer.reset_parameters()
        self.transformer_encoder.reset_parameters()
        self.pooler.reset_parameters()
        self.__reset_parameters()

    def __reset_parameters(self):
        # Init norm layers
        if self.embed_ln_f is not None:
            self.embed_ln_f.bias.data.zero_()
            self.embed_ln_f.weight.data.fill_(1.0)
        # Freeze glu linear layer biases if needed
        if self.freeze_ffn_bias_in_glu:
            freeze_layers = []
            for n, p in self.transformer_encoder.named_parameters():
                if "linear_layer_for_glu.bias" in n:
                    freeze_layers.append(n)
                    # We have two linear layers for glu
                    freeze_layers.append(
                        n.replace("linear_layer_for_glu", "linear_layer")
                    )
            for n, p in self.transformer_encoder.named_parameters():
                if n in freeze_layers:
                    p.data.zero_()
                    p.requires_grad = False

    def get_lr_adjustment_groups(self) -> Dict[str, LRAdjustmentGroup]:
        return self.lr_adjustment_groups

    def compute_input_embeddings(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        segment_ids=None,
    ):
        return self.embedding_layer(
            input_ids,
            position_ids=position_ids,
            segment_ids=segment_ids,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        segment_ids=None,
        attention_span=None,
    ):
        """
        Args:
            input_ids (Tensor): The id of input tokens
                Can be of shape ```[batch_size, seq_length]`
            position_ids (Tensor):
                The position id of input tokens. Can be of shape ``[batch_size, seq_length]``
            segment_ids (Tensor): The segment id of input tokens, indicating which sequence the token belongs to
                Can be of shape ```[batch_size, seq_length]`
            attention_mask (Tensor):
                Can be 2D of shape ``[batch_size, seq_length]``,
                or 3D of shape ``[batch, query_length, seq_length]``,
                or 4D of shape ``[batch, num_heads, query_length, seq_length]``.
            attention_span (Tensor):
                The attention span of input tokens for creating VSL mask. Can be of shape `[batch_size, seq_length]`.
        """
        src_key_padding_mask = None

        hidden_states = self.compute_input_embeddings(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            segment_ids=segment_ids,
        )
        hidden_states = hidden_states * torch.tensor(
            float(self.embeddings_scale), dtype=hidden_states.dtype
        )

        if self.embed_ln_f is not None:
            hidden_states = self.embed_ln_f(hidden_states)
        hidden_states = self.dropout_embd(hidden_states)

        if attention_mask is not None:
            attention_mask = make_key_padding_mask_broadcastable(
                attention_mask, dtype=hidden_states.dtype
            )
            if len(attention_mask.size()) == 2:
                src_key_padding_mask = attention_mask
                attention_mask = None

        # Compute alibi/relative position embeddings bias
        length = input_ids.shape[1]
        self_attn_position_bias = self.embedding_layer.compute_position_bias(
            length, length
        )

        # Computes VSL mask and applies it attention mask.
        if attention_span is not None:
            vsl_attn_mask = create_vsl_mask(
                attention_span=attention_span,
                position_ids=position_ids,
                num_heads=self.num_heads,
                is_causal=False,
                device=input_ids.device,
                dtype=hidden_states.dtype,
                use_neg_inf=True,
            )
            # VSL attention mask contains the padding masking, and we no longer need key padding mask.
            attention_mask = vsl_attn_mask
            src_key_padding_mask = None

        hidden_states = self.transformer_encoder(
            hidden_states,
            mask=attention_mask,
            src_key_padding_mask=src_key_padding_mask,
            rotary_position_embedding_helper=self.embedding_layer.get_rope_helper(),
            self_attn_position_bias=self_attn_position_bias,
        )

        pooled_output = None
        if self.add_pooling_layer:
            pooled_output = self.pooler(hidden_states)

        return hidden_states, pooled_output


class BertPooler(nn.Module):
    def __init__(
        self,
        hidden_size,
        pooler_norm=False,
        layer_norm_epsilon=1.0e-5,
        use_bias=True,
        activation="gelu",
        dropout=None,
        initializer="xavier_uniform",
    ):
        super().__init__()

        self.pooler_norm = None
        if pooler_norm:
            self.pooler_norm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)

        self.pooler = FeedForwardNetwork(
            FeedForwardNetworkConfig(
                input_unit=hidden_size,
                layers_units=[hidden_size],
                layers_activation=[activation],
                layers_dropout_rates=[dropout],
                use_bias=use_bias,
                kernel_initializer=initializer,
            )
        )

    def reset_parameters(self):
        if self.pooler_norm is not None:
            self.pooler_norm.weight.data.fill_(1.0)
            if self.pooler_norm.bias is not None:
                self.pooler_norm.bias.data.zero_()
        self.pooler.reset_parameters()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state
        # corresponding to the first token.
        # shape [batch_size, hidden_size]
        cls_hidden_states = hidden_states[:, 0]
        if self.pooler_norm is not None:
            cls_hidden_states = self.pooler_norm(cls_hidden_states)
        pooled_output = self.pooler(cls_hidden_states)
        return pooled_output
