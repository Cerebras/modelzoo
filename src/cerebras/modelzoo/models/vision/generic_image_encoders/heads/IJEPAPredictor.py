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

import math
from typing import List, Literal, Optional

import torch
import torch.nn as nn

from cerebras.modelzoo.common.utils.model.transformer_utils import (
    replace_with_zero_and_neg_inf,
)
from cerebras.modelzoo.config import ModelConfig
from cerebras.modelzoo.layers.init import (
    InitializerConfig,
    TruncatedNormalInitializer,
    ZerosInitializer,
)
from cerebras.modelzoo.layers.utils import get_2d_fixed_position_embeddings
from cerebras.modelzoo.models.vision.generic_image_encoders.utils import misc
from cerebras.modelzoo.models.vision.vision_transformer.ViTModel import (
    ViTEncoder,
)


class IJEPAPredictorConfig(ModelConfig):
    name: Literal["IJEPAPredictor"]

    num_patches: List[int] = ...
    """Number of patches in height and width of image. 
    Computed using image_height//patch_height, image_width//patch_width
    """
    projection_dim: int = 384
    "Size of transformation of input from hidden_size to projection_dim "

    hidden_size: int = 768
    "The size of the transformer hidden layers."

    # Encoder
    num_hidden_layers: int = 12
    "Number of hidden layers in the Transformer encoder."

    layer_norm_epsilon: float = 1.00e-05
    "The epsilon value used in layer normalization layers."

    # Encoder Attn
    num_heads: int = 12
    "The number of attention heads in the multi-head attention layer."

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
    can set `num_kv_groups: 1` to enable MQA or `num_kv_groups: <groups>` for
    GQA. The number of groups should be divisible by `num_heads`.
    """

    attention_type: Literal["dot_product", "scaled_dot_product"] = (
        "scaled_dot_product"
    )
    """Determines whether the QK dot product should be scaled -
    dot_product -> QK^T
    scaled_dot_product -> QK^T / sqrt(d)

    Note that setting either scale_qk_dot_by_d or scale_qk_dot_by_layer_idx will
    result in different behavior.
    """

    attention_softmax_fp32: bool = True
    "Whether to use fp32 precision for attention softmax."

    dropout_rate: float = 0.0
    "The dropout probability for all fully connected layers."

    nonlinearity: str = "gelu"
    """The non-linear activation function used in the feed forward network
    in each transformer block. Some may have to use autogen_policy: `medium`."""

    attention_dropout_rate: float = 0.0
    "Dropout rate for attention layer."

    use_projection_bias_in_attention: bool = True
    "Whether to include bias in the attention layer for projection."

    use_ffn_bias_in_attention: bool = True
    "Whether to include bias in the attention layer for feed-forward network (FFN)."

    # Encoder ffn
    filter_size: Optional[int] = 3072
    "Dimensionality of the feed-forward layer in the Transformer block."

    use_ffn_bias: Optional[bool] = True
    "Whether to use bias in the feedforward network (FFN)."

    # Task-specific
    use_final_layer_norm: Optional[bool] = True

    initializer_range: float = 0.02
    """FeedForward network"""

    default_initializer: Optional[InitializerConfig] = None

    projection_initializer: Optional[InitializerConfig] = None
    """Initializer for embedding linear layer. Either a string indicating the name of the
    initializer or a dict that includes the name + other params if relevant. If left
    unspecified will apply truncated normal initialization."""

    attention_initializer: Optional[InitializerConfig] = None

    ffn_initializer: Optional[InitializerConfig] = None
    """The name of the initializer for the weights of the ffn kernel."""

    norm_first: bool = True
    """Enables normalization before the Attention & FFN blocks (i.e Pre-LN as
    described in https://arxiv.org/pdf/2002.04745.pdf. When disabled,
    normalization is applied *after* the residual (Post-LN)"""

    prepend_cls_token: bool = False
    "If True, prepends cls token to input."

    layerscale_value: Optional[float] = None

    stochastic_depth_drop_prob: Optional[float] = 0.0

    stochastic_depth_mode: Optional[str] = "batch"

    @property
    def __model_cls__(self):
        return IJEPAPredictor


class IJEPAPredictor(nn.Module):
    def __init__(self, config: IJEPAPredictorConfig):
        if isinstance(config, dict):
            config = IJEPAPredictorConfig(**config)

        super().__init__()

        self.predictor_initial_projection = nn.Linear(
            config.hidden_size, config.projection_dim, bias=True
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.projection_dim))

        position_embeddings = get_2d_fixed_position_embeddings(
            config.num_patches,
            config.projection_dim,
            add_cls_token=False,
        )  # shape: (num_patches, projection_dim)

        self.position_embeddings = nn.Parameter(
            position_embeddings.unsqueeze(0), requires_grad=False
        )  # (1, num_patches, projection_dim)

        self.encoder = ViTEncoder(
            # Embedding
            hidden_size=config.projection_dim,
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
            pooler_nonlinearity=None,
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
            pooler_initializer=None,
            norm_first=config.norm_first,
            use_encoder_pooler_layer=False,
            layerscale_value=config.layerscale_value,
            stochastic_depth_drop_prob=config.stochastic_depth_drop_prob,
            stochastic_depth_mode=config.stochastic_depth_mode,
        )

        self.predictor_final_projection = nn.Linear(
            config.projection_dim, config.hidden_size, bias=True
        )
        self.initializer_range = config.initializer_range
        self.default_initializer = (
            TruncatedNormalInitializer(
                std=self.initializer_range,
                mean=0.0,
                a=self.initializer_range * -2.0,
                b=self.initializer_range * 2.0,
            )
            if config.default_initializer is None
            else config.default_initializer
        )

        self.__reset_parameters()

    def __reset_parameters(self):
        self.default_initializer(self.mask_token.data)
        self.default_initializer(self.predictor_initial_projection.weight.data)
        ZerosInitializer()(self.predictor_initial_projection.bias.data)

        self.default_initializer(self.predictor_final_projection.weight.data)
        ZerosInitializer()(self.predictor_final_projection.bias.data)

        self.encoder.reset_parameters()

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(
            self.encoder.transformer_encoder.layers
        ):
            rescale(
                layer.self_attn.proj_output_dense_layer.weight.data,
                layer_id + 1,
            )
            rescale(
                layer.ffn.ffn[1].linear_layer.weight.data, layer_id + 1
            )  # rescale only fc2

    def forward(
        self,
        x,
        masks_encoder,
        masks_predictor,
        num_valid_mask_patches_encoder=None,
        num_valid_mask_patches_predictor=None,
    ):
        """
        x: torch.Tensor of shape (bsz, num_encoder_masks, max_num_mask_patches_encoder, H)
        masks_encoder: torch.Tensor of shape (bsz, num_encoder_masks, max_num_mask_patches_encoder) to
            indicate masks passed to context encoder
        masks_predictor: torch.Tensor of shape (bsz, num_predictor_masks, max_num_mask_patches_pred) to
            indicate masks passed to Predictor and Target Encoder
        num_valid_mask_patches_encoder: torch.Tensor of shape(bsz, num_encoder_masks)
        num_valid_mask_patches_predictor: torch.Tensor of shape(bsz, num_predictor_masks).
        """
        num_preds_masks, max_num_mask_patches_predictor = (
            masks_predictor.shape[1],
            masks_predictor.shape[2],
        )

        bsz, num_enc_masks, max_num_mask_patches_encoder = (
            masks_encoder.shape[0],
            masks_encoder.shape[1],
            masks_encoder.shape[2],
        )

        # Map from encoder-dim to pedictor-dim
        x = self.predictor_initial_projection(
            x
        )  # shape: (bsz, num_encoder_masks, max_num_mask_patches_encoder, projection_dim)

        # Add positional embedding to x tokens
        x_pos_embed = self.position_embeddings.to(x.dtype).repeat(bsz, 1, 1)

        x_pos_embed_masked = misc.apply_mask(
            x_pos_embed, masks_encoder
        )  # shape: (bsz, num_encoder_masks, max_num_mask_patches_encoder, projection_dim)

        x = x + x_pos_embed_masked

        # Build attention mask for input from ContextEncoder
        # 0 at positions we would like to attend and
        # -inf at positions to be ignored
        encoder_patch_attn_mask = torch.zeros(
            (bsz * num_enc_masks, max_num_mask_patches_encoder),
            device=x.device,
            dtype=x.dtype,
        )
        if num_valid_mask_patches_encoder is not None:
            encoder_patch_attn_mask = self._build_mask(
                num_valid_mask_patches_encoder, max_num_mask_patches_encoder
            )
        encoder_patch_attn_mask = replace_with_zero_and_neg_inf(
            encoder_patch_attn_mask
        )[:, None, None, :]

        encoder_patch_attn_mask = encoder_patch_attn_mask.unsqueeze(
            1
        ).broadcast_to(
            encoder_patch_attn_mask.shape[0],
            num_preds_masks,
            *encoder_patch_attn_mask.shape[1:],
        )
        encoder_patch_attn_mask = encoder_patch_attn_mask.reshape(
            -1,
            *encoder_patch_attn_mask.shape[2:],
        )

        # concat mask tokens at masks_predictor to x
        pos_embs_preds = misc.apply_mask(
            x_pos_embed, masks_predictor
        )  # shape: (bsz, num_predictor_masks, max_num_mask_patches_predictor, projection_dim)

        pos_embs_preds = pos_embs_preds.repeat((1, num_enc_masks, 1, 1))
        pred_tokens = (
            self.mask_token.to(x.dtype)
            .broadcast_to(
                pos_embs_preds.shape[0],
                pos_embs_preds.shape[1] * pos_embs_preds.shape[2],
                pos_embs_preds.shape[3],
            )
            .reshape(pos_embs_preds.shape)
        )

        pred_tokens = pred_tokens + pos_embs_preds

        # Build attention mask for predictor patches
        # 0 at positions we would like to attend and
        # -inf at positions to be ignored
        pred_patch_attn_mask = torch.zeros(
            (bsz * num_preds_masks, max_num_mask_patches_predictor),
            device=x.device,
            dtype=x.dtype,
        )
        if num_valid_mask_patches_predictor is not None:
            pred_patch_attn_mask = self._build_mask(
                num_valid_mask_patches_predictor,
                max_num_mask_patches_predictor,
            )
        pred_patch_attn_mask = replace_with_zero_and_neg_inf(
            pred_patch_attn_mask
        )[:, None, None, :]
        pred_patch_attn_mask = pred_patch_attn_mask.repeat(
            (num_enc_masks, 1, 1, 1)
        )

        attn_mask = torch.cat(
            [encoder_patch_attn_mask, pred_patch_attn_mask], dim=3
        )

        x = torch.repeat_interleave(x, num_preds_masks, dim=1)
        x = torch.cat([x, pred_tokens], dim=2)
        x = x.reshape(bsz * num_enc_masks * num_preds_masks, -1, x.shape[-1])

        x, _ = self.encoder(x, src_mask=attn_mask)  # hidden_states from Encoder

        x = x[
            :, max_num_mask_patches_encoder:
        ]  # predictions at masked_pred positions
        x = self.predictor_final_projection(
            x
        )  # shape: (bsz*num_encoder_masks*num_pred_masks, max_num_mask_patches_predictor, H)
        x = x.reshape(
            bsz, -1, x.shape[1], x.shape[2]
        )  # shape: (bsz, num_encoder_masks*num_pred_masks, max_num_mask_patches_predictor, H)

        return x

    def _build_mask(self, num_valid_mask_patches, max_num_mask_patches):
        # Build attention mask
        _mask = torch.arange(
            max_num_mask_patches,
            device=num_valid_mask_patches.device,
            dtype=num_valid_mask_patches.dtype,
        ).reshape(
            1, -1
        )  # (1, max_num_mask_patches)
        attn_mask = torch.where(
            _mask >= num_valid_mask_patches.reshape(-1, 1), 1.0, 0.0
        )  # (bsz*n_masks, max_num_mask_patches)
        return attn_mask
