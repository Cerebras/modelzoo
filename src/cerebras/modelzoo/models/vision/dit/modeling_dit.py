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
from typing import List, Literal, Optional

import torch
from pydantic import PositiveInt, field_validator
from torch import nn

from cerebras.modelzoo.config import BaseConfig, ModelConfig
from cerebras.modelzoo.layers import ViTEmbeddingLayer
from cerebras.modelzoo.layers.AdaLayerNorm import AdaLayerNorm
from cerebras.modelzoo.layers.create_initializer import create_initializer
from cerebras.modelzoo.layers.init import (
    InitializerConfig,
    NormalInitializer,
    XavierUniformInitializer,
)
from cerebras.modelzoo.models.vision.dit.layers.DiTDecoder import DiTDecoder
from cerebras.modelzoo.models.vision.dit.layers.DiTDecoderLayer import (
    DiTDecoderLayer,
)
from cerebras.modelzoo.models.vision.dit.layers.GaussianDiffusion import (
    GaussianDiffusion,
)
from cerebras.modelzoo.models.vision.dit.layers.RegressionHead import (
    RegressionHead,
)
from cerebras.modelzoo.models.vision.dit.layers.TimestepEmbeddingLayer import (
    TimestepEmbeddingLayer,
)
from cerebras.modelzoo.models.vision.dit.samplers.sampler_utils import (
    configure_sampler_params,
)
from cerebras.modelzoo.models.vision.dit.utils import BlockType


class ReverseProcessConfig(BaseConfig):
    pipeline: dict = {}
    "Pipeline configuration for reverse process."

    sampler: dict = {}
    "Sampler configuration for reverse process."

    batch_size: PositiveInt = 32

    @field_validator("sampler", mode="after")
    @classmethod
    def configure_sampler_params(cls, sampler):
        configure_sampler_params(copy.deepcopy(sampler))
        return sampler


class VAEConfig(BaseConfig):
    act_fn: str = "silu"
    "Activation function to use in VAE autoencoder."

    in_channels: int = 3
    "Number of channels in the input image."

    out_channels: int = 3
    "Number of channels in the output."

    block_out_channels: List[int] = [64]
    "List of block output channels"

    down_block_types: List[str] = ["DownEncoderBlock2D"]
    "List of downsample block types"

    up_block_types: List[str] = ["UpDecoderBlock2D"]
    "List of upsample block types."

    latent_channels: int = 4
    "Number of channels in the latent space."

    latent_size: List[int] = [32, 32]
    "Size of latent space (height, width)."

    layers_per_block: Optional[PositiveInt] = 1
    "Number of layers per block."

    norm_num_groups: int = 32
    "The number of groups to use for the first normalization layer of each block."

    sample_size: Optional[PositiveInt] = None
    "Sample size to use when tiling is enabled"

    scaling_factor: float = 0.18215
    """The component-wise standard deviation of the trained latent space computed using the first batch of the
    training set. This is used to scale the latent space to have unit variance when training the diffusion
    model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
    diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
    / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
    Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper."""


class DiTConfig(ModelConfig):
    name: Literal["dit_model"]

    # Embedding
    embedding_dropout_rate: float = 0.1
    "Dropout rate for embeddings."

    embedding_nonlinearity: str = "silu"
    "Type of nonlinearity to be used in embedding."

    position_embedding_type: Optional[
        Literal["fixed", "relative", "rotary", "learned"]
    ] = "learned"
    """The type of position embedding to use in the model. Can be one of:
    `fixed` - Sinusoidal from original Transformer, (https://arxiv.org/abs/1706.03762).
    `relative` - Relative position embedding, to exploit pairwise,
      relative positional information, (https://arxiv.org/abs/1803.02155).
    `rotary` - a.k.a RoPE
    `learned` - Learned embedding matrix, (https://arxiv.org/pdf/2104.09864v4.pdf)"""

    # Transformer Attention
    attention_type: str = "scaled_dot_product"
    "Type of attention. Accepted values: `dot_product`, `scaled_dot_product`."

    attention_softmax_fp32: bool = True
    "Whether to use fp32 precision for attention softmax."

    attention_dropout_rate: float = 0.0
    "Dropout rate for attention layer."

    use_projection_bias_in_attention: bool = True
    "Whether to include bias in the attention layer for projection."

    use_ffn_bias_in_attention: bool = True
    "Whether to include bias in the attention layer for feed-forward network."

    # Model
    num_classes: int = 1000
    "Number of classes in the training dataset."

    hidden_size: int = 768
    "The size of the transformer hidden layers."

    num_hidden_layers: Optional[PositiveInt] = 12
    "Number of hidden layers in the Transformer encoder/decoder."

    layer_norm_epsilon: Optional[float] = 1.00e-5
    "The epsilon value used in layer normalization layers."

    num_heads: int = 12
    "The number of attention heads in the multi-head attention layer."

    dropout_rate: float = 0.0
    "The dropout probability for all fully connected layers."

    nonlinearity: str = "gelu"
    """The non-linear activation function used in the feed forward network
    in each transformer block. Some may have to use autogen_policy: `medium`."""

    filter_size: int = 3072
    "Dimensionality of the feed-forward layer in the Transformer block."

    use_ffn_bias: Optional[bool] = True
    "Whether to use bias in the feedforward network."

    norm_first: Optional[bool] = True
    """Enables normalization before the Attention & FFN blocks (i.e Pre-LN as
    described in https://arxiv.org/pdf/2002.04745.pdf. When disabled,
    normalization is applied *after* the residual (Post-LN)"""

    latent_size: Optional[List] = None
    "Latent Tensor(output of VAEEncoder) [height, width]."

    latent_channels: Optional[PositiveInt] = None
    "Number of channels in Latent Tensor."

    patch_size: List[int] = [16, 16]
    "Size of patch used to convert image to tokens."

    use_conv_patchified_embedding: bool = False
    "If True, use conv2D to convert image to patches."

    use_conv_transpose_unpatchify: Optional[bool] = True
    "If True, use transposed convolution for unpatchify step."

    frequency_embedding_size: int = 256
    "Size of sinusoidal timestep embeddings."

    label_dropout_rate: float = 0.1
    "Probability of dropout applied to label tensor."

    block_type: Literal["adaln_zero",] = "adaln_zero"
    "DiT Block variant. Accepted values: adaln_zero."

    encoder_nonlinearity: str = "gelu"
    "Type of nonlinearity to be used in encoder."

    vae: Optional[VAEConfig] = None
    """Only used for evaluation. This should be left as None in train mode since we don't run the
    VAE during training."""

    # Initialization
    initializer_range: float = 0.02
    """The standard deviation of the truncated_normal_initializer as the
    default initializer"""

    projection_initializer: InitializerConfig = XavierUniformInitializer(
        gain=1.0
    )
    """Initializer for embedding linear layer. Either a string indicating the name of the
    initializer or a dict that includes the name + other params if relevant. If left
    unspecified will apply truncated normal initialization."""

    position_embedding_initializer: Optional[str] = None
    """Initializer for position embedding layer. Either a string indicating the name of
    the initializer or a dict that includes the name + other params if relevant. If left
    unspecified will apply truncated normal initialization."""

    init_conv_like_linear: Optional[bool] = None
    "If True, modify fan-in fan-out by reshaping before initializing."

    attention_initializer: InitializerConfig = XavierUniformInitializer(
        gain=1.0
    )
    "Attention layer initializer. Defaults to `xavier_uniform`."

    ffn_initializer: InitializerConfig = XavierUniformInitializer(gain=1.0)
    "FFN layer initializer. Defaults to `xavier_uniform`."

    timestep_embedding_initializer: Optional[InitializerConfig] = None
    "Initializer used in timestep embedding."

    label_embedding_initializer: Optional[InitializerConfig] = None
    "Initializer used in label embedding."

    head_initializer: InitializerConfig = "zeros"
    "Initializer used in regression head."

    # Diffusion
    num_diffusion_steps: int = 1000
    "Number of timesteps in the diffusion forward process."

    schedule_name: str = "linear"
    "Scheduler for gaussian diffusion steps."

    beta_start: float = 0.0001
    "The starting beta value of inference."

    beta_end: float = 0.02
    "The final beta value."

    reverse_process: Optional[ReverseProcessConfig] = None
    """Configuration for reverse diffusion process. This is only used for inference
    and should be left as None in train mode."""

    var_loss: bool = False

    def post_init(self, context):
        super().post_init(context)

        if self.latent_channels is None:
            self.latent_channels = self.vae.latent_channels
        if self.latent_size is None:
            self.latent_size = self.vae.latent_size
        if self.init_conv_like_linear is None:
            self.init_conv_like_linear = self.use_conv_patchified_embedding
        if self.timestep_embedding_initializer is None:
            self.timestep_embedding_initializer = NormalInitializer(
                mean=0.0, std=self.initializer_range
            )
        if self.label_embedding_initializer is None:
            self.label_embedding_initializer = NormalInitializer(
                mean=0.0, std=self.initializer_range
            )
        if self.reverse_process:
            self.reverse_process.sampler.setdefault(
                "num_diffusion_steps", self.num_diffusion_steps
            )
            self.reverse_process.pipeline.setdefault(
                "num_classes", self.num_classes
            )
            self.reverse_process.pipeline.setdefault("custom_labels", None)
            if (
                self.reverse_process.sampler["name"] == "ddpm"
                and not self.var_loss
            ):
                self.reverse_process.sampler["variance_type"] = "fixed_small"


class DiT(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()

        # Flags for lowering tests
        self.block_type = BlockType.get(config.block_type)

        self.initializer_range = config.initializer_range
        self.latent_channels = config.latent_channels
        self.patch_size = config.patch_size

        default_initializer = {
            "name": "truncated_normal",
            "std": self.initializer_range,
            "mean": 0.0,
            "a": self.initializer_range * -2.0,
            "b": self.initializer_range * 2.0,
        }

        if config.attention_initializer is None:
            config.attention_initializer = default_initializer
        if config.ffn_initializer is None:
            config.ffn_initializer = default_initializer
        if config.timestep_embedding_initializer is None:
            config.timestep_embedding_initializer = default_initializer
        if config.label_embedding_initializer is None:
            config.label_embedding_initializer = default_initializer
        if config.head_initializer is None:
            config.head_initializer = default_initializer
        # embeddings
        self.patch_embedding_layer = ViTEmbeddingLayer(
            image_size=config.latent_size,
            num_channels=config.latent_channels,
            patch_size=config.patch_size,
            hidden_size=config.hidden_size,
            initializer_range=self.initializer_range,
            embedding_dropout_rate=config.embedding_dropout_rate,
            projection_initializer=config.projection_initializer,
            position_embedding_initializer=config.position_embedding_initializer,
            position_embedding_type=config.position_embedding_type,
            use_conv_patchified_embedding=config.use_conv_patchified_embedding,
            init_conv_like_linear=config.init_conv_like_linear,
        )
        self.projection_initializer = create_initializer(
            config.projection_initializer
        )
        self.use_conv_patchified_embedding = (
            config.use_conv_patchified_embedding
        )

        self.timestep_embedding_layer = TimestepEmbeddingLayer(
            num_diffusion_steps=config.num_diffusion_steps,
            frequency_embedding_size=config.frequency_embedding_size,
            hidden_size=config.hidden_size,
            nonlinearity=config.embedding_nonlinearity,
            kernel_initializer=config.timestep_embedding_initializer,
        )

        use_cfg_embedding = config.label_dropout_rate > 0
        self.label_embedding_layer = nn.Embedding(
            config.num_classes + use_cfg_embedding, config.hidden_size
        )
        self.label_embedding_initializer = create_initializer(
            config.label_embedding_initializer
        )

        norm_layer = (
            AdaLayerNorm
            if self.block_type == BlockType.ADALN_ZERO
            else nn.LayerNorm
        )

        decoder_layer = DiTDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.filter_size,
            dropout=config.dropout_rate,
            activation=config.nonlinearity,
            layer_norm_eps=config.layer_norm_epsilon,
            norm_first=config.norm_first,
            norm_layer=norm_layer,
            attention_module="aiayn_attention",
            extra_attention_params={},
            attention_dropout_rate=config.attention_dropout_rate,
            attention_type=config.attention_type,
            attention_softmax_fp32=config.attention_softmax_fp32,
            use_projection_bias_in_attention=config.use_projection_bias_in_attention,
            use_ffn_bias_in_attention=config.use_ffn_bias_in_attention,
            use_ffn_bias=config.use_ffn_bias,
            attention_initializer=config.attention_initializer,
            ffn_initializer=config.ffn_initializer,
            use_ff_layer1_dropout=False,
            use_ff_layer2_dropout=True,
            gate_res=True if self.block_type == BlockType.ADALN_ZERO else False,
            add_cross_attention=False,
        )

        self.transformer_decoder = DiTDecoder(
            decoder_layer=decoder_layer,
            num_layers=config.num_hidden_layers,
            norm=None,
        )

        # regression heads
        self.noise_head = RegressionHead(
            image_size=config.latent_size,
            hidden_size=config.hidden_size,
            out_channels=config.latent_channels,
            patch_size=config.patch_size,
            use_conv_transpose_unpatchify=config.use_conv_transpose_unpatchify,
            kernel_initializer=config.head_initializer,
        )
        self.final_norm = norm_layer(
            config.hidden_size, eps=config.layer_norm_epsilon
        )

        self.gaussian_diffusion = GaussianDiffusion(
            config.num_diffusion_steps,
            config.schedule_name,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
        )

        self.reset_parameters()

    def reset_parameters(self):
        # Embedding layers
        self.patch_embedding_layer.reset_parameters()
        self.timestep_embedding_layer.reset_parameters()
        self.label_embedding_initializer(self.label_embedding_layer.weight.data)

        # DiT Blocks
        self.transformer_decoder.reset_parameters()

        # Final AdaLayerNorm
        self.final_norm.reset_parameters()

        # Regression Heads for noise and var predictions
        self.noise_head.reset_parameters()

    def forward(
        self,
        input,
        label,
        diffusion_noise,
        timestep,
    ):

        latent = input

        # NOTE: numerical differences observed due to
        # bfloat16 vs float32 `noised_latent` output
        # extract diffusion constants within model
        noised_latent = self.gaussian_diffusion(
            latent, diffusion_noise, timestep
        )

        pred_noise, pred_var = self.forward_dit(noised_latent, label, timestep)

        # We have pred_var = None to be consistent and
        # support other samplers in the future that uses
        # variance to generate samples.
        return pred_noise, pred_var

    def forward_dit(self, noised_latent, label, timestep):
        latent_embeddings = self.patch_embedding_layer(noised_latent)

        context = None
        timestep_embeddings = self.timestep_embedding_layer(timestep)
        label_embeddings = self.label_embedding_layer(label)

        context = timestep_embeddings + label_embeddings
        hidden_states = self.transformer_decoder(latent_embeddings, context)
        hidden_states = self.final_norm(hidden_states, context)
        # We have `pred_var = None` to be consistent and
        # support other samplers in the future that uses
        # variance to generate samples and VLB loss
        pred_var = None

        pred_noise = self.noise_head(hidden_states)

        return pred_noise, pred_var

    def forward_dit_with_cfg(
        self, noised_latent, label, timestep, guidance_scale, num_cfg_channels=3
    ):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        Assumes inputs are already batched with conditional and unconditional parts.

        Note: For exact reproducibility reasons, classifier-free guidance is applied only
        three channels by default, hence `num_cfg_channels` defaults to 3.
        The standard approach to cfg applies it to all channels.
        """
        half = noised_latent[: len(noised_latent) // 2]
        combined = torch.cat([half, half], dim=0)
        pred_noise, pred_var = self.forward_dit(combined, label, timestep)

        eps, rest = (
            pred_noise[:, :num_cfg_channels],
            pred_noise[:, num_cfg_channels:],
        )  # eps shape: (bsz, num_cfg_channels, H, W)
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

        # (1-guidance_scale) * uncond_eps + guidance_scale * cond_eps
        # `guidance_scale`` = 1 disables classifier-free guidance, while
        # increasing `guidance_scale` > 1 strengthens the effect of guidance
        half_eps = uncond_eps + guidance_scale * (
            cond_eps - uncond_eps
        )  # half_eps shape: (bsz//2, num_cfg_channels, H, W)
        eps = torch.cat(
            [half_eps, half_eps], dim=0
        )  # eps shape: (bsz, num_cfg_channels, H, W)
        pred_noise = torch.cat([eps, rest], dim=1)
        return pred_noise, pred_var
