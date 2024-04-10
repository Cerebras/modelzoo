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

import torch
from torch import nn

from cerebras.modelzoo.layers import ViTEmbeddingLayer
from cerebras.modelzoo.layers.AdaLayerNorm import AdaLayerNorm
from cerebras.modelzoo.layers.create_initializer import create_initializer
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
from cerebras.modelzoo.models.vision.dit.utils import BlockType


class DiT(nn.Module):
    def __init__(
        self,
        # Scheduler params
        num_diffusion_steps,
        schedule_name,
        beta_start,
        beta_end,
        # Embedding
        embedding_dropout_rate=0.0,
        embedding_nonlinearity="silu",
        position_embedding_type="learned",
        hidden_size=768,
        # Encoder
        num_hidden_layers=12,
        layer_norm_epsilon=1.0e-5,
        # Encoder Attn
        num_heads=12,
        attention_module_str="aiayn_attention",
        extra_attention_params={},
        attention_type="scaled_dot_product",
        attention_softmax_fp32=True,
        dropout_rate=0.0,
        nonlinearity="gelu",
        attention_dropout_rate=0.0,
        use_projection_bias_in_attention=True,
        use_ffn_bias_in_attention=True,
        # Encoder ffn
        filter_size=3072,
        use_ffn_bias=True,
        # Task-specific
        initializer_range=0.02,
        default_initializer=None,
        projection_initializer=None,
        position_embedding_initializer=None,
        init_conv_like_linear=False,
        attention_initializer=None,
        ffn_initializer=None,
        timestep_embeddding_initializer=None,
        label_embedding_initializer=None,
        head_initializer=None,
        norm_first=True,
        # vision related params
        latent_size=[32, 32],
        latent_channels=4,
        patch_size=[16, 16],
        use_conv_patchified_embedding=False,
        # added DiT params
        frequency_embedding_size=256,
        num_classes=1000,
        label_dropout_rate=0.1,
        block_type=BlockType.ADALN_ZERO,
        use_conv_transpose_unpatchify=False,
    ):
        super(DiT, self).__init__()

        # Flags for lowering tests
        self.block_type = BlockType.get(block_type)

        self.initializer_range = initializer_range
        self.latent_channels = latent_channels
        self.patch_size = patch_size

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
        if timestep_embeddding_initializer is None:
            timestep_embeddding_initializer = default_initializer
        if label_embedding_initializer is None:
            label_embedding_initializer = default_initializer
        if head_initializer is None:
            head_initializer = default_initializer
        # embeddings
        self.patch_embedding_layer = ViTEmbeddingLayer(
            image_size=latent_size,
            num_channels=latent_channels,
            patch_size=patch_size,
            hidden_size=hidden_size,
            initializer_range=self.initializer_range,
            embedding_dropout_rate=embedding_dropout_rate,
            projection_initializer=projection_initializer,
            position_embedding_initializer=position_embedding_initializer,
            position_embedding_type=position_embedding_type,
            use_conv_patchified_embedding=use_conv_patchified_embedding,
            init_conv_like_linear=init_conv_like_linear,
        )
        self.projection_initializer = create_initializer(projection_initializer)
        self.use_conv_patchified_embedding = use_conv_patchified_embedding

        self.timestep_embedding_layer = TimestepEmbeddingLayer(
            num_diffusion_steps=num_diffusion_steps,
            frequency_embedding_size=frequency_embedding_size,
            hidden_size=hidden_size,
            nonlinearity=embedding_nonlinearity,
            kernel_initializer=timestep_embeddding_initializer,
        )

        use_cfg_embedding = label_dropout_rate > 0
        self.label_embedding_layer = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.label_embedding_initializer = create_initializer(
            label_embedding_initializer
        )

        norm_layer = (
            AdaLayerNorm
            if self.block_type == BlockType.ADALN_ZERO
            else nn.LayerNorm
        )

        decoder_layer = DiTDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=filter_size,
            dropout=dropout_rate,
            activation=nonlinearity,
            layer_norm_eps=layer_norm_epsilon,
            norm_first=norm_first,
            norm_layer=norm_layer,
            attention_module=attention_module_str,
            extra_attention_params=extra_attention_params,
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
            gate_res=True if self.block_type == BlockType.ADALN_ZERO else False,
            add_cross_attention=False,
        )

        self.transformer_decoder = DiTDecoder(
            decoder_layer=decoder_layer, num_layers=num_hidden_layers, norm=None
        )

        # regression heads
        self.noise_head = RegressionHead(
            image_size=latent_size,
            hidden_size=hidden_size,
            out_channels=latent_channels,
            patch_size=patch_size,
            use_conv_transpose_unpatchify=use_conv_transpose_unpatchify,
            kernel_initializer=head_initializer,
        )
        self.final_norm = norm_layer(hidden_size, eps=layer_norm_epsilon)

        self.gaussian_diffusion = GaussianDiffusion(
            num_diffusion_steps,
            schedule_name,
            beta_start=beta_start,
            beta_end=beta_end,
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
        Assumes inputs are already batched with conditional and unconditional parts

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
