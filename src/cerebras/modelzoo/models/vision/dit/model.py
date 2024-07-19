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

from copy import deepcopy
from dataclasses import asdict

import torch.nn as nn

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.models.vision.dit.modeling_dit import DiT


@registry.register_model(
    "dit", datasetprocessor=["DiffusionImageNet1KProcessor"]
)
class DiTModel(nn.Module):
    def __init__(self, params):
        super().__init__()

        model_params = deepcopy(params.model)
        self.model = self.build_model(model_params)

    def build_model(self, model_params):
        model = DiT(
            num_diffusion_steps=model_params.num_diffusion_steps,
            schedule_name=model_params.schedule_name,
            beta_start=model_params.beta_start,
            beta_end=model_params.beta_end,
            num_classes=model_params.num_classes,
            # Embedding
            embedding_dropout_rate=model_params.embedding_dropout_rate,
            hidden_size=model_params.hidden_size,
            embedding_nonlinearity=model_params.embedding_nonlinearity,
            position_embedding_type=model_params.position_embedding_type,
            # Encoder
            num_hidden_layers=model_params.num_hidden_layers,
            layer_norm_epsilon=float(model_params.layer_norm_epsilon),
            # Encoder Attn
            num_heads=model_params.num_heads,
            attention_type=model_params.attention_type,
            attention_softmax_fp32=model_params.attention_softmax_fp32,
            dropout_rate=model_params.dropout_rate,
            nonlinearity=model_params.encoder_nonlinearity,
            attention_dropout_rate=model_params.attention_dropout_rate,
            use_projection_bias_in_attention=model_params.use_projection_bias_in_attention,
            use_ffn_bias_in_attention=model_params.use_ffn_bias_in_attention,
            # Encoder ffn
            filter_size=model_params.filter_size,
            use_ffn_bias=model_params.use_ffn_bias,
            # Task-specific
            initializer_range=model_params.initializer_range,
            projection_initializer=(
                asdict(model_params.projection_initializer)
                if model_params.projection_initializer
                else None
            ),
            position_embedding_initializer=model_params.position_embedding_initializer,
            init_conv_like_linear=model_params.init_conv_like_linear,
            attention_initializer=(
                asdict(model_params.attention_initializer)
                if model_params.attention_initializer
                else None
            ),
            ffn_initializer=(
                asdict(model_params.ffn_initializer)
                if model_params.ffn_initializer
                else None
            ),
            timestep_embedding_initializer=(
                asdict(model_params.timestep_embedding_initializer)
                if model_params.timestep_embedding_initializer
                else None
            ),
            label_embedding_initializer=(
                asdict(model_params.label_embedding_initializer)
                if model_params.label_embedding_initializer
                else None
            ),
            head_initializer=(
                asdict(model_params.head_initializer)
                if model_params.head_initializer
                else None
            ),
            norm_first=model_params.norm_first,
            # vision related params
            latent_size=model_params.latent_size,
            latent_channels=model_params.latent_channels,
            patch_size=model_params.patch_size,
            use_conv_patchified_embedding=model_params.use_conv_patchified_embedding,
            # Context embeddings
            frequency_embedding_size=model_params.frequency_embedding_size,
            label_dropout_rate=model_params.label_dropout_rate,
            block_type=model_params.block_type,
            use_conv_transpose_unpatchify=model_params.use_conv_transpose_unpatchify,
        )

        self.mse_loss = nn.MSELoss()

        return model

    def forward(self, data):
        diffusion_noise = data["diffusion_noise"]
        timestep = data["timestep"]

        model_output = self.model(
            input=data["input"],
            label=data["label"],
            diffusion_noise=data["diffusion_noise"],
            timestep=data["timestep"],
        )
        pred_noise = model_output[0]
        loss = self.mse_loss(pred_noise, diffusion_noise)

        return loss
