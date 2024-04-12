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

import torch.nn as nn

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.models.vision.dit.modeling_dit import DiT


@registry.register_model(
    "dit", datasetprocessor=["DiffusionImageNet1KProcessor"]
)
class DiTModel(nn.Module):
    def __init__(self, params):
        super().__init__()

        model_params = params["model"].copy()
        self.model = self.build_model(model_params)

    def build_model(self, model_params):
        model = DiT(
            num_diffusion_steps=model_params["num_diffusion_steps"],
            schedule_name=model_params["schedule_name"],
            beta_start=model_params["beta_start"],
            beta_end=model_params["beta_end"],
            num_classes=model_params.pop("num_classes"),
            # Embedding
            embedding_dropout_rate=model_params.pop(
                "embedding_dropout_rate", 0.0
            ),
            hidden_size=model_params.pop("hidden_size"),
            embedding_nonlinearity=model_params.pop("embedding_nonlinearity"),
            position_embedding_type=model_params.pop("position_embedding_type"),
            # Encoder
            num_hidden_layers=model_params.pop("num_hidden_layers"),
            layer_norm_epsilon=float(model_params.pop("layer_norm_epsilon")),
            # Encoder Attn
            num_heads=model_params.pop("num_heads"),
            attention_type=model_params.pop(
                "attention_type", "scaled_dot_product"
            ),
            attention_softmax_fp32=model_params.pop(
                "attention_softmax_fp32", True
            ),
            dropout_rate=model_params.pop("dropout_rate"),
            nonlinearity=model_params.pop("encoder_nonlinearity", "gelu"),
            attention_dropout_rate=model_params.pop(
                "attention_dropout_rate", 0.0
            ),
            use_projection_bias_in_attention=model_params.pop(
                "use_projection_bias_in_attention", True
            ),
            use_ffn_bias_in_attention=model_params.pop(
                "use_ffn_bias_in_attention", True
            ),
            # Encoder ffn
            filter_size=model_params.pop("filter_size"),
            use_ffn_bias=model_params.pop("use_ffn_bias", True),
            # Task-specific
            initializer_range=model_params.pop("initializer_range", 0.02),
            projection_initializer=model_params.pop(
                "projection_initializer", None
            ),
            position_embedding_initializer=model_params.pop(
                "position_embedding_initializer", None
            ),
            init_conv_like_linear=model_params.pop(
                "init_conv_like_linear", False
            ),
            attention_initializer=model_params.pop(
                "attention_initializer", None
            ),
            ffn_initializer=model_params.pop("ffn_initializer", None),
            timestep_embeddding_initializer=model_params.pop(
                "timestep_embeddding_initializer", None
            ),
            label_embedding_initializer=model_params.pop(
                "label_embedding_initializer", None
            ),
            head_initializer=model_params.pop("head_initializer", None),
            norm_first=model_params.pop("norm_first", True),
            # vision related params
            latent_size=model_params.pop("latent_size"),
            latent_channels=model_params.pop("latent_channels"),
            patch_size=model_params.pop("patch_size"),
            use_conv_patchified_embedding=model_params.pop(
                "use_conv_patchified_embedding", False
            ),
            # Context embeddings
            frequency_embedding_size=model_params.pop(
                "frequency_embedding_size"
            ),
            label_dropout_rate=model_params.pop("label_dropout_rate"),
            block_type=model_params.pop("block_type"),
            use_conv_transpose_unpatchify=model_params.pop(
                "use_conv_transpose_unpatchify"
            ),
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
