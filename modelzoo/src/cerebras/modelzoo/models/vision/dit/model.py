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

from typing import Literal

import torch.nn as nn

from cerebras.modelzoo.models.vision.dit.modeling_dit import DiT, DiTConfig


class DiTModelConfig(DiTConfig):
    name: Literal["dit"]


class DiTModel(nn.Module):
    def __init__(self, config: DiTModelConfig):
        super().__init__()

        self.model = self.build_model(config)

    def build_model(self, config: DiTModelConfig):
        model = DiT(config)

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
