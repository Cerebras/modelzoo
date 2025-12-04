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


class NoiseGenerator(torch.nn.Module):
    def __init__(self, width, height, channels, num_diffusion_steps, seed=None):
        super(NoiseGenerator, self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.num_diffusion_steps = num_diffusion_steps

    def forward(self, input, label):
        """
        Args:
            :param input : Float tensor of size (B, C, H, W).
            :param label : Int tensor of size (B, ).
        Returns:
            A dict corresponding to the noisy images, ground truth noises and
            the timesteps corresponding to the scheduled noise variance with
            the following keys and shapes.

            "input": Tensor of shape (batch_size, C, H, W). This tensor is simply passed through.
            "label": Tensor of shape (batch_size, ) representing labels. This tensor is simply passed through.
            "diffusion_noise": Tensor of shape (batch_size, channels, height, width)
                represents diffusion noise to be applied
            "timestep": Tensor of shape (batch_size, ) that indicates the timesteps for each diffusion sample
            "vae_noise": Tensor of shape (batch_size, latent_channels, latent_height, latent_width)
                represents the noise sample to be used with reparametrization of VAE

        """

        if input.ndim != 4:
            raise ValueError(f"Samples ndim should be 4. Got {input.ndim}")

        # reshaping to (batch_size, 1, ..., 1) for broadcasting
        batch_size = input.shape[0]
        timestep = torch.randint(
            self.num_diffusion_steps, size=(batch_size,), dtype=label.dtype
        )
        noise_shape = (batch_size, self.channels, self.height, self.width)
        diffusion_noise = torch.randn(noise_shape, dtype=input.dtype).to(
            input.device
        )

        vae_noise_shape = (batch_size, self.channels, self.height, self.width)
        vae_noise_sample = torch.randn(vae_noise_shape, dtype=input.dtype)

        return {
            "input": input,
            "label": label,
            "diffusion_noise": diffusion_noise,
            "timestep": timestep,
            "vae_noise": vae_noise_sample,
        }

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"schedule_name={self.schedule_name}"
            f", num_diffusion_steps={self.num_diffusion_steps}"
            f")"
        )


class LabelDropout(torch.nn.Module):
    def __init__(self, dropout_prob, num_classes):
        super(LabelDropout, self).__init__()
        assert dropout_prob > 0
        self.dropout_prob = dropout_prob
        self.num_classes = num_classes

    def token_drop(self, label):
        drop_ids = torch.rand(label.shape[0]) < self.dropout_prob
        return drop_ids

    def forward(self, image, label):
        drop_ids = self.token_drop(label)
        label = torch.where(
            drop_ids, torch.tensor(self.num_classes, dtype=label.dtype), label
        )
        return image, label
