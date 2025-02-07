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

import cerebras.pytorch as cstorch
from cerebras.modelzoo.models.vision.dit.layers.schedulers import (
    get_named_beta_schedule,
)


def index(arr, timestep):
    return torch.index_select(arr, 0, timestep.long())


def extract(arr, timestep, broadcast_shape):
    shape = (broadcast_shape[0],) + (1,) * (len(broadcast_shape) - 1)
    result = index(arr, timestep).view(shape) + torch.zeros(
        broadcast_shape, device=arr.device
    )
    return result.to(timestep.device, dtype=cstorch.amp.get_half_dtype())


class GaussianDiffusion(torch.nn.Module):
    """Generate noisy images via Gaussian diffusion.
    The class implements the noising process as described in Step 5 of Algorithm 1
    in the paper
    `"Denoising Diffusion Probabilistic Models` <https://arxiv.org/abs/2006.11239>`.
    """

    def __init__(
        self,
        num_diffusion_steps,
        schedule_name,
        seed=None,
        beta_start=0.0001,
        beta_end=0.02,
    ):
        """
        :param (int) num_diffusion_steps: Number of diffusion steps.
        :param (float) beta_start: Minimum variance for generated Gaussian noise.
        :param (float) beta_end: Maximum variance for generated Gaussian noise.
        :param (int) seed: Random seed for reproducibility.
        :param (float) beta_start: Initial value of variance schedule i.e beta_1
            (default value according to Ho et al https://arxiv.org/pdf/2006.11239.pdf: Section 4)
        :param (float) beta_end: Final value of variance schedule i.e beta_T
            (default value according to Ho et al https://arxiv.org/pdf/2006.11239.pdf: Section 4)
        """
        super().__init__()

        if num_diffusion_steps <= 0:
            raise ValueError("Number of diffusion steps must be positive.")

        if seed is not None:
            torch.manual_seed(seed)

        self.num_diffusion_steps = num_diffusion_steps
        self.schedule_name = schedule_name
        self.betas = get_named_beta_schedule(
            schedule_name,
            self.num_diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        assert self.betas.dim() == 1, "betas must be 1-D"

        if self.betas.device.type != "lazy":
            assert torch.all(torch.logical_and(self.betas > 0, self.betas <= 1))

        alphas = 1.0 - self.betas
        alphas_cum_prod = torch.cumprod(alphas, dim=0)

        self.sqrt_alphas_cum_prod = torch.nn.Parameter(
            torch.sqrt(alphas_cum_prod).to(torch.float32),
            requires_grad=False,
        )
        self.sqrt_one_minus_alphas_cum_prod = torch.nn.Parameter(
            torch.sqrt(1 - alphas_cum_prod).to(torch.float32),
            requires_grad=False,
        )

    def forward(self, latent, noise, timestep):
        """Lookup alpha-related constants and create noised sample
        Args:
            :param latent (Tensor): Float tensor of size (B, C, H, W).
        Returns:
            A tuple corresponding to the noisy images, ground truth noises and
            the timesteps corresponding to the scheduled noise variance.
        """
        if latent.ndim != 4:
            raise ValueError(f"Samples ndim should be 4. Got {latent.ndim}")

        sqrt_alpha_prod = extract(
            self.sqrt_alphas_cum_prod, timestep, noise.shape
        )
        sqrt_one_minus_alpha_prod = extract(
            self.sqrt_one_minus_alphas_cum_prod, timestep, noise.shape
        )
        noisy_samples = (
            sqrt_alpha_prod * latent + sqrt_one_minus_alpha_prod * noise
        )

        return noisy_samples.to(cstorch.amp.get_half_dtype())

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"schedule_name={self.schedule_name}"
            f", num_diffusion_steps={self.num_diffusion_steps}"
            f")"
        )
