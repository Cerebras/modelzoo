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

# Copyright 2023 Stanford University Team and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This code is strongly influenced by https://github.com/pesser/pytorch_diffusion
# and https://github.com/hojonathanho/diffusion and
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from cerebras.modelzoo.models.vision.dit.layers.schedulers import (
    get_named_beta_schedule,
)
from cerebras.modelzoo.models.vision.dit.samplers.sampler_utils import (
    rescale_zero_terminal_snr,
    set_sampling_timesteps,
    threshold_sample,
)
from cerebras.modelzoo.models.vision.dit.samplers.SamplerBase import SamplerBase


@dataclass
class DDIMSamplerOutput:
    """
    Output class for the scheduler's step function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample (x_{0}) based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None


class DDIMSampler(SamplerBase):
    """
    Denoising diffusion implicit models is a scheduler that extends the denoising procedure introduced in denoising
    diffusion probabilistic models (DDPMs) with non-Markovian guidance.

    For more details, see the original paper: https://arxiv.org/abs/2010.02502

    Args:
        num_diffusion_steps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        schedule_name (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`.
        eta (`float`): weight of noise for added noise in diffusion step.
            Refer to Eqn 16. DDPM when η=1 and DDIM when η=0
        clip_sample (`bool`, default `False`):
            option to clip predicted sample for numerical stability.
        set_alpha_to_one (`bool`, default `True`):
            each diffusion step uses the value of alphas product at that step and at the previous one. For the final
            step there is no previous alpha. When this option is `True` the previous alpha product is fixed to `1`,
            otherwise it uses the value of alpha at step 0.
        thresholding (`bool`, default `False`):
            whether to use the "dynamic thresholding" method (introduced by Imagen, https://arxiv.org/abs/2205.11487).
            Note that the thresholding method is unsuitable for latent-space diffusion models (such as
            stable-diffusion).
        dynamic_thresholding_ratio (`float`, default `0.995`):
            the ratio for the dynamic thresholding method. Default is `0.995`, the same as Imagen
            (https://arxiv.org/abs/2205.11487). Valid only when `thresholding=True`.
        sample_max_value (`float`, default `1.0`):
            the threshold value for dynamic thresholding. Valid only when `thresholding=True`.
        clip_sample_range (`float`, default `1.0`):
            the maximum magnitude for sample clipping. Valid only when `clip_sample=True`.
        rescale_betas_zero_snr (`bool`, default `False`):
            whether to rescale the betas to have zero terminal SNR (proposed by https://arxiv.org/pdf/2305.08891.pdf).
            This can enable the model to generate very bright and dark samples instead of limiting it to samples with
            medium brightness. Loosely related to
            [`--offset_noise`](https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506).
        use_clipped_model_output (`bool`): if `True`, compute "corrected" `model_output` from the clipped
            predicted original sample. Necessary because predicted original sample is clipped to [-1, 1] when
            `self.config.clip_sample` is `True`. If no clipping has happened, "corrected" `model_output` would
            coincide with the one provided as input and `use_clipped_model_output` will have not effect.
        num_inference_steps (`str`): string containing comma-separated numbers,
            indicating the step count per section.
            For example, if there's 300 `num_diffusion_steps` and num_inference_steps=`10,15,20`
            then the first 100 timesteps are strided to be 10 timesteps, the second 100
            are strided to be 15 timesteps, and the final 100 are strided to be 20.
            Can either pass `custom_timesteps` (or) `num_inference_steps`, but not both.
        custom_timesteps (`List[int]`): List of timesteps to be used during sampling.
            Should be in decreasing order.
            Can either pass `custom_timesteps` (or) `num_inference_steps`, but not both.
    """

    def __init__(
        self,
        num_diffusion_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_name: str = "linear",
        eta: float = 0.0,
        clip_sample: bool = False,
        set_alpha_to_one: bool = True,
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        clip_sample_range: float = 1.0,
        rescale_betas_zero_snr: bool = False,
        use_clipped_model_output: bool = False,
        num_inference_steps: int = None,
        custom_timesteps: List[int] = None,
    ):
        # `num_train_timesteps` -> `num_diffusion_steps`
        #  beta_schedule -> schedule_name
        self.num_diffusion_steps = num_diffusion_steps
        self.clip_sample = clip_sample
        self.thresholding = thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.sample_max_value = sample_max_value
        self.clip_sample_range = clip_sample_range
        self.eta = eta
        self.use_clipped_model_output = use_clipped_model_output

        self.betas = get_named_beta_schedule(
            schedule_name,
            self.num_diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end,
        )

        # Rescale for zero SNR
        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # At every step in ddim, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # `set_alpha_to_one` decides whether we set this parameter simply to one or
        # whether we use the final alpha of the "non-previous" one.
        self.final_alpha_cumprod = (
            torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        )

        self.num_inference_steps = num_inference_steps
        self.custom_timesteps = custom_timesteps

        if self.num_inference_steps is None and self.custom_timesteps is None:
            logging.warning(f"Setting `num_inference_steps` to `250")
            self.num_inference_steps = 250

        if self.num_inference_steps and self.custom_timesteps:
            raise ValueError(
                "Can only pass one of str `num_inference_steps` or `custom_timesteps` list."
            )

        self.set_timesteps(
            num_diffusion_steps, num_inference_steps, custom_timesteps
        )

    def _get_variance(self, timestep, prev_timestep):
        """
        Variance Calculation based on Eqn 16 of https://arxiv.org/pdf/2010.02502.pdf
        Args:
            timestep (`int`): current discrete timestep in the diffusion chain.
            prev_timestep (`int`): Previous timestep in diffusion chain
        """
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (
            1 - alpha_prod_t / alpha_prod_t_prev
        )

        return variance

    def set_timesteps(
        self, num_diffusion_steps, num_inference_steps, custom_timesteps
    ):
        """
        Computes timesteps to be used during sampling

        Args:
            num_diffusion_steps (`int`): Total number of steps the model was trained on
            num_inference_steps (`str`): string containing comma-separated numbers,
                indicating the step count per section.
                For example, if there's 300 `num_diffusion_steps` and num_inference_steps=`10,15,20`
                then the first 100 timesteps are strided to be 10 timesteps, the second 100
                are strided to be 15 timesteps, and the final 100 are strided to be 20.
                Can either pass `custom_timesteps` (or) `num_inference_steps`, but not both.
            custom_timesteps (`List[int]`): User specified list of timesteps to be used during sampling.
        """
        self.timesteps = set_sampling_timesteps(
            num_diffusion_steps,
            "ddim" + str(num_inference_steps),
            custom_timesteps,
        )

    def step(
        self,
        pred_noise: torch.FloatTensor,
        pred_var: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator=None,
        return_dict: bool = True,
    ) -> Union[DDIMSamplerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            pred_noise (`torch.FloatTensor`): predicted eps output from learned diffusion model.
            pred_var (`torch.FloatTensor`):  Model predicted values
                used in variance computation.`υ` in Eqn 15.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
            generator: random number generator.
            return_dict (`bool`): option for returning tuple rather than DDIMSchedulerOutput class

        Returns:
            [`DDIMSamplerOutput` (with keys: `prev_sample`, `pred_original_sample`)]
            if `return_dict` is True
            (or) `tuple`.
            When returning a tuple,
            the first element is the `prev_sample` tensor and
            second element is `pred_original_sample`

        """

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"
        # - sample -> x_t

        # 1. get previous step value (=t-1)
        t = timestep
        prev_timestep = self.previous_timestep(t)

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alpha_prod_t)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alpha_prod_t - 1)

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_xstart = (
            sqrt_recip_alphas_cumprod * sample
            - sqrt_recipm1_alphas_cumprod * pred_noise
        )

        # 4. Clip or threshold "predicted x_0"
        if self.thresholding:
            pred_xstart = threshold_sample(
                pred_xstart,
                self.dynamic_thresholding_ratio,
                self.sample_max_value,
            )
        elif self.clip_sample:
            pred_xstart = pred_xstart.clamp(
                -self.clip_sample_range, self.clip_sample_range
            )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = self.eta * variance ** (0.5)

        if self.use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_noise = (
                sample - alpha_prod_t ** (0.5) * pred_xstart
            ) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
            0.5
        ) * pred_noise

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_prev_sample = (
            alpha_prod_t_prev ** (0.5) * pred_xstart + pred_sample_direction
        )

        if self.eta > 0:
            variance_noise = torch.randn(
                sample.size(),
                dtype=sample.dtype,
                layout=sample.layout,
                device=sample.device,
                generator=generator,
            )
            variance = std_dev_t * variance_noise

            pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (pred_prev_sample, pred_xstart)

        return DDIMSamplerOutput(
            prev_sample=pred_prev_sample, pred_original_sample=pred_xstart
        )

    def __len__(self):
        return self.num_diffusion_steps

    def previous_timestep(self, timestep):
        """
        Returns the previous timestep based on current timestep.
        Depends on the timesteps computed in `self.set_timesteps`
        """
        index = (self.timesteps == timestep).nonzero()[0][0]
        if index == self.timesteps.shape[0] - 1:
            prev_timestep = torch.tensor(-1)
        else:
            prev_timestep = self.timesteps[index + 1]
        return prev_timestep
