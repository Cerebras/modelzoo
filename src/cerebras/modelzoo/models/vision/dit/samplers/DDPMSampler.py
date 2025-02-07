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

# Copyright 2023 UC Berkeley Team and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This file is strongly influenced by https://github.com/ermongroup/ddim
# Based on HuggingFace diffusers/schedulers/scheduling_ddpm.py and
# https://github.com/facebookresearch/DiT/blob/main/diffusion/gaussian_diffusion.py

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from cerebras.modelzoo.models.vision.dit.layers.schedulers import (
    get_named_beta_schedule,
)
from cerebras.modelzoo.models.vision.dit.samplers.sampler_utils import (
    set_sampling_timesteps,
    threshold_sample,
)
from cerebras.modelzoo.models.vision.dit.samplers.SamplerBase import SamplerBase


@dataclass
class DDPMSamplerOutput:
    """
    Output class for the sampler's step function output.

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


class DDPMSampler(SamplerBase):
    """
    Denoising diffusion probabilistic models (DDPMs) explores the connections between denoising score matching and
    Langevin dynamics sampling.

    For more details, see the original paper: https://arxiv.org/abs/2006.11239
    and https://arxiv.org/pdf/2102.09672.pdf

    Args:
        num_diffusion_steps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        schedule_name (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model.
            Choose from `linear`
        clip_sample (`bool`, default `False`):
            option to clip predicted sample for numerical stability.
        set_alpha_to_one (`bool`, default `True`):
            each diffusion step uses the value of alphas product at that step and at the previous one. For the final
            step there is no previous alpha. When this option is `True` the previous alpha product is fixed to `1`,
            otherwise it uses the value of alpha at step 0.
        thresholding (`bool`, default `False`):
            whether to use the "dynamic thresholding" method
            (introduced by Imagen, https://arxiv.org/abs/2205.11487).
            Note that the thresholding method is unsuitable for
            latent-space diffusion models (such as stable-diffusion).
        dynamic_thresholding_ratio (`float`, default `0.995`):
            the ratio for the dynamic thresholding method. Default is `0.995`, the same as Imagen
            (https://arxiv.org/abs/2205.11487). Valid only when `thresholding=True`.
        sample_max_value (`float`, default `1.0`):
            the threshold value for dynamic thresholding. Valid only when `thresholding=True`.
        clip_sample_range (`float`, default `1.0`):
            the maximum magnitude for sample clipping. Valid only when `clip_sample=True`.
        variance_type (`str`):
            options to clip the variance used when adding noise to the denoised sample. Choose from
            `learned_range`, `fixed_small`, `fixed_large`.
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
        clip_sample: bool = False,
        set_alpha_to_one: bool = True,
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        clip_sample_range: float = 1.0,
        variance_type: str = "learned_range",
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

        self.betas = get_named_beta_schedule(
            schedule_name,
            self.num_diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
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

        if variance_type not in ["learned_range", "fixed_small", "fixed_large"]:
            raise ValueError(
                f"variance_type={variance_type} unsupported."
                f"Supported values are `learned_range`, `fixed_small`, `fixed_large`"
            )
        self.variance_type = variance_type

        self.set_timesteps(
            num_diffusion_steps, num_inference_steps, custom_timesteps
        )

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
            num_diffusion_steps=num_diffusion_steps,
            num_inference_steps=str(num_inference_steps),
            custom_timesteps=custom_timesteps,
        )

    def _get_variance(self, t, predicted_model_var_values=None):
        """
        Variance calculation https://arxiv.org/pdf/2102.09672.pdf
        Eqn (15)

        Args:
            t (`int`): Current timestep
            predicted_model_var_values (`torch.Tensor`):
                Model predicted values used in variance computation.
                `υ` in Eqn 15
        """

        prev_timestep = self.previous_timestep(t)

        # Section 4 (Improved Sampling Speed) of https://arxiv.org/pdf/2102.09672.pdf
        # For supporting sampling using subsequence of (1, 2, ....T)
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # we always take the log of variance, so clamp it to ensure it's not 0 at t=0
        variance = torch.clamp(variance, min=1e-20)
        log_variance_clipped = torch.log(variance)

        if self.variance_type == "fixed_small":
            model_log_variance = log_variance_clipped
            model_variance = variance
        elif self.variance_type == "fixed_large":
            # This differs from DiT repo where at t=0,
            # `posterior_variance at t=1` is used to prevent log(0)
            model_log_variance = torch.log(current_beta_t)
            model_variance = current_beta_t
        elif self.variance_type == "learned_range":
            min_log = torch.log(variance)
            max_log = torch.log(current_beta_t)
            frac = (predicted_model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = torch.exp(model_log_variance)

        return model_log_variance, model_variance

    def step(
        self,
        pred_noise: torch.FloatTensor,
        pred_var: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator=None,
        return_dict: bool = True,
    ) -> Union[DDPMSamplerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE.
        Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            pred_noise (`torch.FloatTensor`): predicted eps output from learned diffusion model.
            pred_var (`torch.FloatTensor`):  Model predicted values
                used in variance computation.`υ` in Eqn 15.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            generator: random number generator.
            return_dict (`bool`): option for returning tuple rather than DDPMSamplerOutput class

        Returns:
            [`DDPMSamplerOutput` (with keys: `prev_sample`, `pred_original_sample`)]
            if `return_dict` is True
            (or) `tuple`.
            When returning a tuple,
            the first element is the `prev_sample` tensor and
            second element is `pred_original_sample`
        """
        t = timestep
        prev_timestep = self.previous_timestep(t)

        # Section 4 (Improved Sampling Speed) of https://arxiv.org/pdf/2102.09672.pdf
        # For supporting sampling using subsequence of (1, 2, ....T)
        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alpha_prod_t)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alpha_prod_t - 1)
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        # pred_xstart = (sample -  sqrt_beta_prod_t * predicted_noise) / sqrt_alpha_prod_t
        pred_xstart = (
            sqrt_recip_alphas_cumprod * sample
            - sqrt_recipm1_alphas_cumprod * pred_noise
        )

        # 3. Clip or threshold "predicted x_0"
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

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_xstart_sample_coeff = (
            alpha_prod_t_prev ** (0.5) * current_beta_t
        ) / beta_prod_t
        current_sample_coeff = (
            current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
        )

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample_mean = (
            pred_xstart_sample_coeff * pred_xstart
            + current_sample_coeff * sample
        )

        # 6. Compute predicted previous sample variance Σ
        # See formula (11) from https://arxiv.org/pdf/2006.11239.pdf
        # This part differs from HF implementation and uses DiT logic
        pred_prev_std = 0
        if t > 0:
            pred_prev_log_variance, _ = self._get_variance(t, pred_var)
            pred_prev_std = torch.exp(0.5 * pred_prev_log_variance)

        # 7. Add noise
        noise = torch.randn(
            sample.size(),
            dtype=sample.dtype,
            layout=sample.layout,
            device=sample.device,
            generator=generator,
        )
        pred_prev_sample = (
            pred_prev_sample_mean + pred_prev_std * noise
        )  # reparametrization trick

        if not return_dict:
            return (pred_prev_sample, pred_xstart)

        return DDPMSamplerOutput(
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
