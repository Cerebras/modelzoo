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

import inspect

import numpy as np
import torch


def space_timesteps(num_timesteps, section_counts):
    """
    Based on https://github.com/facebookresearch/DiT/blob/main/diffusion/respace.py
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            _ini_sc = section_counts[len("ddim") :]
            section_counts = [
                int(x) for x in section_counts[len("ddim") :].split(",")
            ]
            if len(section_counts) > 1:
                raise ValueError(
                    f"Multiple section count strides not supported for DDIM. "
                    f"This value \"{_ini_sc}\" should NOT be a comma separated list. "
                    f"Instead, it should be a single str that can be converted to integer,"
                    f"for example: \"{str(section_counts[0])}\""
                )
            desired_count = int(section_counts[0])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


def set_sampling_timesteps(
    num_diffusion_steps, num_inference_steps, custom_timesteps
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

    if custom_timesteps is not None:
        for i in range(1, len(custom_timesteps)):
            if custom_timesteps[i] >= custom_timesteps[i - 1]:
                raise ValueError(
                    "`custom_timesteps` must be in descending order."
                )

        if custom_timesteps[0] >= num_diffusion_steps:
            raise ValueError(
                f"`custom_timesteps` must start before `num_diffusion_steps`:"
                f" {num_diffusion_steps}."
            )

        timesteps = np.array(custom_timesteps, dtype=np.int32)
    else:
        t = space_timesteps(num_diffusion_steps, num_inference_steps)
        timesteps = np.array(sorted(t, reverse=True), dtype=np.int32)

    return timesteps


def threshold_sample(
    sample: torch.FloatTensor, dynamic_thresholding_ratio, sample_max_value
) -> torch.FloatTensor:
    """
    "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
    prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
    s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
    pixels from saturation at each step. We find that dynamic thresholding results in significantly better
    photorealism as well as better image-text alignment, especially when using very large guidance weights."

    https://arxiv.org/abs/2205.11487
    """
    dtype = sample.dtype
    batch_size, channels, height, width = sample.shape

    if dtype not in (torch.float32, torch.float64):
        sample = (
            sample.float()
        )  # upcast for quantile calculation, and clamp not implemented for cpu half

    # Flatten sample for doing quantile calculation along each image
    sample = sample.reshape(batch_size, channels * height * width)

    abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

    s = torch.quantile(abs_sample, dynamic_thresholding_ratio, dim=1)
    s = torch.clamp(
        s, min=1, max=sample_max_value
    )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]

    s = s.unsqueeze(
        1
    )  # (batch_size, 1) because clamp will broadcast along dim=0
    sample = (
        torch.clamp(sample, -s, s) / s
    )  # "we threshold xt0 to the range [-s, s] and then divide by s"

    sample = sample.reshape(batch_size, channels, height, width)
    sample = sample.to(dtype)

    return sample


def rescale_zero_terminal_snr(betas):
    """
    Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)


    Args:
        betas (`torch.FloatTensor`):
            the betas that the scheduler is being initialized with.

    Returns:
        `torch.FloatTensor`: rescaled betas with zero terminal SNR
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (
        alphas_bar_sqrt_0 - alphas_bar_sqrt_T
    )

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas


def configure_sampler_params(sampler_config: dict):
    """
    Used for validation in DiT's config class to ensure that
    the values specified to the Sampler's values are only
    those that are specified in the Sampler class.
    """

    from cerebras.modelzoo.models.vision.dit.samplers.get_sampler import (
        get_all_samplers,
    )

    sampler_map = get_all_samplers()

    sampler = sampler_config.pop("name").lower()

    if sampler in sampler_map:
        cls = sampler_map[sampler]

    signature = inspect.signature(cls.__init__)

    init_parameters = set(signature.parameters.keys())

    sampler_keys = set(sampler_config.keys())

    # Check if the sampler dictionary keys are a subset of the __init__ parameters
    if not sampler_keys.issubset(init_parameters):
        extra_params = sampler_keys - init_parameters
        raise ValueError(
            f"Invalid parameters in Sampler dictionary: {extra_params}"
        )
