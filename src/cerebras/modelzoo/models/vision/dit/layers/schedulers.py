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

import numpy as np
import torch


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * torch.ones(num_diffusion_timesteps, dtype=torch.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = torch.linspace(
        beta_start, beta_end, warmup_time, dtype=torch.float64
    )
    return betas


def get_beta_schedule(
    beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps
):
    """
    This is the deprecated API for creating beta schedules.
    See get_named_beta_schedule() for the new library of schedules.
    """
    if beta_schedule == "quad":
        betas = (
            torch.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = torch.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=torch.float64
        )
    elif beta_schedule == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / torch.linspace(
            num_diffusion_timesteps,
            1,
            num_diffusion_timesteps,
            dtype=np.float64,
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def get_named_beta_schedule(
    schedule_name, num_diffusion_timesteps, beta_start=0.0001, beta_end=0.02
):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        # `beta_start`=1e-4 and `beta_end`=0.02 are defaults from Ho et al for T = 1000.
        scale = 1000 / num_diffusion_timesteps
        return get_beta_schedule(
            "linear",
            beta_start=scale * beta_start,
            beta_end=scale * beta_end,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
