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

import logging
from typing import Callable, Optional

import torch
import torch.nn as nn


class DiffusionPipeline(nn.Module):
    def __init__(self, sampler, device="cpu"):
        """
        Args:
            sampler: Instance of one of the supported samplers.
                Refer to ./samplers/get_sampler.py
            device (int): Device info
        """
        super().__init__()
        self.device = device
        self.sampler = sampler
        logging.warning(
            f"This pipeline assumes that the `model_fwd_fn` arg passed to `pipeline.step` "
            f"takes inputs with specific keyword args `noised_latent`, `label`, `timestep` "
            f"Please make sure your function definition follows the same name convention "
            f"when using this pipeline"
        )

    def build_inputs(
        self,
        input_shape,
        num_classes,
        use_cfg,
        custom_labels=None,
        generator=None,
    ):
        """
        Utility to build random inputs to be passed to the model
            for the first pass of reverse diffusion process

        Args:
            input_shape (Tuple): Tuple indicating shape of
                noised_latent to be passed to Diffusion model
            num_classes (int): number of class labels
                in the dataset that the model was trained on
            use_cfg (bool): If True, use classifier guidance during sampling
            custom_labels (List[int]) : Optional list of labels
                that should be used as conditioning during sampling process.
                If specified, the model generates images from these classes only
            generator (torch.Generator): For setting random generator state

        Returns:
            dict with keys: `noised_latent` and `label`
            Note that the keys are chosen to have the same name as
            used in `forward`/`forward_cfg` method of the model
        """

        # Sample inputs:
        bsz, C, H, W = input_shape
        self.batch_size = bsz
        noised_latent = torch.randn(
            input_shape, device=self.device, generator=generator
        )
        if custom_labels is None:
            label = torch.randint(
                0, num_classes, (bsz,), device=self.device, generator=generator
            )
        else:
            if not isinstance(custom_labels, torch.Tensor):
                custom_labels = torch.tensor(custom_labels, device=self.device)

            sample_ids = torch.randint(
                0,
                len(custom_labels),
                size=(bsz,),
                device=self.device,
                generator=generator,
            )
            label = custom_labels[sample_ids]

        # Setup classifier-free guidance:
        if use_cfg:
            noised_latent = torch.cat([noised_latent, noised_latent], 0)
            label_null = torch.tensor(
                [num_classes] * bsz,
                device=self.device,  # unconditional label id = num_classes
            )
            label = torch.cat([label, label_null], 0)

        return {"noised_latent": noised_latent, "label": label}

    @torch.no_grad()
    def forward(
        self,
        model_fwd_fn: Callable,
        generator: Optional[torch.Generator] = None,
        progress: bool = True,
        use_cfg: bool = True,
        **inputs_to_model_fwd_fn,
    ):
        """
        Args:
            model_fwd_fn: Function handle to the desired forward pass
                of the diffusion model.
            generator (torch.Generator): For setting random generator state
            progress (bool): If true, displays progress bar indicating
                the timestep loop in sampling process
            use_cfg (bool): If True, use classifier guidance during sampling
            inputs_to_model_fwd_fn: kwargs that contain all params to be
                passed to `model_fwd_fn`. Assumes that the `model_fwd_fn`
                has inputs by name `noised_latent` indicating
                gaussian diffused latent and `label` indicating
                the conditioning labels to be used.

        Returns:
            torch.Tensor containing final generated sample at
            the end of timestep loop T -> 1
        """

        if not inputs_to_model_fwd_fn:  # if dict empty
            raise ValueError(
                f"Please pass inputs to `model_fwd_fn` as kwargs `inputs_to_model_fwd_fn` "
                f"param by calling `self.build_inputs` with appropriate args"
            )

        latent_model_input = inputs_to_model_fwd_fn["noised_latent"]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            total_timesteps = tqdm(self.sampler.timesteps)
        else:
            total_timesteps = self.sampler.timesteps

        for t in total_timesteps:
            if progress:
                total_timesteps.set_postfix({'timestep': t})

            timestep = t
            if not torch.is_tensor(timestep):
                timestep = torch.tensor(
                    (timestep,), dtype=torch.int32, device=self.device
                )

            timestep = timestep.expand(latent_model_input.shape[0]).to(
                latent_model_input.device
            )  # self.inputs[0].shape[0] -> bsz

            pred_noise, pred_var = model_fwd_fn(
                timestep=timestep, **inputs_to_model_fwd_fn
            )

            # compute previous image: x_t -> x_t-1
            latent_model_input = self.sampler.step(
                pred_noise, pred_var, t, latent_model_input, generator=generator
            ).prev_sample
            inputs_to_model_fwd_fn["noised_latent"] = latent_model_input

        if use_cfg:
            latents, _ = latent_model_input.chunk(2, dim=0)
        else:
            latents = latent_model_input

        return latents
