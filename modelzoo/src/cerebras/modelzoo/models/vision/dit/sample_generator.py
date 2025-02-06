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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained diffusion model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: 
https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample_generator_dit_simple.py.
"""
import logging
import math
import os
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import yaml
from PIL import Image
from tqdm import tqdm

from cerebras.modelzoo.models.vision.dit.pipeline import DiffusionPipeline
from cerebras.modelzoo.models.vision.dit.samplers.get_sampler import get_sampler
from cerebras.modelzoo.models.vision.dit.utils import set_defaults

LOGFORMAT = '%(asctime)s %(levelname)-4s[%(filename)s:%(lineno)d] %(message)s'
logging.basicConfig(level=logging.INFO, format=LOGFORMAT)


def create_npz_from_sample_folder(sample_dir, num_samples=50000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num_samples), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num_samples, samples.shape[1], samples.shape[2], 3)
    npz_path = os.path.join(sample_dir, "sample.npz")
    np.savez(npz_path, arr_0=samples)
    logging.info(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


class SampleGenerator(ABC):
    def __init__(
        self,
        model_ckpt_path,
        vae_ckpt_path,
        params,
        sample_dir,
        seed,
        num_fid_samples=50000,
        per_gpu_batch_size=None,
    ):
        """
        Main BaseClass for model sample generation
        Args:
            model_ckpt_path (str): Path to pretrained diffusion model checkpoint
            vae_ckpt_path (str): Path to pretrained VAE model checkpoint
            params (str): Path to yaml containing model params
            sample_dir (str): Path to folder where generated images
                and npz file to be stored
            seed (int): Seed for random generation process
            num_fid_samples (int): Number of images to be generated
            per_gpu_batch_size (int): Per gpu batch size,
                this input overrides that in yaml if provided.
        """
        self.model_ckpt_path = model_ckpt_path
        self.vae_ckpt_path = vae_ckpt_path
        self.params_path = params
        self.sample_dir = sample_dir
        self.seed = seed
        self.num_fid_samples = num_fid_samples
        # params
        with open(self.params_path, "r") as fh:
            params_data = yaml.safe_load(fh)

        self.params = set_defaults(params_data)

        self.rparams = self.params["model"]["reverse_process"]
        if per_gpu_batch_size is not None:
            self.rparams["batch_size"] = per_gpu_batch_size
            _msg = f"Using command line batch_size = {per_gpu_batch_size} on each GPU: "
        else:
            per_gpu_batch_size = self.rparams["batch_size"]
            _msg = f"Using batch_size = {per_gpu_batch_size} from reverse_params and set_defaults on each GPU"

        self.per_gpu_batch_size = per_gpu_batch_size
        logging.info(_msg)

        assert (
            self.rparams["pipeline"]["guidance_scale"] >= 1.0
        ), "In almost all cases, guidance_scale be >= 1.0"
        self.use_cfg = self.rparams["pipeline"]["guidance_scale"] > 1.0

    @abstractmethod
    def create_diffusion_model(self, params, model_ckpt_path, use_cfg, device):
        """
        Initialize diffusion model, load checkpoint. Also return forward function to use
        """
        raise NotImplementedError(
            f"create_diffusion_model should be implemented in child class"
        )

    @abstractmethod
    def create_vae_model(self, vae_params, vae_ckpt_path, device):
        raise NotImplementedError(
            f"create_vae_model should be implemented in child class"
        )

    def create_sampler(self, sampler_params):
        """
        Create sampler object.
        sampler_params: contains kwargs that can be
            passed as input to __init__ of the sampler class
        """
        sname = sampler_params.pop("name")
        sampler = get_sampler(sname)(**sampler_params)
        return sampler

    def create_pipeline(
        self,
        sampler,
        device,
    ):
        """
        Get Pipeline object that
        creates samples from a batch of random normal noised latent
        using diffusion model and sampler
        """
        # Create pipeline
        diff_pipe = DiffusionPipeline(sampler, device=device).to(device)

        return diff_pipe

    def _save_params(self):
        """
        Save params to yaml
        """
        curr_time = datetime.utcnow().strftime('%m%d%Y_%H%M%S')
        with open(
            os.path.join(self.sample_dir, f"params_{curr_time}.yaml"),
            "w",
        ) as fh:
            yaml.dump(self.params, fh)

    def setup_dist(self):
        # initialize the process group
        dist.init_process_group("nccl")

    def cleanup_dist(self):
        # clean up process group
        dist.destroy_process_group()

    def run(self):
        """
        MAIN function
        """
        assert torch.cuda.is_available(), "Requires at least one GPU."
        torch.set_grad_enabled(False)

        # Set up dist process group
        self.setup_dist()

        # Get ranks and device
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = dist.get_rank()
        device = global_rank % torch.cuda.device_count()
        world_size = dist.get_world_size()

        generator = torch.Generator(device)
        if self.seed is None:
            # large random number chosen as `high` upper bound
            seed = torch.randint(0, 2147483647, (1,), dtype=torch.int64).item()
        else:
            seed = self.seed + global_rank
        generator.manual_seed(seed)

        torch.cuda.set_device(device)
        logging.info(
            f"Starting rank={global_rank}, seed={seed}, world_size={world_size}."
        )

        # Create folder to save samples:
        if global_rank == 0:
            os.makedirs(self.sample_dir, exist_ok=True)
            logging.info(f"\nSaving .png samples at {self.sample_dir}\n")
            self._save_params()
        dist.barrier()

        # Initialize Diffusion Model
        diffusion_model, model_fwd_fn = self.create_diffusion_model(
            self.params, self.model_ckpt_path, self.use_cfg, device=device
        )
        diffusion_model.eval()  # Important

        # Initialize VAE model for decoding latent
        vae_model = self.create_vae_model(
            self.params["model"]["vae"], self.vae_ckpt_path, device
        )
        vae_model.eval()  # Important

        # Create Sampler for Reverse Diffusion process (for loop from T -> 1)
        sampler_params = self.rparams["sampler"]
        sampler = self.create_sampler(sampler_params)

        # Create pipeline
        input_shape = (
            self.rparams["batch_size"],
            self.params["model"]["latent_channels"],
            *self.params["model"]["latent_size"],
        )
        diff_pipe = self.create_pipeline(sampler=sampler, device=device)

        local_bsz = self.per_gpu_batch_size
        global_batch_size = local_bsz * world_size
        total_samples = int(
            math.ceil(self.num_fid_samples / world_size) * world_size
        )
        num_samples_per_gpu = int(total_samples // world_size)

        if num_samples_per_gpu < local_bsz:
            raise ValueError(
                f"`per_gpu_batch_size`(={local_bsz}) > "
                f"number of samples per gpu(={num_samples_per_gpu}). "
                f"Lower batch size in `model.reverse_process` in params yaml"
            )

        if num_samples_per_gpu % local_bsz != 0:
            num_samples_per_gpu = int(
                math.ceil(num_samples_per_gpu / local_bsz) * local_bsz
            )
            total_samples = num_samples_per_gpu * world_size

        iterations = int(num_samples_per_gpu // local_bsz)
        pbar = tqdm(range(iterations), desc="num_batches")

        if global_rank == 0:
            logging.info(
                f"\nTotal number of images that will be sampled: {total_samples} \n"
            )

        total = 0
        diff_inputs = {}
        if self.use_cfg:
            diff_inputs["guidance_scale"] = self.rparams["pipeline"][
                "guidance_scale"
            ]
            diff_inputs["num_cfg_channels"] = self.rparams["pipeline"][
                "num_cfg_channels"
            ]
        for _ in pbar:

            # Random normal noised_latent and random integer labels
            num_classes = self.rparams["pipeline"]["num_classes"]
            custom_labels = self.rparams["pipeline"]["custom_labels"]
            _inputs = diff_pipe.build_inputs(
                input_shape,
                num_classes,
                self.use_cfg,
                generator=generator,
                custom_labels=custom_labels,
            )
            diff_inputs.update(_inputs)

            # Denoised sample
            latent_sample = diff_pipe(
                model_fwd_fn=model_fwd_fn,
                generator=generator,
                progress=False,
                use_cfg=self.use_cfg,
                **diff_inputs,
            )
            samples = vae_model.decode(
                latent_sample / self.params["model"]["vae"]["scaling_factor"]
            ).sample
            samples = (
                torch.clamp(127.5 * samples + 128.0, 0, 255)
                .permute(0, 2, 3, 1)
                .to("cpu", dtype=torch.uint8)
                .numpy()
            )

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + global_rank + total
                Image.fromarray(sample).save(
                    f"{self.sample_dir}/{index:06d}.png"
                )
            total += global_batch_size

        # Make sure all processes have finished saving their
        # samples before attempting to convert to .npz
        dist.barrier()
        if global_rank == 0:
            create_npz_from_sample_folder(self.sample_dir, self.num_fid_samples)
            print("Done.")
        dist.barrier()

        # Clean up dist processes
        self.cleanup_dist()
