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

# isort: off
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
# isort: on
"""
Usage: 

torchrun --nnodes 1 --nproc_per_node 4 \
modelzoo/models/vision/dit/sample_generator_dit.py \
--model_ckpt_path <trained_diffusion_ckpt path> \
--vae_ckpt_path <vae_ckpt_path> \
--params modelzoo/models/vision/dit/configs/params_dit_xlarge_patchsize_2x2.yaml \
--sample_dir=modelzoo/models/vision/dit/sample_dir \
--num_fid_samples=50000

"""

import argparse
import logging

import cerebras.pytorch as cstorch
from cerebras.modelzoo.models.vision.dit.layers.vae.VAEModel import (
    AutoencoderKL,
)
from cerebras.modelzoo.models.vision.dit.model import DiTModel
from cerebras.modelzoo.models.vision.dit.sample_generator import SampleGenerator


class DiTSampleGenerator(SampleGenerator):
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
        Class for DiT model sample generation
        Args:
            model_ckpt_path (str): Path to pretrained diffusion model checkpoint
            vae_ckpt_path (str): Path to pretrained VAE model checkpoint
            params (str): Path to yaml containing model params
            sample_dir (str): Path to folder where generated images
                and npz file to be stored
            seed (int): Seed for random generation process
            num_fid_samples (int): Number of images to be generated
            per_gpu_batch_size (int): Per gpu batch size,
                command line input overrides that in yaml if provided.
        """
        super().__init__(
            model_ckpt_path,
            vae_ckpt_path,
            params,
            sample_dir,
            seed,
            num_fid_samples,
            per_gpu_batch_size,
        )

    def create_diffusion_model(self, params, model_ckpt_path, use_cfg, device):
        """
        Initialize DiT model and load ckpt if provided

        Args:
            params (dict): params to be passed to DiT model initilization
            model_ckpt_path (str): Path to model checkpoint without VAE
            use_cfg (bool): If True, apply classifier free guidance on inputs
                i.e select the appropriate forward fcn based on `use_cfg`
            device (str): Target device for model

        Returns:
            dit_model (nn.Module): DiT model
            fwd_fn (Callable) : Forward fcn to be used by
                pipeline object for sampling
        """
        # Initialize DiT Model
        dit_model = DiTModel(params)

        # Load checkpoint
        if model_ckpt_path:
            _dit_dict = cstorch.load(model_ckpt_path)
            dit_model.load_state_dict(_dit_dict["model"])
            logging.info(f"Initializing DiT model with {model_ckpt_path}")
        else:
            logging.info(f"Initializing DiT model with random weights")
        dit_model = dit_model.model
        dit_model.to(device)

        # Select forward fcn
        if use_cfg:
            fwd_fn = dit_model.forward_dit_with_cfg
        else:
            fwd_fn = dit_model.forward_dit
        return dit_model, fwd_fn

    def create_vae_model(self, vae_params, vae_ckpt_path, device):
        """
        Initialize VAE model and load ckpt if provided

        Args:
            vae_params (dict): params to initialize VAE model
            vae_ckpt_path (str) : Path to VAE model checkpoint
            device (str): Target device for model

        Returns:
            vae_model (nn.Module): VAE model for decoding
        """
        # Initialize VAE model
        vae_model = AutoencoderKL
        vae_model = vae_model(**vae_params)

        # Load checkpoint
        if vae_ckpt_path:
            _vae_dict = cstorch.load(vae_ckpt_path)
            vae_model.load_state_dict(_vae_dict)
            logging.info(f"Initializing VAE model with {vae_ckpt_path}")
        else:
            logging.info(f"Initializing VAE model with random weights")
        vae_model.to(device)
        return vae_model


def get_parser_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--model_ckpt_path",
        type=str,
        default=None,
        help="Optional path to a diffusion model checkpoint",
    )
    parser.add_argument(
        "--vae_ckpt_path",
        type=str,
        default=None,
        help="Optional VAE model checkpoint path",
    )
    parser.add_argument(
        "--params",
        type=str,
        required=True,
        help="Path to params to initialize Diffusion model and VAE models",
    )
    parser.add_argument(
        "--num_fid_samples",
        type=int,
        default=50000,
        help="number of samples to generate",
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        required=True,
        help="Directory to store generated samples",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=None,
        help="per-gpu batch size for forward pass",
    )
    parser.add_argument(
        "--create_grid",
        action="store_true",
        required=False,
        help="If passed, create a grid from images generated",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser_args()
    sample_gen = DiTSampleGenerator(
        args.model_ckpt_path,
        args.vae_ckpt_path,
        args.params,
        args.sample_dir,
        args.seed,
        args.num_fid_samples,
        args.batch_size,
    )
    sample_gen.run()
    if args.create_grid:
        import math

        from cerebras.modelzoo.models.vision.dit.display_images import (
            display_images,
        )

        logging.info(f"Creating grid from samples generated....")
        nrow = math.ceil(math.sqrt(args.num_fid_samples))
        display_images(folder_path=args.sample_dir, nrow=nrow)
