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

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
# isort: on

import argparse
import glob
import json
import logging
import os
import re
import shutil
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

import cerebras.pytorch as cstorch
from cerebras.modelzoo.data.vision.transforms import (
    LambdaWithParam,
    resize_center_crop_pil_image,
)
from cerebras.modelzoo.models.vision.dit.layers.vae.VAEModel import (
    AutoencoderKL,
)

"""
torchrun --nnodes 1 --nproc_per_node 1 modelzoo/data_preparation/vision/dit/create_imagenet_latents.py --image_height 256 --image_width 256 --src_dir=/pathto/imagenet1k_ilsvrc2012 --dest_dir=<path to store created latent files> --log_steps=10 --dataset_split=val --checkpoint_path=<path to vae trained ckpt>
"""

LOGFORMAT = '%(asctime)s %(levelname)-4s[%(filename)s:%(lineno)d] %(message)s'
logging.basicConfig(level=logging.INFO, format=LOGFORMAT)

# Setting backend flags based on
# https://github.com/facebookresearch/DiT/blob/main/train.py#L11-L13
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_VAE_CHECKPOINT = "https://huggingface.co/stabilityai/sd-vae-ft-mse/blob/main/diffusion_pytorch_model.bin"


def get_parser_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        default=None,
        help="Path to VAE model checkpoint",
    )
    parser.add_argument(
        "--params_path",
        type=str,
        required=False,
        default=os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../../../models/vision/dit/configs/params_dit_small_patchsize_2x2.yaml",
            )
        ),
        help="Path to VAE model params yaml",
    )
    parser.add_argument(
        "--horizontal_flip",
        action="store_true",
        help="If passed, flip image horizonatally",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--image_width",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--src_dir", type=str, required=True, help="source data location"
    )
    parser.add_argument(
        "--dest_dir", type=str, required=True, help="Latent data location"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=f"If specified, resumes previous generation process."
        f"The dest_dir should point to previous generation and have "
        f"log_checkpoint saved.",
    )
    parser.add_argument(
        "--resume_ckpt",
        default=None,
        help=f"log ckpt to resume data generation from"
        f"If None, picks latest from log dir",
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=1000,
        help=f"Generation process ckpt and logging frequency",
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        type=int,
        required=False,
        default=64,
        help=f"batch size of input to be passed to VAE model for encoding",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=4,
        help=f"Number of pytorch dataloader workers",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        required=False,
        default="train",
        choices=["train", "val"],
        help=f"Number of pytorch dataloader workers",
    )

    args = parser.parse_args()
    return args


class ImageNet(datasets.ImageNet):
    def __init__(self, root: str, split: str = "train", **kwargs):
        super().__init__(root, split, **kwargs)

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path = self.samples[index][0]
        return {"image": image, "label": label, "path": path}


class LatentImageNetProcessor:
    def __init__(self, args):
        # Set class attributes from args
        self.args_dict = vars(args)
        for k, val in self.args_dict.items():
            setattr(self, k, val)

        # vae params
        with open(self.params_path, "r") as fh:
            self.vae_params = yaml.safe_load(fh)["model"]["vae"]

    def get_latest_latent_checkpoint(self, log_path):
        """
        Get the latest saved log checkpoint
        Args:
            log_path (str): Path to log dir

        Returns:
            latest_filepath (str): Path to the last saved log ckpt
        """
        latest_filepath = None
        list_files = glob.glob(f"{log_path}/logckpt*.json")
        if list_files:
            file_num = [
                re.match(r'.*logckpt.*\.(\d+)\.json', x).groups()[0]
                for x in list_files
            ]
            file_num = [int(x) for x in file_num]
            latest_filepath = sorted(
                zip(file_num, list_files), key=lambda t: t[0], reverse=True
            )
            latest_filepath = latest_filepath[0][
                1
            ]  # 0 index for the latest file, 1 is to index into filename
        return latest_filepath

    def get_resume_data(self, latent_ckpt_path):
        """
        Get data from log ckpt to resume data generation process
        Args:
            latent_ckpt_path (str): Path to log ckpt

        Returns:
            resume_index (int): Index of sample to restart process
            resume_batches (int): Number of batches processed previously
        """
        with open(latent_ckpt_path, "r") as fh:
            ckpt_data = json.load(fh)

        resume_index = ckpt_data["num_samples_processed"]
        resume_batches = ckpt_data["total_batches_processed"]
        # Check if process checkpoint saved can actually be used and data generation resumed
        checks = {
            "src_dir": self.src_dir == ckpt_data["src_dir"],
            "image_height": self.image_height == ckpt_data["image_height"],
            "image_width": self.image_width == ckpt_data["image_width"],
            "dataset_split": self.dataset_split == ckpt_data["dataset_split"],
            "vae_params": self.vae_params == ckpt_data["vae_params"],
            "checkpoint_path": self.checkpoint_path
            == ckpt_data["checkpoint_path"],
            "params_path": self.params_path == ckpt_data["params_path"],
        }
        for k, val in checks.items():
            if not val:
                logging.info(
                    f"{k} differs between input args passed and ckpt saved. Starting from the beginning"
                )
                resume_index = 0
                break

        return resume_index, resume_batches

    def _setup_folders(self):
        """
        Create data and log directories
        """
        # Destination directory setup
        if os.path.exists(self.dest_dir):
            _str = f"\nFolder {self.dest_dir} exists, do you want to delete folder(Y) or not(N)?  "
            while True:
                folder_act = input(_str)
                if folder_act == "N":
                    os.makedirs(
                        os.path.join(self.dest_dir, self.dataset_split),
                        exist_ok=True,
                    )
                    break
                elif folder_act == "Y":
                    logging.info(f"Deleting {self.dest_dir} and creating again")
                    shutil.rmtree(self.dest_dir)
                    os.makedirs(self.dest_dir)
                    os.makedirs(os.path.join(self.dest_dir, self.dataset_split))
                    break

        # Log setup
        log_path = os.path.join(self.dest_dir, "logs", self.dataset_split)
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    def _save_vae_params(self):
        """
        Save VAE model params
        """
        curr_time = datetime.utcnow().strftime('%m%d%Y_%H%M%S')
        with open(
            os.path.join(self.dest_dir, f"params_vae_{curr_time}.yaml"), "w"
        ) as fh:
            data = {"model": {"vae": self.vae_params}}
            yaml.dump(data, fh)

    def set_data_transforms(self, horizontal_flip, image_height, image_width):
        """
        Data transforms used for dataset creation
        Args:
            horizontal_flip (bool): If True, flip the image horizontally
            image_height (int): Height of resized image
            image_width (int): Width of resized image

        Returns:
            transform : torchvision.transforms composition to be applied to image
            target_transform : torchvision.transforms composition to be applied to target label
        """
        transform = [
            transforms.Lambda(
                lambda pil_image: resize_center_crop_pil_image(
                    pil_image,
                    image_height,
                    image_width,
                )
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=False
            ),
        ]

        if horizontal_flip:
            # flip always
            transform.insert(1, transforms.RandomHorizontalFlip(p=1.0))

        transform = transforms.Compose(transform)

        # Target label transform
        def _get_target_transform(x, *args, **kwargs):
            return np.int32(x)

        target_transform = LambdaWithParam(_get_target_transform)

        return transform, target_transform

    def create_dataloader(self, split):
        """
        Build ImageNet Dataloader
        Args:
            split (str): The dataset split, can be one of `train` or `val`
        Returns:
            dataloader: torch.utils.data.Dataloader object that reads from ImageNet dataset
        """
        transform, target_transform = self.set_data_transforms(
            self.horizontal_flip,
            self.image_height,
            self.image_width,
        )
        logging.info(
            f"The following transforms are used for image: {transform} \n"
        )
        logging.info(
            f"The following transforms are used for label: {target_transform}"
        )
        dataset = ImageNet(
            root=self.src_dir,
            split=split,
            transform=transform,
            target_transform=target_transform,
        )
        subdataset = torch.utils.data.Subset(
            dataset, indices=list(range(self.resume_index, len(dataset)))
        )

        # drop_last set to False inorder to preserve all samples.
        # Some samples maybe written twice on account of this since
        # Distributed Sampler pads the last incomplete batch
        # with samples from the beginning to make the data
        # evenly divisible across the replicas
        _sampler = DistributedSampler(
            dataset=subdataset, shuffle=False, drop_last=False
        )

        dataloader = DataLoader(
            subdataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=self.num_workers,
            prefetch_factor=10,
            persistent_workers=True,
            drop_last=False,
            sampler=_sampler,
        )

        return dataloader

    def setup_dist(self):
        # initialize the process group
        dist.init_process_group("nccl")

    def cleanup_dist(self):
        # clean up process group
        dist.destroy_process_group()

    def save_latent_tensors(self, vae_output, label, src_paths):
        """
        Save the output latent tensors from VAE encoder to npz file
        Args:
            vae_output (torch.Tensor): Concatenation of mean and logvar outputs from VAE
                corresponding to images at src_paths,
                shape=(2 * latent_size, latent_height, latent_width)
            label (torch.Tensor): Target label of image
            src_paths List[str]: Path of image
        """
        label = label.cpu().numpy()
        vae_output = vae_output.cpu().numpy()

        for i in range(vae_output.shape[0]):
            path = src_paths[i]
            dest_path = path.replace(
                self.src_dir.rstrip(), self.dest_dir.rstrip()
            )
            dest_path = os.path.splitext(dest_path)[0] + ".npz"
            base_dest_flr = os.path.split(dest_path)[0]

            os.makedirs(base_dest_flr, exist_ok=True)

            kwargs = {
                "src_path": path,
                "dest_path": dest_path,
                "label": label[i],
                "vae_output": vae_output[
                    i
                ],  # includes concat of mean and logvar outputs from VAE, shape=(2 * latent_size, latent_height, latent_width)
            }
            np.savez(dest_path, **kwargs)

    def save_logs(
        self,
        log_path,
        global_rank,
        iter_num,
        total_num_batches,
    ):
        """
        Save data generation log checkpoints to resume process later if needed.
        Args:
            log_path (str): Path to save log ckpt used for data generation resume
            global_rank (int): GPU global rank
            iter_num (int): Current iteration of dataloader on GPU with rank = `global_rank`
            total_num_batches (int): Total number of batches processed so far across all GPUs
                during the current data generation process
        """
        total_num_batches += self.resume_num_batches
        num_samples = total_num_batches * self.batch_size_per_gpu

        logging.info(
            f"[GPU{global_rank}] | iteration num: {iter_num} | total_batches_processed across gpus @ bsz={self.batch_size_per_gpu}: {total_num_batches} | total_samples across gpus: {num_samples}"
        )

        log_dict = {
            "total_batches_processed": total_num_batches,
            "num_samples_processed": num_samples,
            "batch_size_per_gpu": self.batch_size_per_gpu,
            "image_height": self.image_height,
            "image_width": self.image_width,
            "dataset_split": self.dataset_split,
            "src_dir": self.src_dir,
            "dest_dir": self.dest_dir,
            "checkpoint_path": self.checkpoint_path,
            "params_path": self.params_path,
            "vae_params": self.vae_params,
        }
        with open(
            os.path.join(log_path, f"logckpt_numsamples.{num_samples}.json"),
            "w",
        ) as fh:
            json.dump(log_dict, fh)

    def run(self):
        """
        MAIN function
        """
        assert torch.cuda.is_available(), "Requires at least one GPU."

        # Set up dist process group
        self.setup_dist()

        # Get ranks and device
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = dist.get_rank()
        device = global_rank % torch.cuda.device_count()
        world_size = dist.get_world_size()
        torch.cuda.set_device(device)

        global_batch_size = self.batch_size_per_gpu * world_size
        log_path = os.path.join(self.dest_dir, "logs", self.dataset_split)
        if global_rank == 0:
            logging.info(f"Command line args passed: {self.args_dict} \n")
            logging.info(f"VAE model params: {self.vae_params} \n")
            self._setup_folders()
            self._save_vae_params()

        dist.barrier()

        # Resume index
        self.resume_index = 0
        self.resume_num_batches = 0
        if self.resume:
            if self.resume_ckpt is None:
                last_latent_ckpt = self.get_latest_latent_checkpoint(log_path)
            else:
                last_latent_ckpt = self.resume_ckpt
            logging.info(
                f"Resume set to True, restarting from {last_latent_ckpt}"
            )
            if last_latent_ckpt is not None:
                (
                    self.resume_index,
                    self.resume_num_batches,
                ) = self.get_resume_data(last_latent_ckpt)
        dist.barrier()

        logging.info(
            f"Starting [GPU{global_rank}] rank={global_rank}, world_size={world_size}, resume_index: {self.resume_index}"
        )

        # Storing pop value
        # Initialize VAE Model, wrap with DDP
        vae_model = AutoencoderKL
        self.vae_model = vae_model(**self.vae_params)

        # Load state dict for VAE
        if self.checkpoint_path:
            vae_state_dict = cstorch.load(self.checkpoint_path)
            self.vae_model.load_state_dict(vae_state_dict)
            logging.info(f"Initializing VAE model with {self.checkpoint_path}")
        else:
            logging.info("Initializing VAE model with RANDOM weights")

        self.vae_model.eval()
        model = DDP(self.vae_model.to(device), device_ids=[local_rank])

        # Initialize dataloader
        dataloader = self.create_dataloader(self.dataset_split)
        len_dataloader = len(dataloader)

        if len_dataloader == 0:
            logging.info(f"All examples written already, nothing to write")

        # local_num_batches = torch.tensor([0], dtype=torch.int64, device=device)
        total_num_batches = 0

        for i, data in enumerate(dataloader):
            image = data["image"].to(device)
            label = data["label"].to(device)
            path = data["path"]

            # VAE module forward pass
            with torch.no_grad():
                latent = model.module.encode(image).latent_dist
                vae_output = latent.parameters

            self.save_latent_tensors(vae_output, label, path)

            dist.barrier()
            total_num_batches += 1
            total = torch.tensor(total_num_batches).to(device)

            # Collect overall batches processed
            if i % self.log_steps == 0 or i == len_dataloader - 1:
                dist.reduce(total, dst=0, op=dist.ReduceOp.SUM)

            # Save data generation checkpoints
            if global_rank == 0 and (
                i % self.log_steps == 0 or i == len_dataloader - 1
            ):
                self.save_logs(log_path, global_rank, i, total.cpu().item())

            dist.barrier()

        # Clean up dist processes
        self.cleanup_dist()


if __name__ == "__main__":
    args = get_parser_args()
    processor = LatentImageNetProcessor(args)
    processor.run()
