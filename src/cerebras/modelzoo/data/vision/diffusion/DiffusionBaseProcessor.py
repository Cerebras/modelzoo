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
from abc import abstractmethod

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

import cerebras.pytorch as cstorch
from cerebras.modelzoo.common.input_utils import get_streaming_batch_size
from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.data.vision.diffusion.dit_transforms import (
    LabelDropout,
    NoiseGenerator,
)
from cerebras.modelzoo.data.vision.preprocessing import get_preprocess_transform
from cerebras.modelzoo.data.vision.transforms import LambdaWithParam
from cerebras.modelzoo.data.vision.utils import (
    ShardedSampler,
    is_gpu_distributed,
)
from cerebras.modelzoo.models.vision.dit.layers.vae.utils import (
    DiagonalGaussianDistribution,
)


@registry.register_datasetprocessor("DiffusionBaseProcessor")
class DiffusionBaseProcessor:
    def __init__(self, params):
        self.data_dir = params.get("data_dir", ".")
        self.use_worker_cache = params["use_worker_cache"]
        self.mixed_precision = params["mixed_precision"]
        self.allowable_split = ["train", "val"]
        self.num_classes = params["num_classes"]

        if self.mixed_precision:
            self.mp_type = cstorch.amp.get_half_dtype()
        else:
            self.mp_type = torch.float32

        # Preprocessing params
        self.pp_params = dict()
        self.pp_params["noaugment"] = params["noaugment"]
        self.pp_params["mixed_precision"] = params["mixed_precision"]
        self.pp_params["fp16_type"] = params["fp16_type"]
        self.pp_params["transforms"] = params.get("transforms", [])

        # params for data loader
        self.batch_size = get_streaming_batch_size(params["batch_size"])
        self.shuffle = params["shuffle"]
        self.shuffle_seed = params["shuffle_seed"]
        self.drop_last = params["drop_last"]

        # multi-processing params.
        self.num_workers = params["num_workers"]
        self.prefetch_factor = params["prefetch_factor"]
        self.persistent_workers = params["persistent_workers"]
        self.distributed = is_gpu_distributed()

        # DiT related
        # copied to train/eval params in utils.py
        self.vae_scaling_factor = params["vae_scaling_factor"]
        self.dropout_rate = params["label_dropout_rate"]
        self.latent_height = params["latent_size"][0]
        self.latent_width = params["latent_size"][1]
        self.latent_channels = params["latent_channels"]
        self.num_diffusion_steps = params["num_diffusion_steps"]
        self.schedule_name = params["schedule_name"]

    def _passthrough(self, x):
        return x

    def create_dataloader(self, dataset, is_training=False):
        """
        Dataloader returns a dict with keys:
            "input": Tensor of shape (batch_size, latent_channels, latent_height, latent_width)
            "label": Tensor of shape (batch_size, ) with dropout applied with `label_dropout_rate`
            "diffusion_noise": Tensor of shape (batch_size, latent_channels, latent_height, latent_width)
                represents diffusion noise to be applied
            "timestep": Tensor of shape (batch_size, ) that
                indicates the timesteps for each diffusion sample
        """
        shuffle = self.shuffle and is_training
        generator = torch.Generator(device="cpu")
        if self.shuffle_seed is not None:
            generator.manual_seed(self.shuffle_seed)

        if self.distributed:
            data_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                shuffle=shuffle,
                seed=self.shuffle_seed,
            )
        else:
            data_sampler = ShardedSampler(
                dataset, shuffle, self.shuffle_seed, self.drop_last
            )

        if self.num_workers:
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=data_sampler,
                num_workers=self.num_workers,
                pin_memory=self.distributed,
                drop_last=self.drop_last,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=self.persistent_workers,
                generator=generator,
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=data_sampler,
                pin_memory=self.distributed,
                drop_last=self.drop_last,
                generator=generator,
            )

        # Intialize classes
        self.label_dropout = LabelDropout(self.dropout_rate, self.num_classes)
        self.noise_generator = NoiseGenerator(
            self.latent_width,
            self.latent_height,
            self.latent_channels,
            self.num_diffusion_steps,
        )

        self.latent_dist_fn = DiagonalGaussianDistribution

        dataloader.collate_fn = self.custom_collate_fn

        return dataloader

    def custom_collate_fn(self, batch):
        batch = default_collate(batch)
        input, label = batch
        data = self.noise_generator(*self.label_dropout(input, label))

        # Pop `vae_noise` tensor, not needed if not used
        vae_noise = data.pop("vae_noise")

        # torch.clamp in this class does not support half dtypes
        latent_dist = self.latent_dist_fn(input.to(torch.float32))

        # overwrite with latent
        data["input"] = (
            latent_dist.sample(vae_noise)
            .mul_(self.vae_scaling_factor)
            .to(data["diffusion_noise"].dtype)
        )

        return data

    def check_split_valid(self, split):
        if split not in self.allowable_split:
            raise ValueError(
                f"Dataset split {split} is invalid. Only values in "
                f"{self.allowable_split} are allowed."
            )

    def _get_target_transform(self, x, *args, **kwargs):
        return np.int32(x)

    def process_transform(self):
        if self.pp_params["noaugment"]:
            transform_specs = [
                {"name": "to_tensor"},
            ]
            logging.warning(
                "User specified `noaugment=True`. "
                "The input data will only be converted to tensor."
            )
            self.pp_params["transforms"] = transform_specs

        transform = get_preprocess_transform(self.pp_params)

        target_transform = LambdaWithParam(self._get_target_transform)

        return transform, target_transform

    @abstractmethod
    def create_dataset(self, use_training_transforms=True, split="train"):
        raise NotImplementedError(
            "create_dataset must be implemented in a child class!!"
        )
