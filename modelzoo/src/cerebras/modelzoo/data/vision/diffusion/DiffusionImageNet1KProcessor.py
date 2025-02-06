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

import os

import torchvision
from torch.utils.data.dataloader import default_collate

import cerebras.pytorch.distributed as dist
from cerebras.modelzoo.data.vision.diffusion.config import (
    DiffusionImageNet1KProcessorConfig,
)
from cerebras.modelzoo.data.vision.diffusion.DiffusionBaseProcessor import (
    DiffusionBaseProcessor,
)
from cerebras.modelzoo.data.vision.utils import create_worker_cache


# NOTE: Inorder to use this dataloader,
# model side changes as follows are needed
# 1. initializing VAEModel.
# 2. Using `vae_noise` to create latent from forward pass of VAEEncoder
class DiffusionImageNet1KProcessor(DiffusionBaseProcessor):
    def __init__(self, config: DiffusionImageNet1KProcessorConfig):
        if isinstance(config, dict):
            config = DiffusionImageNet1KProcessorConfig(**config)
        super().__init__(config)
        self.num_classes = 1000

    def create_dataset(self):
        if self.use_worker_cache and dist.is_streamer():
            self.data_dir = create_worker_cache(self.data_dir)

        self.check_split_valid(self.split)
        transform, target_transform = self.process_transform()

        if not os.path.isfile(os.path.join(self.data_dir, "meta.bin")):
            raise RuntimeError(
                "The meta file meta.bin is not present in the root directory. "
                "Check data/vision/classification/data/README.md for "
                "more details on downloading the dataset."
            )

        if not os.path.isdir(os.path.join(self.data_dir, self.split)):
            raise RuntimeError(
                f"No directory {self.split} under root dir. Refer to "
                "data/vision/classification/data/README.md on how to "
                "prepare the dataset."
            )

        dataset = torchvision.datasets.ImageNet(
            root=self.data_dir,
            split=self.split,
            transform=transform,
            target_transform=target_transform,
        )
        return dataset

    def _custom_collate_fn(self, batch):
        batch = default_collate(batch)
        input, label = batch
        data = self.noise_generator(*self.label_dropout(input, label))
        return data

    def create_dataloader(self):
        dataloader = super().create_dataloader()
        self.latent_dist_fn = self._passthrough
        dataloader.collate_fn = self._custom_collate_fn

        return dataloader
