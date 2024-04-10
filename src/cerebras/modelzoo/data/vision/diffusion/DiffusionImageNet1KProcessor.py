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
from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.data.vision.diffusion.DiffusionBaseProcessor import (
    DiffusionBaseProcessor,
)
from cerebras.modelzoo.data.vision.utils import create_worker_cache


# NOTE: Inorder to use this dataloader,
# model side changes as follows are needed
# 1. initializing VAEModel.
# 2. Using `vae_noise` to create latent from forward pass of VAEEncoder
@registry.register_datasetprocessor("DiffusionImageNet1KProcessor")
class DiffusionImageNet1KProcessor(DiffusionBaseProcessor):
    def __init__(self, params):
        super().__init__(params)
        self.num_classes = 1000

    def create_dataset(self, use_training_transforms=True, split="train"):
        if self.use_worker_cache and dist.is_streamer():
            self.data_dir = create_worker_cache(self.data_dir)

        self.check_split_valid(split)
        transform, target_transform = self.process_transform()

        if not os.path.isfile(os.path.join(self.data_dir, "meta.bin")):
            raise RuntimeError(
                "The meta file meta.bin is not present in the root directory. "
                "Check data/vision/classification/data/README.md for "
                "more details on downloading the dataset."
            )

        if not os.path.isdir(os.path.join(self.data_dir, split)):
            raise RuntimeError(
                f"No directory {split} under root dir. Refer to "
                "data/vision/classification/data/README.md on how to "
                "prepare the dataset."
            )

        dataset = torchvision.datasets.ImageNet(
            root=self.data_dir,
            split=split,
            transform=transform,
            target_transform=target_transform,
        )
        return dataset

    def _custom_collate_fn(self, batch):
        batch = default_collate(batch)
        input, label = batch
        data = self.noise_generator(*self.label_dropout(input, label))
        return data

    def create_dataloader(self, dataset, is_training=False):
        dataloader = super().create_dataloader(dataset, is_training)
        self.latent_dist_fn = self._passthrough
        dataloader.collate_fn = self._custom_collate_fn

        return dataloader


if __name__ == "__main__":
    import os

    import numpy as np
    import torch
    import yaml

    from cerebras.modelzoo.models.vision.dit.utils import set_defaults

    fpath = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../configs/params_dit_small_patchsize_2x2.yaml",
        )
    )
    with open(fpath, "r") as fid:
        params = yaml.safe_load(fid)

    params = set_defaults(params)

    data_obj = DiffusionImageNet1KProcessor(params['train_input'])
    dataset = data_obj.create_dataset(
        use_training_transforms=True, split=params["train_input"]["split"]
    )
    print("Dataset features: \n")
    sample = dataset[0]
    print(
        f"Dataset 0th sample: {sample}, {sample[0].shape}, unique vals: img: {torch.unique(sample[0])}, label:{np.unique(sample[1])}"
    )
    dataloader = data_obj.create_dataloader(dataset, is_training=True)
    for ii, data in enumerate(dataloader):
        if ii == 1:
            break
        for k, v in data.items():
            print(f"{k} -- {v.shape}, {v.dtype}")
        print("----")
