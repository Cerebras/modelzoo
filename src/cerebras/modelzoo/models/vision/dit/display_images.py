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

import argparse
import logging
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

"""
Usage:

python modelzoo/models/vision/dit/display_images.py \
--folder_path=<path to folder from running `sample_generator_dit.py> \
--file_ext=.png --nrow=8 --image_size=100

"""


def display_images(folder_path, file_ext=".png", nrow=4, image_size=100):
    """
    Display images in a folder in grid
    Args:
        folder_path (str): Path to folder containing images with extension `file_ext`
        file_ext (str): File extension of images in `folder_path`. Used for glob
        nrow (int): Number of images to display in each row of the grid
        image_size (int): Resize image to this size before displaying on grid.
    """
    image_paths = sorted(Path(folder_path).glob("*" + file_ext))
    if len(image_paths) == 0:
        raise ValueError(
            f"Folder {folder_path} does not contain any {file_ext} images, cannot create grid."
        )

    image_tensors = []
    transform = transforms.Compose(
        [
            transforms.Resize(size=(image_size, image_size)),
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x.to(torch.float32)),
        ]
    )
    fname = os.path.basename(folder_path)
    grid_path = os.path.join(folder_path, f"{fname}_grid.png")
    if os.path.exists(grid_path):
        logging.info(f"Image grid already exists, nothing to do, exiting.")
    else:
        for im in tqdm(image_paths):
            img = Image.open(im)
            image_tensors.append(transform(img))

        save_image(
            torch.stack(image_tensors, dim=0),
            grid_path,
            nrow=nrow,
            normalize=True,
        )


def get_parser_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--folder_path", type=str, required=True)
    parser.add_argument(
        "--file_ext",
        type=str,
        default=".png",
    )
    parser.add_argument(
        "--nrow",
        type=int,
        default=4,
        help="Number of images per row",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=100,
        help="Resize image before displaying",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser_args()
    display_images(args.folder_path, args.file_ext, args.nrow, args.image_size)
