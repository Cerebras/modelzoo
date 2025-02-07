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
import glob
import json
import os

import h5py
import torch
import torch.nn as nn
from tqdm import tqdm

# isort: off
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
# isort: on

from cerebras.modelzoo.common.utils.utils import check_and_create_output_dirs


def _resize_and_save(
    input_img_path,
    output_img_path,
    width,
    height,
    depth,
):
    resize_transform = nn.Upsample((width, height, depth))
    with h5py.File(input_img_path, "r") as f:
        echo1 = torch.Tensor(f["echo1"][()])
        echo2 = torch.Tensor(f["echo2"][()])
        seg = torch.Tensor(f["seg"][()])

        img_w, img_h, img_d, num_classes = seg.shape
        # view needed for volumetic interpolation
        echo1_resized = resize_transform(echo1.view(1, 1, img_w, img_h, img_d))
        # convert to np.int16 to be consistent with original dataset format
        echo1_resized = (
            echo1_resized.view(width, height, depth).to(torch.int16).numpy()
        )
        echo2_resized = resize_transform(echo2.view(1, 1, img_w, img_h, img_d))
        echo2_resized = (
            echo2_resized.view(width, height, depth).to(torch.int16).numpy()
        )
        # permuting to move one hot labels to first channels
        seg_resized = resize_transform(
            seg.permute(3, 0, 1, 2).view(1, num_classes, img_w, img_h, img_d)
        )
        seg_resized = (
            seg_resized.permute(0, 2, 3, 4, 1)
            .view(width, height, depth, num_classes)
            .to(torch.bool)
            .numpy()
        )

    with h5py.File(output_img_path, "w") as w:
        w.create_dataset(
            "echo1", data=echo1_resized, shape=(width, height, depth)
        )

        w.create_dataset(
            "echo2", data=echo2_resized, shape=(width, height, depth)
        )

        w.create_dataset(
            "seg",
            data=seg_resized,
            shape=(width, height, depth, num_classes),
        )


def get_parser_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="directory containing original dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="directory where dataset will be stored.",
    )
    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="width of output volume",
    )
    parser.add_argument(
        "--height",
        type=int,
        required=True,
        help="height of output volume",
    )
    parser.add_argument(
        "--depth",
        type=int,
        required=True,
        help="depth of output volume",
    )
    return parser


if __name__ == "__main__":
    args = get_parser_args().parse_args()
    width, height, depth = args.width, args.height, args.depth
    input_json_dir = os.path.join(
        args.input_dir, "v1-release/annotations/v1.0.0"
    )
    input_hdf5_dir = os.path.join(args.input_dir, "v1-release/image_files")
    output_json_dir = os.path.join(
        args.output_dir, "v1-release/annotations/v1.0.0"
    )
    output_hdf5_dir = os.path.join(args.output_dir, "v1-release/image_files")

    check_and_create_output_dirs(output_json_dir, filetype="json")
    check_and_create_output_dirs(output_hdf5_dir, filetype="h5")
    print(f"\nStarting writing data to {args.output_dir}.")

    for file_name in tqdm(glob.glob1(input_hdf5_dir, "*.h5")):
        input_img_path = os.path.join(input_hdf5_dir, file_name)
        _resize_and_save(
            input_img_path,
            f"{output_hdf5_dir}/{file_name}",
            width,
            height,
            depth,
        )

    # save new volume dimensions to dataset dict
    splits = ["train", "val", "test"]
    for split in splits:
        with open(f"{input_json_dir}/{split}.json") as f:
            dataset_dict = json.load(f)
            for i in range(len(dataset_dict["images"])):
                dataset_dict["images"][i]["matrix_shape"] = [
                    width,
                    height,
                    depth,
                ]
            with open(f"{output_json_dir}/{split}.json", "w") as f:
                json.dump(dataset_dict, f, indent=4)
