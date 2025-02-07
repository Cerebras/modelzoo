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
import os
import random

from PIL import Image

# isort: off
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
# isort: on


def _resize_and_save(
    input_img_path, output_img_path, width, height, is_lbl_transform
):
    if is_lbl_transform:
        resample = Image.NEAREST
    else:
        resample = Image.BICUBIC
    img = Image.open(input_img_path)
    img_resized = img.resize((width, height), resample=resample)
    img_resized.save(output_img_path)
    img.close()


def _center_crop_and_save(
    input_img_path, output_img_path, width, height, is_lbl_transform
):
    img = Image.open(input_img_path)
    img_w, img_h = img.size

    left = (img_w - width) // 2
    top = (img_h - height) // 2
    right = (img_w + width) // 2
    bottom = (img_h + height) // 2

    img_cropped = img.crop((left, top, right, bottom))
    img_cropped.save(output_img_path)
    img.close()


def apply_transform_to_files(
    image_names,
    input_dir,
    output_dir,
    width,
    height,
    split,
    transform_op,
    apply_transform_to_lbl=True,
):
    print(f"input_dir: {input_dir}")
    print(f"output_dir: {output_dir}")
    print(f"width: {width}, height: {height}")
    print(f"split: {split}")
    print(
        f"transform_op: {transform_op}, apply_transform_to_lbl: {apply_transform_to_lbl}"
    )
    _in = input("Does everything look ok for above, type y or n \n")
    if _in.lower() in ["n", "no"]:
        raise ValueError("Not proceeding")

    os.makedirs(os.path.join(output_dir, split, "images"))

    if apply_transform_to_lbl:
        os.makedirs(os.path.join(output_dir, split, "gt"))

    if transform_op == "resize":
        transform_fn = _resize_and_save
    elif transform_op == "center-crop":
        transform_fn = _center_crop_and_save
    else:
        raise ValueError(f"Unsupported transform: {transform_op}")

    for fname in image_names:

        img_in_path = os.path.join(input_dir, split, "images", fname)
        img_out_path = os.path.join(output_dir, split, "images", fname)
        transform_fn(
            img_in_path,
            img_out_path,
            width,
            height,
            is_lbl_transform=False,
        )

        if apply_transform_to_lbl:
            lbl_in_path = os.path.join(input_dir, split, "gt", fname)
            lbl_out_path = os.path.join(output_dir, split, "gt", fname)
            transform_fn(
                lbl_in_path,
                lbl_out_path,
                width,
                height,
                is_lbl_transform=True,
            )


def get_parser_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="input image file dir",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="directory for output images",
    )
    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="width of output image",
    )
    parser.add_argument(
        "--height",
        type=int,
        required=True,
        help="height of output image",
    )
    parser.add_argument(
        "--transform",
        type=str,
        required=True,
        choices=["resize", "center-crop"],
        help="transform to apply to the input image, choose from [resize, center-crop]",
    )
    return parser


if __name__ == "__main__":
    args = get_parser_args().parse_args()

    input_dir = args.input_dir
    width = args.width
    height = args.height
    transform_op = args.transform

    if args.output_dir is None:
        output_dir = f"/cb/datasets/cv/scratch/demo/inria_aerial_{width}_{height}_{transform_op}/src_files/AerialImageDataset"
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir)

    for split in ["train", "val", "test"]:
        img_names = sorted(os.listdir(os.path.join(input_dir, split, "images")))
        if split != "test":
            apply_transform_to_lbl = True
        else:
            apply_transform_to_lbl = False

        apply_transform_to_files(
            img_names,
            input_dir,
            output_dir,
            width,
            height,
            split,
            transform_op=transform_op,
            apply_transform_to_lbl=apply_transform_to_lbl,
        )

        # spot check
        out_img_names = sorted(
            os.listdir(os.path.join(output_dir, split, "images"))
        )
        test_imgs = random.sample(out_img_names, 5)
        for out_img in test_imgs:
            print(f"-- testing {out_img} -- ")
            test_img = os.path.join(output_dir, split, "images", out_img)

            im = Image.open(test_img)
            assert im.width == width, "Width mismatch"
            assert im.height == height, "Height mismatch"
            im.close()

            if apply_transform_to_lbl:
                test_lbl = os.path.join(output_dir, split, "gt", out_img)
                im = Image.open(test_lbl)
                assert im.width == width, "Width mismatch"
                assert im.height == height, "Height mismatch"
                im.close()
