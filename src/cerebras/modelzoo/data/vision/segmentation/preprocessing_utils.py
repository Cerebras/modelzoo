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

import torch
from torchvision import transforms


def normalize_tensor_transform(img, normalize_data_method):
    """
    Function to normalize img
    :params img: Input torch.Tensor of any shape
    :params normalize_data_method: One of
        "zero_centered"
        "zero_one"
        "standard_score"
    """
    if normalize_data_method is None:
        pass
    elif normalize_data_method == "zero_centered":
        img = torch.div(img, 127.5) - 1
    elif normalize_data_method == "zero_one":
        img = torch.div(img, 255.0)
    elif normalize_data_method == "standard_score":
        img = (img - img.mean()) / img.std()
    else:
        raise ValueError(
            f"Invalid arg={normalize_data_method} passed to `normalize_data_method`"
        )
    return img


def adjust_brightness_transform(img, p, delta):
    """
    Function equivalent to `tf.image.adjust_brightness`,
    but executed probabilistically.
    :params img: Input torch.Tensor of any shape
    :params p: Integer representing probability
    :params delta: Float value representing the value
        by which img Tensor is increased or decreased.
    """
    if (torch.rand(1) > p).item():
        img = torch.add(img, delta)
    return img


def rotation_90_transform(img, num_rotations):
    """
    Function equivalent to `tf.image.rot90`
    Rotates img in counter clockwise direction
    :params img: torch.Tensor of shape (C, H, W) or (H, W)
    :params num_rotations: int value representing
        number of counter clock-wise rotations of img
    """
    if len(img.shape) == 3:
        # If image of type (C, H, W), rotate along H, W
        # Rotate in counter-clockwise direction
        dims = [1, 2]
    else:
        dims = [0, 1]
    img = torch.rot90(img, k=num_rotations, dims=dims)

    return img


def resize_image_with_crop_or_pad_transform(img, target_height, target_width):
    """
    Function equivalent to `tf.image.resize_with_crop_or_pad`
    :params img: torch.Tensor of shape (C, H, W) or (H, W)
    :params target_height: int value representing output image height
    :params target_width: int value representing output image width
    :returns torch.Tensor of shape (C, target_height, target_width)
    """

    def _pad_image(img):
        """
        Pad image till it reaches target_height and target_width
        """
        img_shape = img.shape

        img_width = img_shape[-1]
        img_height = img_shape[-2]

        lft_rgt_pad = max((target_width - img_width) // 2, 0)
        top_bot_pad = max((target_height - img_height) // 2, 0)

        excess_right_pad = target_width - img_width - 2 * lft_rgt_pad
        excess_bot_pad = target_height - img_height - 2 * top_bot_pad
        pad = [
            lft_rgt_pad,
            lft_rgt_pad + excess_right_pad,
            top_bot_pad,
            top_bot_pad + excess_bot_pad,
        ]
        img = torch.nn.functional.pad(img, pad)

        return img

    def _crop_image(img):
        img_shape = img.shape
        # Crop only when necessary. CenterCrop pads if
        # crop dimensions are greater, hence taking min.
        crop_height = min(img_shape[-2], target_height)
        crop_width = min(img_shape[-1], target_width)
        img = transforms.CenterCrop((crop_height, crop_width))(img)
        return img

    cropped_img = _crop_image(img)
    padded_img = _pad_image(cropped_img)

    assert padded_img.shape[-1] == target_width
    assert padded_img.shape[-2] == target_height

    return padded_img


def tile_image_transform(img, target_height, target_width):
    """
    Function to tile image to tgt_height and target_width
    If target_height < image_height: image is not tiled in this dimension.
    If target_width < image_width: image is not tiled in this dimension.
    :params img: input torch.Tensor of shape (C, H, W)
    :params target_height: int value representing output tiled image height
    :params target_width: int value representing output tiled image width
    :returns torch.Tensor of shape (C, target_height, target_width)
    """

    assert len(img.shape) == 3
    img_channels, img_height, img_width = img.shape
    tgt_img_shape = [img_channels, target_height, target_width]

    def _get_tiled_image(img, tgt_img_shape, axis):

        if tgt_img_shape[axis] <= img.shape[axis]:
            # No tiling since image already satisfies requirement
            return img
        else:
            diff = tgt_img_shape[axis] - img.shape[axis]
            q, r = divmod(diff, img.shape[axis])

            temp_img = img
            for _ in range(q):
                temp_img = torch.concat((img, temp_img), axis=axis)

            if r > 0:
                if axis == 1:
                    sliced_img = temp_img[:, :r, :]
                elif axis == 2:
                    sliced_img = temp_img[:, :, :r]
                else:
                    raise ValueError(
                        f"Incorrect value of {axis} passed. Valid integers are 1, 2"
                    )
                temp_img = torch.concat((temp_img, sliced_img), axis=axis)
            return temp_img

    v_tiled_img = _get_tiled_image(img, tgt_img_shape=tgt_img_shape, axis=1)
    tiled_img = _get_tiled_image(
        v_tiled_img, tgt_img_shape=tgt_img_shape, axis=2
    )
    return tiled_img
