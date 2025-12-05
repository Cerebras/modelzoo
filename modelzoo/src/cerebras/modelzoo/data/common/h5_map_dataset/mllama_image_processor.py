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

import math
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from transformers import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_transforms import get_image_size, pad, resize
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    PILImageResampling,
    infer_channel_dimension_format,
    to_numpy_array,
)
from transformers.utils import TensorType


@lru_cache(maxsize=10)
def get_all_supported_aspect_ratios(
    max_image_tiles: int,
) -> List[Tuple[int, int]]:
    aspect_ratios = []
    for width in range(1, max_image_tiles + 1):
        for height in range(1, max_image_tiles + 1):
            if width * height <= max_image_tiles:
                aspect_ratios.append((width, height))
    return aspect_ratios


def get_image_size_fit_to_canvas(
    image_height: int,
    image_width: int,
    canvas_height: int,
    canvas_width: int,
    tile_size: int,
) -> Tuple[int, int]:
    target_width = np.clip(image_width, tile_size, canvas_width)
    target_height = np.clip(image_height, tile_size, canvas_height)

    scale_h = target_height / image_height
    scale_w = target_width / image_width

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.floor(image_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.floor(image_width * scale_h), target_width)

    return new_height, new_width


@lru_cache(maxsize=100)
def get_optimal_tiled_canvas(
    image_height: int,
    image_width: int,
    max_image_tiles: int,
    tile_size: int,
) -> Tuple[int, int]:
    possible_tile_arrangements = get_all_supported_aspect_ratios(
        max_image_tiles
    )
    possible_canvas_sizes = np.array(possible_tile_arrangements) * tile_size

    target_heights, target_widths = np.array(possible_canvas_sizes).T

    scale_h = target_heights / image_height
    scale_w = target_widths / image_width

    scales = np.where(scale_w > scale_h, scale_h, scale_w)

    upscaling_options = scales[scales >= 1]
    if len(upscaling_options) > 0:
        selected_scale = np.min(upscaling_options)
    else:
        downscaling_options = scales[scales < 1]
        selected_scale = np.max(downscaling_options)

    chosen_canvas = possible_canvas_sizes[scales == selected_scale]

    if len(chosen_canvas) > 1:
        areas = chosen_canvas[:, 0] * chosen_canvas[:, 1]
        optimal_idx = np.argmin(areas)
        optimal_canvas = chosen_canvas[optimal_idx]
    else:
        optimal_canvas = chosen_canvas[0]

    return optimal_canvas


def split_to_tiles(
    image: np.ndarray, num_tiles_height: int, num_tiles_width: int
) -> np.ndarray:
    num_channels, height, width = image.shape
    tile_height = height // num_tiles_height
    tile_width = width // num_tiles_width

    image = image.reshape(
        num_channels, num_tiles_height, tile_height, num_tiles_width, tile_width
    )
    image = image.transpose(1, 3, 0, 2, 4)
    image = image.reshape(
        num_tiles_width * num_tiles_height,
        num_channels,
        tile_height,
        tile_width,
    )

    return np.ascontiguousarray(image)


def build_aspect_ratio_mask(
    aspect_ratios: List[List[Tuple[int, int]]], max_image_tiles: int
) -> np.ndarray:
    batch_size = len(aspect_ratios)
    max_num_images = max([len(row) for row in aspect_ratios])

    aspect_ratio_mask = np.zeros(
        (batch_size, max_num_images, max_image_tiles), dtype=np.int64
    )
    aspect_ratio_mask[:, :, 0] = 1

    for i, sample_aspect_ratios in enumerate(aspect_ratios):
        for j, (num_tiles_w, num_tiles_h) in enumerate(sample_aspect_ratios):
            aspect_ratio_mask[i, j, : num_tiles_w * num_tiles_h] = 1

    return aspect_ratio_mask


def pack_images(
    batch_images: List[List[np.ndarray]],
    max_image_tiles: int,
) -> Tuple[np.ndarray, List[List[int]]]:
    batch_size = len(batch_images)
    max_num_images = max([len(images) for images in batch_images])
    shapes = [image.shape for images in batch_images for image in images]
    _, channels, tile_height, tile_width = shapes[0]

    stacked_images = np.zeros(
        (
            batch_size,
            max_num_images,
            max_image_tiles,
            channels,
            tile_height,
            tile_width,
        ),
        dtype=np.float32,
    )

    all_num_tiles = []
    for i, images in enumerate(batch_images):
        num_sample_tiles = []
        for j, image in enumerate(images):
            num_tiles = image.shape[0]
            stacked_images[i, j, :num_tiles] = image
            num_sample_tiles.append(num_tiles)
        all_num_tiles.append(num_sample_tiles)

    return stacked_images, all_num_tiles


def convert_aspect_ratios_to_ids(
    aspect_ratios: List[List[Tuple[int, int]]], max_image_tiles: int
) -> np.ndarray:
    batch_size = len(aspect_ratios)
    max_num_images = max([len(row) for row in aspect_ratios])
    supported_aspect_ratios = get_all_supported_aspect_ratios(max_image_tiles)

    aspect_ratios_ids = np.zeros((batch_size, max_num_images), dtype=np.int64)
    for i, sample_aspect_ratios in enumerate(aspect_ratios):
        for j, (num_tiles_h, num_tiles_w) in enumerate(sample_aspect_ratios):
            aspect_ratios_ids[i, j] = (
                supported_aspect_ratios.index((num_tiles_h, num_tiles_w)) + 1
            )
    return aspect_ratios_ids


def to_channel_dimension_format(
    image: np.ndarray,
    channel_dim: Union[ChannelDimension, str],
    input_channel_dim: Optional[Union[ChannelDimension, str]] = None,
) -> np.ndarray:
    if input_channel_dim is None:
        input_channel_dim = infer_channel_dimension_format(image)

    target_channel_dim = ChannelDimension(channel_dim)
    if input_channel_dim == target_channel_dim:
        return image

    if target_channel_dim == ChannelDimension.FIRST:
        image = image.transpose((2, 0, 1))
    elif target_channel_dim == ChannelDimension.LAST:
        image = image.transpose((1, 2, 0))
    else:
        raise ValueError(f"Unsupported channel dimension format: {channel_dim}")

    return image


def convert_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    return alpha_composite.convert("RGB")


class MllamaImageProcessor(BaseImageProcessor):
    def __init__(
        self,
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: bool = True,
        max_image_tiles: int = 4,
    ):
        super().__init__()
        self.do_convert_rgb = do_convert_rgb
        self.do_resize = do_resize
        self.size = size or {"height": 224, "width": 224}
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean or IMAGENET_STANDARD_MEAN
        self.image_std = image_std or IMAGENET_STANDARD_STD
        self.do_pad = do_pad
        self.max_image_tiles = max_image_tiles

    def __call__(
        self,
        images,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        return self.preprocess(images, return_tensors=return_tensors, **kwargs)

    def preprocess(
        self,
        images,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        images_list = (
            [images]
            if isinstance(images, (Image.Image, np.ndarray))
            else images
        )

        processed_images = []
        aspect_ratios = []

        for image in images_list:
            image = self.prepare_image(image)
            processed_image, aspect_ratio = self.resize_and_pad_image(image)
            processed_images.append(processed_image)
            aspect_ratios.append(aspect_ratio)

        pixel_values, num_tiles = pack_images(
            [processed_images], self.max_image_tiles
        )
        aspect_ratio_ids = convert_aspect_ratios_to_ids(
            [aspect_ratios], self.max_image_tiles
        )
        aspect_ratio_mask = build_aspect_ratio_mask(
            [aspect_ratios], self.max_image_tiles
        )

        return BatchFeature(
            data={
                "pixel_values": pixel_values,
                "aspect_ratio_ids": aspect_ratio_ids,
                "aspect_ratio_mask": aspect_ratio_mask,
                "num_tiles": num_tiles,
            },
            tensor_type=return_tensors,
        )

    def prepare_image(self, image):
        if self.do_convert_rgb:
            image = convert_to_rgb(image)

        image = to_numpy_array(image)
        image = to_channel_dimension_format(image, ChannelDimension.FIRST)

        if self.do_rescale:
            image = self.rescale(image, scale=self.rescale_factor)

        if self.do_normalize:
            image = self.normalize(
                image, mean=self.image_mean, std=self.image_std
            )

        return image

    def resize_and_pad_image(self, image):
        image_height, image_width = get_image_size(image)
        tile_size = self.size["height"]

        canvas_height, canvas_width = get_optimal_tiled_canvas(
            image_height=image_height,
            image_width=image_width,
            max_image_tiles=self.max_image_tiles,
            tile_size=tile_size,
        )
        num_tiles_height = canvas_height // tile_size
        num_tiles_width = canvas_width // tile_size

        new_height, new_width = get_image_size_fit_to_canvas(
            image_height=image_height,
            image_width=image_width,
            canvas_height=canvas_height,
            canvas_width=canvas_width,
            tile_size=tile_size,
        )

        # convert the image to have range [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-7)

        image = resize(
            image,
            (new_height, new_width),
            resample=self.resample,
        )

        padded_height = num_tiles_height * tile_size
        padded_width = num_tiles_width * tile_size
        pad_size = (
            (0, padded_height - new_height),
            (0, padded_width - new_width),
        )

        image = pad(
            image,
            pad_size,
            mode="constant",
            constant_values=0,
        )

        image = split_to_tiles(image, num_tiles_height, num_tiles_width)

        return image, (num_tiles_height, num_tiles_width)

    def generate_sample(self):
        shape = (3, self.size["height"], self.size["width"])
        return np.zeros(shape, dtype=np.float32)
