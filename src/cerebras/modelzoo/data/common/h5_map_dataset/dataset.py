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
import os
from typing import Dict, List, Optional, Union

import numpy
import torch
from PIL import Image
from pydantic import PositiveInt, model_validator

import cerebras.pytorch as cstorch
import cerebras.pytorch.distributed as dist
from cerebras.modelzoo.common.input_utils import PaddingSample
from cerebras.modelzoo.config import BaseConfig
from cerebras.modelzoo.config.types import ValidatedPath

# for mllama dataset and configs
from cerebras.modelzoo.data.common.h5_map_dataset.mllama_image_processor import (
    MllamaImageProcessor,
)
from cerebras.modelzoo.data.common.h5_map_dataset.readers import (
    H5Reader,
    Mixture,
)
from cerebras.modelzoo.data.vision.preprocessing import get_preprocess_transform
from cerebras.modelzoo.data.vision.utils import create_worker_cache
from cerebras.pytorch.utils.data.sampler import pad_index


class HDF5DatasetConfig(BaseConfig):
    data_dir: Union[ValidatedPath, List[ValidatedPath], None] = None
    """
    The path to the HDF5 files.
    Exactly one of "data_dir" or "mixture" must be specified.
    """

    batch_size: PositiveInt = ...
    """
    Indicates global batch size of the model. This differs from 
    `micro_batch_size`.
    Global batch size refers to the number of samples 
    fed to the model in total for forward and backward pass before
    weights are updated. A global batch size exceeding
    device memory is split into smaller `micro_batch_sizes` and 
    forward and backward pass are executed and accummulated before final update.
    """

    shuffle: bool = False
    "Whether or not to shuffle the dataset."

    shuffle_seed: int = 0
    "The seed used for deterministic shuffling."

    use_worker_cache: bool = False
    """
    Whether or not to copy data to storage that is directly attached to each
    individual worker node. Useful when your network storage is unusually slow,
    but otherwise discouraged.
    """

    max_sequence_length: Optional[int] = None
    """
    The sequence length of samples produced by the dataloader. When using the
    'corpus' data format, the same preprocessed data will work with any max
    sequence length, so this may be set at runtime. When using the 'sample'
    format this must be set to `None`.
    """

    data_subset: Optional[str] = None
    """
    An optional specification to only consider a
    subset of the full dataset, useful for sequence length
    scheduling and multi-epoch testing. Expected to be a comma
    separated list of ranges, e.g. '0.0-0.5' or '0.1-0.3,0.7-1.0'.
    Specifying '0.0-0.5' creates a dataset from the first half of
    the data on disk and disregards the second half.
    """

    mixture: Optional[List[dict]] = None
    """
    An optional specification of multiple datasets to mix over to create one
    single weighted combination. Each element must be a dictionary containing
    keys `data_dir` and `weight`. `data_dir` serves the same purpose as
    mentioned above. `weight` defines the probability with which this dataset
    should be sampled from. Weights are normalized to sum to 1. Optionally, the
    dictionary may also contain a `data_subset` field which functions the same
    as the `data_subset` argument above.
    """

    drop_last: bool = True
    """
    Similar to the PyTorch drop_last setting except that samples that when set
    to `True`, samples that would have been dropped at the end of one epoch are
    yielded at the start of the next epoch so that there is no data loss. This
    is necessary for a data ordering that is independent of the distributed
    setup being used.
    """

    num_samples: Optional[int] = None
    """
    The number of samples to extend or shrink each logical epoch to.
    In multi-epoch training, it is common to set this to the total number of
    samples that you plan to train on and enable shuffling so that epochs
    are not sequential but instead shuffled together for potentially improved
    convergence. If shuffling is off, epoch is shrunk (if num_samples < len(dataset))
    or extended by appending copies of the dataset (if num_samples > len(dataset))
    to fill the requested num_samples. It is not recommended to set this
    setting when shuffle is off. If None, One epoch is sampled from by querying
    the length of the dataset.
    """

    sort_files: bool = True
    """
    Whether or not the reader should sort the input files. This is included for
    backwards compatibility and should almost always be set to `True`.
    """

    use_vsl: bool = False
    """
    Flag to enable variable sequence length training. It requires the dataset
    to have two extra features: the `attention_span` of keys and the
    `position_ids` of tokens.  Defaults to `False`.
    """

    pad_last: bool = False
    """
    Flag to enable padding of the last batch so that the last batch has the same
    batch size as the rest of the batches.
    """

    @model_validator(mode="after")
    def check_mutual_exclusivity(self):
        if self.data_dir is not None:
            if self.mixture is not None:
                raise ValueError(
                    "Only one of `data_dir` or `mixture` must be specified."
                )
        elif self.mixture is None:
            raise ValueError(
                "One of `data_dir` or `mixture` must be specified."
            )

        return self


class HDF5Dataset(torch.utils.data.Dataset):
    """
    Dynamically read samples from disk for using mapping paradigms.

    It supports two different data formats on disk. The first is data stored
    in an H5 file in the shape `(num_tokens,)`, i.e. a series of documents
    tokenized and concatenated together. We call this format the 'corpus' format
    The second format is H5 data of shape `(num_sequences, ...)`, i.e. data has
    already been tokenized and split into sequences. We call this format the 'sample' format.

    The corpus format supports flexible choice of MSL backed by a single copy of
    the data on disk. Both formats support deterministic restart, and a data
    order that is independent of the configuration of the cluster you are
    running on. I.e. you can pause a run, increase or decrease the number of
    systems you are running on, and restart the run with no change in data
    order.

    When used in combination with shuffling, this implementation relies on
    random access reads to disk to dynamically split samples into sequences
    and shuffle. Users with unusually slow storage should look out for data
    loading bottlenecks and might consider using `use_worker_cache=True` if
    disk access is indeed a bottleneck.

    Args:
        config: The configuration used to configure the dataset
    """

    def __init__(self, config: HDF5DatasetConfig):
        if isinstance(config, dict):
            config = HDF5DatasetConfig(**config)

        self.use_worker_cache = config.use_worker_cache
        self.max_sequence_length = config.max_sequence_length
        self.shuffle = config.shuffle
        self.shuffle_seed = config.shuffle_seed
        self.data_dir = config.data_dir
        self.mixture_params = config.mixture
        self.batch_size = config.batch_size
        self.drop_last = config.drop_last
        self.num_samples = config.num_samples
        self.sort_files = config.sort_files
        self.use_vsl = config.use_vsl
        self.pad_last = config.pad_last

        self.map_fn = None

        # Set of member variables that should be ignored when returning state_dict
        self._state_dict_ignore_keys = {
            "map_fn",
            "reader",
            "sampler",
            "_state_dict_ignore_keys",
            "_load_state_ignore_keys",
        }
        # Set of member variables that should be ignored when comparing previous
        # and current state_dict. These variables don't affect the samples returned
        # from the dataset which is why they are ignored.
        self._load_state_ignore_keys = {"use_worker_cache", "batch_size"}

        if self.drop_last and self.pad_last:
            logging.warning(
                "Both drop_last and pad_last were specified to be True. "
                "Note that pad_last only has any effect when drop_last is False."
            )

        if self.data_dir and self.mixture_params:
            raise ValueError(
                "Only one of `data_dir` or `mixture` can be specified."
            )
        if self.data_dir is not None:
            self.reader = self._set_up_reader(self.data_dir, config.data_subset)
        else:
            self.reader = Mixture(
                [
                    self._set_up_reader(
                        x["data_dir"], x.get("data_subset", None)
                    )
                    for x in self.mixture_params
                ],
                [x["weight"] for x in self.mixture_params],
                interleave=not self.shuffle,
                seed=self.shuffle_seed,
            )

        self.sampler = cstorch.utils.data.DistributedSampler(
            self,
            shuffle=self.shuffle,
            seed=self.shuffle_seed,
            shard=True,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            num_samples=self.num_samples,
            pad_last=self.pad_last,
        )

        if self.by_sample and self.shuffle:
            logging.warning(
                "You have chosen to use the sample data format with shuffling. "
                "If you are doing a single-epoch run, it is usually beneficial "
                "to shuffle at preprocessing time instead of runtime. On some "
                "storage setups, shuffling at runtime can cause performance "
                "degredation."
            )

    def state_dict(self):
        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in self._state_dict_ignore_keys
        }

    def load_state_dict(self, state_dict, strict: bool = True):
        if not strict:
            # Don't run any checks
            return

        mismatches = []
        missing = []
        unknown = set(state_dict.keys())
        for k, v in self.state_dict().items():
            unknown.discard(k)
            if k in self._load_state_ignore_keys:
                continue

            if k not in state_dict:
                missing.append(k)
            elif state_dict[k] != v:
                mismatches.append([k, v, state_dict[k]])

        error_str = ""

        if unknown:
            error_str += (
                f"The following keys are unknown in the state_dict: "
                f"{','.join(unknown)}.\n"
            )
        if mismatches:
            error_str += (
                (
                    "The following keys mismatch between the currently loaded dataset "
                    "and the state_dict being loaded onto the dataset:\n    "
                )
                + "\n    ".join(
                    f"key={a}, current_value={b}, state_dict_value={c}"
                    for a, b, c, in mismatches
                )
                + "\n"
            )
        if missing:
            error_str += (
                f"The following keys are missing in the state_dict: "
                f"{','.join(missing)}.\n"
            )

        if error_str:
            raise RuntimeError(
                f"state_dict is incompatible with the dataset settings. "
                f"If these incompatibilities are expected, load with "
                f"`strict=False` setting. \n{error_str}"
            )

    def generate_sample(self):
        """
        Generates an empty tensor with the same shape and dtype
        as a sample from its dataset.
        """
        shape = self.reader.vdataset.shape[1:]
        np_dtype = self.reader.vdataset.dtype
        dtype = cstorch.from_numpy(numpy.empty(0).astype(np_dtype)).dtype
        return PaddingSample(shape, dtype)

    @property
    def by_sample(self):
        return self.reader.by_sample

    def map(self, fn: callable):
        if self.map_fn is not None:
            raise ValueError(
                f"You may only apply one map function to a H5MapDataset"
            )

        if not callable(fn):
            raise ValueError("Mapping function must be a callable.")

        self.map_fn = fn

    def _set_up_reader(self, data_dir, subset):
        if not isinstance(data_dir, list):
            data_dir = [data_dir]
        if self.use_worker_cache and cstorch.use_cs() and dist.is_streamer():
            data_dir = [create_worker_cache(d) for d in data_dir]

        reader = H5Reader(
            data_dirs=data_dir,
            sequence_length=self.max_sequence_length,
            read_extra_token=True,
            data_subset=subset,
            sort=self.sort_files,
            use_vsl=self.use_vsl,
        )
        return reader

    def __getitem__(self, i):
        if i == pad_index:
            if not self.pad_last:
                raise RuntimeError(
                    "Unexpectedly encountered the pad index when pad_last was False"
                )
            x = self.generate_sample()
        else:
            x = self.reader[i]

        if self.map_fn is not None:
            return self.map_fn(x)
        return x

    def __len__(self):
        return len(self.reader)


class MLMHDF5Dataset(HDF5Dataset):
    """Dataset class to handle text preprocessing in bert mlm datasets.

    Args:
        config: The config used to configure the dataset.
    """

    def _set_up_reader(self, data_dir, subset):
        if not isinstance(data_dir, list):
            data_dir = [data_dir]
        if self.use_worker_cache and cstorch.use_cs() and dist.is_streamer():
            data_dir = [create_worker_cache(d) for d in data_dir]

        reader = H5Reader(
            data_dirs=data_dir,
            extra_data_keys=["labels"],
            sequence_length=self.max_sequence_length,
            read_extra_token=True,
            data_subset=subset,
            sort=self.sort_files,
            use_vsl=self.use_vsl,
        )
        return reader

    def generate_sample(self):
        data_sample = super().generate_sample()
        # generate an empty tensor with the same shape and dtype
        # as an processed image from its dataset

        shape = self.reader.vdataset_full["labels"].shape[1:]
        np_dtype = self.reader.vdataset_full["labels"].dtype
        dtype = cstorch.from_numpy(numpy.empty(0).astype(np_dtype)).dtype
        labels_sample = PaddingSample(shape, dtype)

        return data_sample, labels_sample

    def __getitem__(self, i):
        if i == pad_index:
            if not self.pad_last:
                raise RuntimeError(
                    "Unexpectedly encountered the pad index when pad_last was False"
                )
            x = self.generate_sample()
        else:
            x = self.reader[i]

        if self.map_fn is not None:
            data = self.map_fn(x)
            return data

        data, labels = x["data"], x["labels"]

        return data, labels


class MultiModalHDF5DatasetConfig(HDF5DatasetConfig):
    img_data_dir: ValidatedPath = ...
    "The path to the directory containing the images."

    image_data_size: List[int] = ...
    "The final C x H x W shape of the image."

    transforms: List[dict] = ...
    "A specification of the torchvision transforms."

    def post_init(self, context):
        if model_config := context.get("model", {}).get("config"):
            if hasattr(model_config, "image_model"):
                self.image_data_size = [
                    model_config.image_model.num_channels,
                    *model_config.image_model.image_size,
                ]


class MultiModalHDF5Dataset(HDF5Dataset):
    """Dataset class to handle image preprocessing in multimodal datasets.

    This class is largely the same as the parent class `HDF5Dataset` except
    with added image loading and preprocessing.

    Args:
        config: The config used to configure the dataset.
    """

    def __init__(self, config: MultiModalHDF5DatasetConfig):
        if isinstance(config, dict):

            class _MultiModalHDF5DatasetConfig(MultiModalHDF5DatasetConfig):
                model_config = dict(extra="ignore")

            config = _MultiModalHDF5DatasetConfig(**config)

        super().__init__(config)

        self.img_data_dir = config.img_data_dir
        self.image_data_size = config.image_data_size  # (C, H, W)
        self.transforms = get_preprocess_transform(
            {"transforms": config.transforms}
        )

        self._state_dict_ignore_keys.add("transforms")
        self._load_state_ignore_keys.add("img_data_dir")
        self._load_state_ignore_keys.add("image_data_size")

    def generate_sample(self):
        text_sample = super().generate_sample()
        # generate an empty tensor with the same shape and dtype
        # as an processed image from its dataset
        dtype = cstorch.amp.get_half_dtype()
        img_sample = PaddingSample(self.image_data_size, dtype)
        return text_sample, img_sample

    def preprocess_img(self, path):
        path = path[0].decode("utf-8")
        if path != "None":
            image_path = os.path.join(self.img_data_dir, path)
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.new(
                mode="RGB",
                size=(self.image_data_size[2], self.image_data_size[1]),
            )
        return self.transforms(image)

    def _set_up_reader(self, data_dir, subset):
        if not isinstance(data_dir, list):
            data_dir = [data_dir]
        if self.use_worker_cache and cstorch.use_cs() and dist.is_streamer():
            data_dir = [create_worker_cache(d) for d in data_dir]

        reader = H5Reader(
            data_dirs=data_dir,
            extra_data_keys=["img_path"],
            sequence_length=self.max_sequence_length,
            read_extra_token=True,
            data_subset=subset,
            sort=self.sort_files,
            use_vsl=self.use_vsl,
        )
        return reader

    def __getitem__(self, i):
        if i == pad_index:
            if not self.pad_last:
                raise RuntimeError(
                    "Unexpectedly encountered the pad index when pad_last was False"
                )
            text_data, img_data = self.generate_sample()
        else:
            data = self.reader[i]
            text_data, img_path = data["data"], data["img_path"]
            img_data = self.preprocess_img(img_path)

        if self.map_fn is not None:
            data = self.map_fn(text_data)
            data["image_data"] = img_data
            return data

        return text_data, img_data


class MultimodalSimpleHDF5DatasetConfig(MultiModalHDF5DatasetConfig):
    max_num_img: int = 1
    """
    The maximum number of images.
    """

    num_patches: Optional[int] = None
    """
    The number of patches.
    """

    def post_init(self, context):
        super().post_init(context)

        if self.num_patches is None:
            model_config = context.get("model", {}).get("config")

            if hasattr(model_config, "image_model"):
                if len(self.image_data_size) == 3:
                    self.num_patches = (
                        self.image_data_size[-1]
                        // model_config.image_model.patch_size[0]
                    ) * (
                        self.image_data_size[-2]
                        // model_config.image_model.patch_size[1]
                    )
                else:
                    self.num_patches = self.image_data_size[0]


### H5 format
# 1. Data: B x 7 x S -- original 6 + token_modality_idx
# 2. Img_path: list of strings
# 3. image_data_loc: B x 1 x I * num_patches


class MultimodalSimpleHDF5Dataset(MultiModalHDF5Dataset):
    """Dataset class to handle image preprocessing in multimodal datasets.

    This class is largely the same as the parent class `MultimodalHDF5Dataset` except
    with added support for multiple images and intermingling of text and images.

    Args:
        config: The config used to configure the dataset.
    """

    def __init__(self, config: MultimodalSimpleHDF5DatasetConfig):
        if isinstance(config, dict):
            config = MultimodalSimpleHDF5DatasetConfig(**config)

        super().__init__(config)

        self.max_num_img = config.max_num_img
        self.num_patches = config.num_patches
        self.image_data_size = list(self.image_data_size)
        self.image_data_size.insert(0, self.max_num_img)

    def _set_up_reader(self, data_dir, subset):
        if not isinstance(data_dir, list):
            data_dir = [data_dir]
        if self.use_worker_cache and cstorch.use_cs() and dist.is_streamer():
            data_dir = [create_worker_cache(d) for d in data_dir]

        reader = H5Reader(
            data_dirs=data_dir,
            extra_data_keys=["img_path", "img_data_loc"],
            sequence_length=self.max_sequence_length,
            read_extra_token=True,
            data_subset=subset,
            sort=self.sort_files,
            use_vsl=self.use_vsl,
        )
        return reader

    def generate_sample(self):
        text_sample, img_sample = super().generate_sample()
        img_data_loc_sample = PaddingSample(
            [self.max_num_img, self.num_patches], dtype
        )
        return text_sample, img_sample, img_data_loc_sample

    def preprocess_img(self, path_list):
        img_list = []
        for path in path_list:
            path = path.decode("utf-8")
            if path != "None":
                image_path = os.path.join(self.img_data_dir, path)
                image = Image.open(image_path).convert("RGB")
            else:
                image = Image.new(
                    mode="RGB",
                    size=(self.image_data_size[2], self.image_data_size[1]),
                )
            img_list.append(self.transforms(image).unsqueeze(0))

        img = torch.cat(img_list, dim=0)
        return img

    def __getitem__(self, i):
        if i == pad_index:
            if not self.pad_last:
                raise RuntimeError(
                    "Unexpectedly encountered the pad index when pad_last was False"
                )
            text_data, img_data, img_data_loc = self.generate_sample()
        else:
            data = self.reader[i]
            text_data, img_path, img_data_loc = (
                data["data"],
                data["img_path"],
                data["img_data_loc"],
            )
            img_data = self.preprocess_img(img_path)

        if self.map_fn is not None:
            data = self.map_fn(text_data)
            data["image_data"] = img_data
            data["image_data_loc"] = img_data_loc
            return data

        return text_data, img_data, img_data_loc


class MLlamaHDF5DatasetConfig(HDF5DatasetConfig):
    # Image preprocessing parameters for MLlama Vision model
    do_convert_rgb: bool = True  # Convert image to RGB
    do_resize: bool = True  # Resize image
    size: Dict[str, int] = {"height": 448, "width": 448}  # tile size
    resample: int = Image.BICUBIC  # bicubic from PIL.Image
    do_rescale: bool = True  # Rescale image
    rescale_factor: float = 1 / 255  # Rescale factor
    do_normalize: bool = True  # Normalize image
    image_mean: Optional[Union[float, List[float]]] = None  # Image mean
    image_std: Optional[Union[float, List[float]]] = None  # Image std
    do_pad: bool = True  # Pad image
    max_image_tiles: int = (
        4  # Maximum number of tiles an image is to be split into
    )
    img_data_dir: ValidatedPath = (
        ...
    )  # Path to the directory containing the images

    def post_init(self, context):
        super().post_init(context)
        if model_config := context.get("model", {}).get("config"):
            if hasattr(model_config, "image_model"):
                self.size = {
                    "height": model_config.image_model.image_size[0],
                    "width": model_config.image_model.image_size[1],
                }
                self.max_image_tiles = model_config.image_model.max_num_tiles


class MLlamaHDF5Dataset(HDF5Dataset):
    def __init__(self, config: MLlamaHDF5DatasetConfig):
        super().__init__(config)

        self.img_data_dir = config.img_data_dir
        self.image_processor = MllamaImageProcessor(
            do_convert_rgb=config.do_convert_rgb,
            do_resize=config.do_resize,
            size=config.size,
            resample=config.resample,
            do_rescale=config.do_rescale,
            rescale_factor=config.rescale_factor,
            do_normalize=config.do_normalize,
            image_mean=config.image_mean,
            image_std=config.image_std,
            do_pad=config.do_pad,
            max_image_tiles=config.max_image_tiles,
        )

        self._state_dict_ignore_keys.add("image_processor")
        self._load_state_ignore_keys.add("img_data_dir")
        self.padding_token_id = config.pad_token_id
        self.max_num_images = config.max_num_images

    def generate_sample(self):
        text_sample = super().generate_sample()
        img_sample = self.image_processor.generate_sample()
        return text_sample, img_sample

    def preprocess_img(self, path_list):
        img_list = []
        for path in path_list:
            path = path.decode("utf-8")
            if path != "None":
                image_path = os.path.join(self.img_data_dir, path)
                image = Image.open(image_path).convert("RGB")
            else:
                image = Image.new(
                    mode="RGB",
                    size=(
                        self.image_processor.size["width"],
                        self.image_processor.size["height"],
                    ),
                )
            t_img = self.image_processor.preprocess(images=[image])
            img_list.append(t_img)
        return img_list

    def _set_up_reader(self, data_dir, subset):
        if not isinstance(data_dir, list):
            data_dir = [data_dir]
        if self.use_worker_cache and cstorch.use_cs() and dist.is_streamer():
            data_dir = [create_worker_cache(d) for d in data_dir]

        reader = H5Reader(
            data_dirs=data_dir,
            extra_data_keys=["img_path", "img_data_loc", "has_img"],
            sequence_length=None,
            read_extra_token=True,
            data_subset=subset,
            sort=self.sort_files,
            use_vsl=self.use_vsl,
        )
        return reader

    def __getitem__(self, i):
        if i == pad_index:
            if not self.pad_last:
                raise RuntimeError(
                    "Unexpectedly encountered the pad index when pad_last was False"
                )
            text_data, img_data = self.generate_sample()
        else:
            data = self.reader[i]
            text_data, img_path, has_img_data = (
                data["data"],
                data["img_path"],
                data['has_img'],
            )
            img_data = self.preprocess_img(img_path)

        if self.map_fn is not None:
            data = self.map_fn(text_data)
            data['has_img'] = has_img_data
            # TODO: right now img_data is a list containing only data for 1 image -- need to update this for multiple images after clarifying the dataset and HDF5 data preparations
            data.update(img_data[0])
            return data

        return text_data, img_data

    def _create_attention_mask(self, input_ids):
        """
        Create padding attention mask based on input_ids.
        Uses 1 for positions to attend to, 0 for positions to mask out.
        """
        batch_size, seq_length = input_ids.size()

        # Create padding mask
        padding_mask = (input_ids != self.padding_token_id).float()
        padding_mask = padding_mask.view(batch_size, seq_length)

        return padding_mask

    def _create_efficient_batch(self, examples):
        batch_size = len(examples)
        # Pre-allocate tensors based on the first example's shapes
        first_example = examples[0]
        text_id_values = torch.empty(
            (batch_size, *first_example['input_ids'].shape), dtype=torch.long
        )
        pixel_values = torch.empty(
            (batch_size, *first_example['pixel_values'].shape[1:]),
            dtype=torch.float,
        )
        aspect_ratio_ids = torch.empty(
            (batch_size, *first_example['aspect_ratio_ids'].shape),
            dtype=torch.long,
        )
        aspect_ratio_mask = torch.empty(
            (batch_size, *first_example['aspect_ratio_mask'].shape),
            dtype=torch.float,
        )
        labels = torch.empty(
            (batch_size, *first_example['labels'].shape), dtype=torch.long
        )
        has_img = torch.empty((batch_size,), dtype=torch.long)

        # Fill pre-allocated tensors
        for i, example in enumerate(examples):
            text_id_values[i] = torch.from_numpy(example['input_ids'])
            pixel_values[i] = torch.from_numpy(
                example['pixel_values'].squeeze(0)
            )
            aspect_ratio_ids[i] = torch.from_numpy(example['aspect_ratio_ids'])
            aspect_ratio_mask[i] = torch.from_numpy(
                example['aspect_ratio_mask']
            )
            labels[i] = torch.from_numpy(example['labels'])
            has_img[i] = torch.from_numpy(numpy.array(example['has_img']))

        # Create causal attention mask
        attention_mask = self._create_attention_mask(text_id_values)

        # Create cross attention mask
        seq_length = text_id_values.shape[1]
        max_num_tiles = pixel_values.shape[-4]
        cross_attention_mask = torch.zeros(
            (batch_size, seq_length, self.max_num_images, max_num_tiles)
        )
        # TODO: update this for multiple images after clarifying the dataset and HDF5 data preparations
        cross_attention_mask[:, :, 0, :] = (
            1  # All tokens attend to all tiles of Image 0
        )

        # convert relevant tensors to int32
        text_id_values = text_id_values.to(torch.int32)
        aspect_ratio_ids = aspect_ratio_ids.to(torch.int32)
        aspect_ratio_mask = aspect_ratio_mask.to(torch.int32)
        attention_mask = attention_mask.to(torch.int32)
        cross_attention_mask = cross_attention_mask.to(torch.int32)
        labels = labels.to(torch.int32)

        return {
            'input_ids': text_id_values,
            'pixel_values': pixel_values,
            'aspect_ratio_ids': aspect_ratio_ids,
            'aspect_ratio_mask': aspect_ratio_mask,
            'attention_mask': attention_mask,
            'cross_attention_mask': cross_attention_mask,
            'labels': labels,
            'has_img': has_img,
        }

    def collate_fn(self, examples):
        batch = self._create_efficient_batch(examples)
        return batch


class DPRHDF5Dataset(HDF5Dataset):
    def _set_up_reader(self, data_dir, subset):
        if not isinstance(data_dir, list):
            data_dir = [data_dir]
        if self.use_worker_cache and cstorch.use_cs() and dist.is_streamer():
            data_dir = [create_worker_cache(d) for d in data_dir]

        reader = H5Reader(
            data_dirs=data_dir,
            extra_data_keys=["context_data"],
            sequence_length=self.max_sequence_length,
            read_extra_token=True,
            data_subset=subset,
            sort=self.sort_files,
            use_vsl=self.use_vsl,
        )
        return reader

    def __getitem__(self, i):
        if i == pad_index:
            if not self.pad_last:
                raise RuntimeError(
                    "Unexpectedly encountered the pad index when pad_last was False"
                )
            question_data, context_data = self.generate_sample()
        else:
            data = self.reader[i]
            question_data, context_data = data["data"], data["context_data"]

        if self.map_fn is not None:
            return self.map_fn(data)
        return {
            "questions_input_ids": question_data[0],
            "questions_attention_mask": question_data[1],
            "questions_token_type_ids": question_data[2],
            "ctx_input_ids": context_data[0],
            "ctx_attention_mask": context_data[1],
            "ctx_token_type_ids": context_data[2],
        }


class DPREmbedGenHDF5Dataset(HDF5Dataset):
    def _set_up_reader(self, data_dir, subset):
        if not isinstance(data_dir, list):
            data_dir = [data_dir]
        if self.use_worker_cache and cstorch.use_cs() and dist.is_streamer():
            data_dir = [create_worker_cache(d) for d in data_dir]

        reader = H5Reader(
            data_dirs=data_dir,
            extra_data_keys=["id"],
            sequence_length=self.max_sequence_length,
            read_extra_token=True,
            data_subset=subset,
            sort=self.sort_files,
            use_vsl=self.use_vsl,
        )
        return reader

    def __getitem__(self, i):
        if i == pad_index:
            if not self.pad_last:
                raise RuntimeError(
                    "Unexpectedly encountered the pad index when pad_last was False"
                )
            dummy_data = self.generate_sample()
            dummy_data.fill_(1)
            return dummy_data, -1
        else:
            data = self.reader[i]
            question_data, ids = data["data"], data["id"]
            return question_data, ids[0]
