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

import numpy
import torch
from PIL import Image

import cerebras.pytorch as cstorch
import cerebras.pytorch.distributed as dist
from cerebras.modelzoo.common.input_utils import PaddingSample
from cerebras.modelzoo.data.vision.preprocessing import get_preprocess_transform
from cerebras.modelzoo.data.vision.utils import create_worker_cache
from cerebras.pytorch.utils.data.sampler import pad_index

from .readers import H5Reader, Mixture


class HDF5Dataset(torch.utils.data.Dataset):
    """
    Dynamically read samples from disk for using mapping paradigms.

    It supports two different data formats on disk. The first is data stored
    in an H5 file in the shape `(num_tokens,)`, i.e. a series of documents
    tokenized and concatenated together. We call this format the 'corpus' format
    The second format is H5 data of shape `(num_sequences, ...)`, i.e. data has
    already been tokenized and split into sequences. We call this format the
    'sample' format.

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
        params (dict): a dictionary containing the following fields:
            - "data_dir" (str or list[str]): the path to the HDF5 files.
                Exactly one of "data_dir" or "mixture" must be specified.
            - "batch_size" (int): batch size
            - "shuffle" (bool): whether or not to shuffle the dataset. Defaults
                to `False`
            - "shuffle_seed" (int): seed used for deterministic shuffling.
                Defaults to 0.
            - "use_worker_cache" (bool): whether or not to copy data to storage
                that is directly attached to each individual worker node.
                Useful when your network storage is unusually slow, but
                otherwise discouraged.
            - "max_sequence_length" (int): the sequence length of samples
                produced by the dataloader. When using the 'corpus' data format,
                the same preprocessed data will work with any max sequence
                length, so this may be set at runtime. When using the 'sample'
                format this must be set to `None`.
            - "data_subset" (str): an optional specification to only consider a
                subset of the full dataset, useful for sequence length
                scheduling and multi-epoch testing. Expected to be a comma
                separated list of ranges, e.g. '0.0-0.5' or '0.1-0.3,0.7-1.0'.
                Specifying '0.0-0.5' creates a dataset from the first half of
                the data on disk and disregards the second half.
            - "mixture" list[dict]: an optional specification of multiple
                datasets to mix over to create one single weighted combination.
                Each element must be a dictionary containing keys `data_dir`
                and `weight`. `data_dir` serves the same purpose as mentioned
                above. `weight` defines the probability with which this dataset
                should be sampled from. Weights are normalized to sum to 1.
                Optionally, the dictionary may also contain a `data_subset`
                field which functions the same as the `data_subset` argument
                above.
            - "drop_last" (bool): similar to the PyTorch drop_last setting
                except that samples that when set to `True`, samples that would
                have been dropped at the end of one epoch are yielded at the
                start of the next epoch so that there is no data loss. This is
                necessary for a data ordering that is independent of the
                distributed setup being used.
            - "num_samples" (int): the number of samples to shuffle over (if
                shuffling is enabled). In multi-epoch training, it is common to
                set this to the total number of samples that you plan to train
                on so that epochs are not sequential but instead shuffled
                together for potentially improved convergence.
            - "sort_files" (bool): whether or not the reader should sort the input
                files. This is included for backwards compatibility and should
                almost always be set to `True`.
            - "use_vsl" (bool): Flag to enable variable sequence length training.
                It requires the dataset to have two extra features: the
                `attention_span` of keys and the `position_ids` of tokens.
                Defaults to `False`.
            - "pad_last" (bool): Flag to enable padding of the last batch so
                that the last batch has the same batch size as the rest of the
                batches. Defaults to `False`.
    """

    def __init__(self, params):
        self.use_worker_cache = params.get("use_worker_cache", False)
        self.max_sequence_length = params.get("max_sequence_length", None)
        self.shuffle = params.get("shuffle", False)
        self.shuffle_seed = params.get("shuffle_seed", 0)
        self.data_dir = params.get("data_dir", None)
        self.mixture_params = params.get("mixture", None)
        self.batch_size = params["batch_size"]
        self.drop_last = params.get("drop_last", True)
        self.num_samples = params.get("num_samples", None)
        self.sort_files = params.get("sort_files", True)
        self.use_vsl = params.get("use_vsl", False)
        self.pad_last = params.get("pad_last", False)
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
            self.reader = self._set_up_reader(
                self.data_dir, params.get("data_subset", None)
            )
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
        params (dict): A dictionary containing parameters that `HDF5Dataset` accepts
            along with the following add-ons:
            - "data_dir" (str): the path to the directory containing the images.
            - "transforms" (list[dict]): a specification of the torchvision transforms.
    """

    def __init__(self, params):
        super().__init__(params)

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


class MultiModalHDF5Dataset(HDF5Dataset):
    """Dataset class to handle image preprocessing in multimodal datasets.

    This class is largely the same as the parent class `HDF5Dataset` except
    with added image loading and preprocessing.

    Args:
        params (dict): A dictionary containing parameters that `HDF5Dataset` accepts
            along with the following add-ons:
            - "img_data_dir" (str): the path to the directory containing the images.
            - "image_data_size" (list[int]): the final C x H x W shape of the image.
            - "transforms" (list[dict]): a specification of the torchvision transforms.
    """

    def __init__(self, params):
        super().__init__(params)

        self.img_data_dir = params["img_data_dir"]
        self.image_data_size = params["image_data_size"]  # (C, H, W)
        self.transforms = get_preprocess_transform(
            {
                "transforms": params["transforms"],
                "mixed_precision": params["mixed_precision"],
            }
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


### H5 format
# 1. Data: B x 7 x S -- original 6 + token_modality_idx
# 2. Img_path: list of strings
# 3. image_data_loc: B x 1 x I * num_patches


class MultimodalSimpleHDF5Dataset(MultiModalHDF5Dataset):
    """Dataset class to handle image preprocessing in multimodal datasets.

    This class is largely the same as the parent class `MultimodalHDF5Dataset` except
    with added support for multiple images and intermingling of text and images.

    Args:
        params (dict): A dictionary containing parameters that `HDF5Dataset` accepts
            along with the following add-ons:
            - "img_data_dir" (str): the path to the directory containing the images.
            - "image_data_size" (list[int]): the final C x H x W shape of the image.
            - "transforms" (list[dict]): a specification of the torchvision transforms.
    """

    def __init__(self, params):
        super().__init__(params)

        self.max_num_img = params["max_num_img"]
        self.num_patches = params["num_patches"]
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
