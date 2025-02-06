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
from typing import Any, Literal, Optional

import numpy as np
from PIL import Image
from pydantic import Field
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.vision import VisionDataset

from cerebras.modelzoo.data.vision.classification.dataset_factory import (
    VisionClassificationProcessor,
    VisionClassificationProcessorConfig,
)


class SmallNORB(VisionDataset):
    """
    This database is intended for experiments in 3D object recognition from shape.
    It contains images of 50 toys belonging to 5 generic categories: four-legged
    animals, human figures, airplanes, trucks, and cars. The objects were imaged
    by two cameras under 6 lighting conditions, 9 elevations (30 to 70 degrees
    every 5 degrees), and 18 azimuths (0 to 340 every 20 degrees).

    The training set is composed of 5 instances of each category (instances 4,
    6, 7, 8 and 9), and the test set of the remaining 5 instances (instances 0,
    1, 2, 3, and 5).
    """

    _file_dict = {
        "train": {
            "dat": "smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat",
            "cat": "smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat",
            "info": "smallnorb-5x46789x9x18x6x2x96x96-training-info.mat",
        },
        "test": {
            "dat": "smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat",
            "cat": "smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat",
            "info": "smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat",
        },
    }

    def __init__(
        self,
        root,
        split="train",
        task=None,
        transform=None,
        target_transform=None,
    ):
        super().__init__(
            os.path.join(root, "smallnorb"),
            transform=transform,
            target_transform=target_transform,
        )
        self.split = verify_str_arg(split, "split", ("train", "test"))
        if not os.path.exists(self.root):
            raise RuntimeError(
                "Dataset not found. Download from "
                "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/"
            )
        path_dat = os.path.join(self.root, self._file_dict[split]["dat"])
        path_cat = os.path.join(self.root, self._file_dict[split]["cat"])
        path_info = os.path.join(self.root, self._file_dict[split]["info"])

        dat_arr, cat_arr, info_arr = load_chunk(path_dat, path_cat, path_info)
        self.images = []
        self.targets = []
        for image, category, info_vec in zip(dat_arr, cat_arr, info_arr):
            self.images.append(
                {
                    "image": image[0],
                    "image2": image[1],
                }
            )
            record = {
                "label_category": category,
                "instance": info_vec[0],
                "label_elevation": info_vec[1],
                "label_azimuth": info_vec[2],
                "label_lighting": info_vec[3],
            }
            if task is None:
                self.targets.append(record)
            else:
                self.targets.append(record[task])

    def __getitem__(self, index):
        img = np.tile(self.images[index]["image"], (1, 1, 3))
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)


def load_chunk(dat_path, cat_path, info_path):
    dat_array = read_binary_matrix(dat_path)
    # Even if the image is grayscale, still need extra channel dimension to be
    # compatible with PIL.Image
    dat_array = np.expand_dims(dat_array, -1)
    cat_array = read_binary_matrix(cat_path)

    info_array = read_binary_matrix(info_path)
    info_array = np.copy(info_array)  # Make read-only buffer array writable.
    # Azimuth values are 0, 2, 4, .., 34. We divide by 2 to get proper labels.
    info_array[:, 2] = info_array[:, 2] / 2

    return dat_array, cat_array, info_array


def read_binary_matrix(filename):
    """
    Reads and returns binary formatted matrix stored in filename.
    The file format is described on the dataset page:
    https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/

    struct header {
        int magic; // 4 bytes
        int ndim; // 4 bytes, little endian
        int dim[3];
    };
    """
    with open(filename, "rb") as f:
        s = f.read()
        # Data is stored in little-endian byte order.
        int32_dtype = np.dtype("int32").newbyteorder("<")

        # The first 4 bytes contain a magic code that specifies the data type.
        magic = int(np.frombuffer(s, dtype=int32_dtype, count=1))

        if magic == 507333717:
            data_dtype = np.dtype("uint8")  # uint8 does not have a byte order.
        elif magic == 507333716:
            data_dtype = np.dtype("int32").newbyteorder("<")
        else:
            raise ValueError("Invalid magic value for data type!")

        # The second 4 bytes contain an int32 with the number of dimensions
        # of the stored array.
        ndim = int(np.frombuffer(s, dtype=int32_dtype, count=1, offset=4))

        # The next ndim x 4 bytes contain the shape of the array in int32.
        dims = np.frombuffer(s, dtype=int32_dtype, count=ndim, offset=8)

        # If the array has less than three dimensions, three int32 are still
        # used to save the shape info (remaining int32 are simply set to 1).
        # The shape info hence uses max(3, ndim) bytes.
        bytes_used_for_shape_info = max(3, ndim) * 4

        # The remaining bytes are the array.
        data = np.frombuffer(
            s, dtype=data_dtype, offset=8 + bytes_used_for_shape_info
        )

    return data.reshape(tuple(dims))


class SmallNORBProcessorConfig(VisionClassificationProcessorConfig):
    data_processor: Literal["SmallNORBProcessor"]

    use_worker_cache: bool = ...

    split: Literal["train", "test"] = "train"
    "Dataset split."

    num_classes: Optional[Any] = Field(None, deprecated=True)


class SmallNORBProcessor(VisionClassificationProcessor):
    def __init__(self, config: SmallNORBProcessorConfig):
        super().__init__(config)
        self.split = config.split
        self.shuffle = self.shuffle and (self.split == "train")
        self.num_classes = 5

    def create_dataset(self):
        use_training_transforms = self.split == "train"
        transform, target_transform = self.process_transform(
            use_training_transforms
        )
        dataset = SmallNORB(
            root=self.data_dir,
            split=self.split,
            transform=transform,
            target_transform=target_transform,
        )
        return dataset
