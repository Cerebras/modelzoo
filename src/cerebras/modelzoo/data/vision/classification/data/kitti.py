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

import collections
import os

import numpy as np
import torchvision

from cerebras.modelzoo.data.vision.classification.dataset_factory import (
    Processor,
    VisionSubset,
)


def _closest_vehicle_distance_pp(target):
    """Predict the distance to the closest vehicle"""
    # Location feature contains (x, y, z) in meters w.r.t. the camera.
    vehicles = [
        i
        for i in range(len(target))
        if target[i]["type"] in ["Car", "Van", "Truck"]
    ]
    vehicle_z = [target[idx]["location"][2] for idx in vehicles]
    vehicle_z.append(1000.0)
    dist = min(vehicle_z)

    # Results in a uniform distribution over three distances, plus one class for
    # "no vehicle".
    thrs = np.array([-100.0, 8.0, 20.0, 999.0])
    label = np.max(np.where((thrs - dist) < 0))
    return label


class KITTIProcessor(Processor):
    _TASK_DICT = {
        "closest_vehicle_distance": {
            "preprocess_fn": _closest_vehicle_distance_pp,
            "num_classes": 4,
        },
    }

    # The percentage of trainset videos to put into validation and test sets.
    # The released test images do not have labels.
    _VALIDATION_SPLIT_PERCENT_VIDEOS = 10
    _TEST_SPLIT_PERCENT_VIDEOS = 10

    def __init__(self, params):
        super().__init__(params)
        self.allowable_split = ["train", "test"]
        self.allowable_task = self._TASK_DICT.keys()

    def create_dataset(self, use_training_transforms=True, split="train"):
        self.check_split_valid(split)
        transform, target_transform = self.process_transform(
            use_training_transforms
        )
        dataset = torchvision.datasets.Kitti(
            root=self.data_dir,
            train=True if split == "train" else False,
            transform=transform,
            target_transform=target_transform,
            download=False,
        )
        return dataset

    def create_vtab_dataset(
        self, task="closest_vehicle_distance", use_1k_sample=True, seed=42
    ):
        if task not in self.allowable_task:
            raise ValueError(
                f"Task {task} is not supported, choose from "
                f"{self.allowable_task} instead"
            )

        train_transform, train_tgt_transform = self.process_transform(
            use_training_transforms=True
        )
        eval_transform, eval_tgt_transform = self.process_transform(
            use_training_transforms=False
        )

        tgt_transform = self._TASK_DICT[task]["preprocess_fn"]

        train_target_transform = torchvision.tranforms.Compose(
            [tgt_transform, train_tgt_transform]
        )
        eval_target_transform = torchvision.tranforms.Compose(
            [tgt_transform, eval_tgt_transform]
        )
        dataset = torchvision.datasets.Kitti(
            root=self.data_dir,
            train=True,
            transform=None,
            target_transform=None,
            download=False,
        )

        train_idx, val_idx, test_idx = self._build_vtab_splits()

        if use_1k_sample:
            rng = np.random.default_rng(seed)
            rng.shuffle(train_idx)
            rng.shuffle(val_idx)
            train_set = VisionSubset(dataset, train_idx[:800])
            val_set = VisionSubset(dataset, val_idx[:200])
        else:
            train_set = VisionSubset(dataset, train_idx)
            val_set = VisionSubset(dataset, val_idx)

        test_set = VisionSubset(dataset, test_idx)

        train_set.set_transforms(
            transform=train_transform, target_transform=train_target_transform
        )
        val_set.set_transforms(
            transform=eval_transform, target_transform=eval_target_transform
        )
        test_set.set_transforms(
            transform=eval_transform, target_transform=eval_target_transform
        )

        return train_set, val_set, test_set

    def _build_vtab_splits(self):
        """
        Splits the training data into train/val/test by video. This ensures that
        images from the same video do not traverse the splits.
        """
        mapping_line_ids = None
        mapping_lines = None
        with open(
            os.path.join(self.data_dir, "Kitti", "mapping", "train_rand.txt"),
            "r",
        ) as f:
            # convert 1-based line index to 0-based
            mapping_line_ids = [
                int(line.strip()) - 1 for line in f.readline().split(",")
            ]
        with open(
            os.path.join(
                self.data_dir, "Kitti", "mapping", "train_mapping.txt"
            ),
            "r",
        ) as f:
            mapping_lines = f.readlines()

        assert (
            mapping_line_ids
        ), "train_rand.txt is empty! Cannot assign index to image."
        assert (
            mapping_lines
        ), "train_mapping.txt is empty! Cannot map index to raw dataset file."

        video_to_image = collections.defaultdict(list)
        for image_id, mapping_lineid in enumerate(mapping_line_ids):
            line = mapping_lines[mapping_lineid]
            video_id = line.split(" ")[1]
            video_to_image[video_id].append(image_id)

        # sets numpy random state
        numpy_original_state = np.random.get_state()
        np.random.seed(seed=123)

        # max 1 for testing
        num_test_videos = max(
            1, self._TEST_SPLIT_PERCENT_VIDEOS * len(video_to_image) // 100
        )
        num_val_videos = max(
            1,
            self._VALIDATION_SPLIT_PERCENT_VIDEOS * len(video_to_image) // 100,
        )
        test_videos = set(
            np.random.choice(
                sorted(list(video_to_image.keys())),
                num_test_videos,
                replace=False,
            )
        )
        val_videos = set(
            np.random.choice(
                sorted(list(set(video_to_image.keys()) - set(test_videos))),
                num_val_videos,
                replace=False,
            )
        )
        train_img, val_img, test_img = [], [], []
        for k, v in video_to_image.items():
            if k in test_videos:
                test_img.extend(v)
            elif k in val_videos:
                val_img.extend(v)
            else:
                train_img.extend(v)

        # reset numpy random state
        np.random.set_state(numpy_original_state)
        return train_img, val_img, test_img
