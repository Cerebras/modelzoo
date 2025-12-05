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

import pandas as pd
import torch
from torchvision.io import ImageReadMode, read_image
from torchvision.utils import draw_bounding_boxes, save_image

from cerebras.modelzoo.data.multimodal.datasets import BaseDataset
from cerebras.modelzoo.data.multimodal.datasets.features import (
    Bbox,
    ImageLabels,
    ObjectDetectionFeaturesDict,
)

_URLS = {
    "webpage": "https://storage.googleapis.com/openimages/web/download_v4.html",
    # Images: Refer to webpage: Install awscli, dowload using `aws s3 --no-sign-request cp <url> [target_dir]`
    "train_images": [
        f"s3://open-images-dataset/tar/train_{i}.tar.gz"
        for i in "0123456789abcdef"
    ],
    "test_images": [f"s3://open-images-dataset/tar/test.tar.gz"],
    "validation_images": [f"s3://open-images-dataset/tar/validation.tar.gz"],
    # Label info: Download using `wget -P <path_to_download_folder> <url>`
    "train_human_labels": "https://storage.googleapis.com/openimages/2018_04/train/train-annotations-human-imagelabels.csv",
    "train_machine_labels": "https://storage.googleapis.com/openimages/2018_04/train/train-annotations-machine-imagelabels.csv",
    "test_human_labels": "https://storage.googleapis.com/openimages/2018_04/test/test-annotations-human-imagelabels.csv",
    "test_machine_labels": "https://storage.googleapis.com/openimages/2018_04/test/test-annotations-machine-imagelabels.csv",
    "validation_human_labels": "https://storage.googleapis.com/openimages/2018_04/validation/validation-annotations-human-imagelabels.csv",
    "validation_machine_labels": "https://storage.googleapis.com/openimages/2018_04/validation/validation-annotations-machine-imagelabels.csv",
    "train-annotations-bbox": "https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv",
    "test-annotations-bbox": "https://storage.googleapis.com/openimages/2018_04/test/test-annotations-bbox.csv",
    "validation-annotations-bbox": "https://storage.googleapis.com/openimages/2018_04/validation/validation-annotations-bbox.csv",
    "class-image-labels": "https://storage.googleapis.com/openimages/2018_04/class-descriptions.csv",
    "trainable-image-labels": "https://storage.googleapis.com/openimages/2018_04/classes-trainable.txt",
    # Trainable classes are those with at least 100 positive human-verifications in the V4 training set
    "bbox-labels": "https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv",
}


class OpenImagesv4(BaseDataset):
    """
    This class builds an OpenImages dataset based on version v4 and all metadata info
    """

    def __init__(self, data_dir, split, *args):

        if split not in ["train", "validation", "test"]:
            raise ValueError(
                f"Split={split} invalid. Accepted values are one of ('train', 'validation', 'test')"
            )
        self.split = split
        self.data_dir = data_dir
        self.images_dir = os.path.join(self.data_dir, self.split)

        # Read relevant files for data
        self._csv_image_labels_desc = os.path.join(
            self.data_dir, f"class-descriptions.csv"
        )
        self._csv_trainable_labels_desc = os.path.join(
            self.data_dir, f"classes-trainable.txt"
        )
        self._csv_bbox_labels_desc = os.path.join(
            self.data_dir, f"class-descriptions-boxable.csv"
        )
        self._csv_image_human_labels = os.path.join(
            self.data_dir, f"{self.split}-annotations-human-imagelabels.csv"
        )
        self._csv_image_machine_labels = os.path.join(
            self.data_dir, f"{self.split}-annotations-machine-imagelabels.csv"
        )
        self._csv_bbox_anns = os.path.join(
            self.data_dir, f"{self.split}-annotations-bbox.csv"
        )

        # Get label and bbox data
        self.label_data = self._get_label_data()
        # Dict is faster than dataframe for label level lookups
        self.label_data_dict = self.label_data.to_dict(orient="index")
        self.labelid_str_dict = {
            v["ClassIntID"]: v["ClassLabel"]
            for v in self.label_data_dict.values()
        }
        self._label_id_range = [
            min(self.label_data["ClassIntID"]),
            max(self.label_data["ClassIntID"]) + 1,
        ]
        self._num_labels = len(self.label_data)

        self.bbox_data = self._get_bbox_data()
        self.imgid_bbox = sorted(
            list(self.bbox_data.groups.keys())
        )  # ImageIDs with bboxes
        (
            self.img_human_anns,
            self.img_machine_anns,
        ) = self._get_imagelabels_data()

        self.imgid_human_anns = sorted(list(self.img_human_anns.groups.keys()))
        self.imgid_machine_anns = sorted(
            list(self.img_machine_anns.groups.keys())
        )

    @property
    def label_id_range(self):
        # [start, end)
        return self._label_id_range

    @property
    def num_labels(self):
        # Return length of all Labels in the dataset
        return self._num_labels

    def index_to_sample_path(self, index):
        # Return path to imagepath based on index
        image_id = self.imgid_bbox[index]
        return os.path.join(self.images_dir, f"{image_id}.jpg")

    def sample_path_to_index(self, sample_path):
        # Return path to sample based on index
        image_id = os.path.basename(sample_path).split(".")[0]
        return self.imgid_bbox.index(image_id)

    def __getitem__(self, index):
        image_path = self.index_to_sample_path(index)
        image = read_image(image_path, ImageReadMode.RGB)
        image_id = os.path.basename(image_path).split(".jpg")[0]
        bbox_data = (
            self.bbox_data.get_group(image_id)
            if image_id in self.bbox_data.groups
            else []
        )
        img_human_anns = (
            self.img_human_anns.get_group(image_id)
            if image_id in self.img_human_anns.groups
            else []
        )
        img_machine_anns = (
            self.img_machine_anns.get_group(image_id)
            if image_id in self.img_machine_anns.groups
            else []
        )

        bboxes = self.process_bboxes(bbox_data)
        objects = self.process_image_labels(img_human_anns, img_machine_anns)

        data_dict = {
            "Image": image,
            "ImagePath": image_path,
            "ImageID": image_id,
            "Bboxes": bboxes,
            "Objects": objects,
        }
        features = ObjectDetectionFeaturesDict(**data_dict)
        return features

    def __repr__(self):
        s = f"OpenImagesv4(split={self.split}, data_dir={self.data_dir})"
        return s

    def __len__(self):
        return len(self.imgid_bbox)

    @staticmethod
    def display_sample(features_dict):
        """
        Plot and display features_dict
        """
        image = features_dict.Image  # tensor of shape (C, H, W)
        h, w = image.shape[1], image.shape[2]

        bboxes = [b.bbox_to_tensor(format="xyxy") for b in features_dict.Bboxes]
        bboxes = torch.stack(bboxes, dim=0)  # (num_boxes, 4)

        bboxes = bboxes * torch.tensor([w, h, w, h])
        label_str = [
            f"{b.ClassLabel}_{b.ClassIntID}" for b in features_dict.Bboxes
        ]
        result = draw_bounding_boxes(
            image, bboxes, label_str, colors=["blue"] * len(label_str), width=4
        )
        save_image(
            result.unsqueeze(0).to(torch.float32),
            f"{features_dict.ImageID}_bbox.jpg",
            nrow=1,
            normalize=True,
        )

    ## Helper classes below ##

    def _process_helper(self, df_rowtup):
        df_rowtup = df_rowtup._asdict()
        # Get Label info
        df_rowtup["ClassID"] = df_rowtup["LabelName"]
        cls_id = self.label_data_dict[df_rowtup["ClassID"]]
        for k in ("ClassLabel", "ClassIntID", "IsTrainable"):
            df_rowtup[k] = cls_id[k]
        # Pop unncessary values for easy Bbox creation
        for kp in ("Index", "LabelName", "ImageID"):
            df_rowtup.pop(kp)
        return df_rowtup

    def process_bboxes(self, bbox_data):
        bboxes = []
        if not bbox_data.empty:
            for r in bbox_data.itertuples():
                r_dict = self._process_helper(r)
                bboxes.append(Bbox(**r_dict))
        return bboxes

    def process_image_labels(self, img_human_anns, img_machine_anns):
        objects = []

        def process(input_data):
            data = []
            for r in input_data.itertuples():
                r_dict = self._process_helper(r)
                data.append(ImageLabels(**r_dict))
            return data

        if not img_human_anns.empty:
            objects.extend(process(img_human_anns))

        if not img_machine_anns.empty:
            objects.extend(process(img_machine_anns))

        return objects

    def _get_label_data(self):
        bbox_labels = pd.read_csv(
            self._csv_bbox_labels_desc,
            header=None,
            index_col=0,
            names=["ClassLabel"],
        )
        bbox_labels["ClassID"] = bbox_labels.index

        image_labels_train = pd.read_csv(
            self._csv_trainable_labels_desc, header=None, names=["ClassID"]
        )

        image_labels = pd.read_csv(
            self._csv_image_labels_desc,
            header=None,
            index_col=0,
            names=["ClassLabel"],
        )
        image_labels["ClassID"] = image_labels.index

        image_labels["IsTrainable"] = image_labels.ClassID.map(
            lambda row: row in image_labels_train.ClassID.values
        )
        image_labels["IsBboxLabel"] = image_labels.ClassID.map(
            lambda row: row in bbox_labels.ClassID.values
        )
        image_labels["ClassIntID"] = range(0, len(image_labels))

        return image_labels

    def _get_bbox_data(self):
        bbox_data = pd.read_csv(self._csv_bbox_anns)
        return bbox_data.groupby("ImageID")

    def _get_imagelabels_data(self):
        img_labels_human = pd.read_csv(self._csv_image_human_labels)
        img_labels_machine = pd.read_csv(self._csv_image_machine_labels)
        return (
            img_labels_human.groupby("ImageID"),
            img_labels_machine.groupby("ImageID"),
        )

    # TODO: Re-evaluate: do we really need these ?
    def _convert_helper(self, from_data, from_key, to_key):
        if not isinstance(from_data, list):
            from_data = [from_data]
        from_data = pd.DataFrame(from_data, columns=[from_key]).astype(
            self.label_data[from_key].dtype
        )
        to_data = from_data.merge(
            self.label_data, left_on=from_key, right_on=from_key, how="inner"
        )[to_key].to_list()
        return to_data

    def classid_to_label(self, class_id):
        # Complement of `self.label_to_classid`
        # class_id = [/m/011k07, /m/011k07] -> return class_label = [Tortoise, Tortoise]
        return self._convert_helper(
            class_id, from_key="ClassID", to_key="ClassLabel"
        )

    def label_to_classid(self, label):
        # Complement of `self.classid_to_label` (Case-sensitive)
        # class_label = [Tortoise, Tortoise] -> return class_id = [/m/011k07, /m/011k07]
        return self._convert_helper(
            label, from_key="ClassLabel", to_key="ClassID"
        )

    def classintid_to_label(self, class_int_id):
        # class_int_id= [45, 45] -> return class_label = [Tortoise, Tortoise]
        return self._convert_helper(
            class_int_id, from_key="ClassIntID", to_key="ClassLabel"
        )

    def label_to_classintid(self, label):
        # class_label = [Tortoise, Tortoise] -> return class_int_id=[45, 45]
        return self._convert_helper(
            label, from_key="ClassLabel", to_key="ClassIntID"
        )

    def classid_to_classintid(self, class_id):
        # class_id = [/m/011k07, /m/011k07] -> return class_int_id = [45, 45]
        return self._convert_helper(
            class_id, from_key="ClassID", to_key="ClassIntID"
        )

    def classintid_to_classid(self, class_int_id):
        # class_int_id = [45, 45] -> return class_id = [/m/011k07, /m/011k07]
        return self._convert_helper(
            class_int_id, from_key="ClassIntID", to_key="ClassID"
        )


if __name__ == "__main__":
    import random

    obj = OpenImagesv4(
        "/cb/cold/multimodal_datasets/open_images/v4", "validation"
    )

    print(obj)

    idx = random.randint(0, len(obj))
    features_dict = obj[idx]

    print(idx, features_dict, features_dict.ImageID)
    obj.display_sample(features_dict)
