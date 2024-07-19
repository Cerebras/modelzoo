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

from dataclasses import dataclass
from typing import List

import torch


# TODO: edit dataclass attrs
@dataclass
class Bbox:
    """
    Source: indicates how the box was made:
        xclick: are manually drawn boxes using the method presented in [1], were the annotators click on the four extreme points of the object. In V6 we release the actual 4 extreme points for all xclick boxes in train (13M), see below.
        activemil: are boxes produced using an enhanced version of the method [2]. These are human verified to be accurate at IoU>0.7.
    LabelName: the MID of the object class this box belongs to.
    Confidence: a dummy value, always 1.
    XMin, XMax, YMin, YMax: coordinates of the box, in normalized image coordinates. XMin is in [0,1], where 0 is the leftmost pixel, and 1 is the rightmost pixel in the image. Y coordinates go from the top pixel (0) to the bottom pixel (1).
    IsOccluded: Indicates that the object is occluded by another object in the image.
    IsTruncated: Indicates that the object extends beyond the boundary of the image.
    IsGroupOf: Indicates that the box spans a group of objects (e.g., a bed of flowers or a crowd of people). We asked annotators to use this tag for cases with more than 5 instances which are heavily occluding each other and are physically touching.
    IsDepiction: Indicates that the object is a depiction (e.g., a cartoon or drawing of the object, not a real physical instance).
    IsInside: Indicates a picture taken from the inside of the object (e.g., a car interior or inside of a building).
    For each of them, value 1 indicates present, 0 not present, and -1 unknown.
    """

    XMin: float
    YMin: float
    XMax: float
    YMax: float
    ClassLabel: str
    ClassIntID: int
    ClassID: str = None
    IsOccluded: bool = None
    IsTruncated: bool = None
    IsGroupOf: bool = None
    IsDepiction: bool = None
    IsInside: bool = None
    IsTrainable: bool = None
    Source: str = None
    Confidence: int = None  # dummy value of 1 in OpenImages

    def bbox_to_tensor(self, format="yxyx"):
        if format == "yxyx":
            out = torch.tensor(
                [self.YMin, self.XMin, self.YMax, self.XMax],
                dtype=torch.float32,
            )
        elif format == "xyxy":
            out = torch.tensor(
                [self.XMin, self.YMin, self.XMax, self.YMax],
                dtype=torch.float32,
            )
        else:
            raise ValueError(
                f"Unsupported format: {format}, supported values are ('xyxy', 'yxyx')"
            )
        return out

    def labelID_to_tensor(self):
        return torch.tensor(self.ClassIntID, dtype=torch.int64)


@dataclass
class ImageLabels:
    ClassLabel: str
    ClassIntID: str
    ClassID: str = None
    Confidence: float = None
    IsTrainable: bool = None
    Source: str = None


@dataclass
class ObjectDetectionFeaturesDict:
    """
    ImageID: Name of image
    """

    ImagePath: str
    ImageID: str
    Image: torch.Tensor = None  # (C, H, W)
    Bboxes: List[Bbox] = None
    Objects: List[ImageLabels] = None

    def compare(self, other):
        for k in ["ImagePath", "ImageID"]:
            if getattr(self, k) != getattr(other, k):
                return False

        if self.Image is None and other.Image is None:
            pass
        elif isinstance(self.Image, torch.Tensor) and isinstance(
            other.Image, torch.Tensor
        ):
            if not torch.all(torch.eq(self.Image, other.Image)):
                return False
        else:
            return False

        for k in ["Bboxes", "Objects"]:
            obj_1 = getattr(self, k)
            obj_2 = getattr(other, k)
            if obj_1 is None and obj_2 is None:
                pass
            elif isinstance(obj_1, list) and isinstance(obj_2, list):
                obj_1 = sorted(obj_1, key=lambda x: x.ClassIntID)
                obj_2 = sorted(obj_2, key=lambda x: x.ClassIntID)
                if obj_1 != obj_2:
                    return False
            else:
                return False

        return True

    def __eq__(self, other):
        return self.compare(other)


@dataclass
class VQAAnswer:
    answer_id: int
    answer: str
    answer_confidence: str
    answer_language: str


@dataclass
class VQAQuestion:
    question_id: int
    question: str
    question_language: str


@dataclass
class VQAFeaturesDict:
    """
    ImageID: Name of image
    """

    image_path: str
    image_id: str
    question: VQAQuestion
    answers: List[VQAAnswer]
    multiple_choice_answer: str  # most frequent ground-truth answer.
    multiple_choice_answer_language: str
    answer_type: str = None
    image: torch.Tensor = None  # (C, H, W)

    def __repr__(self):

        s = (
            f"VQAFeaturesDict.image_id: {self.image_id}, \n"
            + f"VQAFeaturesDict.image_path: {self.image_path}, \n\n"
        )
        s += repr(self.question) + f"\n\n"

        for a in self.answers:
            s += repr(a) + "\n"

        s += f"VQAFeaturesDict.multiple_choice_answer: {self.multiple_choice_answer}, \n"
        s += f"VQAFeaturesDict.answer_type: {self.answer_type}, \n"

        return s
