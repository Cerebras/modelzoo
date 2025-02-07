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

# __author__ = "aagrawal"
# __version__ = "0.9"
# Based on https://github.com/GT-Vision-Lab/VQA/blob/master/PythonHelperTools/vqaTools/vqa.py
# Interface for accessing the VQA dataset.

# This code is based on the code written by Tsung-Yi Lin for MSCOCO Python API available at the following link:
# (https://github.com/pdollar/coco/blob/master/PythonAPI/pycocotools/coco.py).

# The following functions are defined:
#  VQA        - VQA class that loads VQA annotation file and prepares data structures.
#  getQuesIds - Get question ids that satisfy given filter conditions.
#  getImgIds  - Get image ids that satisfy given filter conditions.
#  loadQA     - Load questions and answers with the specified question ids.
#  showQA     - Display the specified questions and answers.
#  loadRes    - Load result file and create result object.

# Help on each function can be accessed by: "help(COCO.function)"

url_questions = {
    "train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Train_mscoco.zip",
    "val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Val_mscoco.zip",
    "test": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Test_mscoco.zip",
}
url_annotations = {
    "train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Train_mscoco.zip",
    "val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Val_mscoco.zip",
}

filename_questions = {
    "train": "OpenEnded_mscoco_train2014_questions.json",
    "val": "OpenEnded_mscoco_val2014_questions.json",
    "test": "OpenEnded_mscoco_test2015_questions.json",
    "test-dev": "OpenEnded_mscoco_test-dev2015_questions.json",
}

filename_annotations = {
    "train": "mscoco_train2014_annotations.json",
    "val": "mscoco_val2014_annotations.json",
}

import datetime
import json
import os

import torch
from torchvision.io import ImageReadMode, read_image
from torchvision.utils import save_image

from cerebras.modelzoo.data.multimodal.datasets import BaseDataset
from cerebras.modelzoo.data.multimodal.datasets.features import (
    VQAAnswer,
    VQAFeaturesDict,
    VQAQuestion,
)


class VQAv2(BaseDataset):
    def __init__(self, data_dir, split, *args):
        """
        Constructor of VQA helper class for reading and visualizing questions and answers.
                Assumes data_dir structure as mentioned here: https://github.com/GT-Vision-Lab/VQA/tree/master#files
        :param data_dir (str): Parent directory which contains Images, Annotations and Questions
        :param split (str): Split of dataset. One of {"train", "validation", "test"}
        """
        self.check_if_exists(data_dir)
        self.data_dir = data_dir

        if split not in ["train", "validation", "test"]:
            raise ValueError(
                f"Split={split} invalid. Accepted values are one of (`train`, `validation`, `test`)"
            )
        self.split = split

        # get filepaths
        self._init_filepaths()

        # load dataset
        self.dataset = {}
        self.questions = {}
        self.qa = {}
        self.qqa = {}
        self.imgToQA = {}
        self.idxToqid = {}

        if os.path.exists(self.annotation_file) and os.path.exists(
            self.question_file
        ):
            print("loading VQA annotations and questions into memory...")
            time_t = datetime.datetime.utcnow()
            dataset = json.load(open(self.annotation_file, "r"))
            questions = json.load(open(self.question_file, "r"))
            print(datetime.datetime.utcnow() - time_t)
            self.dataset = dataset
            self.questions = questions
            self.createIndex()

    def _init_filepaths(self):
        self.subtype = {
            "train": "train2014",
            "validation": "val2014",
            "test": "test2015",
        }
        self.version_type = "v2_"
        self.tasktype = "OpenEnded"
        self.datatype = "mscoco"
        self.datasubtype = self.subtype[self.split]

        self.annotation_file = os.path.join(
            self.data_dir,
            "Annotations",
            f"{self.version_type}{self.datatype}_{self.datasubtype}_annotations.json",
        )
        self.check_if_exists(self.annotation_file)

        self.question_file = os.path.join(
            self.data_dir,
            "Questions",
            f"{self.version_type}{self.tasktype}_{self.datatype}_{self.datasubtype}_questions.json",
        )
        self.check_if_exists(self.question_file)
        self.img_dir = os.path.join(
            self.data_dir, "Images", self.datatype, self.datasubtype
        )
        self.check_if_exists(self.img_dir)
        self._img_filepath_str = os.path.join(self.img_dir, "COCO_{}_{}.jpg")

    def check_if_exists(self, filepath):
        if not os.path.exists(filepath):
            raise ValueError(f"File does not exist at {filepath}")

    def createIndex(self):
        # create index
        print("creating index...")
        imgToQA = {ann["image_id"]: [] for ann in self.dataset["annotations"]}
        qa = {ann["question_id"]: [] for ann in self.dataset["annotations"]}
        qqa = {ann["question_id"]: [] for ann in self.dataset["annotations"]}
        for ann in self.dataset["annotations"]:
            imgToQA[ann["image_id"]] += [ann]
            qa[ann["question_id"]] = ann
        for ques in self.questions["questions"]:
            qqa[ques["question_id"]] = ques

        idx_qid = {id: qa_id for id, qa_id in enumerate(sorted(qa.keys()))}

        # create class members
        self.qa = qa  # key: question_id, val: annotation for question_id. There is always a 1-1 mapping between question_id and annotation
        self.qqa = qqa  # key: question_id, val: question for question_id. There is always a 1-1 mapping between question_id and question.
        self.imgToQA = imgToQA  # key: imgid, val: annotation, single image_id can have multiple question answers
        self.idxToqid = idx_qid  # key: index, val: question_id. There is always a 1-1 mapping

        print("index created!")

    def info(self):
        """
        Print information about the VQA annotation file.
        :return:
        """
        for key, value in self.datset["info"].items():
            print(f"{key}: {value}") % (key, value)

    def getQuesIds(self, imgIds=[], quesTypes=[], ansTypes=[]):
        """
        Get question ids that satisfy given filter conditions. default skips that filter
        :param 	imgIds    (int array)   : get question ids for given imgs
                        quesTypes (str array)   : get question ids for given question types
                        ansTypes  (str array)   : get question ids for given answer types
        :return:    ids   (int array)   : integer array of question ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
        ansTypes = ansTypes if type(ansTypes) == list else [ansTypes]

        if len(imgIds) == len(quesTypes) == len(ansTypes) == 0:
            anns = self.dataset["annotations"]
        else:
            if not len(imgIds) == 0:
                anns = sum(
                    [
                        self.imgToQA[imgId]
                        for imgId in imgIds
                        if imgId in self.imgToQA
                    ],
                    [],
                )
            else:
                anns = self.dataset["annotations"]
            anns = (
                anns
                if len(quesTypes) == 0
                else [ann for ann in anns if ann["question_type"] in quesTypes]
            )
            anns = (
                anns
                if len(ansTypes) == 0
                else [ann for ann in anns if ann["answer_type"] in ansTypes]
            )
        ids = [ann["question_id"] for ann in anns]
        return ids

    def getImgIds(self, quesIds=[], quesTypes=[], ansTypes=[]):
        """
         Get image ids that satisfy given filter conditions. default skips that filter
         :param quesIds   (int array)   : get image ids for given question ids
        quesTypes (str array)   : get image ids for given question types
        ansTypes  (str array)   : get image ids for given answer types
         :return: ids     (int array)   : integer array of image ids
        """
        quesIds = quesIds if type(quesIds) == list else [quesIds]
        quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
        ansTypes = ansTypes if type(ansTypes) == list else [ansTypes]

        if len(quesIds) == len(quesTypes) == len(ansTypes) == 0:
            anns = self.dataset["annotations"]
        else:
            if not len(quesIds) == 0:
                anns = sum(
                    [
                        self.qa[quesId]
                        for quesId in quesIds
                        if quesId in self.qa
                    ],
                    [],
                )
            else:
                anns = self.dataset["annotations"]
            anns = (
                anns
                if len(quesTypes) == 0
                else [ann for ann in anns if ann["question_type"] in quesTypes]
            )
            anns = (
                anns
                if len(ansTypes) == 0
                else [ann for ann in anns if ann["answer_type"] in ansTypes]
            )
        ids = [ann["image_id"] for ann in anns]
        return ids

    def loadQA(self, ids=[]):
        """
        Load questions and answers with the specified question ids.
        :param ids (int array)       : integer ids specifying question ids
        :return: qa (object array)   : loaded qa objects
        """
        if type(ids) == list:
            return [self.qa[id] for id in ids]
        elif type(ids) == int:
            return [self.qa[ids]]

    def showQA(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        for ann in anns:
            quesId = ann["question_id"]
            print(f"Question: {self.qqa[quesId]['question']}")
            print(
                f"multiple_choice_answer: {self.qa[quesId]['multiple_choice_answer']}"
            )
            for ans in ann["answers"]:
                print(
                    f"Answer {ans['answer_id']}: {ans['answer']}, answer_confidence: {ans['answer_confidence']}"
                )

    # TODO: Aarti: Fix this fcn, it doesnt work currently. We"ll need this for VQA eval
    # def loadRes(self, resFile, quesFile):
    #     """
    # 	Load result file and return a result object.
    # 	:param   resFile (str)     : file name of result file
    # 	:return: res (obj)         : result api object
    # 	"""
    #
    #     res = VQA()
    #     res.questions = json.load(open(quesFile))
    #     res.dataset["info"] = copy.deepcopy(self.questions["info"])
    #     res.dataset["task_type"] = copy.deepcopy(self.questions["task_type"])
    #     res.dataset["data_type"] = copy.deepcopy(self.questions["data_type"])
    #     res.dataset["data_subtype"] = copy.deepcopy(
    #         self.questions["data_subtype"]
    #     )
    #     res.dataset["license"] = copy.deepcopy(self.questions["license"])

    #     print("Loading and preparing results...     ")
    #     time_t = datetime.datetime.utcnow()
    #     anns = json.load(open(resFile))
    #     assert type(anns) == list, "results is not an array of objects"
    #     annsQuesIds = [ann["question_id"] for ann in anns]
    #     assert set(annsQuesIds) == set(
    #         self.getQuesIds()
    #     ), "Results do not correspond to current VQA set. Either the results do not have predictions for all question ids in annotation file or there is atleast one question id that does not belong to the question ids in the annotation file."
    #     for ann in anns:
    #         quesId = ann["question_id"]
    #         if res.dataset["task_type"] == "Multiple Choice":
    #             assert (
    #                 ann["answer"] in self.qqa[quesId]["multiple_choices"]
    #             ), "predicted answer is not one of the multiple choices"
    #         qaAnn = self.qa[quesId]
    #         ann["image_id"] = qaAnn["image_id"]
    #         ann["question_type"] = qaAnn["question_type"]
    #         ann["answer_type"] = qaAnn["answer_type"]
    #     t = (datetime.datetime.utcnow() - time_t).total_seconds()
    #     print(f"DONE (t=t:.2fs)")

    #     res.dataset["annotations"] = anns
    #     res.createIndex()
    #     return res

    def __len__(self):
        return len(self.qa)

    def __getitem__(self, index):
        qid = self.idxToqid[index]

        question = self.qqa[qid]
        q_anns = self.qa[qid]

        features = self._create_features_dict(question, q_anns)
        return features

    def _create_features_dict(self, qa_dict, q_anns_dict):
        ques = VQAQuestion(
            **{
                "question_id": qa_dict["question_id"],
                "question": qa_dict["question"],
                "question_language": "EN",
            }
        )

        ans = []
        for a in q_anns_dict["answers"]:
            a["answer_language"] = "EN"
            ans.append(VQAAnswer(**a))

        img_fname = self._img_filepath_str.format(
            self.datasubtype, str(qa_dict["image_id"]).zfill(12)
        )
        img_path = os.path.join(self.img_dir, img_fname)
        self.check_if_exists(img_path)

        img = read_image(img_path, ImageReadMode.RGB)
        features = VQAFeaturesDict(
            image_path=img_path,
            image_id=q_anns_dict["image_id"],
            question=ques,
            answers=ans,
            multiple_choice_answer=q_anns_dict["multiple_choice_answer"],
            multiple_choice_answer_language="EN",
            answer_type=q_anns_dict["answer_type"],
            image=img,
        )

        return features

    def __repr__(self):
        s = f"VQA(data_dir={self.data_dir}, split={self.split})"
        return s

    @staticmethod
    def display_sample(features_dict):
        print(features_dict)
        if features_dict.image is None:
            I = read_image(features_dict.image_path, ImageReadMode.RGB)
        else:
            I = features_dict.image
        img_fname = os.path.basename(features_dict.image_path)
        save_image(
            I.unsqueeze(0).to(torch.float32),
            f"vqa_{img_fname}",
            nrow=1,
            normalize=True,
        )


if __name__ == "__main__":
    import random

    vqa_obj = VQAv2(
        data_dir="/cb/cold/multimodal_datasets/vqa_v2/VQA", split="validation"
    )
    print(vqa_obj)
    idx = random.randint(0, len(vqa_obj))
    features_dict = vqa_obj[idx]
    vqa_obj.display_sample(features_dict)
