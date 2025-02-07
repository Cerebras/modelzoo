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
import glob
import io
import json
import logging
import os
import random
import warnings

import pandas as pd
from PIL import Image


def parse_arguments():
    parser = argparse.ArgumentParser("Pre-process LLaVa datasets")

    subparsers = parser.add_subparsers(dest="dataset")

    # subparser for ai2d datset
    preprocess_ai2d = subparsers.add_parser(
        "ai2d", help="Pre-process AI2D dataset"
    )
    preprocess_ai2d.add_argument(
        "--question_dir",
        type=str,
        required=True,
        help="Path to the AI2D question directory, which contains json files describing the question and answer corresponding to an image.",
    )
    preprocess_ai2d.add_argument(
        "--output_jsonl_dir",
        type=str,
        required=True,
        help="Folder to write the AI2D output jsonl files, which is in LLaVa format describing the image and associated question and answer.",
    )

    # subparser for arxivcap dataset
    preprocess_arxivcap = subparsers.add_parser(
        "arxivcap", help="Pre-process ArxivCAP dataset"
    )
    preprocess_arxivcap.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory of ArxivCAP dataset parquet files.",
    )
    preprocess_arxivcap.add_argument(
        "--output_jsonl_dir",
        type=str,
        required=True,
        help="Output directory of ArxivCAP processed jsonl files with LLaVa jsonl format.",
    )
    preprocess_arxivcap.add_argument(
        "--output_parquet_dir",
        type=str,
        required=True,
        help="Output directory of ArxivCAP processed parquet files.",
    )
    preprocess_arxivcap.add_argument(
        "--parquet_range",
        type=int,
        required=True,
        nargs="+",
        help="Range of ArxivCAP parquet files to be selected.",
    )
    preprocess_arxivcap.add_argument(
        "--output_image_dir",
        type=str,
        required=True,
        help="Directory of ArxivCAP image files.",
    )
    preprocess_arxivcap.add_argument(
        "--image_prefix",
        type=str,
        required=True,
        help="Relative path prefix for ArxivCAP image files.",
    )

    # subparser for arxivqa
    preprocess_arxivqa = subparsers.add_parser(
        "arxivqa", help="Pre-process ArxivQA dataset"
    )
    preprocess_arxivqa.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the ArxivQA question file, which contains the question and answer corresponding to an image.",
    )
    preprocess_arxivqa.add_argument(
        "--output_jsonl_dir",
        type=str,
        required=True,
        help="Folder to write the ArxivQA output jsonl files, which is in LLaVa format describing the image and associated question and answer.",
    )

    # subparser for chartqa
    preprocess_chartqa = subparsers.add_parser(
        "chartqa", help="Pre-process ChartQA dataset"
    )
    preprocess_chartqa.add_argument(
        "--dataset_folder",
        type=str,
        required=True,
        help="Path to the ChartQA dataset folder with the data split folders.",
    )

    # subparser for sp_docvqa
    preprocess_sp_docvqa = subparsers.add_parser(
        "sp_docvqa", help="Pre-process SP-DocVQA dataset"
    )
    preprocess_sp_docvqa.add_argument(
        "--dataset_folder",
        type=str,
        required=True,
        help="Path to the SP-DocVQA dataset folder with the data files.",
    )

    # subparser for infographics_docvqa
    preprocess_infographics_docvqa = subparsers.add_parser(
        "infographics_docvqa", help="Pre-process Infographics-DocVQA dataset"
    )
    preprocess_infographics_docvqa.add_argument(
        "--dataset_folder",
        type=str,
        required=True,
        help="Path to the Inforgraphics-DocVQA dataset folder with the data files.",
    )

    # subparser for dvqa
    preprocess_dvqa = subparsers.add_parser(
        "dvqa", help="Pre-process DVQA dataset"
    )
    preprocess_dvqa.add_argument(
        "--dataset_folder",
        type=str,
        required=True,
        help="Path to the DVQA dataset folder with the data files.",
    )

    # subparser for synthdog_en
    preprocess_synthdog_en = subparsers.add_parser(
        "synthdog_en", help="Pre-process Synthdog_EN dataset"
    )
    preprocess_synthdog_en.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory of Synthdog-EN dataset parquet files.",
    )
    preprocess_synthdog_en.add_argument(
        "--output_jsonl_dir",
        type=str,
        required=True,
        help="Output directory of Synthdog-EN processed json files with LLaVa jsonl format.",
    )

    preprocess_synthdog_en.add_argument(
        "--output_parquet_dir",
        type=str,
        required=True,
        help="Output directory of Synthdog-EN processed parquet files.",
    )
    preprocess_synthdog_en.add_argument(
        "--parquet_range",
        type=int,
        required=True,
        nargs="+",
        help="Range of Synthdog-EN parquet files to be selected.",
    )
    preprocess_synthdog_en.add_argument(
        "--output_image_dir",
        type=str,
        required=True,
        help="Directory of Synthdog-EN image files.",
    )
    preprocess_synthdog_en.add_argument(
        "--image_prefix",
        type=str,
        required=True,
        help="Relative path prefix for Synthdog-EN image files.",
    )

    # subparser for simply converting from json to jsonl
    preprocess_json_to_jsonl = subparsers.add_parser(
        "convert_json2jsonl", help="Pre-process json files to jsonl files"
    )
    preprocess_json_to_jsonl.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to a folder of json files that need to be converted to jsonl format.",
    )

    # returned parsed arguments
    args = parser.parse_args()
    return args


def convert_json_to_jsonl(new_data):
    output_folder = f"{input_folder}_to_jsonl"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    json_files = list(glob.glob(os.path.join(input_folder, "*.json")))

    for jfile in json_files:
        out_file = os.path.join(
            output_folder, os.path.basename(jfile).replace("json", "jsonl")
        )
        logging.info(f"Processing {jfile} -> {out_file}")

        with open(jfile, "r") as fh:
            data = json.load(fh)

        # Convert and save to JSONL
        with open(out_file, "w") as jsonl_file:
            for entry in data:
                jsonl_file.write(json.dumps(entry) + "\n")

    logging.info(f"--- jsonl files saved at {output_folder} ---")


def process_ai2d(args):
    question_dir = args.question_dir
    output_jsonl_dir = args.output_jsonl_dir

    if not os.path.exists(output_jsonl_dir):
        os.makedirs(output_jsonl_dir, exist_ok=False)

    input_file_list = os.listdir(question_dir)

    label_list = ["a", "b", "c", "d"]

    def process_options(options):
        ret_options = []
        for idx in range(len(options)):
            ret_options.append(f"{label_list[idx]}) {options[idx]}")
        return ret_options

    def get_user_string(question, options):
        option_str = " ".join(options)
        return "<image>\n" + question + " " + option_str

    new_data = []
    for input_file in input_file_list:
        filename = os.path.join(question_dir, input_file)
        with open(filename, "r") as json_file:
            data = json.load(json_file)
        image_name = f"ai2d/images/{data['imageName']}"
        for quest in data["questions"].keys():
            options = process_options(data["questions"][quest]["answerTexts"])
            new_d = {
                "id": data["questions"][quest]["questionId"],
                "image": image_name,
                "conversations": [
                    {
                        "from": "human",
                        "value": get_user_string(quest, options),
                    },
                    {
                        "from": "gpt",
                        "value": options[
                            data["questions"][quest]["correctAnswer"]
                        ],
                    },
                ],
            }
            new_data.append(new_d)

    out_file = os.path.join(output_jsonl_dir, "ai2d_llava.jsonl")
    with open(out_file, "w") as jsonl_file:
        for entry in new_data:
            jsonl_file.write(json.dumps(entry) + "\n")
    logging.info(f"--- jsonl files saved at {output_jsonl_dir} ---")


def process_arxivcap(args):
    # Only handling single-figure captioning for now
    question_dict = {
        "Single-Figure Captioning": "Create a caption for the provided figure.",
        "Multiple-Figure Captioning": "Create a caption for the provided figures.",
        "Title Generation": "According to the figures and captions, generate a title for this paper. Title:",
        "Contextualized Captioning": None,  # depends on the figure type
    }

    def preprocess_parquet_to_llava(
        in_filename,
        out_jsonl_filename,
        out_parquet_fname,
        image_foldername,
        img_relpath_prefix,
    ):
        logging.info(f"preprocessing: {in_filename}")

        data = pd.read_parquet(in_filename)

        # write id
        data["id"] = data.index
        data["id"] = data["id"].apply(lambda x: "{:07d}".format(x))

        def convert_to_llava(caption_images):
            img_with_subcaption, img_single = 0, 0
            llava_samples = []
            for caption_img in caption_images:
                image_path = caption_img["cil_pairs"][0]["image_file"]
                if len(caption_img["cil_pairs"]) == 1:
                    image_filename = os.path.join(
                        img_relpath_prefix, image_path
                    )
                    caption = caption_img["caption"]
                    out = {
                        "id": image_path.split("/")[1][: -len(".jpg")],
                        "image": image_filename,
                        "conversations": [
                            {"from": "human", "value": None},
                            {"from": "gpt", "value": None},
                        ],
                    }
                    conversations = out["conversations"]
                    question = question_dict["Single-Figure Captioning"]
                    conversations[0]["value"] = f"<image>\n{question}"
                    conversations[1]["value"] = caption
                    llava_samples.append(out)
                    img_single += 1
                else:
                    for subcaption in caption_img["cil_pairs"]:
                        img_with_subcaption += 1

                if not os.path.exists(
                    os.path.join(image_foldername, image_path.split("/")[0])
                ):
                    os.makedirs(
                        os.path.join(image_foldername, image_path.split("/")[0])
                    )
                for img in caption_img["cil_pairs"]:
                    image_name = os.path.join(
                        image_foldername, img["image_file"]
                    )
                    image = Image.open(io.BytesIO(img["image"]["bytes"]))
                    image.save(image_name)

            return llava_samples

        data["llava"] = data.apply(
            lambda x: convert_to_llava(x.caption_images), axis=1
        )

        logging.info(f"Writing preprocessed parquet")
        data.to_parquet(out_parquet_fname, compression=None)

        with open(out_jsonl_filename, "w") as jsonl_file:
            for entry in data["llava"].tolist():
                jsonl_file.write(json.dumps(entry) + "\n")

    input_dir = args.input_dir
    parquet_range = args.parquet_range

    all_files = glob.glob(os.path.join(input_dir, "*.parquet"))
    all_files = sorted(
        all_files, key=lambda x: int(os.path.basename(x).split("_")[2])
    )

    def file_filter(x, range):
        bname = os.path.basename(x)
        f_month = int(bname.split("_")[2])
        if range[0] <= f_month < range[1]:
            return True
        else:
            False

    select_files = list(
        filter(lambda x: file_filter(x, parquet_range), all_files)
    )

    logging.info("selected_files:", len(select_files))

    if not os.path.exists(args.output_jsonl_dir):
        os.makedirs(args.output_jsonl_dir)

    if not os.path.exists(args.output_parquet_dir):
        os.makedirs(args.output_parquet_dir)

    for file in select_files:
        logging.info(f"---------- Parsing file: {file} ----------")
        output_jsonl_fname = os.path.basename(file).replace(
            ".parquet", ".jsonl"
        )
        out_jsonl_filename = os.path.join(
            args.output_jsonl_dir, output_jsonl_fname
        )
        out_parquet_fname = os.path.join(
            args.output_parquet_dir,
            output_jsonl_fname.replace("jsonl", "parquet"),
        )

        logging.info(f"in_filename: {file}")
        logging.info(f"out_jsonl_filename: {out_jsonl_filename}")
        logging.info(f"out_parquet_filename: {out_parquet_fname}")
        logging.info(f"image_foldername: {args.output_image_dir}")
        logging.info(f"img_relpath_prefix: {args.image_prefix}")

        preprocess_parquet_to_llava(
            in_filename=file,
            out_jsonl_filename=out_jsonl_filename,
            out_parquet_fname=out_parquet_fname,
            image_foldername=args.output_image_dir,
            img_relpath_prefix=args.image_prefix,
        )

    logging.info(f"--- jsonl files saved at {args.output_jsonl_dir} ---")


def process_arxivqa(args):
    input_file = args.input_file
    output_jsonl_dir = args.output_jsonl_dir

    if not os.path.exists(output_jsonl_dir):
        os.makedirs(output_jsonl_dir, exist_ok=False)

    # Load your JSONL file
    with open(input_file, "r") as jsonl_file:
        json_list = list(jsonl_file)

    def get_user_string(question, options):
        option_str = " ".join(options)
        return "<image>\n" + question + " " + option_str

    label_dict = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6,
        "H": 7,
        "I": 8,
    }

    def get_gpt_string(options, label, rationale):
        # No response
        if label == "" and rationale == "":
            return None
        # Label of type "[xxxxxx]"
        elif not (label[0] in label_dict.keys()):
            return label + " " + rationale
        # Label of type "A"
        else:
            if label[0] in label_dict.keys():
                # Apparently there are labels that are beyond options...
                if label_dict[label[0]] >= len(options):
                    return None
                else:
                    label_str = options[label_dict[label[0]]]
                return label_str + " " + rationale
            else:
                warnings.warn(
                    "This sample's label is not part of the label_dict. Ignoring this sample."
                )

    """
    # Note: some options don"t have A/B/C/D and some options has format A) or A.
    # Labels may or may not contain full option string and are not consistent.
    # As a cleanup step, we will replace the label with the full text of the option,
    # regardless of the format for the options.
    """
    new_data = []
    for idx, d in enumerate(json_list):
        d = json.loads(d)
        new_d = {
            "id": d["id"],
            "image": f"ArxivQA/{d['image']}",
            "conversations": [
                {
                    "from": "human",
                    "value": get_user_string(d["question"], d["options"]),
                },
                {
                    "from": "gpt",
                    "value": get_gpt_string(
                        d["options"], d["label"], d["rationale"]
                    ),
                },
            ],
        }
        if new_d["conversations"][1]["value"] is not None:
            new_data.append(new_d)

    out_file = os.path.join(output_jsonl_dir, "arxivqa_llava.jsonl")
    with open(out_file, "w") as jsonl_file:
        for entry in new_data:
            jsonl_file.write(json.dumps(entry) + "\n")
    logging.info(f"--- jsonl files saved at {output_jsonl_dir} ---")


def process_chartqa(args):
    def generate(split, subset):
        input_file = f"{args.dataset_folder}/{split}/{split}_{subset}.json"
        output_file = f"{args.dataset_folder}/{split}/{split}_{subset}_llava_jsonl/{split}_{subset}_llava.jsonl"
        output_jsonl_dir = os.path.dirname(output_file)

        if not os.path.exists(output_jsonl_dir):
            os.makedirs(output_jsonl_dir, exist_ok=False)

        # Load your JSON file
        with open(input_file, "r") as json_file:
            data = json.load(json_file)

        new_data = []
        for idx, d in enumerate(data):
            new_d = {
                "id": idx,
                "image": f"ChartQA_Dataset/{split}/png/{d['imgname']}",
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{d['query']}",
                    },
                    {
                        "from": "gpt",
                        "value": d["label"],
                    },
                ],
            }
            new_data.append(new_d)

        with open(output_file, "w") as jsonl_file:
            for entry in new_data:
                jsonl_file.write(json.dumps(entry) + "\n")
        logging.info(f"--- jsonl files saved at {output_jsonl_dir} ---")

    for split in ["train", "val", "test"]:
        for subset in ["human", "augmented"]:
            generate(split, subset)


def process_sp_docvqa(args):
    def generate(split):
        input_file = f"{args.dataset_folder}/{split}.json"
        output_file = (
            f"{args.dataset_folder}/{split}_llava_jsonl/{split}_llava.jsonl"
        )
        output_jsonl_dir = os.path.dirname(output_file)

        if not os.path.exists(output_jsonl_dir):
            os.makedirs(output_jsonl_dir, exist_ok=False)

        new_data = []
        with open(input_file, "r") as json_file:
            data = json.load(json_file)["data"]

        for quest in data:
            image_name = quest["image"].split("/")[-1]
            new_d = {
                "id": quest["questionId"],
                "image": f"DocVQA/sp_docvqa/images/{image_name}",
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{quest['question']}",
                    },
                    {
                        "from": "gpt",
                        # only use the first answer
                        "value": quest["answers"][0],
                    },
                ],
            }
            new_data.append(new_d)

        with open(output_file, "w") as jsonl_file:
            for entry in new_data:
                jsonl_file.write(json.dumps(entry) + "\n")
        logging.info(f"--- jsonl files saved at {output_jsonl_dir} ---")

    for split in ["train_v1.0_withQT", "val_v1.0_withQT"]:
        generate(split)


def process_infographics_docvqa(args):
    def generate(split):
        input_file = f"{args.dataset_folder}/infographicsVQA_{split}.json"
        output_file = f"{args.dataset_folder}/infographicsVQA_{split}_llava_jsonl/infographicsVQA_{split}_llava.jsonl"
        output_jsonl_dir = os.path.dirname(output_file)

        if not os.path.exists(output_jsonl_dir):
            os.makedirs(output_jsonl_dir, exist_ok=False)

        new_data = []
        with open(input_file, "r") as json_file:
            data = json.load(json_file)["data"]

        for quest in data:
            new_d = {
                "id": quest["questionId"],
                "image": f"DocVQA/Infographicsvqa/images/{quest['image_local_name']}",
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{quest['question']}",
                    },
                    {
                        "from": "gpt",
                        # only use the first answer
                        "value": quest["answers"][0],
                    },
                ],
            }
            new_data.append(new_d)

        with open(output_file, "w") as jsonl_file:
            for entry in new_data:
                jsonl_file.write(json.dumps(entry) + "\n")
        logging.info(f"--- jsonl files saved at {output_jsonl_dir} ---")

    for split in ["train_v1.0", "val_v1.0_withQT"]:
        generate(split)


def process_dvqa(args):
    # "train_qa" #"val_easy_qa" #"val_hard_qa"

    subset = "train_qa"

    input_file = f"{args.dataset_folder}/{subset}.json"
    output_file = (
        f"{args.dataset_folder}/{subset}_llava_jsonl/{subset}_llava.jsonl"
    )
    output_jsonl_dir = os.path.dirname(output_file)

    if not os.path.exists(output_jsonl_dir):
        os.makedirs(output_jsonl_dir, exist_ok=False)

    # Load your JSON file
    with open(input_file, "r") as json_file:
        data = json.load(json_file)

    new_data = []
    for idx, d in enumerate(data):
        new_d = {
            "id": d["question_id"],
            "image": f"DVQA/images/{d['image']}",
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{d['question']}",
                },
                {
                    "from": "gpt",
                    "value": d["answer"],
                },
            ],
        }
        new_data.append(new_d)

    with open(output_file, "w") as jsonl_file:
        for entry in new_data:
            jsonl_file.write(json.dumps(entry) + "\n")
    logging.info(f"--- jsonl files saved at {output_jsonl_dir} ---")


def process_synthdog_en(args):
    question_list = [
        "Describe the image concisely.",
        "Provide a brief description of the given image.",
        "Offer a succinct explanation of the picture presented.",
        "Summarize the visual content of the image.",
        "Give a short and clear explanation of the subsequent image.",
        "Share a concise interpretation of the image provided.",
        "Present a compact description of the photo's key features.",
        "Relay a brief, clear account of the picture shown.",
        "Render a clear and concise summary of the photo.",
        "Write a terse but informative summary of the picture.",
        "Create a compact narrative representing the image presented.",
    ]

    def preprocess_parquet_to_llava(
        in_filename,
        out_jsonl_filename,
        out_parquet_fname,
        image_foldername,
        img_relpath_prefix,
    ):
        logging.info(f"preprocessing: {in_filename}")

        data = pd.read_parquet(in_filename)

        # write id
        data["id"] = data.index
        data["id"] = data["id"].apply(lambda x: "{:07d}".format(x))

        def convert_to_llava(id, ground_truth):
            out = {
                "id": id,
                "image": os.path.join(img_relpath_prefix, f"{id}.png"),
                "conversations": [
                    {"from": "human", "value": None},
                    {"from": "gpt", "value": None},
                ],
            }
            ground_truth = eval(ground_truth)
            conversations = out["conversations"]
            question_idx = random.randint(0, len(question_list) - 1)
            question = question_list[question_idx]
            conversations[0]["value"] = f"<image>\n{question}"
            conversations[1]["value"] = ground_truth["gt_parse"][
                "text_sequence"
            ]
            return out

        def save_image(id, image):
            image = Image.open(io.BytesIO(image["bytes"]))
            p = os.path.join(image_foldername, f"{id}.png")
            image.save(p)

        data["llava"] = data.apply(
            lambda x: convert_to_llava(x.id, x.ground_truth), axis=1
        )

        logging.info(f"Writing preprocessed parquet")
        data.to_parquet(out_parquet_fname, compression=None)

        with open(out_jsonl_filename, "w") as jsonl_file:
            for entry in data["llava"].tolist():
                jsonl_file.write(json.dumps(entry) + "\n")

        logging.info(f"Saving images now")
        data.apply(lambda x: save_image(x.id, x.image), axis=1)
        logging.info(f"DONE: saving images")

    input_dir = args.input_dir
    parquet_range = args.parquet_range

    all_files = glob.glob(os.path.join(input_dir, "*.parquet"))
    all_files = sorted(
        all_files, key=lambda x: int(os.path.basename(x).split("-")[1])
    )

    def file_filter(x, range):
        bname = os.path.basename(x)
        fnum = int(bname.split("-")[1])
        if range[0] <= fnum < range[1]:
            return True
        else:
            False

    select_files = list(
        filter(lambda x: file_filter(x, parquet_range), all_files)
    )

    logging.info(f"selected_files: {select_files}")

    if not os.path.exists(args.output_jsonl_dir):
        os.makedirs(args.output_jsonl_dir)

    if not os.path.exists(args.output_parquet_dir):
        os.makedirs(args.output_parquet_dir)

    for file in select_files:
        logging.info(f"---------- Parsing file: {file} ----------")
        output_jsonl_fname = os.path.basename(file).replace(".parquet", ".json")
        out_jsonl_filename = os.path.join(
            args.output_jsonl_dir, output_jsonl_fname
        )
        out_parquet_fname = os.path.join(
            args.output_parquet_dir,
            output_jsonl_fname.replace("json", "parquet"),
        )

        splits = os.path.basename(file).split("-")
        image_subdir = os.path.join(args.output_image_dir, splits[0], splits[1])

        if not os.path.exists(image_subdir):
            os.makedirs(image_subdir)
        assert args.image_prefix in image_subdir
        image_prefix = os.path.join(args.image_prefix, splits[0], splits[1])

        logging.info(f"in_filename: {file}")
        logging.info(f"out_jsonl_filename: {out_jsonl_filename}")
        logging.info(f"out_parquet_filename: {out_parquet_fname}")
        logging.info(f"image_foldername: {image_subdir}")
        logging.info(f"img_relpath_prefix: {image_prefix}")

        preprocess_parquet_to_llava(
            in_filename=file,
            out_jsonl_filename=out_jsonl_filename,
            out_parquet_fname=out_parquet_fname,
            image_foldername=image_subdir,
            img_relpath_prefix=image_prefix,
        )

    logging.info(f"--- jsonl files saved at {args.output_jsonl_dir} ---")


if __name__ == "__main__":
    args = parse_arguments()

    if args.dataset == "ai2d":
        process_ai2d(args)
    elif args.dataset == "arxivcap":
        process_arxivcap(args)
    elif args.dataset == "arxivqa":
        process_arxivqa(args)
    elif args.dataset == "chartqa":
        process_chartqa(args)
    elif args.dataset == "sp_docvqa":
        process_sp_docvqa(args)
    elif args.dataset == "infographics_docvqa":
        process_infographics_docvqa(args)
    elif args.dataset == "dvqa":
        process_dvqa(args)
    elif args.dataset == "synthdog_en":
        process_synthdog_en(args)
    elif args.dataset == "convert_json2jsonl":
        convert_json_to_jsonl(args.input_dir)
    else:
        raise ValueError(
            "Dataset currently not supported. Feel free to adapt codebase to include your dataset."
        )
