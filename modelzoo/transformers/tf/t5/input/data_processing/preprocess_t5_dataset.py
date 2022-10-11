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
import json
import os

import sentencepiece as spm


def tokenize_text(text, sp_model):
    text = text.strip()
    tokens = sp_model.encode(text, out_type=str)
    return tokens


def tokenize_dataset(input_dir, output_dir, sp_model_path, input_file_name):
    sp_model = spm.SentencePieceProcessor()
    sp_model.load(sp_model_path)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if input_file_name is not None:
        file_names = [input_file_name]
    else:
        file_names = os.listdir(input_dir)

    for file_name in file_names:
        input_file = os.path.join(input_dir, file_name)
        output_file = os.path.join(output_dir, file_name)
        if output_file.endswith(".json"):
            output_file = output_file[: -len(".json")] + ".txt"

        with open(input_file, "r") as fin:
            with open(output_file, "w") as fout:
                for line in fin:
                    data = json.loads(line)
                    tokens = tokenize_text(data["text"], sp_model)
                    fout.write(" ".join(tokens))
                    fout.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        required=True,
        type=str,
        help="Path to the input directory with a raw C4 dataset. "
        "The downloaded C4 dataset should be stored in `tf.data.TextLineDataset` format.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Path to the output directory where to store tokenized C4 dataset.",
    )

    parser.add_argument(
        "--spiece_model",
        default="./spiece.model",
        type=str,
        help="Path to the pre-trained sentencepiece model for dataset tokenization.",
    )

    parser.add_argument(
        "--file_name",
        default=None,
        type=str,
        help="File to process. If not specified, all files in the directory will be processed.",
    )

    params = parser.parse_args()

    tokenize_dataset(
        params.input_dir,
        params.output_dir,
        params.spiece_model,
        params.file_name,
    )
