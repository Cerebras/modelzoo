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
import os

# isort: off
import sys

# isort: on
import subprocess
import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from cerebras.modelzoo.common.utils.utils import check_and_create_output_dirs


def parse_args():
    """Argparser definition for command line arguments from user.

    Returns:
        Argparse namespace object with command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download the raw Pile data and associated vocabulary for pre-processing."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Base directory where raw data is to be downloaded.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="pile",
        help=(
            "Sub-directory where raw data is to be downloaded."
            + " Defaults to `pile`."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Checks if a given split exists in remote location.",
    )
    return parser.parse_args()


def get_urls_from_split(split):
    """Get urls given split of dataset.

    Args:
        split (str): Split of dataset to get urls for.

    Returns:
        List of urls, containing jsonl.zst file names for downloading.
    """
    if split == "train":
        warnings.warn(
            message=(
                f"Starting a large download process for full training data."
                + f" This process takes time and needs a storage with"
                + f" at least 500GB space."
            ),
            category=UserWarning,
        )
        urls = [
            f"https://mystic.the-eye.eu/public/AI/pile/train/{i:02}.jsonl.zst"
            for i in range(30)
        ]
    elif split == "val":
        urls = ["https://mystic.the-eye.eu/public/AI/pile/val.jsonl.zst"]
    elif split == "test":
        urls = ["https://mystic.the-eye.eu/public/AI/pile/test.jsonl.zst"]

    return urls


def get_urls_for_tokenizer_files():
    """Get urls for downloading files for tokenization.

    Returns:
        A dictionary containing urls for original GPT2 tokenizaiton and GPT-NeoX
        tokenization schemes
    """
    return {
        "gpt2-vocab.bpe": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt",
        "gpt2-encoder.json": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json",
        "neox-20B-tokenizer.json": "https://mystic.the-eye.eu/public/AI/models/GPT-NeoX-20B/slim_weights/20B_tokenizer.json",
    }


def debug_or_download_individual_file(url, filepath, debug=False):
    """Download a single file from url to specified filepath.

    Args:
        url (str): Url to download the data from.
        filepath (str): Filename (with path) to download the data to.
        debug (bool): Check if remote file exists. Defaults to `False`.
    """

    if debug:
        # use --no-check-certificate as eye.ai throws the below error:
        # `cannot verify mystic.the-eye.eu's certificate, issued by ‘/C=US/O=Let's Encrypt/CN=R3’,
        #    Issued certificate has expired.`
        cmd = f"wget --no-check-certificate --spider {url}"
        subprocess.run(cmd.split(" "), check=True)
        return

    execute = False
    # check for each individual file, because the train split has 30
    # individual files and in a potential previous download attempt,
    # some files may not have downloaded to the specified path.
    if not os.path.isfile(filepath):
        execute = True
    elif os.stat(filepath).st_size == 0:
        # Previous attempt at downloading file failed, but wget stats
        # the file. Check if filesize is 0, if so, delete and execute the
        # download process again.
        execute = True
        print(f"Got empty file at {filepath}, deleting and downloading again.")
        cmd = f"rm -rf {filepath}"
        subprocess.run(cmd.split(" "), check=True)
    else:
        print(
            f"{os.path.basename(filepath)} exists at {os.path.dirname(filepath)}"
            + f", skipping download."
        )

    # use --no-check-certificate as eye.ai throws the below error:
    # `cannot verify mystic.the-eye.eu's certificate, issued by ‘/C=US/O=Let's Encrypt/CN=R3’,
    #    Issued certificate has expired.`
    if execute:
        cmd = f"wget --no-check-certificate {url} -O {filepath}"
        subprocess.run(cmd.split(" "), check=True)


def download_pile(args, split):
    """Download The Pile dataset from eye.ai website.

    Args:
        args (argparse namespace): Arguments for downloading the dataset.
        split (str): The subset of the PILE dataset to download.
    """
    check_and_create_output_dirs(
        os.path.join(args.data_dir, args.name, split),
        filetype="jsonl.zst",
    )
    urls = get_urls_from_split(split)

    for url in urls:
        filepath = os.path.join(
            args.data_dir, args.name, split, os.path.basename(url)
        )
        debug_or_download_individual_file(url, filepath, args.debug)


def download_tokenizer_files(args):
    """Download files needed for tokenization for dataset creation.

    Args:
        args (argparse namespace): Arguments for downloading the tokenizer files.
    """
    check_and_create_output_dirs(
        os.path.join(args.data_dir, args.name, "vocab"),
        filetype="json",
    )
    check_and_create_output_dirs(
        os.path.join(args.data_dir, args.name, "vocab"),
        filetype="bpe",
    )

    urls_to_download = get_urls_for_tokenizer_files()
    for key, value in urls_to_download.items():
        if args.debug:
            cmd = f"wget --no-check-certificate --spider {value}"
            subprocess.run(cmd.split(" "), check=True)
            # continue since we want to run only debug, but for all items
            # in the url dictionary
            continue

        filepath = os.path.join(args.data_dir, args.name, "vocab", key)
        cmd = f"wget --no-check-certificate {value} -O {filepath}"
        subprocess.run(cmd.split(" "), check=True)


def main():
    """Main function for execution."""
    args = parse_args()

    # download all subsets and the corresponding tokenizer files
    for split in ["train", "val", "test"]:
        download_pile(args, split)
    download_tokenizer_files(args)


if __name__ == "__main__":
    main()
