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
import copy
import json
import logging
import math
import os
import re
import shutil
import signal
import sys
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from multiprocessing import Event, Value
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
import yaml
from PIL import Image
from pydantic import model_validator
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from tqdm import tqdm
from typing_extensions import Self

from cerebras.modelzoo.config import BaseConfig

logger = logging.getLogger("utils")
logger.setLevel(logging.INFO)

## Added .parquet extension to the list of valid extensions
VALID_EXTENSIONS = [
    '.jsonl',
    '.jsonl.zst',
    '.jsonl.zst.tar',
    '.txt',
    '.json.gz',
    '.parquet',
    '.fasta',
]


SYSTEM_PROMPT_REGISTRY = {
    "zephyr": "<|system|>\n</s>",
    "vicuna_v0": (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions."
    ),
    "vicuna_v1": (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    ),
    "llava_plain": "",
    "llava_v0": (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions."
    ),
    "llava_v1": (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions."
    ),
    "mistral_instruct": "",
}


def get_writer_process_num(shuffle: bool, processes: int) -> int:
    """
    Compute writer_process_num based on shuffle and processes.

    Args:
        shuffle (bool): Whether data shuffling is enabled.
        processes (int): Total number of processes.

    Returns:
        int: Computed writer_process_num value.
    """
    if shuffle:
        return (processes - 1) // 2
    else:
        return math.ceil((processes - 1) / 10)


def convert_fractions_or_floats(values):
    """
    Convert a fraction or float value, or a flat list of fraction/float values, into a float or a list of normalized floats.

    - If input is a single value, returns a single float.
    - If input is a list, returns a list of floats, normalized to sum to 1.

    Args:
        values (str, float, int, list of str/float/int): Input value(s).

    Returns:
        float or list: A single float if input is a single value, or a normalized list of floats if input is a list.

    Raises:
        ValueError: If any value cannot be converted to a float.
    """
    try:
        # If it's a single value, convert and return it directly
        if not isinstance(values, list):
            return (
                float(Fraction(values))
                if isinstance(values, str) and '/' in values
                else float(values)
            )

        # Convert list elements to floats
        float_values = [
            float(Fraction(v)) if isinstance(v, str) and '/' in v else float(v)
            for v in values
        ]

        # Normalize the list so it sums to 1
        total = sum(float_values)
        if total > 0:
            float_values = [v / total for v in float_values]

        return float_values

    except Exception as e:
        raise ValueError(f"Error converting values to floats: {e}")


def has_valid_extension(file):
    return any([file.endswith(ext) for ext in VALID_EXTENSIONS])


def _listdir_or_file(x):
    if isinstance(x, list):
        return reduce(lambda x, y: x + y, map(listdir_or_file, sorted(x)))
    if os.path.isfile(x):
        return [x]
    elif os.path.isdir(x):
        return [str(Path(x) / fn) for fn in sorted(os.listdir(x))]
    else:
        raise FileNotFoundError(f"{x} not found")


def listdir_or_file(x):
    return list(filter(has_valid_extension, _listdir_or_file(x)))


def dump_result(results, json_params_file, lock):
    """
    Write outputs of execution directly to a JSON file.

    Parameters:
    - results (dict): Dictionary containing the results to append/update in the JSON file.
    - json_params_file (str): Path to the JSON file.
    - lock (multiprocessing.Lock): Lock object for process safety.
    """
    with lock:
        with open(json_params_file, "r+") as f:
            data = json.load(f)

            if not data.get("post-process"):
                data["post-process"] = {}

            # Update or append new results
            for key, value in results.items():
                data["post-process"][key] = (
                    data["post-process"].get(key, 0) + value
                )

            # Calculate averages if necessary
            if data["post-process"].get("n_examples", 0) > 0:
                data["post-process"]["average_chars_per_sequence"] = math.ceil(
                    data["post-process"]["raw_chars_count"]
                    / data["post-process"]["n_examples"]
                )
                data["post-process"]["average_bytes_per_sequence"] = math.ceil(
                    data["post-process"]["raw_bytes_count"]
                    / data["post-process"]["n_examples"]
                )
            else:
                data["post-process"]["average_chars_per_sequence"] = 0
                data["post-process"]["average_bytes_per_sequence"] = 0

            # Calculate packing factor if applicable
            num_sequences_before_packing = data["post-process"].get(
                "num_sequences_before_packing", None
            )
            if num_sequences_before_packing:
                data["post-process"]["vsl_packing_factor"] = round(
                    num_sequences_before_packing
                    / data["post-process"]["n_examples"],
                    2,
                )

            # Write updated data back to the file
            f.seek(0)  # Move to the beginning of the file
            json.dump(data, f, indent=4, sort_keys=True)
            f.truncate()  # Truncate any remaining content if file size reduces


def dump_args(args, json_params_file):
    """
    Write the input params to file.
    """
    logger.info(f"User arguments can be found at {json_params_file}.")

    redundant_params = [
        "eos_id",
        "pad_id",
        "display_pbar",
        "files_per_record",
        "output_name",
        "write_remainder",
    ]

    relevant_args = copy.deepcopy(args)
    # Iterate through the dictionary and remove the redundant params
    for key in redundant_params:
        for sub_dict in relevant_args.values():
            if key in sub_dict:
                del sub_dict[key]

    # write initial params to file
    with open(json_params_file, "w") as _fout:
        json.dump(args, _fout, indent=4, sort_keys=True)


def update_args(args, json_params_file):
    "Update eos_id and pad_id in data_params"

    with open(json_params_file, "r") as _file:
        data = json.load(_file)

    data['processing']['pad_id'] = args.get(
        'pad_id', data['processing'].get('pad_id')
    )
    data['processing']['eos_id'] = args.get(
        'eos_id', data['processing'].get('eos_id')
    )
    data['processing']['vocab_size'] = args.get(
        'vocab_size', data['processing'].get('vocab_size')
    )
    data['features'] = args.get('features', None)

    with open(json_params_file, "w") as _fout:
        json.dump(data, _fout, indent=4, sort_keys=True)


def get_parser(desc):
    """Argparser definition for command line arguments from user.

    Returns:
        Argparse namespace object with command line arguments.
    """
    parser = argparse.ArgumentParser(description=desc)
    add_preprocess_args(parser)
    return parser.parse_args()


def add_preprocess_args(parser):
    """Add arguments to the data preprocessing parser."""
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the YAML config file for setting dataset preprocessing hyper-parameters.",
    )


def update_params(params, args):
    """
    Update config parameters with CLI arguments
    """
    setup_params = [
        "data",
        "metadata_files",
        "output_dir",
        "image_dir",
        "processes",
        "mode",
    ]
    processing_params = [
        "custom_tokenizer",
        "huggingface_tokenizer",
        "tokenizer_params",
        "eos_id",
        "pad_id",
        "max_seq_length",
        "min_sequence_len",
        "input_ids_dtype",
        "input_mask_dtype",
        "inverted_mask",
        "use_ftfy",
        "ftfy_normalizer",
        "wikitext_detokenize",
        "short_seq_prob",
        "write_in_batch",
        "resume_from_checkpoint",
        "seed",
        "read_chunk_size",
        "write_chunk_size",
        "shuffle",
        "shuffle_seed",
        "fraction_of_RAM_alloted",
        "read_hook",
        "read_hook_kwargs",
        "semantic_drop_mask",
        "semantic_loss_weight",
        "semantic_attention_mask",
    ]
    dataset_params = [
        "use_vsl",
        "truncate_to_msl",
        "max_prompt_length",
        "is_multimodal",
        "training_objective",
        "pack_sequences",
        "sep_token",
        "fim_rate",
        "spm_rate",
        "fim_prefix_tok",
        "fim_middle_tok",
        "fim_suffix_tok",
        "fold_long_doc",
        "split_text_to_tokenize",
        "chunk_len_to_split",
        "remove_bos_in_chunks",
        "user_role",
        "assistant_role",
        "chat_template",
        "respose_delimiter",
        "num_patches",
        "mlm_fraction",
        "mlm_with_gather",
        "ignore_index",
        "excluded_tokens",
        "max_num_img",
    ]
    cli_params = [
        "cmd",
        "func",
    ]

    for key, value in args.items():
        if value in ["True", "False"]:
            value = value == "True"
        if value is not None:
            if key in setup_params:
                params["setup"][key] = value
            elif key in processing_params:
                params["processing"][key] = value
            elif key in dataset_params:
                params["dataset"][key] = value
            elif key in cli_params:
                continue
            else:
                raise ValueError(f"Unexpected arguments: {key}")

    # Sections to check
    sections = {
        "setup": setup_params,
        "processing": processing_params,
        "dataset": dataset_params,
    }

    for section, allowed_params in sections.items():

        params_in_yaml = params.get(section, {})

        # Check for misplaced parameters
        for param in params_in_yaml:
            if param not in allowed_params:
                correct_section = next(
                    (s for s, p in sections.items() if param in p),
                    "unknown section",
                )
                if correct_section != "unknown section":
                    raise ValueError(
                        f"Error: Parameter '{param}' in section '{section}' is misplaced. It should be in '{correct_section}'."
                    )


def args_to_params(args):
    """Process data preprocessing CLI arguments to parameters
    Returns:
        params (Dict): Dictionary contains the parameters used to configure
            the data processing.
    """
    args = vars(args)

    params_file = args.pop("config", None)
    if params_file:
        with open(params_file, 'r') as stream:
            params = yaml.safe_load(stream)
    else:
        params = {}

    for section in ["setup", "processing", "dataset"]:
        if not params.get(section, None):
            params[section] = {}

    update_params(params, args)
    return params


def get_params(desc):
    """Retrieve configuration parameters
    Returns:
        params (Dict): Dictionary contains the parameters used to configure
            the data processing.
    """
    args = get_parser(desc)
    return args_to_params(args)


def dump_args(args, json_params_file):
    """
    Write the input params to file.
    """
    # write initial params to file
    with open(json_params_file, "w") as _fout:
        json.dump(args, _fout, indent=4, sort_keys=True)


def setup_warning_logging(output_dir, module_name):
    """
    Set up logging to log warnings to a file in the specified output directory.

    Args:
        output_dir (str): The directory where the warnings log file should be stored.
    """

    logger = logging.getLogger(module_name)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)
    os.makedirs(output_dir, exist_ok=True)
    # Create a file handler that logs to 'output_dir/warnings.log'
    log_file_path = os.path.join(output_dir, 'warnings.log')
    file_handler = logging.FileHandler(log_file_path)

    # Create a formatter and set it for the file handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)
    # Remove the default StreamHandler to prevent logging to stdout
    logger.propagate = False

    return logger


def get_files(input_dir=None, filetypes=None, metadata_files=None):
    """Get all files of given filetypes from input directory.

    Args:
        input_dir (str): Input directory to read files from.
        filetypes (list): File types to fetch from the given input
            directory. Defaults to `None`.
        metadata_files (str): Comma separated string of metadata files.

    Returns:
        List of lists containing all file paths as strings
    """
    if not filetypes:
        filetypes = [
            '.jsonl',
            '.json.gz',
            '.jsonl.zst',
            '.jsonl.zst.tar',
            '.txt',
            '.parquet',
            '.fasta',
        ]
    if isinstance(filetypes, str):
        filetypes = [filetypes]
    filetypes = tuple(filetypes)
    assert input_dir or metadata_files, (
        "User need to provide `input_dir` or `metadata_files`, "
        "but neither was provided."
    )
    if metadata_files:
        if isinstance(metadata_files, str):
            metadata_files = [metadata_files]

        if input_dir:
            logger.warning(
                "Both `input_dir` and `metadata_files` were provided, "
                "ignoring `input_dir` and using `metadata_files`."
            )

        input_files = []
        for _file in metadata_files:
            with open(_file, "r") as _fin:
                input_files.extend(_fin.readlines())

        input_files_list = [x.strip() for x in input_files if x]
        flattened_list = [x for x in input_files_list if x.endswith(filetypes)]
    else:
        files = [list(Path(input_dir).rglob(f"*{ft}")) for ft in filetypes]
        # flatten list of list -> list and stringify Paths
        flattened_list = [str(item) for sublist in files for item in sublist]
    if not flattened_list:
        raise Exception(
            f"Did not find any files at this path {input_dir}, please "
            f"ensure your files are in format {filetypes}."
        )
    return flattened_list


def wikitext_detokenizer(string):
    """Detokenizer for wikitext. Used for special handling of data for substrings.

    Args:
        string (str): String to detoknize before tokenization.

    Returns:
        Detokenized string
    """
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string


def clean_text(
    data: str, use_ftfy: bool, wikitext_detokenize: bool, ftfy_normalizer: str
) -> str:
    """
    Clean the provided text using ftfy normalization and wikitext detokenization.

    Args:
        data (str): The text to be cleaned.
        use_ftfy (bool): Whether to use the `ftfy` library to fix text encoding issues.
        wikitext_detokenize (bool): Whether to apply wikitext detokenization to the text.
        ftfy_normalizer (str): The normalization method to use with `ftfy` if enabled.

    Returns:
        str: The cleaned text after applying the specified operations.
    """
    import ftfy

    if use_ftfy:
        data = ftfy.fix_text(data, normalization=ftfy_normalizer)
    if wikitext_detokenize:
        data = wikitext_detokenizer(data)

    return data


def get_data_stats(
    sample: np.ndarray,
    pad_id: int,
    eos_id: int,
    max_seq_length: int,
    loss_valid_tokens: Optional[int] = None,
) -> Dict[str, int]:
    """
    Get data statistics from the sample.

    Args:
        sample (np.ndarray): Tokenized sample in the form of a NumPy array.
        pad_id (int): The ID used for padding tokens.
        eos_id (int): The ID used for end-of-sequence tokens.
        max_seq_length (int): The maximum sequence length.
        loss_valid_tokens (Optional[int]): The number of valid tokens for loss computation. If not provided, it will be calculated from the sample.

    Returns:
        Dict[str, int]: A dictionary containing the following data statistics:
            - "num_pad_tokens": Number of padding tokens in the sample.
            - "non_pad_tokens": Number of tokens that are neither padding nor end-of-sequence tokens.
            - "num_tokens": Total number of tokens in the sample.
            - "loss_valid_tokens": Number of valid tokens for loss computation.
            - "num_masked_tokens": Number of masked tokens based on the maximum sequence length.
    """
    stats = defaultdict(int)
    if len(sample) == 0:
        return stats
    stats["num_pad_tokens"] = int((sample[0, :] == pad_id).sum())
    stats["non_pad_tokens"] = int(
        np.logical_and(sample[0, :] != eos_id, sample[0, :] != pad_id).sum()
    )
    stats["num_tokens"] = int(sample[0, :].shape[0])

    if loss_valid_tokens:
        stats["loss_valid_tokens"] = loss_valid_tokens
    else:
        stats["loss_valid_tokens"] = int(sample[1, :].sum())
    stats["num_masked_tokens"] = max_seq_length - stats["loss_valid_tokens"]
    stats["n_examples"] = 1
    return stats


def check_and_create_output_dirs(
    output_dir, filetype, dir_name="Output", overwrite=False
):
    contains_filetype = False
    if os.path.isdir(output_dir):
        for dirpath, dirnames, filenames in os.walk(output_dir):
            for file in filenames:
                if file.endswith(filetype):
                    contains_filetype = True
                    break
            if contains_filetype:
                break

    if contains_filetype and not overwrite:
        _in = input(
            f"{dir_name} directory already contains {filetype} file(s)."
            + " Do you want to delete the folder to write"
            + " new records in the same output folder name? (yes/no): "
        )
        if _in.lower() in ["y", "yes"]:
            shutil.rmtree(output_dir)
        elif _in.lower() in ["n", "no"]:
            raise IsADirectoryError(
                "Create a new folder for the files you want to write!!"
            )
        else:
            raise ValueError(f"Inputs can be yes, no, y or n. Received {_in}!!")

    os.makedirs(output_dir, exist_ok=True)


# routine to split the text into smaller sequences
def split_text_and_tokenize(
    text, tokenizer, max_tok_len=2000, remove_bos_in_chunks=True
):
    """Function to split the text into smaller sequences of length max_tok_len
    and then tokenize each of the smaller sequences. This is done to avoid
    performance issues with tokenizers like LlamaTokenizer which are slow for
    long sequences.

    Args:
        text (str): text to be tokenized
        tokenizer (Tokenizer): tokenizer to be used
        max_tok_len (int, optional): max length of each sequence. Defaults to 2000.
        remove_bos_in_chunks (bool, optional): whether to ignore bos token id in
            chunks. Defaults to True.
    Returns:
        tok_ids (list): list of token ids for the text
    """
    if len(text) == 0:
        return []

    curr_start = 0
    tok_ids = []

    while curr_start < len(text):
        curr_end = min(text.find(' ', curr_start + max_tok_len), len(text))
        if curr_end < 0:
            curr_substr = text[curr_start:]
            curr_end = len(text)
        else:
            curr_substr = text[curr_start:curr_end]
        if curr_start == 0:
            # keep special tokens for the first chunk
            bos_token_id = [tokenizer.encode(curr_substr)[0]]
        curr_tok_ids = (
            tokenizer.encode(curr_substr)[1:]
            if remove_bos_in_chunks
            else tokenizer.encode(curr_substr)
        )
        tok_ids.extend(curr_tok_ids)
        curr_start = curr_end
    # concatenated tok_ids chunks together by using `extend` to return full sequence of tokens

    # NOTE: add bos token id if it is needed here, eos id is added in the next line
    # which calls this function
    return bos_token_id + tok_ids if remove_bos_in_chunks else tok_ids


def chunk(
    sample,
    tokenizer,
    fim_rate,
    spm_rate,
):
    """
    Since we do character-level FIM we need to detokenize, determine boundaries
    to split, and re-tokenize after splitting. We chunk but do not shuffle and add
    special tokens because we might have to truncate or pad the tokens since they
    have been split at the character-level and re-tokenized, leading to potentially
    different lengths than the original sequence.
    If the sub-context is designated to be an AR (auto-regressive) sequence and not FIM, we store
    as [[], [], [sequence]] for convenience in the truncate_helper function.

    Args:
        sample (np.array):
        tokenizer (Tokenizer):
        fim_rate (float):
        spm_rate (float):

    Returns:
        List[List[int]], str: List of token lists corresponding to the
          prefix/middle/suffix tokens, or 2 empty lists plus the whole
          sequence in case of auto-regressive (AR) sequence. Also returns
          string representing the format of the sequence (i.e. SPM or
          PSM or AR)
    """
    if np.random.binomial(1, fim_rate):  # sample bernoulli dist
        contents = tokenizer.decode(sample, skip_special_tokens=False)
        try:
            # A boundary can be =0 (prefix will be empty)
            # a boundary can be =len(contents) (suffix will be empty)
            # The two boundaries can be equal (middle will be empty)
            boundaries = list(
                np.random.randint(low=0, high=len(contents) + 1, size=2)
            )
            boundaries.sort()
        except ValueError as e:
            logging.info(len(contents))
            logging.info(contents)
            logging.info(e)
            raise e

        prefix = contents[: boundaries[0]]
        middle = contents[boundaries[0] : boundaries[1]]
        suffix = contents[boundaries[1] :]

        prefix = tokenizer.encode(prefix)
        middle = tokenizer.encode(middle)
        suffix = tokenizer.encode(suffix)

        is_spm = np.random.binomial(1, spm_rate)
        fim_format = "SPM" if is_spm else "PSM"
        return [prefix, middle, suffix], fim_format
    else:
        # don't do FIM preproc
        fim_format = "AR"
        return [[], [], sample.tolist()], fim_format


def format_fim(
    segment_fim_format_pairs,
    max_seq_len,
    suffix_tok_id,
    prefix_tok_id,
    middle_tok_id,
    eos_tok_id,
    opt_bos_tok_id,
):
    """
    Takes in list of prefix/middle/suffix token lists, along with respective FIM (or AR) formats.
    Performs the correct transformation according to the format, adding the special tokens
    and shuffling the sections, before concatenating everything together.

    Args:
        segments_fim_format_pairs (List[Tuple[List[List[int]], str]]): This list of tuples is used
        to store the prefix/middle/suffix token-id lists and the corresponding FIM formats (PSM/SPM) to
        be used downstream in the FIM formatting.
        max_seq_len (int): Max sequence length that each sequence is expected
          to match
        suffix_tok_id (int): Id for suffix token
        prefix_tok_id (int): Id for suffix token
        middle_tok_id (int): Id for suffix token
        eos_tok_id (int): Id for suffix token
        opt_bos_tok_id (list): Optionally a list containing the bos token id,
          otherwise will be empty list. Empty list will be a no-op in the
          concatenation. Bos-token will only exist if model's tokenizer adds
          bos-token by default. Both have to be lists so that np concat works

    Returns:
        sample (np.array): Array of token ids in the FIMed order
          along with special tokens
        mask (np.array): Array of 1's and 0's corresponding to true
          tokens and padding respectively
        label (np.array): Token i of label corresponds to token i+1 in
          sample array. Same elements except that label ends in eos
          (end-of-sequence) token
    """

    prefix_idx, middle_idx, suffix_idx = 0, 1, 2
    sample = []
    total_padding_len = 0
    for sample_i, fim_format in segment_fim_format_pairs:
        optional_padding = sample_i[-1] if len(sample_i) > 3 else []
        total_padding_len += len(optional_padding)
        if fim_format == "PSM":
            sample_i = np.concatenate(
                [
                    opt_bos_tok_id,
                    [prefix_tok_id],
                    sample_i[prefix_idx],
                    [suffix_tok_id],
                    sample_i[suffix_idx],
                    [middle_tok_id],
                    sample_i[middle_idx],
                    [eos_tok_id],
                ]
            )
        elif fim_format == "SPM":
            sample_i = np.concatenate(
                [
                    opt_bos_tok_id,
                    [prefix_tok_id, suffix_tok_id],
                    sample_i[suffix_idx],
                    [middle_tok_id],
                    sample_i[prefix_idx],
                    sample_i[middle_idx],
                    [eos_tok_id],
                ]
            )
        else:
            sample_i = np.concatenate(
                [
                    opt_bos_tok_id,
                    sample_i[prefix_idx],
                    sample_i[middle_idx],
                    sample_i[suffix_idx],
                    [eos_tok_id],
                ]
            )
        sample_i = np.concatenate([sample_i, optional_padding])
        sample.append(sample_i)
    sample = np.concatenate(sample).astype(np.int64)
    label = sample[1:]
    sample = sample[:-1]
    sample_mask = np.ones(max_seq_len - total_padding_len)
    padding_mask = np.zeros(total_padding_len)
    mask = np.concatenate([sample_mask, padding_mask])
    return sample, mask, label


def truncate_helper(samples_lst, diff, sample_idx):
    """
    The goal of our truncation scheme is to avoid removing tokens from the
    middle section. We first remove from the end of suffix, and then from the
    beginning of the prefix. We store the chunks in lists in the original order
    so that we can easily perform this truncation. Since each sub-context can have
    different amounts of tokens in suffix/prefix, we store unique indices for the
    section to remove from. If we run out of tokens to remove from, we switch to the next.
    This way we can switch to the prefix of one context while still removing from suffix
    of another. If the sub-context is AR (auto-regressive) and not FIM, the AR sequence
    is stored as [[], [], [sequence]] so that the remove_idx being 2 will simultaneously
    work for the AR and FIM sequences.

    Args:
        samples_lst (List[List[int]]): List of lists that contain token ids
        diff (int): Number of tokens to pad
        sample_idx (int): Index for the sample from the dataset, for use in
          logging if we remove from the middle.

    Returns:
        (List[List[int]]): List of lists of token ids that have been truncated
    """
    num_groups = len(samples_lst)
    remove_idxs = [2] * num_groups  # remove from suffixes first
    i = 0

    while diff:
        remove_idx_i = remove_idxs[i]
        sample_i = samples_lst[i]
        if sample_i[remove_idx_i]:
            pop_idx = (
                -1 if remove_idx_i == 2 else 0
            )  # remove from end of suffix but beginning of prefix
            sample_i[remove_idx_i].pop(pop_idx)
            diff -= 1
        else:
            remove_idxs[i] = (
                remove_idxs[i] + 1
            ) % 3  # order of removal is end of suffix, beginning of prefix, then beginning of middle
            if remove_idxs[i] == 1:
                logging.info(
                    f"""Context {i} in the {sample_idx}-th data sample has
                        begun truncating from the middle section, meaning
                        the prefix and suffix sections have been exhausted.
                      """
                )
        i = (i + 1) % num_groups

    return samples_lst


def pad_helper(samples_lst, diff, fim_pad_tok_id):
    """
    Helper for padding. We put all padding tokens into the last sequence.

    Args:
        samples_lst (List[List[int]]): List of lists that contain token ids
        diff (int): Number of tokens to pad
        fim_pad_tok_id (int): Id for padding token

    Returns:
        (List[List[int]]): List of lists of token ids with padding
    """
    padding = np.full(np.abs(diff), fim_pad_tok_id)
    samples_lst[-1].append(padding)
    return samples_lst


def truncate_or_pad_helper(
    segments_fim_format_pairs, diff, fim_pad_tok_id, sample_idx
):
    """
    Since we perform FIM at character-level, we potentially split characters
    in the middle of a word. This can lead to non-standard token sequences,
    and after re-tokenizing we might need to truncate or pad to get back to
    the original context length. This function ensures that our outputs are
    back at their original length.

    Args:
        segments_fim_format_pairs (List[Tuple[List[List[int]], str]]): This list of tuples is used
        to store the prefix/middle/suffix token-id lists and the corresponding FIM formats (PSM/SPM) to
        be used downstream in the FIM formatting.
        diff (int): The number of tokens to add or remove. Positive means truncate, negative means pad
        fim_pad_tok_id (int): Id of padding token

    Returs:
        (List[Tuple[List[List[int]], str]]): The element of the tuples will
        now be lists that are truncated or padded such that the concatenation of all these tokens, along
        with the special tokens, will be equal to the original sequence length.
    """
    segments = [pair[0] for pair in segments_fim_format_pairs]
    fim_formats = [pair[1] for pair in segments_fim_format_pairs]
    if diff >= 0:
        segments = truncate_helper(segments, diff, sample_idx)
    else:
        segments = pad_helper(segments, diff, fim_pad_tok_id)
    return [(segments[i], fim_formats[i]) for i in range(len(segments))]


def fim(
    sample_array,
    sample_idx,
    tokenizer,
    fim_rate,
    spm_rate,
    suffix_tok_id,
    prefix_tok_id,
    middle_tok_id,
    fim_pad_tok_id,
    eos_tok_id,
    opt_bos_tok_id,
):
    """
    Takes in an array of input_ids, mask, and labels, and performs the
    FIM operation to re-arrange into PSM and SPM format with some probability

    Args:
        sample_array (np.array): Stack of input_ids, mask, and labels after tokenization. Labels are off-by-one of input_ids
        as in standard auto-regressive training
        i (int): Index of sample from dataset, used for logging.
        tokenizer (Tokenizer): Tokenizer object
        fim_rate (float): Determines what percentage of contexts are FIM'ed
        spm_rate (float): Determines what percentage of FIM'ed contexts are in SPM format. 1 - spm_rate determines PSM
        suffix_tok_id (int): Id for special token denoting suffix section in a FIM'ed context
        prefix_tok_id (int): Id for special token denoting prefix section in a FIM'ed context
        middle_tok_id (int): Id for special token denoting middle section in a FIM'ed context
        fim_pad_tok_id (int): Id for padding
        eos_tok_id (int): Id for the end-of-seqence
        opt_bos_tok_id (list): Optionally a list containing the bos token id,
          otherwise will be empty list. Empty list will be a no-op in the
          concatenation. Bos-token will only exist if model's tokenizer adds
          bos-token by default.

    Returns:
        fim_outputs (np.array): Stack of input_ids, mask, and labels after FIM transformation. Mask and labels have been
        adjusted to still filter padding tokens and represent the following token, respectively.
    """
    assert (
        fim_rate <= 1 and fim_rate >= 0
    ), "FIM rate must be a probability 0 <= rate <= 1"
    sample = sample_array[0, :]
    mask = sample_array[1, :]
    max_seq_len = sample.shape[0]

    segment_breaks = np.argwhere(
        sample == eos_tok_id
    )  # split sample by document
    segments_fim_format_pairs = []
    if segment_breaks.shape != (0, 1):  # FIM each sub-context
        curr_start_position = 0
        for loc in np.nditer(segment_breaks):
            # Only permute non-empty segments.
            if loc - curr_start_position > 0:
                segments, fim_format = chunk(
                    sample=sample[curr_start_position:loc],
                    tokenizer=tokenizer,
                    fim_rate=fim_rate,
                    spm_rate=spm_rate,
                )
                segments_fim_format_pairs.append((segments, fim_format))
            curr_start_position = loc + 1  # jump over the EOD token
        # Permute the segment after the last EOD
        segments, fim_format = chunk(
            sample=sample[curr_start_position:],
            tokenizer=tokenizer,
            fim_rate=fim_rate,
            spm_rate=spm_rate,
        )
        segments_fim_format_pairs.append((segments, fim_format))
    else:  # FIM over full context
        segments, fim_format = chunk(
            sample=sample,
            tokenizer=tokenizer,
            fim_rate=fim_rate,
            spm_rate=spm_rate,
        )
        segments_fim_format_pairs.append((segments, fim_format))

    def flatten_2d(arr):
        return np.concatenate([np.concatenate(subarr) for subarr in arr])

    total_len = flatten_2d(
        [pair[0] for pair in segments_fim_format_pairs]
    ).shape[0]
    # we factor in the final EOS, which we add before splitting into
    # inputs and labels, i.e. sequence[:-1] and sequence[1:], and the
    # optional bos token
    add_constant = -1
    for _, fmt in segments_fim_format_pairs:
        if fmt == "AR":
            add_constant += 1
        else:
            add_constant += 4
        if opt_bos_tok_id:
            add_constant += 1
    diff = (total_len + add_constant) - max_seq_len
    segments_fim_format_pairs = truncate_or_pad_helper(
        segments_fim_format_pairs,
        diff,
        fim_pad_tok_id,
        sample_idx,
    )
    inputs, mask, labels = format_fim(
        segments_fim_format_pairs,
        max_seq_len,
        suffix_tok_id,
        prefix_tok_id,
        middle_tok_id,
        eos_tok_id,
        opt_bos_tok_id,
    )

    try:
        assert inputs.shape[0] == max_seq_len
        assert mask.shape[0] == max_seq_len
        assert labels.shape[0] == max_seq_len
    except:
        logging.error(
            "The inputs/masks/labels were not the correct\
                      sized after FIM process. Shapes of each are printed\
                      below, along with the correct max seqeunce length\
                      that each sequence should be."
        )
        logging.error(inputs.shape, max_seq_len)
        logging.error(mask.shape, max_seq_len)
        logging.error(labels.shape, max_seq_len)
        raise AssertionError
    try:
        assert labels[-1] == eos_tok_id
    except:
        logging.error("The sequence did not end with an EOS token")
        raise AssertionError
    # end FIM-specific code
    fim_outputs = np.stack([inputs, mask, labels], axis=0)
    return fim_outputs


def get_tokenizer_vocab(tokenizer):
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

    from cerebras.modelzoo.data_preparation.nlp.tokenizers.BPETokenizer import (
        BPETokenizer,
    )
    from cerebras.modelzoo.data_preparation.nlp.tokenizers.HFTokenizer import (
        HFTokenizer,
    )

    if isinstance(tokenizer, BPETokenizer):
        tokenizer_vocab = tokenizer.encoder
    elif isinstance(tokenizer, HFTokenizer):
        tokenizer_vocab = tokenizer.tokenizer.get_vocab()
    elif isinstance(tokenizer, PreTrainedTokenizer) or isinstance(
        tokenizer, PreTrainedTokenizerFast
    ):
        tokenizer_vocab = tokenizer.vocab
    else:
        raise NotImplementedError(
            "We do not support specified tokenizer\
                                  type."
        )
    return tokenizer_vocab


def check_fim_special_tokens(params, tokenizer):
    # Check that input config lists the FIM special tokens
    assert (
        "fim_suffix_tok" in params['dataset']
        and "fim_prefix_tok" in params['dataset']
        and "fim_middle_tok" in params['dataset']
    ), """Configs for FIM pre-processing must include the special tokens that
    denote prefix, middle, and suffix tokens."""
    # Check that the provided tokens are in the tokenizer
    pre_tok = params['dataset'].get("fim_prefix_tok")
    mid_tok = params['dataset'].get("fim_middle_tok")
    suf_tok = params['dataset'].get("fim_suffix_tok")
    tokenizer_vocab = get_tokenizer_vocab(tokenizer)
    assert (
        pre_tok in tokenizer_vocab
        and mid_tok in tokenizer_vocab
        and suf_tok in tokenizer_vocab
    ), """Please ensure that the provided FIM special tokens are in the
    specified tokenizer."""


def handle_bos_token_default(tokenizer):
    """
    When performing FIM, we tokenize each chunk again after splitting.
    Therefore, if the tokenizer adds bos-token by default, we will get
    extra bos-tokens in the middle of the sequence. In this function,
    we set the tokenizer bos default to False, and return a flag that
    indicates whether we will need to add bos-token in the final
    fim formatting function.
    """
    if hasattr(tokenizer, "add_bos_token") and tokenizer.add_bos_token:
        tokenizer.add_bos_token = False
        bos_tok_id = tokenizer.encode(tokenizer.bos_token)[-1]
        return True, [bos_tok_id]
    return False, []


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(
        obj, (str, bytes, bytearray)
    ):
        size += sum([get_size(i, seen) for i in obj])
    return size


def append_eos_to_multiple_semantic_regions(
    formatted_data,
    data_ranges,
    eos_token,
    image_token,
    is_chat_data,
):

    if data_ranges == [] or not eos_token:
        return data_ranges
    eos_indices = []
    start_search_index = data_ranges[0].get("indices")[0]
    while start_search_index < len(formatted_data):
        eos_start_idx = formatted_data.find(eos_token, start_search_index)
        if eos_start_idx == -1:
            ## No eos found. Break
            break
        eos_end_idx = eos_start_idx + len(eos_token)
        start_search_index = eos_end_idx
        eos_indices.append((eos_start_idx, eos_end_idx))

    current_eos_pos = 0
    current_data_range_pos = 0
    while current_eos_pos < len(eos_indices) and current_data_range_pos < len(
        data_ranges
    ):
        eos_start_idx, eos_end_idx = eos_indices[current_eos_pos]
        region_start_idx, region_end_idx = data_ranges[
            current_data_range_pos
        ].get("indices")
        ## EOS occurs in the current region
        if region_start_idx <= eos_start_idx < region_end_idx:
            current_eos_pos += 1
            continue

        if current_data_range_pos + 1 < len(data_ranges):
            next_region_start_idx, next_region_end_idx = data_ranges[
                current_data_range_pos + 1
            ].get("indices")
            ## Check if eos occurs between current and next region
            if region_end_idx <= eos_start_idx < next_region_start_idx:
                image_start_idx = (
                    -1
                    if image_token is None
                    else formatted_data[region_end_idx:eos_start_idx].find(
                        image_token
                    )
                )
                if image_start_idx == -1:
                    indices_incl_eos = (region_start_idx, eos_end_idx)
                    data_ranges[current_data_range_pos][
                        "indices"
                    ] = indices_incl_eos
                    current_eos_pos += 1
        else:
            ## insert EOS in the last region
            image_start_idx = (
                -1
                if image_token is None
                else formatted_data[region_end_idx:eos_start_idx].find(
                    image_token
                )
            )
            if image_start_idx == -1:
                indices_incl_eos = (region_start_idx, eos_end_idx)
                data_ranges[current_data_range_pos][
                    "indices"
                ] = indices_incl_eos
                current_eos_pos += 1
        current_data_range_pos += 1

    if (
        not is_chat_data or len(eos_indices) > 1
    ):  ## 1 because the last eot could be eos
        return data_ranges

    for i in range(1, len(data_ranges)):
        start_idx, end_idx = data_ranges[i].get("indices")
        previous_start_idx, previous_end_idx = data_ranges[i - 1].get("indices")
        if previous_end_idx != start_idx:
            handle_turn_token = True
            data_ranges[i - 1]["handle_turn_token"] = True
        if i == len(data_ranges) - 1:
            if end_idx < len(formatted_data):
                data_ranges[i]["handle_turn_token"] = True

    return data_ranges


def find_region_in_formatted_string(text_semantic_region_list, formatted_data):

    string_search_idx = 0
    for semantic_region in text_semantic_region_list:
        region_identifier = semantic_region.pop("region_identifier", "")
        region_len = semantic_region.get("region_len")
        region_identifier_start_idx = formatted_data.find(
            region_identifier, string_search_idx
        )
        assert (
            region_identifier_start_idx != -1
        ), f"Unable to find region_identifier - {region_identifier} in the string - {formatted_data}"
        formatted_data = formatted_data.replace(region_identifier, "")
        start_idx = region_identifier_start_idx
        end_idx = start_idx + region_len
        string_search_idx = end_idx
        semantic_region.update({"indices": (start_idx, end_idx)})

    return formatted_data, text_semantic_region_list


def find_token_range(region, offsets, starting_offset_position):

    string_start, string_end = region.pop('indices')
    token_start = next(
        (
            i
            for i in range(starting_offset_position, len(offsets))
            if (offsets[i][0] <= string_start and offsets[i][1] > string_start)
            or (
                offsets[i][0] > string_start
            )  ## this condition is useful for neox tokenizer which treats space as an additional token
        ),
        None,
    )
    if token_start is None:
        raise ValueError(
            f"The implementation of offset mapping of this tokenizer may be incorrect. Check the huggingface implementation for more details."
        )
    token_end = next(
        (
            i
            for i in range(token_start, len(offsets))
            if (offsets[i][1] >= string_end and offsets[i][0] < string_end)
            or (
                offsets[i][1] < string_end
                and ((i + 1) >= len(offsets) or offsets[i + 1][0] >= string_end)
            )
        ),
        None,
    )
    if token_end is None:
        raise ValueError(
            f"The huggingface implementation of offset mapping of this tokenizer may be incorrect. Check the huggingface implementation for more details."
        )
    data = {
        "indices": (token_start, token_end + 1),
        "loss_weight": region.get("loss_weight"),
        "attention_mask": region.get("attention_mask"),
    }

    return data


def truncate_sequence(
    token_ids,
    tokenized_semantic_region_list,
    max_sequence_length,
    max_turn_length,
    prompt_truncation_mode,
):
    """
    Truncates token sequences to fit within a specified MSL, parameterized by max_turn_length.

    Args:
        token_ids (list): List of token IDs representing the entire sequence.
        tokenized_semantic_region_list (list): List of tokenized semantic regions.
        max_sequence_length (int): Maximum allowed length of the sequence after truncation.
        max_turn_length (int): Maximum length of any single segment that can be present, after truncation.
        prompt_truncation_mode (str): Mode of truncation for prompt/user part of chat. Can be 'keep_start' or 'keep_end'.

    Returns:
        tokenized_semantic_region_list (list): Returned with indices updated for region after truncation.
        list: The truncated sequence of token IDs that fits within the max_sequence_length constraint.
    """

    def update_semantic_regions(
        part_one_list,
        part_two_list,
        part_one_indices_to_remove,
        part_two_indices_to_remove,
    ):
        combined_list = part_one_list + part_two_list
        combined_list.sort(key=lambda x: x[2][0])

        combined_rem = part_one_indices_to_remove + part_two_indices_to_remove
        combined_rem_dict = OrderedDict()

        for element in combined_rem:
            key = (element[0], element[1])
            value = (element[2], element[3])
            combined_rem_dict[key] = value

        updated_ranges = []
        cumulative_shift = 0

        for index, part, (original_start, original_end) in combined_list:
            removed_item = combined_rem_dict.get((index, part))

            if removed_item is not None:
                mode, (removed_start, removed_end) = removed_item
                current_shift = removed_end - removed_start

                if mode == "keep_start":
                    new_start, new_end = (
                        original_start - cumulative_shift,
                        removed_start - cumulative_shift,
                    )
                elif mode == "keep_end":
                    new_start, new_end = (
                        removed_end - cumulative_shift - current_shift,
                        original_end - cumulative_shift - current_shift,
                    )

                cumulative_shift += current_shift
            else:
                current_shift = 0
                new_start, new_end = (
                    original_start - cumulative_shift,
                    original_end - cumulative_shift,
                )
                cumulative_shift += current_shift

            updated_ranges.append((new_start, new_end))

        no_of_regions = 0
        for region in tokenized_semantic_region_list:
            no_of_regions += 1

        assert (
            len(updated_ranges) == no_of_regions
        ), "Mismatch in number of regions of tokenized_semantic_region_list and the updated ranges."

        index = 0
        for region in tokenized_semantic_region_list:
            region['indices'] = updated_ranges[index]
            index += 1

        return tokenized_semantic_region_list

    def _truncate(
        tokenized_semantic_region_list,
        part_one_list,
        part_two_list,
        truncate_length,
    ):
        """
        Helper function to truncate two parts of the sequence based on the provided length.

        Args:
            tokenized_semantic_region_list (list): List of semantic regions that are present.
            part_one_list (list): List of (start, end) tuples for the first part of the sequence.
            part_two_list (list): List of (start, end) tuples for the second part of the sequence.
            truncate_length (int): Total length that needs to be truncated from the sequence.

        Returns:
            list: Truncated sequence of token IDs.
        """

        # Enumerating the lists, to maintain indices (which are used later).
        part_one_list = list(enumerate(part_one_list))
        part_one_list = [
            (item[0], 'part_one', item[1]) for item in part_one_list
        ]

        part_two_list = list(enumerate(part_two_list))
        part_two_list = [
            (item[0], 'part_two', item[1]) for item in part_two_list
        ]

        part_one_indices_to_remove = []

        # Sort the ordered list by maximum turn length, with the maximum length indices coming first.
        sorted_part_one = sorted(
            part_one_list, key=lambda x: x[2][1] - x[2][0], reverse=True
        )

        # Truncate from the first part of the sequence.
        for index, part, (start, end) in sorted_part_one:
            length_of_turn = end - start

            """
                We also have to always maintain (max_turn_length) in every turn, after truncation.
                Therefore, the max amount that can be truncated = (length_of_turn - max_turn_length)

                What happens if length of turn is < max_turn_length?
                Then we keep the entire turn, and move to the next user and try truncating from there.
            """

            if max_turn_length >= length_of_turn:
                # Keep the entire turn; no truncation at all.
                continue
            else:
                # max_turn_length < length_of_turn i.e truncation is possible from this turn.
                available_truncate = length_of_turn - max_turn_length

                if available_truncate < truncate_length:
                    # Truncate the max you can, move to the next turn.
                    truncate_length -= available_truncate

                    if prompt_truncation_mode == "keep_start":
                        part_one_indices_to_remove.append(
                            (
                                index,
                                part,
                                'keep_start',
                                (end - available_truncate, end),
                            )
                        )
                    elif prompt_truncation_mode == "keep_end":
                        part_one_indices_to_remove.append(
                            (
                                index,
                                part,
                                'keep_end',
                                (start, start + available_truncate),
                            )
                        )
                else:
                    # Here, available_truncate >= truncate_length i.e we have more than what we need.
                    # Therefore, we'll take only what we need, and we have finished truncation from Part 1 solely.
                    if prompt_truncation_mode == "keep_start":
                        part_one_indices_to_remove.append(
                            (
                                index,
                                part,
                                'keep_start',
                                (end - truncate_length, end),
                            )
                        )
                    elif prompt_truncation_mode == "keep_end":
                        part_one_indices_to_remove.append(
                            (
                                index,
                                part,
                                'keep_end',
                                (start, start + truncate_length),
                            )
                        )

                    # Sorting this, in order to not mess up the indices while removing.
                    range_of_indices_to_remove_part_one = sorted(
                        part_one_indices_to_remove,
                        key=lambda x: x[3][0],
                        reverse=True,
                    )

                    for (
                        index,
                        part,
                        mode,
                        (start, end),
                    ) in range_of_indices_to_remove_part_one:
                        del token_ids[start:end]

                    assert (
                        len(token_ids) == max_sequence_length
                    ), "After truncation, the length of token IDs should be equal to MSL."

                    # Now, update tokenized_semantic_region_list.
                    tokenized_semantic_region_list = update_semantic_regions(
                        part_one_list,
                        part_two_list,
                        part_one_indices_to_remove,
                        [],
                    )

                    return tokenized_semantic_region_list, token_ids

        assert (
            truncate_length > 0
        ), "Truncation from second part should only happen if truncation from the first part is exhausted."

        # Calculate the total possible truncation length from the second part.
        total_possible_truncation = 0
        for index, part, (start, end) in part_two_list:
            total_possible_truncation += (end - start) - max_turn_length

        if total_possible_truncation < truncate_length:
            return (
                tokenized_semantic_region_list,
                {},
            )  # If the total truncation possible is not enough to meet the truncation length.
        else:
            part_two_indices_to_remove = []

            # Sorting this by max turn length, so that most of the truncation happens from the longest range.
            sorted_part_two = sorted(
                part_two_list, key=lambda x: x[2][1] - x[2][0], reverse=True
            )

            for index, part, (start, end) in sorted_part_two:
                length_of_turn = end - start

                if max_turn_length >= length_of_turn:
                    # Keep the entire turn; no truncation.
                    continue
                else:
                    # Truncate the maximum you can, move to the next turn. By default, we keep the end i.e "keep_start" for completion.
                    # This is done to maintain recent context as much as possible.
                    available_truncate = length_of_turn - max_turn_length

                    if available_truncate < truncate_length:
                        # We need to truncate more than what is availabe; thus truncate max you can and move to next turn.
                        truncate_length -= available_truncate
                        part_two_indices_to_remove.append(
                            (
                                index,
                                part,
                                'keep_start',
                                (end - available_truncate, end),
                            )
                        )
                    else:
                        # We can finish the truncation here, as what we have is more than what we need.
                        part_two_indices_to_remove.append(
                            (
                                index,
                                part,
                                'keep_start',
                                (end - truncate_length, end),
                            )
                        )
                        break

        # Sorting the indices in descending order, to maintain correctness while deleting.
        range_of_indices_to_remove = (
            part_one_indices_to_remove + part_two_indices_to_remove
        )
        range_of_indices_to_remove.sort(key=lambda x: x[3][0], reverse=True)

        for index, part, mode, (start, end) in range_of_indices_to_remove:
            del token_ids[start:end]

        assert (
            len(token_ids) == max_sequence_length
        ), "After truncation, the length of token IDs should be equal to MSL."

        tokenized_semantic_region_list = update_semantic_regions(
            part_one_list,
            part_two_list,
            part_one_indices_to_remove,
            part_two_indices_to_remove,
        )

        return tokenized_semantic_region_list, token_ids

    def _get_truncation_indices(tokenized_semantic_region_list):
        truncation_indices = {}
        for regions in tokenized_semantic_region_list:
            if regions['role'] not in truncation_indices:
                truncation_indices[regions['role']] = []

            truncation_indices[regions['role']].append(regions['indices'])
        return truncation_indices

    if prompt_truncation_mode not in ['keep_start', 'keep_end']:
        raise ValueError(
            "prompt_truncation_mode should only be 'keep_start' or 'keep_end'."
        )

    # Generate truncation indices
    truncation_indices = _get_truncation_indices(tokenized_semantic_region_list)

    # Determine which keys are present in the truncation indices dictionary.
    keys = set(truncation_indices.keys())

    # Total length to truncate.
    truncate_length = len(token_ids) - max_sequence_length

    if "prompt" in keys and "completion" in keys:
        # Adjusting for BOS token in prompt/completion.
        if truncation_indices['prompt'][0][0] != 0:
            truncation_indices['prompt'][0][0] = 0

        interaction_type = "prompt_completion"
        return _truncate(
            tokenized_semantic_region_list,
            truncation_indices['prompt'],
            truncation_indices['completion'],
            truncate_length,
        )
    elif "user" in keys and "assistant" in keys:
        interaction_type = "user_assistant"
        return _truncate(
            tokenized_semantic_region_list,
            truncation_indices['user'],
            truncation_indices['assistant'],
            truncate_length,
        )
    else:
        raise ValueError(
            "Truncation is only supported for 'prompt'/'completion' or 'user'/'assistant'."
        )


def default_chat_template():
    """
    This template formats inputs in the standard ChatML format. See
    https://github.com/openai/openai-python/blob/main/chatml.md
    """
    return (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    )


class EmbeddingStatCollector:
    def __init__(
        self,
        use_ftfy: bool,
        ftfy_normalizer: str,
        wikitext_detokenize: bool,
        eos_id: int,
        pad_id: int,
    ):
        '''
        Statistics Collector class , contains methods to calculate statistics of raw data and tokenized data
        '''
        self.use_ftfy = use_ftfy
        self.ftfy_normalizer = ftfy_normalizer
        self.wikitext_detokenize = wikitext_detokenize
        self.pad_id = pad_id
        self.eos_id = eos_id

    def get_raw_stats(self, content: str) -> Dict[str, int]:
        '''
        Args:
            content: the string whose raw statistics will be calculated

        Returns:
            Dict[str, int]: The dictionary which will contain raw data statistics
        '''
        stats = {
            "raw_chars_count": 0,
            "raw_bytes_count": 0,
            "normalized_chars_count": 0,
            "normalized_bytes_count": 0,
        }

        stats['raw_chars_count'] = len(content)
        stats["raw_bytes_count"] += len(content.encode("utf-8"))
        cleaned_content = clean_text(
            content,
            self.use_ftfy,
            self.wikitext_detokenize,
            self.ftfy_normalizer,
        )
        stats["normalized_chars_count"] += len(cleaned_content)
        stats["normalized_bytes_count"] += len(cleaned_content.encode("utf-8"))

        return cleaned_content, stats

    def get_tokenized_stats(self, tokens: List) -> Dict[str, int]:
        """
        Get tokenized data statistics from the sample.

        Args:
            tokens (List): Tokenized sample.

        Returns:
            Dict[str, int]: Tokenized Data statistics.
        """
        stats = defaultdict(int)
        if tokens == []:
            return stats
        stats["num_pad_tokens"] = sum(
            ele == self.pad_id for ele in tokens['input_ids']
        )
        stats["non_pad_tokens"] = sum(
            ele != self.eos_id and ele != self.pad_id
            for ele in tokens['input_ids']
        )

        stats["num_tokens"] = len(tokens['input_ids'])
        stats["n_examples"] = 1

        return stats

    def combine_stats(
        self,
        stat_dict: Dict[str, int],
        raw_dict: Dict[str, int],
        tokenized_dict: Dict[str, int],
    ) -> Dict[str, int]:
        '''
        Args:
            stat_dict: Main dictionary which contains previous stats
            raw_dict: dictionary which contains raw data stats of current string
            tokenized_dict: dictionary which contains tokenized data stats of current string

        Returns:
            combined dictionary of all the data encountered so far.
        '''
        for key, val in raw_dict.items():
            stat_dict[key] += val

        for key, val in tokenized_dict.items():
            stat_dict[key] += val

        return stat_dict


class EmbedGenElement(BaseConfig):
    '''
    Class responsible for validating each element of the embed generation sda
    '''

    type: Literal['embedding', 'id']
    content: Union[str, int]

    @model_validator(mode='after')
    def check_type(self) -> Self:

        if self.type == 'id':
            if not isinstance(self.content, int):
                raise ValueError('Content for type id should be a digit')

        if self.type == 'embedding':
            if not isinstance(self.content, str):
                raise ValueError(
                    'Content for type embedding should be a string'
                )
        return self


class EmbedGenList(BaseConfig):
    '''
    Class responsible for validating the embed generation sda
    '''

    sda: List[EmbedGenElement]

    @model_validator(mode='after')
    def check_count_of_sda(self) -> Self:
        '''
        Num of embedding type entry in the list == 1
        Num of id type entry in the list == 1
        '''

        emb_count = 0
        id_count = 0

        for item in self.sda:
            if item.type == 'id':
                id_count += 1
            if item.type == 'embedding':
                emb_count += 1

        if emb_count != 1:
            raise ValueError(
                f'There should be exactly 1 embedding type entry in the list but found {emb_count}'
            )
        if id_count != 1:
            raise ValueError(
                f'There should be exactly 1 id type entry in the list but found {id_count}'
            )

        return self


class EmbedTrainElement(BaseConfig):
    '''
    Class responsible for validating each element of the embed training sda
    '''

    type: Literal['question', 'context_positive', 'context_negative']
    content: str


class EmbedTrainList(BaseConfig):
    '''
    Class responsible for validating the embed training sda
    '''

    sda: List[EmbedTrainElement]

    @model_validator(mode='after')
    def check_list(self) -> Self:

        question_count = 0
        ctx_positive_count = 0
        ctx_negative_count = 0

        for item in self.sda:
            if item.type == 'question':
                question_count += 1
            elif item.type == 'context_positive':
                ctx_positive_count += 1
            elif item.type == 'context_negative':
                ctx_negative_count += 1

        if question_count != 1:
            raise ValueError(
                f'There should be exactly 1 question type entry in the list but found {question_count}'
            )
        if ctx_positive_count != 1:
            raise ValueError(
                f'There should be exactly 1 positive context type entry in the list but found {ctx_positive_count}'
            )
        if ctx_negative_count < 1:
            raise ValueError(
                f'There should be more than 0 negative contexts type entry in the list but found {ctx_negative_count}'
            )

        return self


def normalize_msl(msl_value):
    if isinstance(msl_value, str):
        # Check if 'k' or 'K' is present
        if re.search(r'k', msl_value, flags=re.IGNORECASE):
            msl_value = re.sub(r'k', '', msl_value, flags=re.IGNORECASE)
            return int(float(msl_value) * 1024)
        # If no 'k' is found, treat it as a numeric string and return as an int
        return int(msl_value)

    # For numeric inputs (int/float)
    if not isinstance(msl_value, (int, float)):
        raise ValueError(f"Invalid msl value type: {type(msl_value)}")
    if int(msl_value) != msl_value:
        raise ValueError("Fractional msl values are not allowed.")
    return int(msl_value)


def check_and_create_dir(dir: Optional[str], split_dir: Optional[str]) -> str:
    """
    Ensures a directory exists, optionally handling a subdirectory. It prompts
    the user for action if the directory already has files.

    Args:
    dir (Optional[str]): Base directory path. Defaults to 'input_dir' in cwd.
    split_dir (Optional[str]): Subdirectory to add to the base directory.

    Returns:
    str: The final directory path ensured to exist.
    """
    # Set default directory if none provided
    if dir is None:
        dir = os.path.join(os.getcwd(), 'input_dir')

    # Append subdirectory if provided
    if split_dir:
        dir = os.path.join(dir, split_dir)

    # Check for existing files and handle existing files
    if os.path.isdir(dir) and os.listdir(dir):
        _in = input(
            "Input directory already contains file(s). Do you want to delete "
            "the folder and download the dataset again? "
            "(yes/no): "
        )
        if _in.lower() in ["y", "yes"]:
            shutil.rmtree(dir)
            os.makedirs(dir)  # Recreate directory after removal
        elif _in.lower() in ["n", "no"]:
            return dir
        else:
            raise ValueError(
                f"Inputs can be yes, no, y, or n. Received {_in}!!"
            )
    else:
        # Create directory if it does not exist
        os.makedirs(dir, exist_ok=True)

    return dir


def save_image_locally(example, idx, image_key, image_dir):
    """
    Saves image data locally to a specified directory, processing both individual
    and batched image data. Updates the dataset example with the relative paths
    of the saved images.

    Args:
    example (dict): A single dataset example containing the image data.
    idx (int): The index of the current example in the dataset.
    image_key (str): The key in the dataset example that contains the image data.
    image_dir (str): The directory where the images should be saved.

    Returns:
    dict: The updated dataset example with the `image_key` field containing
          the relative paths of saved images or their original paths if already a string.

    Raises:
    ValueError: If the image data is of an unsupported format.

    Notes:
    - If the image data is a list, it processes each element in the list and saves
      the images with filenames in the format `{idx}_{i}.png`, where `i` is the
      position of the image in the list.
    - If the image data is a single image, it saves the image with the filename
      `{idx}.png`.
    - If the image data is a string (e.g., an existing image path), it is retained as is.
    - Supports image data in `PIL.Image.Image` format for saving locally.
    """
    image_data = example[image_key]

    if isinstance(image_data, list):
        image_paths = []
        for i, img_data in enumerate(image_data):
            if img_data is None:
                image_paths.append(None)
            else:
                image_path = os.path.join(image_dir, f"{idx}_{i}.png")
                if img_data is None:
                    image_paths.append(None)
                    continue
                if isinstance(img_data, Image.Image):
                    img_data.save(image_path)
                    image_paths.append(f"{idx}_{i}.png")
                elif isinstance(img_data, str):
                    image_paths.append(img_data)
                else:
                    raise ValueError(
                        f" Image data format - {type(image_data)} is not supported"
                    )

        example[image_key] = image_paths
    else:
        if image_data is None:
            example[image_key] = None
        else:
            image_path = os.path.join(image_dir, f"{idx}.png")
            if isinstance(image_data, Image.Image):
                image_data.save(image_path)
                example[image_key] = f"{idx}.png"
            elif isinstance(image_data, str):
                example[image_key] = image_data
            else:
                raise ValueError(
                    f" Image data format - {type(image_data)} is not supported"
                )
    return example


def load_dataset_wrapper(
    input_data_params: Dict[str, Optional[str]], **kwargs
) -> str:
    """
    Loads a dataset from a specified source and saves it in a specified format
    in the given directory, potentially within a subdirectory denoted by a 'split'.

    Args:
    input_data_params (Dict[str, Optional[str]]): Parameters for dataset loading
        including 'source', 'split' (optional), and 'format'.
    **kwargs: Additional parameters.
        - image_key (str): Column name in the dataset that contains image paths.
        - image_dir (str): Directory to save processed images.
        - processes (int): Number of processes to use for parallel image processing.

    Returns:
    str: The directory where the dataset has been saved.

    Raises:
    ValueError: If the specified format is not supported or required parameters are missing.
    """
    from datasets import load_dataset

    split_type = input_data_params.pop('split', None)
    cache_dir = input_data_params.pop('cache_dir', None)
    cache_dir = check_and_create_dir(cache_dir, split_type)
    source_dataset = input_data_params.pop('source')

    # Validate split_type
    if split_type is None:
        raise ValueError(
            "A dataset split is required. Specify it in the 'split' key of input_data_params."
        )
    # Load the dataset
    dataset = load_dataset(
        source_dataset,
        split=split_type,
        cache_dir=cache_dir,
        **input_data_params,
    )

    # Handle image processing
    image_key = kwargs.get("image_key")
    if image_key and image_key in dataset.column_names:
        process_images_fn = partial(
            save_image_locally,
            image_key=image_key,
            image_dir=kwargs.get("image_dir"),
        )
        dataset = dataset.map(
            process_images_fn,
            with_indices=True,
            num_proc=kwargs.get("processes", 1),
        )

    # Determine and validate file format
    format_type = input_data_params.get('format', 'parquet')
    file_path = os.path.join(cache_dir, "data", f"dataset.{format_type}")
    if format_type == 'parquet':
        dataset.to_parquet(file_path)
    elif format_type == 'jsonl':
        dataset.to_json(file_path, orient='records', lines=True)
    else:
        raise ValueError(f"Unsupported format: {format_type}.")

    logger.info(f"Dataset saved in {format_type} format at {file_path}")
    return os.path.join(cache_dir, "data")


def get_compression_factor(filename: str) -> int:
    """
    Calculate and return the compression factor based on a file's extension.

    Args:
        filename (str): The name of the file.

    Returns:
        int: Compression factor. Returns 3 for all compressed and parquet formats,
             otherwise returns 1 for uncompressed formats.
    """
    compressed_formats = [
        ".jsonl.zst",
        ".jsonl.zst.tar",
        ".json.gz",
        ".parquet",
    ]

    for format in compressed_formats:
        if filename.endswith(format):
            return 3  # compression factor for compressed/parquet formats

    return 1  # default factor for uncompressed formats


def format_time(seconds):
    """
    Format seconds into a human-readable string showing hours:minutes:seconds,
    minutes:seconds, or seconds.
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h:{minutes:02d}m:{seconds:02d}s"
    elif minutes:
        return f"{minutes}m:{seconds:02d}s"
    else:
        return f"{seconds}s"


def update_progress(
    pbar: tqdm,
    progress_counter: Value,
    total_chunks: int,
    start_time: float,
    stop_event: Event,
    json_params_file: str = None,
    task: str = "preprocessing",
    data_stats: dict = None,
) -> None:
    """
    Update the progress bar based on the current progress.

    Args:
        pbar (tqdm): The progress bar instance.
        progress_counter (Value): A shared counter to track progress across processes.
        data_stats (dict): A dictionary of dataset statistics to diplay while execution of task.
        total_chunks (int): Total chunks to process.
        start_time (float): The start time of the process.
        stop_event (Event): Event to signal when to stop updating progress.
        task (str): The task for which the progress bar is being displayed for. Can be either `preprocessing` or `splitting`.

    Returns:
        None
    """

    pbar.total = total_chunks
    last_mtime = None
    while not stop_event.is_set():
        progress = progress_counter.value
        # Update total if actual progress exceeds estimated total_chunks
        if progress >= pbar.total:
            pbar.total = progress + 1  # Update the progress bar total

        if progress > pbar.n:
            num_processed = progress - pbar.n
            pbar.update(num_processed)
            elapsed_time = time.time() - start_time
            avg_time_per_chunk = elapsed_time / pbar.n
            remaining_chunks = pbar.total - pbar.n
            estimated_remaining = avg_time_per_chunk * remaining_chunks
            formatted_estimated_remaining = format_time(estimated_remaining)

            if task == "preprocessing":
                # Update the progress bar postfix with avg processing time and estimated time
                # Check if the file has been modified before reading
                current_mtime = os.path.getmtime(json_params_file)
                if last_mtime is None or current_mtime != last_mtime:
                    with open(json_params_file, "r") as _fin:
                        data = json.load(_fin)
                    last_mtime = current_mtime  # Update the last modified time
                postfix_items = OrderedDict(
                    avg_time=f"{avg_time_per_chunk:.3f}s/chunk",
                    eta=formatted_estimated_remaining,
                    discarded=data.get("post-process", {}).get(
                        "discarded_files", 0
                    ),
                    processed=data.get("post-process", {}).get(
                        "processed_files", 0
                    ),
                )
                # Update progress bar description with processed/total chunks
                pbar.set_description(f"Processing {pbar.n}/{pbar.total} chunks")
            else:

                ##Calculate remaining number of docs to split by estimating average number of docs per chunk.
                avg_docs_per_chunk = data_stats["split_docs"].value / progress
                remaining_docs = math.ceil(
                    remaining_chunks * avg_docs_per_chunk
                )
                # Update the progress bar postfix with avg processing time and estimated time
                postfix_items = OrderedDict(
                    avg_time=f"{avg_time_per_chunk:.3f}s/chunk",
                    eta=formatted_estimated_remaining,
                    split_docs=data_stats["split_docs"].value,
                    est_remaining_docs=remaining_docs,
                )
                # Update progress bar description with processed/total chunks
                pbar.set_description(f"Splitting")
            pbar.set_postfix(
                postfix_items,
                refresh=True,
            )
        time.sleep(0.5)


def calculate_total_size(input_files) -> int:
    """
    Calculate the total size of all input files, taking compression
    factors into consideration.

    Returns:
        int: The total size of all input files in bytes.
    """
    total_size = sum(
        os.path.getsize(file) * get_compression_factor(file)
        for file in input_files
    )
    return total_size


class YamlReader:
    def __init__(self, yaml_source: Union[str, dict]):
        self._yaml_path = None
        self.logger = logging.getLogger(f"YamlReader[{id(self)}]")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        if not self.logger.handlers:
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        if isinstance(yaml_source, str):
            self._yaml_path = yaml_source
            self.config = self._load_yaml()
        elif isinstance(yaml_source, dict):
            self.config = yaml_source

    def _load_yaml(self) -> dict:
        try:
            with open(self._yaml_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load YAML: {e}")
            raise

    def get(
        self,
        dotted_key: str,
        default: Optional[Any] = None,
        strict: bool = False,
    ) -> Any:
        keys = dotted_key.split(".")
        value = self.config
        for key in keys:
            try:
                value = value[key]
            except (KeyError, TypeError):
                if strict:
                    raise KeyError(f"Key '{dotted_key}' not found")
                if default is None:
                    self.logger.warning(
                        f"Missing key '{dotted_key}', returning default=None"
                    )
                return default
        return value

    def set(self, dotted_key: str, value: Any) -> Dict[str, Any]:
        keys = dotted_key.split(".")
        d = self.config
        for key in keys[:-1]:
            if key not in d or not isinstance(d[key], dict):
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value
        self.logger.info(f"Set '{dotted_key}' = {value}")
        return self.config

    @property
    def params(self) -> dict:
        return self.config


class SLURMPipe:
    def __init__(
        self,
        partition: str,
        log_dir: str = "slurm_logs",
        timeout_min: int = 30,
        cpus_per_task: int = 2,
        mem_gb: int = 4,
        # Additional SLURM configurations
        ntasks_per_node: int = 1,
        array_parallelism: Optional[int] = None,
        # Job management
        max_retries: int = 3,
        retry_delay: int = 60,
        launch_delay: int = 5,
    ):
        import submitit

        self.submitit = submitit
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        # Initialize executor with comprehensive parameters
        self.executor = submitit.AutoExecutor(folder=self.log_dir)

        # Build parameters dictionary
        params = {
            "timeout_min": timeout_min,
            "cpus_per_task": cpus_per_task,
            "mem_gb": mem_gb,
            "slurm_partition": partition,
            "slurm_ntasks_per_node": ntasks_per_node,
        }

        # Add optional parameters
        if array_parallelism:
            params["slurm_array_parallelism"] = array_parallelism

        self.executor.update_parameters(**params)

        # Job management
        self.jobs = []
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.launch_delay = launch_delay

        # Set up comprehensive logging
        self._setup_logging()

        # Job status tracking
        self.completed_jobs = []
        self.failed_jobs = []
        self.retried_jobs = {}

    def _setup_logging(self):
        self.logger = logging.getLogger("SLURMPipe")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            # Main log file
            main_handler = logging.FileHandler(self.log_dir / "slurm_pipe.log")
            main_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
            self.logger.addHandler(main_handler)

            # Console handler for real-time feedback
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            self.logger.addHandler(console_handler)

    def launch(self, fn: Callable, *args, **kwargs) -> object:
        for attempt in range(self.max_retries + 1):
            try:
                job = self.executor.submit(fn, *args, **kwargs)
                self.logger.info(
                    f"Launched job {job.job_id} for function {fn.__name__} "
                    f"(attempt {attempt + 1}/{self.max_retries + 1})"
                )
                self.jobs.append(job)
                return job

            except Exception as e:
                if attempt < self.max_retries:
                    self.logger.warning(
                        f"Job launch failed (attempt {attempt + 1}): {e}. "
                        f"Retrying in {self.retry_delay} seconds..."
                    )
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(
                        f"Job launch failed after {self.max_retries + 1} attempts: {e}"
                    )
                    raise

    def launch_batch(
        self, fn: Callable, param_list: List[tuple]
    ) -> List[object]:
        jobs = []
        for i, params in enumerate(param_list):
            args, kwargs = (
                params
                if isinstance(params, tuple) and len(params) == 2
                else (params, {})
            )
            try:
                job = self.launch(fn, *args, **kwargs)
                jobs.append(job)
                time.sleep(self.launch_delay)
            except Exception as e:
                self.logger.error(f"Failed to launch job {i}: {e}")

        return jobs

    def track_jobs(
        self, poll_interval: int = 30, detailed_status: bool = True
    ) -> Dict[str, List]:
        if not self.jobs:
            self.logger.warning("No jobs to track")
            return {"completed": [], "failed": []}

        self.logger.info(f"Tracking {len(self.jobs)} jobs...")

        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            self.logger.info(
                f"Received signal {signum}. Attempting to cancel running jobs..."
            )
            self.cancel_all_jobs()
            sys.exit(1)

        # Register handlers for all signals that might terminate the process
        for sig in [
            signal.SIGINT,
            signal.SIGTERM,
            signal.SIGHUP,
            signal.SIGQUIT,
        ]:
            signal.signal(sig, signal_handler)

        completed_jobs = []
        failed_jobs = []

        try:
            while self.jobs:
                running_jobs = []

                for job in self.jobs:
                    try:
                        if job.done():
                            if job.exception() is None:
                                result = job.result()
                                self.logger.info(
                                    f"Job {job.job_id} completed successfully"
                                )
                                if detailed_status:
                                    self.logger.info(
                                        f"Result: {str(result)[:200]}..."
                                    )
                                completed_jobs.append(job)
                                self.completed_jobs.append(job)
                            else:
                                error = job.exception()
                                self.logger.error(
                                    f"Job {job.job_id} failed with exception: {error}"
                                )
                                failed_jobs.append(job)
                                self.failed_jobs.append(job)
                        else:
                            running_jobs.append(job)
                            if detailed_status:
                                self.logger.info(
                                    f"Job {job.job_id} status: {job.state}"
                                )
                    except Exception as e:
                        self.logger.error(
                            f"Error checking job {job.job_id}: {e}"
                        )
                        failed_jobs.append(job)

                self.jobs = running_jobs

                if self.jobs:
                    self.logger.debug(
                        f"Jobs remaining: {len(self.jobs)}, "
                        f"Completed: {len(completed_jobs)}, "
                        f"Failed: {len(failed_jobs)}"
                    )
                    time.sleep(poll_interval)

        except KeyboardInterrupt:
            self.logger.info("Job tracking interrupted by user")
            self.cancel_all_jobs()

        # Final summary
        self.logger.info(
            f"Job tracking complete. "
            f"Completed: {len(completed_jobs)}, Failed: {len(failed_jobs)}"
        )

        return {"completed": completed_jobs, "failed": failed_jobs}

    def cancel_all_jobs(self):
        cancelled_count = 0
        for job in self.jobs:
            try:
                if not job.done():
                    job.cancel()
                    cancelled_count += 1
                    self.logger.info(f"Cancelled job {job.job_id}")
            except Exception as e:
                self.logger.error(f"Failed to cancel job {job.job_id}: {e}")

        self.logger.info(f"Cancelled {cancelled_count} jobs")

    def get_job_status_summary(self) -> Dict[str, int]:
        """Get summary of job statuses"""
        status_counts = {
            "running": 0,
            "completed": 0,
            "failed": 0,
            "pending": 0,
        }

        for job in self.jobs:
            try:
                if job.done():
                    if job.exception() is None:
                        status_counts["completed"] += 1
                    else:
                        status_counts["failed"] += 1
                else:
                    state = job.state.lower()
                    if "running" in state:
                        status_counts["running"] += 1
                    else:
                        status_counts["pending"] += 1
            except Exception:
                status_counts["failed"] += 1

        return status_counts

    def wait_for_completion(
        self, timeout: Optional[int] = None
    ) -> Dict[str, List]:
        if not self.jobs:
            self.logger.warning("No jobs to wait for")
            return {"completed": [], "failed": []}

        start_time = time.time()

        while self.jobs:
            if timeout and (time.time() - start_time) > timeout:
                self.logger.warning(
                    f"Timeout reached ({timeout}s). Some jobs may still be running."
                )
                # Return current state of completed and failed jobs
                return {
                    "completed": self.completed_jobs.copy(),
                    "failed": self.failed_jobs.copy(),
                }

            # Use track_jobs to handle the polling and return results when done
            result = self.track_jobs(poll_interval=30, detailed_status=False)

            # If track_jobs returns (meaning all jobs are done), return the result
            if not self.jobs:
                return result

        # This should not be reached, but included for completeness
        return {
            "completed": self.completed_jobs.copy(),
            "failed": self.failed_jobs.copy(),
        }


from multiprocessing import Event as MEvent


class MultiprocessingExitEvent:
    """
    A simple wrapper around multiprocessing.Event environments.
    """

    def __init__(self, name=None):
        self._event = MEvent()
        self.name = name or "exit-event"

    def set(self):
        """Set the exit flag"""
        logger.info(f"Setting local multiprocessing exit event: {self.name}")
        self._event.set()

    def is_set(self):
        """Check if the exit flag is set"""
        return self._event.is_set()

    def clear(self):
        """Clear the exit flag"""
        self._event.clear()


@dataclass
class NodeProgress:
    node_id: str
    max_step: int = 0
    total_steps: int = 0
    last_update: float = 0
    processes: Dict[int, int] = None

    def __post_init__(self):
        if self.processes is None:
            self.processes = {}

    @property
    def progress_percent(self) -> float:
        # Calculate based on sum of all process chunks
        total_processed = sum(self.processes.values())
        return (
            (total_processed / self.total_steps * 100)
            if self.total_steps > 0
            else 0.0
        )

    @property
    def is_stalled(self) -> bool:
        return (
            time.time() - self.last_update > 300
            if self.last_update > 0
            else False
        )


class ProgressMonitor:
    """
    Progress monitor class
    """

    def __init__(
        self,
        checkpoint_dir: str,
        expected_nodes: List[str],
        processes_per_node: int,
        total_chunks: int,
        refresh_rate: float = 1.0,
        stall_threshold: float = 300.0,  # 5 minutes
        logger: Optional[logging.Logger] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.expected_nodes = expected_nodes
        self.processes_per_node = processes_per_node
        self.total_chunks = total_chunks
        self.refresh_rate = max(0.5, refresh_rate)  # Minimum 0.5s refresh
        self.stall_threshold = stall_threshold
        self.logger = logger or logging.getLogger(__name__)

        # Threading and control
        self._thread = None
        self._running = False
        self._lock = threading.RLock()

        # Progress tracking
        self._node_progress: Dict[str, NodeProgress] = {}
        self._last_file_checks: Dict[str, float] = {}
        self._console = Console()

        # Performance optimization
        self._file_cache_ttl = 2.0  # Cache file reads for 2 seconds

        # Initialize node progress
        self._init_node_progress()

    def _init_node_progress(self):
        with self._lock:
            for node in self.expected_nodes:
                self._node_progress[node] = NodeProgress(
                    node_id=node, total_steps=self.total_chunks
                )

    def _read_checkpoint_file(self, filepath: Path) -> Optional[int]:
        """
        Read checkpoint file and return the last value (current chunk being processed).
        """
        try:
            # Simple caching to avoid excessive file I/O
            current_time = time.time()
            cache_key = str(filepath)

            if (
                cache_key in self._last_file_checks
                and current_time - self._last_file_checks[cache_key]
                < self._file_cache_ttl
            ):
                return None  # Skip read, too recent

            self._last_file_checks[cache_key] = current_time

            if not filepath.exists():
                return None

            with open(filepath, 'r') as f:
                line = f.readline().strip()
                if not line:
                    return None

                parts = line.split(",")
                if len(parts) >= 3:
                    # Return the last value (current chunk being processed)
                    current_chunk = int(parts[-1].strip())
                    return current_chunk

        except (IOError, ValueError, IndexError) as e:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Error reading checkpoint {filepath}: {e}")

        return None

    def _update_node_progress(self):
        with self._lock:
            for node in self.expected_nodes:
                node_progress = self._node_progress[node]
                updated = False

                for process_id in range(self.processes_per_node):
                    filename = f"checkpoint_process_{process_id}_{node}.txt"
                    filepath = self.checkpoint_dir / filename

                    current_chunk = self._read_checkpoint_file(filepath)
                    if current_chunk is not None:
                        # Update process progress with current chunk count
                        old_chunk = node_progress.processes.get(process_id, 0)
                        if current_chunk > old_chunk:
                            node_progress.processes[process_id] = current_chunk
                            updated = True

                # Update node status if any process was updated
                if updated:
                    node_progress.last_update = time.time()

    def _build_summary_table(self) -> Table:
        table = Table(
            title="Multi-node Progress Monitor", title_style="bold blue"
        )
        table.add_column("Node", style="cyan", no_wrap=True)
        table.add_column("Progress", justify="right")
        table.add_column("Completion", justify="right")
        table.add_column("Status", justify="center")

        with self._lock:
            total_completed = 0
            total_expected = len(self.expected_nodes) * self.total_chunks

            for node in self.expected_nodes:
                progress = self._node_progress[node]

                # Calculate total chunks processed for this node
                total_chunks_processed = sum(progress.processes.values())

                # Determine status
                if total_chunks_processed >= self.total_chunks:
                    status = "[green]✓ Complete[/green]"
                elif progress.is_stalled:
                    status = "[yellow]⚠ Stalled[/yellow]"
                elif total_chunks_processed > 0:
                    status = "[blue]→ Running[/blue]"
                else:
                    status = "[dim]○ Waiting[/dim]"

                table.add_row(
                    f"Node {node}",
                    f"{total_chunks_processed:,}/{self.total_chunks:,}",
                    f"{progress.progress_percent:.1f}%",
                    status,
                )

                total_completed += total_chunks_processed

            # Add summary row
            overall_percent = (
                (total_completed / total_expected * 100)
                if total_expected > 0
                else 0
            )
            table.add_section()
            table.add_row(
                "[bold]Overall[/bold]",
                f"[bold]{total_completed:,}/{total_expected:,}[/bold]",
                f"[bold]{overall_percent:.1f}%[/bold]",
                (
                    "[bold green]✓[/bold green]"
                    if overall_percent >= 100
                    else "[bold blue]→[/bold blue]"
                ),
            )

        return table

    def _create_layout(self) -> Layout:
        layout = Layout()
        layout.add_split(Layout(self._build_summary_table(), name="summary"))
        return layout

    def _run_monitor(self):
        with Live(
            self._create_layout(),
            refresh_per_second=1 / self.refresh_rate,
            console=self._console,
        ) as live:
            while self._running:
                try:
                    self._update_node_progress()
                    live.update(self._create_layout())
                    time.sleep(self.refresh_rate)
                except Exception as e:
                    self.logger.error(f"Error in progress monitor: {e}")
                    time.sleep(self.refresh_rate)

    def start(self):
        if self._thread and self._thread.is_alive():
            self.logger.warning("Progress monitor already running")
            return

        if self.total_chunks <= 0:
            raise ValueError("total_chunks must be set to a positive integer")

        self._running = True
        self._thread = threading.Thread(target=self._run_monitor, daemon=True)
        self._thread.start()
        self.logger.info("Progress monitor started")

    def stop(self):
        if not self._running:
            return

        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)  # 5 second timeout
            if self._thread.is_alive():
                self.logger.warning(
                    "Progress monitor thread did not stop gracefully"
                )

        self.logger.info("Progress monitor stopped")

    def get_progress_summary(self) -> Dict:
        with self._lock:
            summary = {
                'nodes': {},
                'overall': {
                    'total_completed': 0,
                    'total_expected': len(self.expected_nodes)
                    * self.total_chunks,
                    'completion_percent': 0.0,
                },
            }

            total_completed = 0
            for node in self.expected_nodes:
                progress = self._node_progress[node]
                total_chunks_processed = sum(progress.processes.values())

                summary['nodes'][node] = {
                    'total_chunks_processed': total_chunks_processed,
                    'total_steps': progress.total_steps,
                    'completion_percent': progress.progress_percent,
                    'is_stalled': progress.is_stalled,
                    'processes': dict(progress.processes),
                }
                total_completed += total_chunks_processed

            summary['overall']['total_completed'] = total_completed
            summary['overall']['completion_percent'] = (
                total_completed / summary['overall']['total_expected'] * 100
                if summary['overall']['total_expected'] > 0
                else 0
            )

            return summary
