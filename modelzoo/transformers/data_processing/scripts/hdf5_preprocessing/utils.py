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
import gzip
import io
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from itertools import repeat
from math import ceil
from multiprocessing import Pool
from pathlib import Path

import h5py
import jsonlines
import numpy as np
import yaml
import zstandard
from lm_dataformat import listdir_or_file, tarfile_reader
from tqdm import tqdm

from modelzoo.transformers.data_processing.utils import split_list

logger = logging.getLogger("utils")
logger.setLevel(logging.INFO)


def add_common_args(parser):
    """
    For the argparse to parse arguments for subcommands, we add common
    command line arguments to each subcommand parser here.
    """
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Path to the YAML config file for setting dataset preprocessing hyper-parameters.",
    )
    parser.add_argument(
        "--input_dir", type=str, help="Directory where raw data is stored.",
    )
    parser.add_argument(
        "--metadata_files",
        type=str,
        default=None,
        help="Path to text file containing a list of file names "
        "corresponding to the raw input documents to be "
        "processed and stored; can handle multiple metadata files "
        "separated by comma.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory where HDF5 files will be stored.",
    )
    parser.add_argument(
        "--processes", type=int, help="Number of processes to use.",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        choices=["GPT2Tokenizer", "NeoXTokenizer", "HuggingFaceTokenizer"],
        help=(
            "Type of tokenizer to use for HDF5 dataset generation. "
            "Can be one of `GPT2Tokenizer`, `NeoXTokenizer` or `HuggingFaceTokenizer`."
        ),
    )
    parser.add_argument(
        "--huggingface_tokenizer",
        type=str,
        default=None,
        help=(
            "Name/Path to the HuggingFace tokenizer."
            "Only used when tokenizer_type=HuggingFaceTokenizer"
        ),
    )
    parser.add_argument(
        "--vocab_file", type=str, help="Path to the vocabulary file."
    )
    parser.add_argument(
        "--encoder_file", type=str, help="Path to the encoder file."
    )
    parser.add_argument(
        "--eos_id", type=int, help="Token id of the end of sentence token",
    )
    parser.add_argument(
        "--pad_id", type=int, help="Token id of the padding token."
    )
    parser.add_argument(
        "--max_seq_length", type=int, help="Maximum sequence length.",
    )
    parser.add_argument(
        "--short_seq_prob",
        type=float,
        default=0.0,
        help=(
            "Probability of creating sequences which are shorter than the"
            + " maximum sequence length."
        ),
    )
    parser.add_argument(
        "--use_ftfy",
        type=str,
        choices=["True", "False"],
        help="Whether to fix text with ftfy. Defaults to `True`.",
    )
    parser.add_argument(
        "--ftfy_normalizer",
        type=str,
        choices=["NFC", None],
        help=(
            "Choose what kind of unicode normalization is applied. Usually, we "
            "apply `NFC` normalization, so that letters followed by combining "
            "characters become single combined characters. Using `None` "
            "applies no normalization while fixing text."
        ),
    )
    parser.add_argument(
        "--wikitext_detokenize",
        type=str,
        choices=["True", "False"],
        help="Whether to use wikitext detokenizer to fix text. Defaults to `False`.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="examples",
        help=(
            "Name of the dataset; i.e. prefix to use for HDF5 file names."
            + "Defaults to `examples`."
        ),
    )
    parser.add_argument(
        "--files_per_record",
        type=int,
        help="Text files to write per HDF5 file.",
    )
    parser.add_argument(
        "--write_in_batch",
        type=str,
        choices=["True", "False"],
        help="Whether to write the samples in batch for the HDF5 format, "
        "setting to false will save memory but a bit slower. Defaults to "
        "`True`.",
    )
    parser.add_argument(
        "--write_remainder",
        type=str,
        choices=["True", "False"],
        help="Write the remainder files when data is left over from "
        "processing. Defaults to `True`.",
    )
    parser.add_argument(
        "--pack_sequences",
        type=str,
        choices=["True", "False"],
        help="Concatenate a document smaller than maximum sequence length with "
        "other documents, instead of filling it with Padding token. Defaults "
        "to `True`.",
    )
    parser.add_argument(
        "--min_sequence_len",
        type=int,
        default=6,
        help=(
            "sequences shorter than min_sequence_len tokens in length will be skipped"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        choices=["True", "False"],
        help="Resume record writing from a given checkpoint. Defaults to `False`.",
    )
    parser.add_argument(
        "--display_pbar",
        type=str,
        choices=["True", "False"],
        help="Display progress while runs. Defaults to `False`.",
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed.",
    )


def get_parser(desc):

    """Argparser definition for command line arguments from user.

    Returns:
        Argparse namespace object with command line arguments.
    """
    parser = argparse.ArgumentParser(description=desc)
    subparser = parser.add_subparsers(
        description="Sub command for HDF5 conversion.",
        dest="mode",
        required=True,
        help="Sub command to choose saving the raw text into HDF5 files or "
        "pre-processed text converted into token ids at desired maximum "
        "sequence length.",
    )

    lm_parser = subparser.add_parser(
        "LMData", help="Language modeling dataset in `.jsonl` or `.txt` format."
    )
    add_common_args(lm_parser)
    lm_parser.add_argument(
        "--jsonl_key",
        type=str,
        default=None,
        help="The key name in input jsonl files from which the raw text will be "
        "extracted in order to further process it.",
    )
    lm_parser.add_argument(
        "--split_text_to_tokenize",
        type=str,
        choices=["True", "False"],
        help="Whether to split the text into smaller chunks before tokenizing.",
    )
    lm_parser.add_argument(
        "--chunk_len_to_split",
        type=int,
        help="Length of the chunk size to split the text document into.",
    )
    lm_parser.add_argument(
        "--remove_bos_in_chunks",
        type=str,
        choices=["True", "False"],
        help="Whether to ignore bos token id in chunks when splitting the text.",
    )

    summarization_parser = subparser.add_parser(
        "Summarization", help="Fine-tuning dataset in plane text format."
    )
    add_common_args(summarization_parser)
    summarization_parser.add_argument(
        "--sep_token",
        type=str,
        default=None,
        help="Token added between prompt and completion in preprocessed sequences.",
    )
    summarization_parser.add_argument(
        "--prompt_key", type=str, help="Json key for the prompt.",
    )
    summarization_parser.add_argument(
        "--completion_key", type=str, help="Json key for the completion.",
    )

    custom_parser = subparser.add_parser(
        "Customize", help="Provide customized dataset processor."
    )
    add_common_args(custom_parser)
    custom_parser.add_argument(
        "--module",
        type=str,
        help="Python file name contains the custom dataset processor.",
    )
    custom_parser.add_argument(
        "--dataset_processor",
        type=str,
        help="Name of the custom dataset processor.",
    )

    return parser.parse_args()


def update_params(params, args):
    """
    Update config parameters with CLI arguments
    """
    setup_params = [
        "input_dir",
        "metadata_files",
        "output_dir",
        "processes",
        "module",
        "dataset_processor",
    ]
    processing_params = [
        "tokenizer_type",
        "huggingface_tokenizer",
        "vocab_file",
        "encoder_file",
        "eos_id",
        "pad_id",
        "split_text_to_tokenize",
        "chunk_len_to_split",
        "remove_bos_in_chunks",
        "max_seq_length",
        "short_seq_prob",
        "output_name",
        "files_per_record",
        "write_in_batch",
        "write_remainder",
        "resume_from_checkpoint",
        "display_pbar",
        "seed",
    ]
    dataset_params = [
        "use_ftfy",
        "ftfy_normalizer",
        "wikitext_detokenize",
        "jsonl_key",
        "pack_sequences",
        "min_sequence_len",
        "input_ids_dtype",
        "input_mask_dtype",
        "inverted_mask",
        "prompt_key",
        "completion_key",
        "sep_token",
    ]
    processor_map = {
        "lmdata": "LMDataPreprocessor",
        "summarization": "SummarizationPreprocessor",
    }

    mode = args.pop("mode").lower()
    if mode != "customize":
        params["setup"]["dataset_processor"] = processor_map[mode]

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
            else:
                raise ValueError(f"Unexpected arguments: {key}")

    set_defaults(params)


def set_defaults(params):
    params["processing"]["eos_id"] = params["processing"].get("eos_id")
    params["processing"]["pad_id"] = params["processing"].get("pad_id")
    params["processing"]["split_text_to_tokenize"] = params["processing"].get(
        "split_text_to_tokenize", False
    )
    params["processing"]["chunk_len_to_split"] = params["processing"].get(
        "chunk_len_to_split", 2000
    )
    params["processing"]["remove_bos_in_chunks"] = params["processing"].get(
        "remove_bos_in_chunks", False
    )
    params["processing"]["write_in_batch"] = params["processing"].get(
        "write_in_batch", True
    )
    params["processing"]["write_remainder"] = params["processing"].get(
        "write_remainder", True
    )
    params["processing"]["resume_from_checkpoint"] = params["processing"].get(
        "resume_from_checkpoint", False
    )
    params["processing"]["display_pbar"] = params["processing"].get(
        "display_pbar", False
    )
    params["dataset"]["use_ftfy"] = params["dataset"].get("use_ftfy", True)
    params["dataset"]["ftfy_normalizer"] = params["dataset"].get(
        "ftfy_normalizer", "NFC"
    )
    params["dataset"]["wikitext_detokenize"] = params["dataset"].get(
        "wikitext_detokenize", False
    )
    params["dataset"]["pack_sequences"] = params["dataset"].get(
        "pack_sequences", True
    )


def get_params(desc):
    """Retrieve configuration parameters
    Returns:
        params (Dict): Dictionary contains the parameters used to configure
            the data processing.
    """
    args = get_parser(desc)
    args = vars(args)

    params_file = args.pop("params", None)
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


def dump_args(args, json_params_file):
    """
    Write the input params to file.
    """
    logger.info(f"User arguments can be found at {json_params_file}.")

    # write initial params to file
    with open(json_params_file, "w") as _fout:
        json.dump(args, _fout, indent=4, sort_keys=True)


def dump_result(
    results,
    dataset_stats,
    json_params_file,
    eos_id=None,
    pad_id=None,
    vocab_size=None,
):
    """
    Write outputs of execution
    """
    with open(json_params_file, "r") as _fin:
        data = json.load(_fin)

    post_process = {}
    post_process["discarded_files"] = results["discarded"]
    post_process["processed_files"] = results["processed"]
    post_process["successful_files"] = results["successful"]
    post_process["n_examples"] = results["examples"]
    post_process["raw_chars_count"] = results["raw_chars_count"]
    post_process["raw_bytes_count"] = results["raw_bytes_count"]

    if eos_id is not None:
        post_process["eos_id"] = eos_id
    if pad_id is not None:
        post_process["pad_id"] = pad_id
    if vocab_size is not None:
        post_process["vocab_size"] = vocab_size

    data["post-process"] = post_process
    data["h5_dataset_stats"] = asdict(dataset_stats)

    with open(json_params_file, "w") as _fout:
        json.dump(data, _fout, indent=4, sort_keys=True)


@dataclass
class VerificationArgs:
    processes: int
    files_per_record: int
    max_seq_length: int
    tokenizer_obj: object
    eos_id: int
    pad_id: int


def get_verification_args(processes, data_processor):
    """Get arguments for verifying HDF5 dataset.
    Args:
        params (dict): Dictionary containing parameters for verifying HDF5 dataset.
        data_processor: Class containing methods that specify how the dataset
            will be processed and written into HDF5 files.
    """
    verification_args = VerificationArgs(
        processes,
        data_processor.files_per_record,
        data_processor.max_seq_length,
        data_processor.tokenizer,
        data_processor.eos_id,
        data_processor.pad_id,
    )

    return verification_args


def process_dataset(files, dataset_processor, processes):
    """Process a dataset and write it into HDF5 format.

    Args:
        files (list): List of files to process.
        dataset_processor: Class containing methods that specify how the dataset
            will be processed and written into HDF5 files.
        processes (int): Number of processes to use.

    Returns:
        Dictionary containing results of execution, specifically as number of
            processed, discarded, and successful files as well as number of examples 
            from all processes.
    """
    if processes < 2:
        # Run only single process run, with process number set as 0.
        return dataset_processor.create_dataset((files, 0))

    try:
        n_proc = processes
        n_chunks = ceil(len(files) / n_proc)
        remain = len(files) % n_proc
        if n_chunks == 1 and remain:
            n_proc = remain
            logger.warning(
                f"There aren't enough files to distribute to {processes} "
                f"processes, resetting it to {n_proc}. If you're working with a "
                "small number of compressed archives and could extract it into "
                "txt files, you might be able to get more benefits from the "
                f"available {processes} processes."
            )
        files = split_list(files, n_chunks)
    except ValueError as e:
        # We hit errors in two potential scenarios,
        # 1) Files is an empty list, in which case there is nothing to split
        # 2) There are more processes than files, in which case we cannot split
        #    the files to processes correctly, as there will be many idle
        #    processes which are not doing anything.
        logger.error(e)
        raise

    with Pool(processes=n_proc) as pool:
        pbar = tqdm(
            pool.imap(
                dataset_processor.create_dataset,
                zip(files, range(len(files)),),
            ),
            total=len(files),
        )
        meta = {
            "discarded": 0,
            "processed": 0,
            "successful": 0,
            "examples": 0,
            "raw_chars_count": 0,
            "raw_bytes_count": 0,
        }
        for results in pbar:
            pbar.update()
            for k, v in results.items():
                meta[k] += v

        return meta


@dataclass
class DatasetStats:
    num_sequences: int
    num_tokens: int
    detokenized_bytes: int
    detokenized_chars: int
    non_pad_tokens: int
    loss_valid_tokens: int


def collect_stats(data_arr, args):
    """Collect statistics of the dataset.

    Args:
        data_arr (numpy.ndarray): Numpy array containing the dataset.
        args (ValidationArgs): Arguments for verifying HDF5 dataset.
    """
    num_sequences = data_arr.shape[0]
    num_tokens = data_arr.shape[0] * data_arr.shape[2]
    non_pad_tokens = np.logical_and(
        data_arr[:, 0, :] != args.eos_id, data_arr[:, 0, :] != args.pad_id
    ).sum()
    loss_valid_tokens = data_arr[:, 1, :].sum()
    detokenized_bytes = 0
    detokenized_chars = 0

    for i in range(data_arr.shape[0]):
        line_str = args.tokenizer_obj.decode(data_arr[i, 0, :])
        detokenized_bytes += len(line_str.encode("utf-8"))
        detokenized_chars += len(line_str)

    return DatasetStats(
        num_sequences,
        num_tokens,
        detokenized_bytes,
        detokenized_chars,
        int(non_pad_tokens),  # cast to int to support saving to json
        int(loss_valid_tokens),  # cast to int to support saving to json
    )


def verify_saved_hdf5_files(params):
    """
    This function is used to do sanity checks at the end of the creation
    of hdf5 files.
    This function loads every .h5 files generated and checks:
        1. The data type
        2. Shape of the dataset
        3. Fact that labels and inputs are as expected
    """
    h5_files_path, args, vocab_size = params
    h5_stats = DatasetStats(
        0, 0, 0, 0, 0, 0
    )  # stats over list of files in a process
    for h5_file_path in h5_files_path:
        with h5py.File(h5_file_path, mode="r") as h5_file:
            n_examples = h5_file.attrs["n_examples"]
            dataset = h5_file["data"]
            data_arr = dataset[()]
            expected_dtype = "i4"
            assert dataset.dtype == expected_dtype, (
                f"Error in {h5_file}, conversion is corrupted as the "
                f"datatype is unexpected. Expected: {expected_dtype}, "
                f"received {dataset.dtype}."
            )
            data_shape = data_arr.shape
            assert (
                data_shape[1:] == (3, args.max_seq_length)
                or args.max_seq_length == -1
            ), (
                f"Error in {h5_file}, conversion is corrupted as the "
                f"shape of example is unexpected. Expected:"
                f" {(3, args.max_seq_length)}, received {data_shape[1:]}."
            )
            assert (data_arr < vocab_size).all(), (
                f"Error in {h5_file}, conversion is corrupted as the "
                f"input ids are greater than vocab size."
                f"Please ensure that a correct tokenizer is used "
                f"and the eos_id and pad_id are correct within the "
                f"tokenizer vocabulary size."
            )
            file_stats = collect_stats(data_arr, args)
            assert n_examples == file_stats.num_sequences, (
                f"Error in {h5_file}, conversion is corrupted as the "
                f"number of examples in file is unexpected. Expected:"
                f" {n_examples}, collected {file_stats.num_sequences}."
            )
            assert file_stats.num_tokens == n_examples * args.max_seq_length, (
                f"Error in {h5_file}, conversion is corrupted as the "
                f"number of tokens in file is unexpected. Expected:"
                f" {n_examples * args.max_seq_length}, collected "
                f"{file_stats.num_tokens}."
            )
        h5_stats.num_sequences += file_stats.num_sequences
        h5_stats.num_tokens += file_stats.num_tokens
        h5_stats.detokenized_bytes += file_stats.detokenized_bytes
        h5_stats.detokenized_chars += file_stats.detokenized_chars
        h5_stats.non_pad_tokens += file_stats.non_pad_tokens
        h5_stats.loss_valid_tokens += file_stats.loss_valid_tokens

    return h5_stats


def verify_saved_hdf5_files_mp(files, args, vocab_size):
    """Verify the generated HDF5 dataset.
    Args:
        files (list): List of files to process.
        args (VerificationArgs): Arguments for verifying HDF5 dataset.
        vocab_size (int): Size of the vocabulary from data_processor.
    """
    if args.processes == 1:
        return verify_saved_hdf5_files((files, args, vocab_size))

    try:
        n_proc = args.processes
        n_chunks = ceil(len(files) / n_proc)
        remain = len(files) % n_proc
        if n_chunks == 1 and remain:
            n_proc = remain
            logger.warning(
                f"There aren't enough files to distribute to {args.processes} "
                f"processes, resetting it to {n_proc}."
            )
        files = split_list(files, n_chunks)
    except ValueError as e:
        # We hit errors in one potential scenario:
        # Files is an empty list, in which case there is nothing to split
        logger.error(e)
        raise

    dataset_stats = DatasetStats(0, 0, 0, 0, 0, 0)

    with Pool(processes=n_proc) as pool:
        pbar = tqdm(desc="Verifying HDF5 files", total=len(files),)
        for stats in pool.imap(
            verify_saved_hdf5_files,
            zip(files, repeat(args), repeat(vocab_size),),
        ):
            dataset_stats.num_sequences += stats.num_sequences
            dataset_stats.num_tokens += stats.num_tokens
            dataset_stats.detokenized_bytes += stats.detokenized_bytes
            dataset_stats.detokenized_chars += stats.detokenized_chars
            dataset_stats.non_pad_tokens += stats.non_pad_tokens
            dataset_stats.loss_valid_tokens += stats.loss_valid_tokens
            pbar.update()

    return dataset_stats


def handle_jsonl(
    jsonl_reader, get_meta, autojoin_paragraphs, para_joiner, key=None
):
    for ob in jsonl_reader:
        # naive jsonl where each object is just the string itself, with no meta. For legacy compatibility.
        if isinstance(ob, str):
            assert not get_meta
            yield ob
            continue

        if key == None:
            yield ob
        else:
            text = ob[key]

            if autojoin_paragraphs and isinstance(text, list):
                text = para_joiner.join(text)

            if get_meta:
                yield text, (ob['meta'] if 'meta' in ob else {})
            else:
                yield text


# Slightly modified version of the Reader class from lm_dataformat.
# from https://github.com/leogao2/lm_dataformat/blob/master/lm_dataformat/__init__.py
class Reader:
    def __init__(self, in_path):
        self.in_path = in_path

    def stream_data(self, get_meta=False, jsonl_key=None):
        self.f_name = ""
        files = listdir_or_file(self.in_path)
        if not files:
            raise FileNotFoundError(f"No valid file(s) found in {self.in_path}")
        for f in files:
            self.f_name = f
            if f.endswith('.jsonl'):
                yield from self.read_jsonl(f, get_meta, key=jsonl_key)
            elif f.endswith('.jsonl.zst'):
                yield from self.read_jsonl_zst(f, get_meta, key=jsonl_key)
            elif f.endswith('.jsonl.zst.tar'):
                yield from self.read_jsonl_tar(f, get_meta, key=jsonl_key)
            elif f.endswith('.json.zst'):
                assert not get_meta

                yield from self.read_json(f)
            elif f.endswith('.txt'):
                assert not get_meta

                yield from self.read_txt(f)
            elif f.endswith('.json.gz'):
                assert not get_meta

                yield from self.read_jsongz(f)
            else:
                # shouldn't be reached
                print(
                    f'Skipping {f} as streaming for that filetype is not implemented'
                )

    def read_txt(self, file):
        with open(file, 'r') as fh:
            yield fh.read()

    def read_gz(self, file):
        with gzip.open(file, 'rb') as f:
            for line in f:
                yield line.decode('utf-8')

    def read_jsongz(self, file):
        for line in self.read_gz(file):
            yield json.loads(line)

    def read_json(self, file):
        with open(file, 'rb') as fh:
            cctx = zstandard.ZstdDecompressor()
            reader = cctx.stream_reader(fh)
            ob = json.load(reader)
            yield from ob

    def read_jsonl(
        self,
        file,
        get_meta=False,
        autojoin_paragraphs=True,
        para_joiner='\n\n',
        key=None,
    ):
        with jsonlines.open(file) as rdr:
            yield from handle_jsonl(
                rdr, get_meta, autojoin_paragraphs, para_joiner, key
            )

    def read_jsonl_zst(
        self,
        file,
        get_meta=False,
        autojoin_paragraphs=True,
        para_joiner='\n\n',
        key=None,
    ):
        with open(file, 'rb') as fh:
            cctx = zstandard.ZstdDecompressor()
            reader = io.BufferedReader(cctx.stream_reader(fh))
            rdr = jsonlines.Reader(reader)
            yield from handle_jsonl(
                rdr, get_meta, autojoin_paragraphs, para_joiner, key
            )

    def read_jsonl_tar(
        self,
        file,
        get_meta=False,
        autojoin_paragraphs=True,
        para_joiner='\n\n',
        key=None,
    ):
        with open(file, 'rb') as fh:
            for f in tarfile_reader(fh, streaming=True):
                cctx = zstandard.ZstdDecompressor()
                reader = io.BufferedReader(cctx.stream_reader(f))
                rdr = jsonlines.Reader(reader)
                yield from handle_jsonl(
                    rdr, get_meta, autojoin_paragraphs, para_joiner, key
                )
                f.close()


def validate_tokens(tokens, min_len=2):
    is_valid = len(tokens) >= min_len
    if not is_valid:
        logger.warning(
            f"token_ids must have at least {min_len} elements, skipping this example..."
        )
    return is_valid


def create_features_auto_lm(
    token_ids,
    max_sequence_length,
    short_seq_prob=0,
    inverted_mask=False,
    pad_id=0,
    min_len=10,
    input_ids_dtype="int32",
    input_mask_dtype="int32",
    labels_dtype="int32",
    rng=None,
):
    """Given a list of token_ids, generate input sequence and labels.

    Args:
        token_ids (sequence): List containing token ids for creating features,
            labels and input mask from.
        max_sequence_length (int): Maximum sequence length for data writes.
        short_seq_prob (float): Probability of generating short sequences from
            data. Defaults to `0`.
        inverted_mask (bool): Invert mask if specified for runtime execution.
            Defaults to `False`.
        min_len (int): Minimum length of token_ids to be considered a valid 
            sequence.
        pad_id (int): Id for pad token. Defaults to `0`.
        input_ids_dtype (str): Dtype as string for input ids.
            Defaults to `int32`.
        input_mask_dtype (str): Dtype as string for input mask.
            Defaults to `int32`.
        labels_dtype (str): Dtype as string for labels. Defaults to `int32`.
        rng (random.Random obj): Instance of random object, with states set.
            Defaults to `None`.

    Returns:
        Tuple containing features and labels
    """
    if not validate_tokens(token_ids, min_len=min_len):
        return []

    if rng.random() < short_seq_prob:
        token_ids = token_ids[0 : rng.randint(2, max_sequence_length - 1)]

    input_ids = token_ids[:-1]
    labels = token_ids[1:]
    input_mask = [1] * len(input_ids)

    # padding
    num_pad = max_sequence_length - len(input_ids)
    padding = [pad_id] * num_pad

    input_ids.extend(padding)
    labels.extend(padding)
    input_mask.extend([0] * num_pad)

    # assertions to ensure correct output shapes
    assert (
        len(input_ids) == max_sequence_length
        and len(labels) == max_sequence_length
        and len(input_mask) == max_sequence_length
    ), "Wrong sequence length"

    # create feature dict
    features = dict()
    features["input_ids"] = getattr(np, input_ids_dtype)(input_ids)
    features["input_mask"] = getattr(np, input_mask_dtype)(input_mask)

    if inverted_mask:
        features["input_mask"] = np.equal(features["input_mask"], 0).astype(
            features["input_mask"].dtype
        )
    labels = getattr(np, labels_dtype)(labels)

    return np.stack([features["input_ids"], features["input_mask"], labels])


def create_features_summarization(
    prompt_ids,
    completion_ids,
    max_sequence_length,
    eos_id=0,
    sep_id=None,
    pad_id=0,
    min_len=10,
    inverted_mask=False,
    input_ids_dtype="int32",
    input_mask_dtype="int32",
    labels_dtype="int32",
):
    """
    Given a list of prompt_ids and completion_ids, generate input sequence 
    and labels.
    
    Args:
        prompt_ids (sequence): List containing token ids for the prompt to 
            create features,labels and input mask from.
        completion_ids (sequence): List containing token ids for the completion
            create features,labels and input mask from.
        max_sequence_length (int): Maximum sequence length for data writes.
        eos_id (int): Id for end of sequence token. Defaults to `0`.
        sep_id (int): Id for separator token. Defaults to `None`.
        pad_id (int): Id for pad token. Defaults to `0`.
        min_len (int): Minimum length of token_ids to be considered a valid
            sequence.
        inverted_mask (bool): Invert mask if specified for runtime execution.
            Defaults to `False`.
        input_ids_dtype (str): Dtype as string for input ids.
            Defaults to `int32`.
        input_mask_dtype (str): Dtype as string for input mask. 
            Defaults to `int32`.
        labels_dtype (str): Dtype as string for labels. Defaults to `int32`.
    """

    # extra <EOS>
    total_len = len(prompt_ids) + len(completion_ids) + 1
    if sep_id is not None:
        total_len += 1
    if total_len > max_sequence_length:
        logger.warning(
            "prompt_ids + completion_ids > max_sequence_length, skipping this example..."
        )
        return []
    if total_len < min_len:
        logger.warning(
            "prompt_ids + completion_ids < min_sequence_len, skipping this example..."
        )
        return []

    token_ids = prompt_ids
    if sep_id is not None:
        token_ids = token_ids + [sep_id]
    token_ids = token_ids + completion_ids + [eos_id]

    token_mask = [0] * (len(prompt_ids))
    if sep_id is not None:
        token_mask += [1]
    else:
        # if no sep_id, prediction starts at the last token of prompt_ids
        token_mask[-1] = 1
    token_mask += [1] * len(completion_ids)
    token_mask += [0]  # EOS

    # add padding
    token_ids_pad = max_sequence_length + 1 - len(token_ids)
    input_mask_pad = max_sequence_length - len(token_mask)

    token_ids.extend([pad_id] * token_ids_pad)
    token_mask.extend([0] * input_mask_pad)

    input_ids = token_ids[:-1]
    labels = token_ids[1:]

    assert (
        len(input_ids) == max_sequence_length
        and len(labels) == max_sequence_length
        and len(token_mask) == max_sequence_length
    ), "Wrong sequence length"

    features = dict()
    features["input_ids"] = getattr(np, input_ids_dtype)(input_ids)
    features["input_mask"] = getattr(np, input_mask_dtype)(token_mask)

    if inverted_mask:
        features["input_mask"] = np.equal(features["input_mask"], 0).astype(
            features["input_mask"].dtype
        )
    labels = getattr(np, labels_dtype)(labels)

    return np.stack([features["input_ids"], features["input_mask"], labels])


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


def read_checkpoint(checkpoint_path, resume_from_checkpoint=True):
    """Checkpoint reader for execution.

    Args:
        checkpoint_path (str): Path to read checkpoint data from
        resume_from_checkpoint (bool): Resume from checkpoint for execution.
            Defaults to `True`.

    Returns:
        Tuple containing number of files processed and the count of tfrecords/HDF5 files
            written to output directory.
    """
    if resume_from_checkpoint and os.path.isfile(checkpoint_path):
        try:
            resume_files_processed, count = [
                int(i) for i in open(checkpoint_path, "r").read().split(", ")
            ]
            logger.info(
                f"Resuming from file number: {count}, "
                f"with raw file number processed: {resume_files_processed}"
            )
            return resume_files_processed, count
        except Exception as e:
            # if checkpoint path is at initialization,
            # file may exist, but no data might be written in the file
            # in that event, do not do anything, go to the final return
            logger.error(e)
    return 0, 0
