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
import logging
import random
from itertools import repeat
from math import ceil
from multiprocessing import Pool, cpu_count

import h5py
import numpy as np
from tqdm import tqdm

from modelzoo.transformers.data_processing.scripts.pile.tokenizer import (
    get_tokenizer,
)
from modelzoo.transformers.data_processing.scripts.utils import (
    get_single_example,
    read_checkpoint,
    wikitext_detokenizer,
)
from modelzoo.transformers.data_processing.utils import split_list

logger = logging.getLogger("utils")
logger.setLevel(logging.INFO)


def add_common_args(parser):
    """
    For the argparse to parse arguments for subcommands, we add common 
    command line arguments to each subcommand parser here.
    """
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory where raw data is stored.",
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
        "--jsonl_key",
        type=str,
        default="text",
        help="The key name in input jsonl files from which the raw text will be "
        "extracted in order to further process it. Default: 'text'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data_dir/",
        help="Directory where HDF5 files will be stored. Defaults to `./data_dir/`.",
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
        "--seed", type=int, default=0, help="Random seed. Defaults to `0`.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=0,
        help="Number of processes to use. Default to cpu count.",
    )
    parser.add_argument(
        "--write_remainder",
        action="store_true",
        help="Write the remainder files when data is left over from "
        "processing.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Resume record writing from a given checkpoint.",
    )
    parser.add_argument(
        "--display_pbar",
        action="store_true",
        help="Display progress while runs.",
    )
    parser.add_argument(
        "--files_per_record",
        type=int,
        default=50000,
        help="Text files to write per HDF5 file.",
    )
    parser.add_argument(
        "--write_in_batch",
        action="store_true",
        help="Whether to write the samples in batch for the HDF5 format, "
        "setting to false will save memory but a bit slower.",
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
    raw_text_parser = subparser.add_parser(
        "raw_text", help="Convert input files into hdf5 files with raw text."
    )
    add_common_args(raw_text_parser)
    token_id_parser = subparser.add_parser(
        "preprocessed_text",
        help="Convert input files into hdf5 files with the input text "
        "that is tokenized and converted into token ids. Along with it "
        "also save the labels and input attention mask for each sample.",
    )
    add_common_args(token_id_parser)
    token_id_parser.add_argument(
        "--tokenizer_type",
        type=str,
        required=True,
        choices=["GPT2Tokenizer", "NeoXTokenizer"],
        help=(
            "Type of tokenizer to use for tfrecord/HDF5 dataset generation. "
            "Can be one of `GPT2Tokenizer` or `NeoXTokenizer`."
        ),
    )
    token_id_parser.add_argument(
        "--vocab_file",
        type=str,
        default=None,
        help="path to the vocabulary file. Defaults to None.",
    )
    token_id_parser.add_argument(
        "--encoder_file",
        type=str,
        default=None,
        help="Path to the encoder file. Defaults to None.",
    )
    token_id_parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length. Defaults to `2048`.",
    )
    token_id_parser.add_argument(
        "--short_seq_prob",
        type=float,
        default=0.0,
        help=(
            "Probability of creating sequences which are shorter than the"
            + " maximum sequence length. Defaults to `0.0`."
        ),
    )
    token_id_parser.add_argument(
        "--ftfy", action="store_true", help="Fix text with ftfy."
    )
    token_id_parser.add_argument(
        "--ftfy_normalizer",
        type=str,
        default="NFC",
        choices=["NFC", "None"],
        help=(
            "Choose what kind of unicode normalization is applied. Usually, we"
            + " apply `NFC` normalization, so that letters followed by combining"
            + " characters become single combined characters. Using `None`"
            + " applies no normalization while fixing text."
        ),
    )
    token_id_parser.add_argument(
        "--wikitext-detokenize",
        action="store_true",
        help="Use wikitext detokenizer to fix text.",
    )
    token_id_parser.add_argument(
        "--eos_id",
        type=int,
        default=50256,
        help=(
            "Id for padding out shorter sequences. Defaults to 50256, which"
            + " is `<|endoftext|>` in tokens."
        ),
    )
    token_id_parser.add_argument(
        "--pad_id",
        type=int,
        default=50256,
        help=(
            "Id for padding out shorter sequences. Defaults to 50256, which"
            + " is `<|endoftext|>` in tokens."
        ),
    )
    return parser


def set_defaults(args):
    if args.mode == "preprocessed_text":
        # fix args for ftfy normalization
        if args.ftfy_normalizer == "None":
            args.ftfy_normalizer = None

    if args.processes == 0:
        # if nothing is specified, then set number of processes to CPU count.
        args.processes = cpu_count()


def dump_args(args, json_params_file):
    """
    Write the input params to file.
    """
    logger.info(f"User arguments can be found at {json_params_file}.")

    # write initial params to file
    params = vars(args)
    with open(json_params_file, "w") as _fout:
        json.dump(params, _fout, indent=4, sort_keys=True)


def dump_result(results, json_params_file, eos_id=None, pad_id=None):
    """
    Write outputs of execution
    """
    with open(json_params_file, "r") as _fin:
        data = json.load(_fin)

    data["discarded_files"] = results["discarded"]
    data["processed_files"] = results["processed"]
    data["successful_files"] = results["successful"]
    data["n_examples"] = results["examples"]
    if eos_id:
        data["eos_id"] = eos_id[0] if isinstance(eos_id, list) else eos_id
    if pad_id:
        data["pad_id"] = pad_id

    with open(json_params_file, "w") as _fout:
        json.dump(data, _fout, indent=4, sort_keys=True)


def get_doc_or_tokens(f, tokenizer, args, prefix=[]):
    """Generator that yields the contents of the files in an archive
    if data_to_prepend is not None, prepend data_to_preprend + an EOS separator
    to the encoded data.
    Optionally, it also tokenizes the docs and encodes them based on the input 
    args.

    Args:
        f (file): Archive file to read.
        tokenizer (BPETokenizer obj): Tokenizer used to encode raw data.
        args (argparse namespace): Arguments for writing out tfrecords/HDF5.
        prefix (list): Data to prefix before splitting to given context length.
            Used to add remainder data from previous iteration of data reads.
            Defaults to `[]`, i.e, empty list.

    Yields:
        A list of lists with tokenized data + EOS separator appended at the end.
    """
    import ftfy
    from lm_dataformat import Reader

    reader = Reader(f)
    # We're not using `stream_data()` from lm_dataformat.Reader object because
    # it has a bug where it can't support custom keys, so we use the so
    # called private method `_stream_data()`
    for doc in reader._stream_data(jsonl_key=args.jsonl_key):
        if args.mode == "raw_text":
            yield doc
        else:
            if args.ftfy:
                doc = ftfy.fix_text(doc, normalization=args.ftfy_normalizer)
            if args.wikitext_detokenize:
                doc = wikitext_detokenizer(doc)

            doc = tokenizer.encode(doc) + args.eos_id
            doc = prefix + doc
            yield [
                doc[i : i + args.max_seq_length + 1]
                for i in range(0, len(doc), args.max_seq_length)
            ]


def create_dataset(params):
    """Creates HDF5 dataset from given parameters.

    Args:
        params (tuple): Tuple containing the files, arguments and process number
            for current execution.

    Returns:
        Dictionary containing results of execution, specifically as number of
            processed, discarded, and successful files as well as number of examples.
    """
    files, args, write_files_fn, process_no = params
    rng = seed_runs(args.seed, rank=process_no)

    write_remainder = args.write_remainder
    resume_from_checkpoint = args.resume_from_checkpoint
    display_pbar = args.display_pbar

    discarded_files = 0
    files_processed = 0
    pbar = tqdm(
        desc=f"Parsed 0 input files. Files written ", disable=not display_pbar,
    )
    checkpoint_path = f"{args.output_dir}/checkpoint_{process_no}.txt"
    resume_files_processed, df_count = read_checkpoint(
        checkpoint_path, resume_from_checkpoint
    )

    data_to_prepend = []
    doc_object_array = []

    tokenizer = None
    if args.mode == "preprocessed_text":
        tokenizer, eos_id, pad_id = get_tokenizer(args)

        # re-assign eos_id, pos_id to the args value since it is used for generation
        if not isinstance(eos_id, list):
            eos_id = [eos_id]
        args.eos_id = eos_id
        args.pad_id = pad_id

    for _file in files:
        for doc_object in get_doc_or_tokens(
            _file, tokenizer, args, prefix=data_to_prepend
        ):
            files_processed += 1
            if files_processed < resume_files_processed:
                continue  # enable resuming from checkpoint

            if args.mode == "preprocessed_text":
                # if the last chunk < chunk size, but > minimum_size, take it
                # and append it to the beginning of the next file
                if data_to_prepend:
                    data_to_prepend.clear()
                n_tokens = len(doc_object[-1])
                if n_tokens < args.max_seq_length:
                    data = doc_object.pop(-1)
                    if n_tokens > 0:
                        data_to_prepend.extend(data)
                    else:
                        discarded_files += 1

            # add tokenized files > chunk size to main array
            if args.mode == "preprocessed_text":
                doc_object_array.extend(doc_object)
            else:
                doc_object_array.append(doc_object)

            if len(doc_object_array) >= args.files_per_record:
                _df_count, remainder = write_files_fn(
                    doc_object_array,
                    args,
                    start_number=df_count,
                    process_number=process_no,
                    rng=rng,
                )
                pbar.update(_df_count - df_count)
                pbar.set_description(
                    f"Parsed {files_processed} input files. Files written "
                )

                df_count = _df_count
                doc_object_array = (
                    remainder if remainder is not None else []
                )  # add remaining files to next chunk
                with open(checkpoint_path, "w") as checkpoint_file:
                    checkpoint_file.write(f"{files_processed}, {df_count}")

    if len(doc_object_array) >= args.files_per_record:
        _df_count, remainder = write_files_fn(
            doc_object_array,
            args,
            start_number=df_count,
            process_number=process_no,
            rng=rng,
        )
        pbar.update(_df_count - df_count)
        pbar.set_description(
            f"Parsed {files_processed} input files. Files written "
        )
        df_count = _df_count
        with open(checkpoint_path, "w") as checkpoint_file:
            checkpoint_file.write(f"{files_processed}, {df_count}")
    else:
        remainder = doc_object_array

    n_examples = df_count * args.files_per_record
    if write_remainder:
        n_examples += len(remainder)
        _df_count, _ = write_files_fn(
            remainder,
            args,
            start_number=df_count,
            write_remainder=True,
            process_number=process_no,
            rng=rng,
        )
        pbar.update(_df_count - df_count)
        pbar.set_description(
            f"Parsed {files_processed} input files. Files written "
        )
        with open(checkpoint_path, "w") as checkpoint_file:
            checkpoint_file.write(f"{files_processed}, {_df_count}")

    successful_files = files_processed - discarded_files
    return {
        "discarded": discarded_files,
        "processed": files_processed,
        "successful": successful_files,
        "examples": n_examples,
    }


def create_dataset_mp(files, args, write_files_fn):
    """Create HDF5 dataset using multiple processes.

    Args:
        files (list): List of files to process.
        args (argparse namespace): Arguments for writing out HDF5 dataset.

    Returns:
        Dictionary containing results of execution, specifically as number of
            processed, discarded, and successful files as well as number of examples 
            from all processes.
    """
    try:
        n_proc = args.processes
        n_chunks = ceil(len(files) / n_proc)
        remain = len(files) % n_proc
        if n_chunks == 1 and remain:
            n_proc = remain
            logger.warning(
                f"There aren't enough files to distribute to {args.processes} "
                f"processes, resetting it to {n_proc}. If you're working with a "
                "small number of compressed archives and could extract it into "
                "txt files, you might be able to get more benefits from the "
                f"available {args.processes} processes."
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
                create_dataset,
                zip(
                    files,
                    repeat(args),
                    repeat(write_files_fn),
                    range(len(files)),
                ),
            ),
            total=len(files),
        )
        meta = {"discarded": 0, "processed": 0, "successful": 0, "examples": 0}
        for results in pbar:
            pbar.update()
            for k, v in results.items():
                meta[k] += v

        return meta


def seed_runs(seed, rank=0):
    """Set seed for run based on user provided seed and rank.

    Args:
        seed (int): Seed value to set.
        rank (int): Rank to set, based on process number for execution.
            Defaults to 0.

    Returns:
        Object of type random.Random, with seed set.
    """
    rng = random.Random()
    rng.seed(seed + rank)
    np.random.seed(seed + rank)

    return rng


def write_hdf5_file(
    file_path,
    files,
    args,
    rng,
    n_examples,
    chunks,
    dtype="i4",
    compression="gzip",
):
    """Write data to HDF5 file.

    Args:
        file_path (string): HDF5 file path.
        files (sequence): List of lists containing tokenized data to write.
        args (argparse namespace): Arguments for writing out HDF5.
        rng (random.Random obj): Instance of random object, with states set.
        n_examples (int): Number of examples that will be written in the file.
        chunks (tuple or bool): Chunk shape, or True to enable auto-chunking.
        dtype (string): Data type for the HDF5 dataset.
        compression (string): Compression strategy.
    """
    if args.mode == "raw_text":
        data_label = "text"
        data_shape = (n_examples,)
        args.write_in_batch = False
    else:
        data_label = "data"
        data_shape = (n_examples, 3, args.max_seq_length)
    if args.write_in_batch:
        data_buffer = [get_single_example(f, args, rng) for f in files]
        # Below will convert list of strings into numpy 'U' type and h5py
        # doesn't allow storing such format
        # https://docs.h5py.org/en/stable/strings.html#what-about-numpy-s-u-type
        _data = np.stack(data_buffer)
        with h5py.File(file_path, mode="w") as h5_file:
            h5_file.attrs["n_examples"] = n_examples
            h5_file.create_dataset(
                data_label,
                data=_data,
                dtype=dtype,
                chunks=chunks,
                compression=compression,
            )
    else:
        with h5py.File(file_path, mode="w") as h5_file:
            h5_file.attrs["n_examples"] = n_examples
            dset = h5_file.create_dataset(
                data_label,
                shape=data_shape,
                dtype=dtype,
                chunks=chunks,
                compression=compression,
            )
            for idx, f in enumerate(files):
                dset[idx] = (
                    get_single_example(f, args, rng)
                    if args.mode != "raw_text"
                    else f
                )


def write_hdf5_files(
    files,
    args,
    start_number,
    write_remainder=False,
    process_number=None,
    rng=random.Random(),
):
    """Writes a list of files to HDF5.

    Args:
        files (sequence): List of lists containing tokenized data to write.
        args (argparse namespace): Arguments for writing out HDF5
        start_number (int): Continual count of HDF5 files written out.
        write_remainder (bool): Write out remaining data from files, if
            files per record is not met. Defaults to `False`.
        process_number (int): Process number for execution. Defaults to `None`.
        rng (random.Random obj): Instance of random object, with states set.
            Defaults to new instance created for write.

    Returns:
        start_number (int): Continual count of HDF5 files written out.
        remainder (list): Remaining sequences not written out, if length of
            files to write is greater than the file per record.
    """
    if not files:
        return

    files_per_record = args.files_per_record
    file_chunks = split_list(files, files_per_record)
    if not file_chunks:
        return

    if len(file_chunks[-1]) != files_per_record and not write_remainder:
        remainder = file_chunks.pop(-1)
    else:
        remainder = None
        files_per_record = len(file_chunks[-1])

    if args.mode == "raw_text":
        hdf5_chunk_size = (1,)
        hdf5_dtype = h5py.string_dtype(encoding="utf-8")
    else:
        hdf5_chunk_size = (1, 3, args.max_seq_length)
        hdf5_dtype = "i4"

    for files in file_chunks:
        fp = f"{args.output_dir}/{args.output_name}_{start_number}"
        if process_number is not None:
            fp += f"_{process_number}"

        write_hdf5_file(
            file_path=fp + f".h5",
            files=files,
            args=args,
            rng=rng,
            n_examples=files_per_record,
            chunks=hdf5_chunk_size,
            dtype=hdf5_dtype,
        )

        start_number += 1

    return start_number, remainder


def verify_saved_hdf5_files(params):
    """
    This function is used to do sanity checks at the end of the creation 
    of hdf5 files.
    This function loads every .h5 files generated and 
    - for `raw_text` mode it checks:
        1. The data type
        2. Attributes in the dataset
    - for `preprocessed_text` mode it checks:
        1. The data type
        2. Shape of the dataset
        3. Fact that labels and inputs are as expected
    """
    h5_files_path, args = params
    for h5_file_path in h5_files_path:
        with h5py.File(h5_file_path, mode="r") as h5_file:
            if args.mode == "raw_text":
                # verify the raw text h5 file content
                n_docs = h5_file.attrs["n_examples"]
                dataset = h5_file["text"]
                expected_dtype = h5py.string_dtype(encoding="utf-8")
                assert dataset.dtype == expected_dtype, (
                    f"Error in {h5_file}, conversion is corrupted as the "
                    f"datatype is unexpected. Expected: {expected_dtype}, "
                    f"received {dataset.dtype}."
                )
                assert n_docs <= args.files_per_record, (
                    f"Error in {h5_file}, conversion is corrupted as the "
                    f"number of docs in file is unexpected. Expected:"
                    f" {args.files_per_record}, received {n_docs}."
                )
            else:
                # verify the preprocessed text h5 file content
                n_examples = h5_file.attrs["n_examples"]
                dataset = h5_file["data"]
                expected_dtype = "i4"
                assert dataset.dtype == expected_dtype, (
                    f"Error in {h5_file}, conversion is corrupted as the "
                    f"datatype is unexpected. Expected: {expected_dtype}, "
                    f"received {dataset.dtype}."
                )
                data_shape = dataset[()].shape
                assert n_examples <= args.files_per_record, (
                    f"Error in {h5_file}, conversion is corrupted as the "
                    f"number of examples in file is unexpected. Expected:"
                    f" {args.files_per_record}, received {n_examples}."
                )
                assert data_shape[1:] == (3, args.max_seq_length), (
                    f"Error in {h5_file}, conversion is corrupted as the "
                    f"number shape of example is unexpected. Expected:"
                    f" {(3, args.max_seq_length)}, received {data_shape[1:]}."
                )


def verify_saved_hdf5_files_mp(files, args):
    """Create HDF5 dataset using multiple processes.

    Args:
        files (list): List of files to process.
        args (argparse namespace): Arguments for verifying HDF5 dataset.
    """
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

    with Pool(processes=n_proc) as pool:
        pbar = tqdm(
            pool.imap(verify_saved_hdf5_files, zip(files, repeat(args))),
            total=len(files),
        )
