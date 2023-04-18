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

"""
Common utils.py file sharing the utility functions that could be shared by the 
special scripts in any of the sub folders. 
"""
import logging
import os
import re
from pathlib import Path

import numpy as np

from modelzoo.transformers.data_processing.utils import split_list

logger = logging.getLogger("utils")
logger.setLevel(logging.INFO)


def create_features_labels(
    token_ids,
    max_sequence_length,
    short_seq_prob=0,
    inverted_mask=False,
    pad_id=0,
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
    assert len(token_ids) >= 2, "token_ids must have at least 2 elements."

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

    return features, labels


def get_files(input_dir=None, filetypes=None, metadata_files=None):
    """Get all files of given filetypes from input directory.

    Args:
        input_dir (str): Input directory to read files from.
        filetypes (sequence): File types to fetch from the given input
            directory. Defaults to `None`.
        metadata_files (str): Comma separated string of metadata files.

    Returns:
        List of lists containing all file paths as strings
    """
    if not filetypes:
        filetypes = [
            '.jsonl',
            '.jsonl.zst',
            '.jsonl.zst.tar',
            '.txt',
        ]

    assert input_dir or metadata_files, (
        "User need to provide `input_dir` or `metadata_files`, "
        "but neither was provided."
    )
    if metadata_files:
        if isinstance(metadata_files, str):
            metadata_files = [metadata_files]

        input_files = []
        for _file in metadata_files:
            with open(_file, "r") as _fin:
                input_files.extend(_fin.readlines())

        input_files_list = [x.strip() for x in input_files if x]
        flattened_list = [
            x for x in input_files_list if os.path.splitext(x)[1] in filetypes
        ]
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


def archive_to_tokens(f, tokenizer, args, prefix=[]):
    """Generator that yields the contents of the files in an archive
    if data_to_prepend is not None, prepend data_to_preprend + an EOS separator
    to the encoded data.

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
    for doc in reader.stream_data(threaded=False):
        if args.ftfy:
            doc = ftfy.fix_text(doc, normalization=args.ftfy_normalizer)
        if args.wikitext_detokenize:
            doc = wikitext_detokenizer(doc)

        doc = tokenizer.encode(doc) + args.eos_id
        yield split_list(prefix + doc, args.max_seq_length + 1)


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


def get_single_example(tokens, args, rng):
    """Create features, labels from tokens for HDF5.
    Args:
        tokens (list): List containing tokenized data to write.
        args (argparse namespace): Arguments for writing out HDF5 dataset.
        rng (random.Random obj): Instance of random object, with states set.
    
    Returns:
        Numpy array contains features for a single example (shape: [3, max_sequence_length])
    """
    features, labels = create_features_labels(
        tokens,
        args.max_seq_length,
        short_seq_prob=args.short_seq_prob,
        pad_id=args.pad_id,
        rng=rng,
    )
    return np.stack([features["input_ids"], features["input_mask"], labels])
