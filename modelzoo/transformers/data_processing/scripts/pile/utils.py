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

# This code is adapated from
# https://github.com/EleutherAI/gpt-neo/blob/master/data/create_tfrecords.py
#
# Copyright (c) 2020 EleutherAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
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
import random
import re
from pathlib import Path

import ftfy
import h5py
import numpy as np
import tensorflow as tf
from lm_dataformat import Reader

from modelzoo.common.tf.input.utils import create_int_feature


def _create_features_labels(
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
        features['input_mask'] = np.equal(features['input_mask'], 0).astype(
            features['input_mask'].dtype
        )
    labels = getattr(np, labels_dtype)(labels)

    return features, labels


def get_files(input_dir, filetypes=None):
    """Get all files of given filetypes from input directory.

    Args:
        input_dir (str): Input directory to read files from.
        filetypes (sequence): File types to fetch from the given input
            directory. Defaults to `None`.

    Returns:
        List of lists containing all file paths as strings
    """
    if not filetypes:
        filetypes = ["jsonl.zst", ".txt", ".xz", ".tar.gz"]

    files = [list(Path(input_dir).glob(f"*{ft}")) for ft in filetypes]
    # flatten list of list -> list and stringify Paths
    flattened_list = [str(item) for sublist in files for item in sublist]
    if not flattened_list:
        raise Exception(
            f"Did not find any files at this path {input_dir}, please also"
            + f" ensure your files are in format {filetypes}."
        )
    return flattened_list


def split_list(l, n):
    """Splits list/string into n sized chunks.

    Args:
        l (list, str): List or string to split.
        n (int): Number of chunks to split to.

    Returns:
        List of lists containing split list/string.
    """
    return [l[i : i + n] for i in range(0, len(l), n)]


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
            print(
                f"Resuming from tfrecord/HDF5 file number: {count}, "
                + f" with raw file number processed: {resume_files_processed}"
            )
            return resume_files_processed, count
        except Exception as e:
            # if checkpoint path is at initialization,
            # file may exist, but no data might be written in the file
            # in that event, do not do anything, go to the final return
            print(e)
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
    features, labels = _create_features_labels(
        tokens,
        args.max_seq_length,
        short_seq_prob=args.short_seq_prob,
        pad_id=args.pad_id,
        rng=rng,
    )
    return np.stack([features["input_ids"], features["input_mask"], labels])


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
        args (argparse namespace): Arguments for writing out tfrecords/HDF5.
        rng (random.Random obj): Instance of random object, with states set.
        n_examples (int): Number of examples that will be written in the file.
        chunks (tuple or bool): Chunk shape, or True to enable auto-chunking.
        dtype (string): Data type for the HDF5 dataset.
        compression (string): Compression strategy.
    """
    if args.write_in_batch:
        data_buffer = [get_single_example(f, args, rng) for f in files]
        _data = np.stack(data_buffer)
        with h5py.File(file_path, mode='w') as h5_file:
            h5_file.attrs["n_examples"] = n_examples
            h5_file.create_dataset(
                "data",
                data=_data,
                dtype=dtype,
                chunks=chunks,
                compression=compression,
            )
    else:
        with h5py.File(file_path, mode='w') as h5_file:
            h5_file.attrs["n_examples"] = n_examples
            dset = h5_file.create_dataset(
                "data",
                shape=(n_examples, 3, args.max_seq_length),
                dtype=dtype,
                chunks=chunks,
                compression=compression,
            )
            for idx, f in enumerate(files):
                dset[idx] = get_single_example(f, args, rng)


def write_to_file(writer, tokens, args, rng):
    """Create features, labels from tokens and write to tfrecord file.

    Args:
        writer (TFRecord writer obj): Instance of TFRecord writer to
            write out tfrecords.
        tokens (list): List containing tokenized data to write.
        args (argparse namespace): Arguments for writing out tfrecords.
        rng (random.Random obj): Instance of random object, with states set.
    """
    features, labels = _create_features_labels(
        tokens,
        args.max_seq_length,
        short_seq_prob=args.short_seq_prob,
        pad_id=args.pad_id,
        rng=rng,
    )

    features_dict = dict()
    features_dict["input_ids"] = create_int_feature(features["input_ids"])
    features_dict["input_mask"] = create_int_feature(features["input_mask"])
    features_dict["labels"] = create_int_feature(labels)
    tf_example = tf.train.Example(
        features=tf.train.Features(feature=features_dict)
    )
    writer.write(tf_example.SerializeToString())


def write_files(
    files,
    args,
    start_number,
    write_remainder=False,
    process_number=None,
    rng=random.Random(),
):
    """Writes a list of files to tfrecords/HDF5.

    Args:
        files (sequence): List of lists containing tokenized data to write.
        args (argparse namespace): Arguments for writing out tfrecords/HDF5
        start_number (int): Continual count of tfrecords/HDF5 files written out.
        write_remainder (bool): Write out remaining data from files, if
            files per record is not met. Defaults to `False`.
        process_number (int): Process number for execution. Defaults to `None`.
        rng (random.Random obj): Instance of random object, with states set.
            Defaults to new instance created for write.

    Returns:
        start_number (int): Continual count of tfrecords/HDF5 files written out.
        remainder (list): Remaining sequences not written out, if length of
            files to write is greater than the file per record.
    """
    if not files:
        return

    files_per_record = args.files_per_record
    chunks = split_list(files, files_per_record)
    if not chunks:
        return

    if len(chunks[-1]) != files_per_record and not write_remainder:
        remainder = chunks.pop(-1)
    else:
        remainder = None
        files_per_record = len(chunks[-1])

    for files in chunks:
        fp = f"{args.output_dir}/{args.output_name}_{start_number}"
        if process_number is not None:
            fp += f"_{process_number}"

        if args.file_format == "tfrecords":
            fp += f".tfrecords"
            with tf.io.TFRecordWriter(fp) as writer:
                for f in files:
                    write_to_file(writer, f, args, rng)
        elif args.file_format == "HDF5":
            fp += f".h5"
            write_hdf5_file(
                file_path=fp,
                files=files,
                args=args,
                rng=rng,
                n_examples=files_per_record,
                chunks=(1, 3, args.max_seq_length),
            )
        else:
            raise Exception(
                "Only supports `tfrecords` or `HDF5` file formats for now."
            )

        start_number += 1

    return start_number, remainder


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
    tf.random.set_seed(seed + rank)

    return rng
