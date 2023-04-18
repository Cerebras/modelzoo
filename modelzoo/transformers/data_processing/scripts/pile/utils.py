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

import random

import h5py
import numpy as np
import tensorflow as tf

from modelzoo.common.tf.input.utils import create_int_feature
from modelzoo.transformers.data_processing.scripts.utils import (
    create_features_labels,
    get_single_example,
)
from modelzoo.transformers.data_processing.utils import split_list


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
    features, labels = create_features_labels(
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
