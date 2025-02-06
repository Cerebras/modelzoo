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
Script to write HDF5 files for MLM_only and MLM + NSP datasets.

Usage:
    # For help related to MLM_only dataset creation
    python mlm_only -h 

    # For help related to MLM + NSP dataset creation
    python mlm_nsp -h

"""

import json
import logging
import os
import pickle

# isort: off
import sys

# isort: on

from collections import Counter, defaultdict
from itertools import repeat
from multiprocessing import Pool, cpu_count

import h5py
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
import cerebras.modelzoo.data_preparation.nlp.bert.dynamic_processor as mlm_nsp_processor
import cerebras.modelzoo.data_preparation.nlp.bert.mlm_only_processor as mlm_only_processor
from cerebras.modelzoo.common.utils.utils import check_and_create_output_dirs
from cerebras.modelzoo.data_preparation.nlp.bert.parser_utils import (
    get_parser_args,
)
from cerebras.modelzoo.data_preparation.utils import (
    get_files_in_metadata,
    get_output_type_shapes,
    get_vocab,
    split_list,
)


def _get_data_generator(args, metadata_file):

    output_type_shapes = get_output_type_shapes(
        args.max_seq_length,
        args.max_predictions_per_seq,
        mlm_only=(args.__mode == "mlm_only"),
    )

    if args.__mode == "mlm_only":
        data_generator = mlm_only_processor.data_generator
        kwargs = {
            "metadata_files": metadata_file,
            "vocab_file": args.vocab_file,
            "do_lower": args.do_lower_case,
            "disable_masking": True,
            "mask_whole_word": args.mask_whole_word,
            "max_seq_length": args.max_seq_length,
            "max_predictions_per_seq": args.max_predictions_per_seq,
            "masked_lm_prob": args.masked_lm_prob,
            "dupe_factor": 1,
            "output_type_shapes": output_type_shapes,
            "multiple_docs_in_single_file": args.multiple_docs_in_single_file,
            "multiple_docs_separator": args.multiple_docs_separator,
            "single_sentence_per_line": args.single_sentence_per_line,
            "overlap_size": args.overlap_size,
            "min_short_seq_length": args.min_short_seq_length,
            "buffer_size": args.buffer_size,
            "short_seq_prob": args.short_seq_prob,
            "spacy_model": args.spacy_model,
            "inverted_mask": False,
            "allow_cross_document_examples": args.allow_cross_document_examples,
            "document_separator_token": args.document_separator_token,
            "seed": args.seed,
            "input_files_prefix": args.input_files_prefix,
        }
    elif args.__mode == "mlm_nsp":
        data_generator = mlm_nsp_processor.data_generator
        kwargs = {
            "metadata_files": metadata_file,
            "vocab_file": args.vocab_file,
            "do_lower": args.do_lower_case,
            "split_num": args.split_num,
            "max_seq_length": args.max_seq_length,
            "short_seq_prob": args.short_seq_prob,
            "mask_whole_word": args.mask_whole_word,
            "max_predictions_per_seq": args.max_predictions_per_seq,
            "masked_lm_prob": args.masked_lm_prob,
            "dupe_factor": 1,
            "output_type_shapes": output_type_shapes,
            "min_short_seq_length": args.min_short_seq_length,
            "multiple_docs_in_single_file": args.multiple_docs_in_single_file,
            "multiple_docs_separator": args.multiple_docs_separator,
            "single_sentence_per_line": args.single_sentence_per_line,
            "inverted_mask": False,
            "seed": args.seed,
            "spacy_model": args.spacy_model,
            "input_files_prefix": args.input_files_prefix,
        }
    else:
        raise ValueError

    def _data_generator():
        return data_generator(**kwargs)

    return _data_generator


def create_h5(params):
    files, args, process_no = params
    n_docs = len(files)

    # Write each process metadata file
    metadata_file = os.path.join(args.output_dir, f"metadata_{process_no}.txt")
    with open(metadata_file, "w") as mfh:
        mfh.writelines([x + "\n" for x in files])

    num_output_files = max(args.num_output_files // args.num_processes, 1)

    output_files = [
        os.path.join(
            args.output_dir,
            f"{args.name}-{fidx + num_output_files*process_no}_p{process_no}.h5",
        )
        for fidx in range(num_output_files)
    ]

    ## Create hdf5 writers for each hdf5 file
    writers = []
    meta_data = defaultdict(int)
    writer_num_examples = 0
    vocab = get_vocab(args.vocab_file, args.do_lower_case)
    hist_tokens = Counter({key: 0 for key in vocab})
    hist_lengths = Counter({x: 0 for x in range(args.max_seq_length + 1)})
    for output_file in output_files:
        writers.append(
            [h5py.File(output_file, "w"), writer_num_examples, output_file]
        )

    _data_generator = _get_data_generator(args, metadata_file)

    writer_index = 0
    total_written = 0
    _dt = h5py.string_dtype(encoding='utf-8')

    ## Names of keys of instance dictionary
    if args.__mode == "mlm_only":
        fieldnames = ["tokens"]
    elif args.__mode == "mlm_nsp":
        # MLM + NSP fields
        fieldnames = ["tokens", "segment_ids", "is_random_next"]
    else:
        raise ValueError

    for features in _data_generator():
        ## write dictionary into hdf5
        writer, writer_num_examples, output_file = writers[writer_index]
        grp_name = f"example_{writer_num_examples}"

        if args.__mode == "mlm_only":
            writer.create_dataset(
                f"{grp_name}/tokens",
                data=np.array(features, dtype=_dt),
                compression="gzip",
            )
            hist_tokens.update(features)
            hist_lengths.update([len(features)])
        elif args.__mode == "mlm_nsp":
            features_dict = features.to_dict()
            writer.create_dataset(
                f"{grp_name}/tokens",
                data=np.array(features_dict["tokens"], dtype=_dt),
                compression="gzip",
            )
            writer.create_dataset(
                f"{grp_name}/segment_ids",
                data=np.array(features_dict["segment_ids"]),
                compression="gzip",
            )
            # h5 cannot write scalars, hence converting to array
            writer.create_dataset(
                f"{grp_name}/is_random_next",
                data=np.array([features_dict["is_random_next"]]),
                compression="gzip",
            )
            hist_tokens.update(features_dict["tokens"])
            hist_lengths.update([len(features_dict["tokens"])])
        else:
            raise ValueError

        total_written += 1
        writers[writer_index][1] += 1
        writer_index = (writer_index + 1) % len(writers)

        ## Update meta info with number of lines in the input data.
        meta_data[output_file] += 1

    for writer, writer_num_examples, output_file in writers:
        assert len(writer) == writer_num_examples
        assert len(writer) == meta_data[output_file]
        writer.flush()
        writer.close()

    return {
        "total_written": total_written,
        "meta_data": meta_data,
        "n_docs": n_docs,
        "hist_tokens": hist_tokens,
        "hist_lengths": hist_lengths,
    }


def create_h5_mp(input_files, args):

    try:
        files = split_list(input_files, len(input_files) // args.num_processes)
    except ValueError as e:
        # We hit errors in two potential scenarios,
        # 1) Files is an empty list, in which case there is nothing to split
        # 2) There are more processes than files, in which case we cannot split
        #    the files to processes correctly, as there will be many idle
        #    processes which are not doing anything.
        print(e)
        raise

    with Pool(processes=args.num_processes) as pool:
        results = pool.imap(
            create_h5, zip(files, repeat(args), range(len(files)))
        )
        meta = {
            "total_written": 0,
            "n_docs": 0,
            "meta_data": {},
            "hist_tokens": Counter({}),
            "hist_lengths": Counter(
                {x: 0 for x in range(args.max_seq_length + 1)}
            ),
        }
        for r in results:
            for k, v in r.items():
                if not isinstance(v, dict):
                    meta[k] += v
                else:
                    # Valid for both Counter and Dict objects
                    # For `Counter`` objects, values corresponding
                    # to same key are added.
                    # For `dict` objects, values corresponding
                    # to same key are updated with the new value `v`
                    meta[k].update(v)
        return meta


def main():
    args = get_parser_args()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"hdf5_{args.__mode}_unmasked",
        )

    check_and_create_output_dirs(args.output_dir, filetype="h5")
    print(args)

    json_params_file = os.path.join(args.output_dir, "data_params.json")
    print(
        f"\nStarting writing data to {args.output_dir}."
        + f" User arguments can be found at {json_params_file}."
    )

    # write initial params to file
    params = vars(args)
    with open(json_params_file, 'w') as _fout:
        json.dump(params, _fout, indent=4, sort_keys=True)

    if args.num_processes == 0:
        # if nothing is specified, then set number of processes to CPU count.
        args.num_processes = cpu_count()

    input_files = get_files_in_metadata(args.metadata_files)

    if args.num_processes > 1:
        results = create_h5_mp(input_files, args)
    else:
        # Run only single process run, with process number set as 0.
        results = create_h5((input_files, args, 0))

    # Write outputs of execution
    ## Store Token Histogram
    hist_tokens_file = os.path.join(args.output_dir, "hist_tokens.pkl")
    with open(hist_tokens_file, "wb") as handle:
        pickle.dump(
            results["hist_tokens"], handle, protocol=pickle.HIGHEST_PROTOCOL
        )

    ## Store Lengths Histogram
    hist_lengths_file = os.path.join(args.output_dir, "hist_lengths.pkl")
    with open(hist_lengths_file, "wb") as handle:
        pickle.dump(
            results["hist_lengths"], handle, protocol=pickle.HIGHEST_PROTOCOL
        )

    ## Update data_params file with new fields
    with open(json_params_file, 'r') as _fin:
        data = json.load(_fin)

    data["n_docs"] = results["n_docs"]
    data["total_written"] = results["total_written"]
    data["hist_tokens_file"] = hist_tokens_file
    data["hist_lengths_file"] = hist_lengths_file

    with open(json_params_file, 'w') as _fout:
        json.dump(data, _fout, indent=4, sort_keys=True)

    print(
        f"\nFinished writing data to HDF5 to {args.output_dir}."
        + f" Runtime arguments and outputs can be found at {json_params_file}."
    )

    ## Store meta file.
    meta_file = os.path.join(args.output_dir, "meta.dat")
    with open(meta_file, "w") as fout:
        for output_file, num_lines in results["meta_data"].items():
            fout.write(f"{output_file} {num_lines}\n")


if __name__ == "__main__":
    main()
