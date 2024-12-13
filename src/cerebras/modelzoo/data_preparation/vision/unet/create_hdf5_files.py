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
Script to write HDF5 files for UNet datasets.

Usage:
    # For help:
    python create_hdf5_files.py -h

    # Step-1:
    Set image shape to desired shape in 
    `train_input.image_shape` and `eval_input.image_shape` 
    i.e. [H, W, 1] in config: 
    /path_to_modelzoo/vision/pytorch/unet/configs/params_severstal_binary.yaml

    # Step-2: Run the script 
    python modelzoo.data_preparation.vision.unet.create_hdf5_files.py --params=/path_to_modelzoo/vision/pytorch/unet/configs/params_severstal_binary.yaml --output_dir=/path_to_outdir/severstal_binary_classid_3_hdf --num_output_files=10 --num_processes=5

"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from itertools import repeat
from multiprocessing import Pool, cpu_count

import h5py
from tqdm import tqdm

# isort: off
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
# isort: on
from cerebras.modelzoo.common.utils.run.cli_parser import read_params_file
from cerebras.modelzoo.common.utils.utils import check_and_create_output_dirs
from cerebras.modelzoo.data.vision.classification.dataset_factory import (
    VisionSubset,
)
from cerebras.modelzoo.data_preparation.utils import split_list

from cerebras.modelzoo.data.vision.segmentation.SeverstalBinaryClassDataProcessor import (  # noqa
    SeverstalBinaryClassDataProcessor,
)


def update_params_from_args(args, params):
    """
    Sets command line arguments from args into params.

    :param argparse namespace args: Command line arguments
    :param dict params: runconfig dict we want to update
    """

    if args:
        for k, v in list(vars(args).items()):
            params[k] = v if v is not None else params.get(k)


def _get_dataset(params):
    params["use_worker_cache"] = False
    return getattr(sys.modules[__name__], params["data_processor"])(
        params
    ).create_dataset()


def _get_data_generator(params, is_training, dataset_range):
    dataset = _get_dataset(params, is_training)
    sub_dataset = VisionSubset(dataset, dataset_range)
    sub_dataset.set_transforms()

    for idx, feature in enumerate(sub_dataset):
        image, label = feature
        yield (image, label, image.shape, label.shape)


def create_h5(params):
    dataset_range, data_params, args, process_no = params
    n_docs = len(dataset_range)

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

    for output_file in output_files:
        w = h5py.File(output_file, "w")
        w.attrs["n_examples"] = 0
        writers.append([w, writer_num_examples, output_file])

    writer_index = 0
    total_written = 0

    ## Names of keys of instance dictionary
    fieldnames = ["image", "label"]
    is_training = "train" in args.split

    data_generator = lambda: _get_data_generator(
        data_params, is_training, dataset_range
    )

    for features in tqdm(data_generator(), total=n_docs):
        image, label, image_shape, label_shape = features
        ## write dictionary into hdf5
        writer, writer_num_examples, output_file = writers[writer_index]
        grp_name = f"example_{writer_num_examples}"

        writer.create_dataset(
            f"{grp_name}/image", data=image, shape=image_shape
        )
        writer.create_dataset(
            f"{grp_name}/label", data=label, shape=label_shape
        )

        total_written += 1
        writers[writer_index][1] += 1
        writer_index = (writer_index + 1) % len(writers)

        ## Update meta info with number of lines in the input data.
        meta_data[output_file] += 1

    for writer, writer_num_examples, output_file in writers:
        assert len(writer) == writer_num_examples
        assert len(writer) == meta_data[output_file]
        writer.attrs["n_examples"] = writer_num_examples
        writer.flush()
        writer.close()

    return {
        "total_written": total_written,
        "meta_data": meta_data,
        "n_docs": n_docs,
        "dataset_range": {process_no: (min(dataset_range), max(dataset_range))},
    }


def create_h5_mp(dataset_range, data_params, args):
    try:
        sub_dataset_range = split_list(
            dataset_range, len(dataset_range) // args.num_processes
        )
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
            create_h5,
            zip(
                sub_dataset_range,
                repeat(data_params),
                repeat(args),
                range(len(sub_dataset_range)),
            ),
        )
        meta = {
            "total_written": 0,
            "n_docs": 0,
            "meta_data": {},
            "dataset_range": {},
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


def get_parser_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=0,
        help="Number of parallel processes to use, defaults to cpu count",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="directory where HDF5 files will be stored.",
    )
    parser.add_argument(
        "--num_output_files",
        type=int,
        default=10,
        help="number of output files in total i.e each process writes num_output_files//num_processes number of files"
        "Defaults to 10.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="preprocessed_data",
        help="name of the dataset; i.e. prefix to use for hdf5 file names. "
        "Defaults to 'preprocessed_data'.",
    )
    parser.add_argument(
        "--params", type=str, required=True, help="params config yaml file"
    )
    return parser


def main():
    args = get_parser_args().parse_args()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    output_dir = args.output_dir
    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"hdf5_dataset",
        )

    check_and_create_output_dirs(args.output_dir, filetype="h5")
    json_params_file = os.path.join(args.output_dir, "data_params.json")
    print(
        f"\nStarting writing data to {args.output_dir}."
        + f" User arguments can be found at {json_params_file}."
    )

    # write initial params to file
    params = read_params_file(args.params)
    params["input_args"] = {}
    update_params_from_args(args, params["input_args"])

    with open(json_params_file, 'w') as _fout:
        json.dump(params, _fout, indent=4, sort_keys=True)

    if args.num_processes == 0:
        # if nothing is specified, then set number of processes to CPU count.
        args.num_processes = cpu_count()

    splits = ["train_input", "eval_input"]
    for split in splits:
        # set split specific output dir
        args.output_dir = os.path.join(output_dir, split)
        check_and_create_output_dirs(args.output_dir, filetype="h5")
        args.split = split

        dataset = _get_dataset(params[split])
        len_dataset = len(dataset)
        dataset_range = list(range(len_dataset))

        # Set defaults
        # Data augmentation should be on the fly when training model.
        params[split]["augment_data"] = False

        # Write generic data, the data gets converted to appropriate dtypes in
        # `transform_image_and_mask` fcn.
        params[split]["mixed_precision"] = False

        # Write data without hardcoding normalization.
        # This helps use the same files with HDFDataProcessor
        # and different normalization schemes
        params[split]["normalize_data_method"] = None

        if args.num_processes > 1:
            results = create_h5_mp(dataset_range, params[split], args)
        else:
            # Run only single process run, with process number set as 0.
            results = create_h5((dataset_range, params[split], args, 0))

        ## Update data_params file with new fields
        with open(json_params_file, 'r') as _fin:
            data = json.load(_fin)

        data[split].update(params[split])

        _key = f"{split}_hdf"
        data[_key] = {}
        data[_key]["n_docs"] = results["n_docs"]
        data[_key]["total_written"] = results["total_written"]
        data[_key]["dataset_range"] = results["dataset_range"]

        with open(json_params_file, 'w') as _fout:
            json.dump(data, _fout, indent=4, sort_keys=True)

        print(
            f"\nFinished writing {split} data to HDF5 to {args.output_dir}."
            + f" Runtime arguments and outputs can be found at {json_params_file}."
        )

        ## Store meta file.
        meta_file = os.path.join(output_dir, f"meta_{split}.dat")
        with open(meta_file, "w") as fout:
            for output_file, num_lines in results["meta_data"].items():
                fout.write(f"{output_file} {num_lines}\n")


if __name__ == "__main__":
    main()
