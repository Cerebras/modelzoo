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
Wrapper script to generate PubMed dataset
Reference: https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT

"""

import argparse
import glob
import json
import math
import multiprocessing
import os
import pprint
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../.."))


from modelzoo.common.input.utils import check_and_create_output_dirs
from modelzoo.transformers.data_processing.scripts.pubmed.preprocess import (
    Downloader,
    TextFormatting,
    TextSharding,
)
from modelzoo.transformers.tf.bert.input.scripts.create_tfrecords import (
    create_tfrecords,
)


def main(args):
    data_dir = args.output_dir

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if os.path.basename(data_dir) != args.dataset:
        data_dir = os.path.join(data_dir, args.dataset)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    print('Data Directory:', data_dir)
    print('Action:', args.action)
    print('Dataset Name:', args.dataset)

    # if args.input_files:
    #     args.input_files = args.input_files.split(',')

    ### Generate TFrecord folder name

    directory_structure = {
        'download': data_dir + '/download',  # Downloaded and decompressed
        'extracted': data_dir
        + '/download'
        + '/extracted',  # Extracted from whatever the initial format is (e.g., wikiextractor)
        'formatted': data_dir
        + '/formatted',  # This is the level where all sources should look the same
        'sharded': data_dir + '/sharded',
        'tfrecord': data_dir + '/tfrecord',
    }

    print('\nDirectory Structure:')
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(directory_structure)
    print('')

    if args.action == 'download':

        downloader = Downloader.Downloader(
            args.dataset, directory_structure['download']
        )
        downloader.download()

    elif args.action == 'text_formatting':

        if args.input_files is None:
            pubmed_path = directory_structure['extracted']
        else:
            pubmed_path = args.input_files

        filesize_limit = 5 * (10 ** 9)
        output_filename = os.path.join(
            directory_structure['formatted'],
            f"{args.dataset}_one_article_per_line.txt",
        )

        pubmed_formatter = TextFormatting.TextFormatting(
            pubmed_path, output_filename, filesize_limit, recursive=True
        )
        pubmed_formatter.merge(args.dataset)

    elif args.action == 'sharding':
        # Note: requires user to provide list of input_files
        if 'pubmed' in args.dataset:
            if args.input_files is None:
                args.input_files = glob.glob(
                    os.path.join(directory_structure['formatted'], '*.txt')
                )
            else:
                args.input_files = glob.glob(
                    os.path.join(args.input_files, '*.txt')
                )

            args.input_files = [
                x for x in args.input_files if os.stat(x).st_size > 0
            ]

            output_file_prefix = os.path.join(
                directory_structure['sharded'], args.dataset
            )

            if not os.path.exists(directory_structure['sharded']):
                os.makedirs(directory_structure['sharded'])

            if not os.path.exists(
                os.path.join(directory_structure['sharded'], 'training')
            ):
                os.makedirs(
                    os.path.join(directory_structure['sharded'], 'training')
                )

            if not os.path.exists(
                os.path.join(directory_structure['sharded'], 'test')
            ):
                os.makedirs(
                    os.path.join(directory_structure['sharded'], 'test')
                )

            # Segmentation is here because all datasets look the same in one article/book/whatever per line format, and
            # it seemed unnecessarily complicated to add an additional preprocessing step to call just for this.
            # Different languages (e.g., Chinese simplified/traditional) may require translation and
            # other packages to be called from here -- just add a conditional branch for those extra steps

            if args.dataset in [
                "pubmed_baseline",
                "pubmed_daily_update",
                "pubmed_fulltext",
                "pubmed_open_access",
            ]:
                segmenter = TextSharding.NLTKSegmenter()
                num_input_files = len(args.input_files)

                fname_size = {
                    x: (os.stat(x).st_size) / 10 ** 9 for x in args.input_files
                }  ## calculate number of GB's
                total_files_size = sum(list(fname_size.values()))
                num_train_shards_per_file = {}
                num_test_shards_per_file = {}

                # shard each "formatted" file into test and train subsets based on proportion of total size of files

                for i, f_sz in enumerate(fname_size.items()):
                    file, si = f_sz
                    num_train_shards_per_file[file] = math.ceil(
                        (si / total_files_size) * args.n_training_shards
                    )
                    num_test_shards_per_file[file] = math.ceil(
                        (si / total_files_size) * args.n_test_shards
                    )

                print("================================")
                print(f"file size:{fname_size}")
                print(
                    f"num_train_shards_per_file :{num_train_shards_per_file}, {sum(list(num_train_shards_per_file.values()))}"
                )
                print(
                    f"num_test_shards_per_file : {num_test_shards_per_file}, {sum(list(num_test_shards_per_file.values()))}"
                )
                print(f"total_files_size:{total_files_size}")
                print("================================")

                processes = []

                def multiprocessing_func(
                    filepath, num_train_shards, num_test_shards
                ):
                    print(
                        f"*************************\n",
                        f"process id: {os.getpid()}",
                        f"filepath: {filepath}",
                        f"filesize: {(os.stat(filepath).st_size)/10**9}",
                        f"num_train_shards:{num_train_shards}",
                        f"num_test_shards:{num_test_shards}",
                    )
                    fname = os.path.basename(filepath)
                    fname = fname.split('.txt')[0]
                    output_file_prefix = os.path.join(
                        directory_structure['sharded'], fname
                    )

                    sharding = TextSharding.Sharding(
                        [file],
                        output_file_prefix,
                        num_train_shards,
                        num_test_shards,
                        args.fraction_test_set,
                    )
                    sharding.load_articles()
                    sharding.segment_articles_into_sentences(segmenter)
                    sharding.distribute_articles_over_shards()
                    sharding.write_shards_to_disk()

                for file in args.input_files:
                    print("**************************************")
                    p = multiprocessing.Process(
                        target=multiprocessing_func,
                        args=(
                            file,
                            num_train_shards_per_file[file],
                            num_test_shards_per_file[file],
                        ),
                    )
                    processes.append(p)
                    p.start()

                for process in processes:
                    process.join()

        else:
            assert False, 'Unsupported dataset for sharding'

    elif args.action == 'create_tfrecord_files':

        def _write_metadata_files(input_folder, output_folder):
            files = glob.glob(os.path.join(input_folder, '*.txt'))
            for filepath in files:
                filename = os.path.basename(filepath)
                filename = 'meta_' + filename
                out_filename = os.path.join(output_folder, filename)
                with open(out_filename, 'w') as fh:
                    fh.write(filepath)

        # Create tfrecord_prefix
        tfrecord_folder_prefix = ""
        if args.do_lower_case:
            tfrecord_folder_prefix += "/uncased_"
        else:
            tfrecord_folder_prefix += "/cased_"

        tfrecord_folder_prefix = (
            tfrecord_folder_prefix
            + f"msl{args.max_seq_length}"
            + f"_mp{args.max_predictions_per_seq}"
            + f"_wwm{str(args.mask_whole_word)}"
            + f"_dupe{args.dupe_factor}"
        )

        directory_structure["tfrecord"] = (
            directory_structure["tfrecord"] + tfrecord_folder_prefix
        )

        pp.pprint(directory_structure)

        ############## Write train TFrecords #############
        if args.input_files is None:
            train_sharded_folder = os.path.join(
                directory_structure['sharded'], 'training'
            )
        else:
            train_sharded_folder = os.path.join(args.input_files, 'training')

        train_output_dir = os.path.join(
            directory_structure['tfrecord'], 'training'
        )

        check_and_create_output_dirs(train_output_dir, filetype="tfrecord")

        train_meta_folder = os.path.join(train_output_dir, 'metadata_files')

        if not os.path.exists(train_meta_folder):
            os.makedirs(train_meta_folder)

            _write_metadata_files(train_sharded_folder, train_meta_folder)

        train_files = glob.glob(os.path.join(train_meta_folder, "*.txt"))

        n_processes = args.n_processes
        total_train_examples = 0

        pool = multiprocessing.Pool(processes=n_processes)
        for idx in range(0, len(train_files), n_processes):
            # Pass one file to one pool process
            _files = train_files[idx : idx + n_processes]

            pool_results = []
            for file in _files:
                fname = os.path.basename(file)
                tfrecord_name_prefix = fname.split('.txt')[0]
                tfrecord_name_prefix = tfrecord_name_prefix.replace("meta_", "")

                kwargs = {
                    "metadata_files": file,
                    "single_sentence_per_line": True,
                    "multiple_docs_in_single_file": True,
                    "multiple_docs_separator": "\\n",
                    "sentence_pair": True,
                    "vocab_file": args.vocab_file,
                    "do_lower_case": args.do_lower_case,
                    "split_num": 1,
                    "max_seq_length": args.max_seq_length,
                    "short_seq_prob": args.short_seq_prob,
                    "mask_whole_word": args.mask_whole_word,
                    "max_predictions_per_seq": args.max_predictions_per_seq,
                    "masked_lm_prob": args.masked_lm_prob,
                    "dupe_factor": args.dupe_factor,
                    "inverted_mask": False,
                    "seed": args.random_seed,
                    "tfrecord_name_prefix": tfrecord_name_prefix,
                    "output_dir": train_output_dir,
                    "num_output_files": 1,
                }

                pool_results.append(
                    pool.apply_async(create_tfrecords, kwds=kwargs)
                )

            # Wait till `n_processes` number of files are written to TFrecords. This can be problematic if one process takes significantly longer time than others.
            for res in pool_results:
                total_train_examples += res.get()

        print(f"Total examples in training dataset: {total_train_examples}")

        ############# Write Test TFrecords #############
        if args.input_files is None:
            test_sharded_folder = os.path.join(
                directory_structure['sharded'], 'test'
            )
        else:
            test_sharded_folder = os.path.join(args.input_files, 'test')

        test_output_dir = os.path.join(directory_structure['tfrecord'], 'test')

        check_and_create_output_dirs(test_output_dir, filetype="tfrecord")

        test_meta_folder = os.path.join(test_output_dir, 'metadata_files')

        if not os.path.exists(test_meta_folder):
            os.makedirs(test_meta_folder)

            _write_metadata_files(test_sharded_folder, test_meta_folder)

        test_files = glob.glob(os.path.join(test_meta_folder, "*.txt"))

        n_processes = args.n_processes
        total_test_examples = 0

        for idx in range(0, len(test_files), n_processes):
            # Pass one file to one pool process
            _files = test_files[idx : idx + n_processes]

            pool_results = []
            for file in _files:
                fname = os.path.basename(file)
                tfrecord_name_prefix = fname.split('.txt')[0]
                tfrecord_name_prefix = tfrecord_name_prefix.replace("meta_", "")

                kwargs = {
                    "metadata_files": file,
                    "single_sentence_per_line": True,
                    "multiple_docs_in_single_file": True,
                    "multiple_docs_separator": "\\n",
                    "sentence_pair": True,
                    "vocab_file": args.vocab_file,
                    "do_lower_case": args.do_lower_case,
                    "split_num": 1,
                    "max_seq_length": args.max_seq_length,
                    "short_seq_prob": args.short_seq_prob,
                    "mask_whole_word": args.mask_whole_word,
                    "max_predictions_per_seq": args.max_predictions_per_seq,
                    "masked_lm_prob": args.masked_lm_prob,
                    "dupe_factor": args.dupe_factor,
                    "inverted_mask": False,
                    "seed": args.random_seed,
                    "tfrecord_name_prefix": tfrecord_name_prefix,
                    "output_dir": test_output_dir,
                    "num_output_files": 1,
                }
                pool_results.append(
                    pool.apply_async(create_tfrecords, kwds=kwargs)
                )

            # Wait till `n_processes` number of files are written to TFrecords. Can be further optimized by passing similar size files
            for res in pool_results:
                total_test_examples += res.get()

        print(f"Total examples in Test dataset: {total_test_examples}")
        pool.close()
        pool.join()

        params_file = os.path.join(
            directory_structure['tfrecord'], "params.json"
        )
        params = vars(args)
        params['num_train_examples'] = total_train_examples
        params['num_test_examples'] = total_test_examples

        with open(params_file, 'w') as outfile:
            print("Writing json")
            json.dump(vars(args), outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocessing Application for PubMedBert Dataset'
    )

    parser.add_argument(
        '--action',
        type=str,
        help='Specify the action you want the app to take. e.g., generate vocab, segment, create tfrecords',
        choices={
            'download',  # Download and verify mdf5/sha sums
            'text_formatting',  # Convert into a file that contains one article/book per line
            'sharding',  # Convert previous formatted text into shards containing one sentence per line
            'create_tfrecord_files',  # Turn each shard into a TFrecord with masking and next sentence prediction info
        },
    )

    parser.add_argument(
        '--dataset',
        type=str,
        help='Specify the dataset to perform --action on',
        choices={
            'pubmed_baseline',
            'pubmed_daily_update',
            'pubmed_fulltext',
            'pubmed_open_access',
            'all',
        },
        required=True,
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        help='Output parent folder where files/subfolders are written to',
        required=True,
    )

    parser.add_argument(
        '--input_files', type=str, help='Specify the input path for files',
    )

    parser.add_argument(
        '--n_training_shards',
        type=int,
        help='Specify the number of training shards to generate',
        default=1472,
    )

    parser.add_argument(
        '--n_test_shards',
        type=int,
        help='Specify the number of test shards to generate',
        default=1472,
    )

    parser.add_argument(
        '--fraction_test_set',
        type=float,
        help='Specify the fraction (0..1) of the data to withhold for the test data split (based on number of sequences)',
        default=0.1,
    )

    parser.add_argument(
        '--segmentation_method',
        type=str,
        help='Specify your choice of sentence segmentation',
        choices={'nltk'},
        default='nltk',
    )

    parser.add_argument(
        '--n_processes',
        type=int,
        help='Specify the max number of processes to allow at one time',
        default=multiprocessing.cpu_count(),
    )

    parser.add_argument(
        '--random_seed',
        type=int,
        help='Specify the base seed to use for any random number generation',
        default=0,
    )

    parser.add_argument(
        '--dupe_factor',
        type=int,
        help='Specify the duplication factor',
        default=10,
    )

    parser.add_argument(
        '--masked_lm_prob',
        type=float,
        help='Specify the probability for masked lm',
        default=0.15,
    )

    parser.add_argument(
        '--max_seq_length',
        type=int,
        help='Specify the maximum sequence length',
        default=128,
    )

    parser.add_argument(
        '--max_predictions_per_seq',
        type=int,
        help='Specify the maximum number of masked words per sequence',
        default=20,
    )

    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="pass this flag to lower case the input text; should be "
        "True for uncased models and False for cased models",
    )

    parser.add_argument(
        "--mask_whole_word",
        action="store_true",
        help="whether to use whole word masking rather than per-WordPiece "
        "masking.",
    )

    parser.add_argument(
        '--vocab_file',
        type=str,
        required=False,
        help='Specify absolute path to vocab file to use)',
    )

    parser.add_argument(
        "--short_seq_prob",
        type=float,
        default=0.1,
        help="probability of creating sequences which are shorter "
        "than the maximum sequence length",
    )

    args = parser.parse_args()
    main(args)
