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

# isort: off
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../.."))
# isort: on

import json
import logging
import os
import traceback
from argparse import ArgumentParser
from copy import deepcopy

import h5py
import numpy as np
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from cerebras.modelzoo.data_preparation.data_preprocessing.tokenflow import (
    tokenizer,
)
from cerebras.modelzoo.data_preparation.data_preprocessing.tokenflow.utils import (
    construct_attention_mask,
)

app = Flask(__name__)

logging.basicConfig(
    level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s'
)


class TokenFlowDataProcessor:
    def __init__(self, filepath, data_params):
        self.filepath = filepath
        self.data_params = data_params

        assert (
            'processing' in data_params
        ), "'processing' key must be in data_params"
        assert 'setup' in data_params, "'setup' key must be in data_params"
        assert 'features' in data_params, "'features' must be in data_params"

        assert (
            'max_seq_length' in data_params['processing']
        ), "'max_seq_length' must be in 'processing'"
        assert (
            'pad_id' in data_params['processing']
        ), "'pad_id' must be in 'processing'"
        assert (
            'eos_id' in data_params['processing']
        ), "'eos_id' must be in 'processing'"
        assert 'mode' in data_params['setup'], "'mode' must be in 'setup'"

        self.msl = data_params["processing"].get("max_seq_length")
        self.pad_id = data_params['processing'].get('pad_id')
        self.eos_id = data_params['processing'].get('eos_id')
        self.mode = data_params['setup'].get('mode')
        self.features = data_params.get('features')
        self.is_multimodal = self.data_params['dataset'].get(
            'is_multimodal', False
        )

        self.input_ids = 'text_input_ids' if self.is_multimodal else 'input_ids'
        self.tokenizer = tokenizer.GenericTokenizer(
            deepcopy(data_params), filepath
        )

        self.datadict = {}

        # Special handling for MLM mode while loading data.
        if data_params['dataset'].get('training_objective') == 'mlm':
            # Handle load_data() for MLM separately as it returns an extra array.
            self.mlm_with_gather = data_params['dataset'].get('mlm_with_gather')
            self.ignore_index = data_params['dataset'].get('ignore_index', -100)

            self.data, labels, self.image_paths, image_data_locs = (
                self.load_data()
            )

            if not self.mlm_with_gather:
                self.datadict['labels'] = labels[:, 0, :].copy()
            else:
                self.datadict['labels'] = labels
        else:
            # image_paths and image_data_locs are used in multimodal datasets
            self.data, self.image_paths, image_data_locs = self.load_data()

        self.datadict['image_paths'] = self.image_paths

        if self.mode == 'dpo':
            for i, feature in enumerate(self.features):
                if feature == "chosen_attention_mask":
                    feature = "chosen_loss_mask"

                if feature == "rejected_attention_mask":
                    feature = "rejected_loss_mask"

                self.datadict[feature] = self.data[:, i, :].copy()

            chosen_nstrings = self.datadict['chosen_input_ids'].shape[0]
            rejected_nstrings = self.datadict['rejected_input_ids'].shape[0]
            self.nstrings = chosen_nstrings + rejected_nstrings
        else:
            # Special handling for MLM mode; we don't need to construct labels as we've done it above.
            ## Also, rename attention_mask to loss_mask. Since dataloader incorrectly renames loss mask to attention mask.
            if data_params['dataset'].get('training_objective') == 'mlm':
                self.datadict['input_ids'] = self.data[:, 0, :].copy()
                self.datadict['loss_mask'] = self.data[:, 1, :].copy()
            else:
                for i, feature in enumerate(self.features):
                    if feature == "attention_mask":
                        feature = "loss_mask"

                    self.datadict[feature] = self.data[:, i, :].copy()

            self.nstrings = self.datadict[self.input_ids].shape[0]

        ## Inverted attention_mask is named as key_padding_mask in multimodal in other cases as it is not a part of data
        if "key_padding_mask" in self.datadict:
            self.datadict['attention_mask'] = (
                1 - self.datadict['key_padding_mask']
            ).tolist()
        else:
            if self.mode == 'dpo':
                self.datadict['chosen_attention_mask'] = (
                    construct_attention_mask(
                        self.datadict,
                        self.eos_id,
                        self.pad_id,
                        input_key='chosen_input_ids',
                    )
                )
                self.datadict['rejected_attention_mask'] = (
                    construct_attention_mask(
                        self.datadict,
                        self.eos_id,
                        self.pad_id,
                        input_key='rejected_input_ids',
                    )
                )
            else:
                self.datadict['attention_mask'] = construct_attention_mask(
                    self.datadict,
                    self.eos_id,
                    self.pad_id,
                    input_key=self.input_ids,
                )

        # This should modify the dict attr
        if self.mode == 'dpo':
            self.datadict['images_bitmap'] = np.zeros(
                self.datadict['chosen_input_ids'].shape
            )
        else:
            self.datadict['images_bitmap'] = np.zeros(
                self.datadict[self.input_ids].shape
            )

        if image_data_locs.size:
            for i in range(image_data_locs.shape[0]):
                image_index = 1
                for j in range(image_data_locs.shape[1]):
                    if image_data_locs[i][j][0] == self.msl:
                        break
                    for k in image_data_locs[i][j]:
                        self.datadict['images_bitmap'][i][k] = image_index
                    image_index += 1

        # Construct dummy input_strings and label_strings that are overwritten once decoded on-demand
        if self.mode == 'dpo':
            # Chosen responses.
            self.datadict['chosen_input_strings'] = np.full(
                self.datadict['chosen_input_ids'].shape, '', dtype='<U20'
            )
            self.datadict['chosen_label_strings'] = np.full(
                self.datadict['chosen_labels'].shape, '', dtype='<U20'
            )

            # Rejected responses.
            self.datadict['rejected_input_strings'] = np.full(
                self.datadict['rejected_input_ids'].shape, '', dtype='<U20'
            )
            self.datadict['rejected_label_strings'] = np.full(
                self.datadict['rejected_labels'].shape, '', dtype='<U20'
            )
        else:
            self.datadict['input_strings'] = np.full(
                self.datadict[self.input_ids].shape, '', dtype='<U20'
            )
            self.datadict['label_strings'] = np.full(
                self.datadict['labels'].shape, '', dtype='<U20'
            )

    def load_data(self):
        try:
            with h5py.File(self.filepath, mode='r') as h5_file:
                # Multimodal data has this format
                if self.data_params['dataset'].get('is_multimodal', False):
                    return (
                        np.array(h5_file['data']),
                        np.array(
                            [
                                [i.decode('utf-8') for i in paths]
                                for paths in h5_file.get('img_path')
                            ]
                        ),
                        np.array(h5_file.get('img_data_loc')),
                    )

                # MLM data has this format.
                elif (
                    self.data_params['dataset'].get('training_objective')
                    == 'mlm'
                ):
                    if not self.mlm_with_gather:
                        labels = np.array(h5_file['labels'])
                        return (
                            np.array(h5_file['data']),
                            labels,
                            np.array([]),
                            np.array([]),
                        )
                    else:
                        labels = np.array(h5_file['labels'][:, 0, :])
                        masked_lm_positions_list = np.array(
                            h5_file['labels'][:, 1, :]
                        )
                        masked_lm_weights_list = np.array(
                            h5_file['labels'][:, 2, :]
                        )
                        updated_shape = (labels.shape[0], self.msl)
                        updated_labels = np.full(
                            updated_shape, self.ignore_index
                        )
                        for i in range(labels.shape[0]):
                            positions = masked_lm_positions_list[i]
                            updated_labels[i, positions] = labels[i]

                        return (
                            np.array(h5_file['data']),
                            updated_labels,
                            np.array([]),
                            np.array([]),
                        )
                elif h5_file.get('data'):
                    return np.array(h5_file['data']), np.array([]), np.array([])

                return (
                    np.array(h5_file['data_data']),
                    np.array([]),
                    np.array([]),
                )
        except Exception as e:
            logging.error(f"Failed to load data from {self.filepath}: {str(e)}")
            logging.error(traceback.format_exc())
            raise RuntimeError(
                f"Error while loading data from {self.filepath}: {str(e)}"
            )

    def get_stats(self):
        stats = self.data_params['processing']
        stats.update(self.data_params['setup'])
        stats.update(self.data_params['dataset'])
        stats['multimodal'] = self.data_params['dataset'].get(
            'is_multimodal', False
        )

        # Removed empty keys
        for attr in list(stats.keys()):
            if stats[attr] is None:
                stats.pop(attr)
        # Data is trivial to the user
        if stats.get('data'):
            stats.pop('data')
        if stats.get('input_dir'):
            stats.pop('input_dir')
        return stats

    def get_datadict(self, sequence):
        response = {}

        if self.mode != 'dpo':
            response['input_ids'] = self.datadict[self.input_ids][sequence]
            response['labels'] = self.datadict['labels'][sequence]
            response['input_strings'] = self.tokenizer.convert_ids_to_tokens(
                response['input_ids']
            )
            response['label_strings'] = self.tokenizer.convert_ids_to_tokens(
                response['labels']
            )

            if 'images_bitmap' in self.datadict:
                response['images_bitmap'] = self.datadict['images_bitmap'][
                    sequence
                ]

            response['image_paths'] = []
            if self.data_params['dataset'].get('is_multimodal', False):
                response['image_paths'] = self.datadict['image_paths'][sequence]
            if 'loss_mask' in self.datadict:
                response['loss_mask'] = self.datadict['loss_mask'][sequence]
            response['attention_mask'] = self.datadict['attention_mask'][
                sequence
            ]

            # Store back the decoded strings for faster call next time
            self.datadict['input_strings'][sequence] = response['input_strings']
            self.datadict['label_strings'][sequence] = response['label_strings']
        else:
            # Update response for chosen.
            response['chosen_input_ids'] = self.datadict['chosen_input_ids'][
                sequence
            ]
            response['chosen_labels'] = self.datadict['chosen_labels'][sequence]

            response['chosen_input_strings'] = (
                self.tokenizer.convert_ids_to_tokens(
                    response['chosen_input_ids']
                )
            )
            response['chosen_label_strings'] = (
                self.tokenizer.convert_ids_to_tokens(response['chosen_labels'])
            )

            if 'chosen_loss_mask' in self.datadict:
                response['chosen_loss_mask'] = self.datadict[
                    'chosen_loss_mask'
                ][sequence]

            response['chosen_attention_mask'] = self.datadict[
                'chosen_attention_mask'
            ][sequence]

            self.datadict['chosen_input_strings'][sequence] = response[
                'chosen_input_strings'
            ]
            self.datadict['chosen_label_strings'][sequence] = response[
                'chosen_label_strings'
            ]

            # Update response for rejected.
            response['rejected_input_ids'] = self.datadict[
                'rejected_input_ids'
            ][sequence]
            response['rejected_labels'] = self.datadict['rejected_labels'][
                sequence
            ]

            response['rejected_input_strings'] = (
                self.tokenizer.convert_ids_to_tokens(
                    response['rejected_input_ids']
                )
            )
            response['rejected_label_strings'] = (
                self.tokenizer.convert_ids_to_tokens(
                    response['rejected_labels']
                )
            )

            if 'rejected_loss_mask' in self.datadict:
                response['rejected_loss_mask'] = self.datadict[
                    'rejected_loss_mask'
                ][sequence]

            response['rejected_attention_mask'] = self.datadict[
                'rejected_attention_mask'
            ][sequence]

            self.datadict['rejected_input_strings'][sequence] = response[
                'rejected_input_strings'
            ]
            self.datadict['rejected_label_strings'][sequence] = response[
                'rejected_label_strings'
            ]

        # Update responses related to image.
        if 'images_bitmap' in self.datadict:
            response['images_bitmap'] = self.datadict['images_bitmap'][sequence]

        response['image_paths'] = []
        if self.data_params['dataset'].get('is_multimodal', False):
            response['image_paths'] = self.datadict['image_paths'][sequence]

        response.update({'stats': self.get_stats()})
        for key, val in response.items():
            if isinstance(val, np.ndarray):
                response[key] = val.tolist()

        return response


def process_file_for_sequence_distribution(filename, bin_edges, pad_id, msl):
    import os

    import h5py
    import numpy as np

    length_of_sequences = np.zeros(len(bin_edges) - 1, dtype=int)
    sequence_lengths = []

    filename = os.path.abspath(filename)

    try:
        with h5py.File(filename, mode='r') as h5_file:
            data = h5_file["data"][:]
            no_of_sequences = data.shape[0]

            for i in range(no_of_sequences):
                tokens = data[i, 0]

                if pad_id in tokens:
                    sequence_length = np.argmax(tokens == pad_id)
                else:
                    sequence_length = (
                        msl  # Sequence has no padding, use max length
                    )

                sequence_lengths.append(sequence_length)

                # Calculate the bin
                percentage = (sequence_length * 100) // msl
                bin_number = min(percentage // 5, len(length_of_sequences) - 1)

                length_of_sequences[bin_number] += 1
    except Exception as e:
        return np.zeros(len(bin_edges) - 1, dtype=int), []

    return length_of_sequences, sequence_lengths


def save_sequence_distribution(file_directory, data_params_path):
    import glob
    import json
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from os import cpu_count

    import matplotlib.pyplot as plt
    import numpy as np

    h5_files = glob.glob(os.path.join(file_directory, "*.h5"))

    sequence_dist_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'static',
        'images',
    )

    with open(data_params_path, 'r') as json_file:
        data_params = json.load(json_file)

    all_sequence_lengths = []

    pad_id = data_params['processing'].get('pad_id')
    msl = data_params['processing'].get('max_seq_length')

    num_bins = 20
    bin_edges = np.linspace(0, msl, num_bins + 1)
    length_of_sequences_total = np.zeros(len(bin_edges) - 1, dtype=int)

    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = [
            executor.submit(
                process_file_for_sequence_distribution,
                filename,
                bin_edges,
                pad_id,
                msl,
            )
            for filename in h5_files
        ]
        for future in as_completed(futures):
            length_of_sequences, sequence_lengths = future.result()
            length_of_sequences_total += length_of_sequences
            all_sequence_lengths.extend(sequence_lengths)

    total_sequences = np.sum(length_of_sequences_total)

    # Calculate percentage distribution
    if total_sequences > 0:
        length_of_sequences_percentage = (
            length_of_sequences_total / total_sequences
        ) * 100
    else:
        length_of_sequences_percentage = length_of_sequences_total

    # Calculate mean and standard deviation of the sequence lengths
    if all_sequence_lengths:
        mean_sequence_length = np.mean(all_sequence_lengths)
        std_sequence_length = np.std(all_sequence_lengths)
    else:
        mean_sequence_length = 0
        std_sequence_length = 0

    plt.figure(figsize=(12, 6))

    ranges = [f'{int(bin_edges[i+1])}' for i in range(len(bin_edges) - 1)]
    text_ranges = [
        f'{int(bin_edges[i])}--{int(bin_edges[i+1])}'
        for i in range(len(bin_edges) - 1)
    ]

    plt.bar(ranges, length_of_sequences_percentage, color='skyblue')

    # Create a string that contains the information for each range
    percentage_info = "\n".join(
        [
            f"{range_label}: {v:.1f}%"
            for range_label, v in zip(
                text_ranges, length_of_sequences_percentage
            )
        ]
    )

    # Add a text box inside the plot with mean and std
    text_info = f"Mean: {mean_sequence_length:.1f}\nStd: {std_sequence_length:.1f}\n\n{percentage_info}"

    plt.gca().text(
        1.05,
        0.5,
        text_info,
        transform=plt.gca().transAxes,
        bbox=dict(
            facecolor='white', edgecolor='black', boxstyle='round,pad=1.0'
        ),
        verticalalignment='center',
        fontsize=6,
    )

    plt.gca().set_position([0.1, 0.1, 0.75, 0.8])

    plt.xlabel('MSL Length')
    plt.ylabel('% of sequences')

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    image_filename = f'sequence_distribution.png'
    image_path = os.path.join(sequence_dist_dir, image_filename)
    plt.title(f'Sequence Distribution Plot')
    plt.savefig(image_path)
    plt.close()


def get_data_or_error(filename):
    global data_processors
    global data_params
    try:
        if not data_processors[filename]:
            data_processors[filename] = TokenFlowDataProcessor(
                filename, data_params
            )
        return data_processors[filename]
    except Exception as e:
        logging.error(f"Error processing the file {filename}: {str(e)}")
        logging.error(traceback.format_exc())

        return (
            f"The requested file is not found (or error in processing -- please check the logs for more details.): {e}",
            400,
        )


def load_params(args):
    try:
        with open(args.data_params, 'r') as json_file:
            return json.load(json_file)
    except:
        return


@app.route('/')
def index():
    global args
    global data_params
    global data_processors
    files = []
    initial_data = None
    # Load data_params here to make it a one time operation
    data_params = load_params(args)
    if not data_params:
        return (
            "Error in loading data_params.json! Please check if the output directory contains the data_params.json file, or specify it as a CLI argument.",
            400,
        )
    if os.path.isdir(args.output_dir):
        files = [
            os.path.join(args.output_dir, f)
            for f in os.listdir(args.output_dir)
            if f.endswith('.h5')
        ]
        if not files:
            return (
                "There are no HDF5 files present in the directory. Please check the directory.",
                404,
            )

    elif os.path.isfile(args.output_dir) and args.output_dir.endswith('.h5'):
        files = [args.output_dir]
        if not files:
            return (
                "The passed file is not a valid HDF5 file. Please check the file.",
                404,
            )

    # Get initial data that is supposed to be loaded.
    data_processors = {file: None for file in files}
    initial_data = get_data_or_error(files[0])
    if not isinstance(initial_data, TokenFlowDataProcessor):
        return jsonify({"error": initial_data[0], "code": initial_data[1]})

    return render_template(
        'index.html',
        files=files,
        initial_data=initial_data.get_datadict(0),
        nstrings=initial_data.nstrings,
    )


@app.route('/data', methods=['POST'])
def data():
    filename = request.form['filename']
    sequence = request.form['sequence']
    processor = get_data_or_error(filename)
    if not isinstance(processor, TokenFlowDataProcessor):
        return jsonify({"error": processor[0], "code": processor[1]})
    response = processor.get_datadict(int(sequence))
    response['nstrings'] = processor.nstrings
    return response


@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(
        os.path.join(
            os.path.dirname(os.path.abspath(args.data_params)),
            data_params['setup']['image_dir'],
        ),
        filename,
    )


@app.route('/get_data_params')
def get_data():
    with open(args.data_params, 'r') as file:
        data = json.load(file)
    return jsonify(data)


@app.route('/generate_sequence_distribution', methods=['POST'])
def serve_sequence_distribtion():
    try:
        file_directory = args.output_dir
        save_sequence_distribution(file_directory, args.data_params)
        image_path = url_for(
            'static', filename='images/sequence_distribution.png'
        )
        return jsonify({'image_path': image_path}), 200
    except Exception as e:
        return (
            jsonify({'Unable to retrieve sequence distribution!': str(e)}),
            500,
        )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory/location of one or more HDF5 files. In case a single file is passed, data_params should also be specified",
        required=True,
    )
    parser.add_argument(
        "--data_params",
        type=str,
        help="Location of data_params, required for loading heruistics related to the preprocessed data",
    )
    parser.add_argument(
        "--port", type=int, help="Port to run the Flask app on", default=5000
    )

    global args
    args = parser.parse_args()
    if not args.data_params:
        if os.path.isdir(args.output_dir):
            args.data_params = os.path.join(args.output_dir, 'data_params.json')
        else:
            exit(
                "Use --data_params <path/to/file> to specify the path of data_params.json. Required when passing a single HDF5 file."
            )
    app.run(debug=True, host='0.0.0.0', port=args.port)
