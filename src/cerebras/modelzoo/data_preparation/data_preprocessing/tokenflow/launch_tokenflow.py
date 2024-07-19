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
import os
from argparse import ArgumentParser
from copy import deepcopy

import h5py
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory

from cerebras.modelzoo.data_preparation.data_preprocessing.tokenflow import (
    tokenizer,
)

app = Flask(__name__)


class TokenFlowDataProcessor:
    def __init__(self, filepath, data_params):
        self.filepath = filepath
        self.data_params = data_params
        self.msl = data_params["processing"].get("max_seq_length")
        self.top_tokens = None
        self.pad_id = data_params['post-process'].get('pad_id')
        self.eos_id = data_params['post-process'].get('eos_id')
        self.sample_features = data_params['post-process'].get(
            'sample_features'
        )
        self.tokenizer = tokenizer.GenericTokenizer(
            deepcopy(data_params['processing']), filepath
        )
        # image_paths and image_data_locs are used in multimodal datasets
        self.data, self.image_paths, image_data_locs = self.load_data()
        self.datadict = {'image_paths': self.image_paths}
        for i, feature in enumerate(self.sample_features):
            ## rename attention_mask to loss_mask. Since dataloader incorrectly renames loss mask to attention mask.
            if feature == "attention_mask":
                feature = "loss_mask"
            self.datadict[feature] = self.data[:, i, :].copy()

        self.nstrings = self.datadict['input_ids'].shape[0]
        ## Inverted attention_mask is named as key_padding_mask in multimodal in other cases as it is not a part of data
        if "key_padding_mask" in self.datadict:
            self.datadict['attention_mask'] = (
                1 - self.datadict['key_padding_mask']
            ).tolist()
        else:
            # Manually construct the attention mask
            self.datadict['attention_mask'] = []
            for i in range(self.datadict['input_ids'].shape[0]):
                pad_indices = np.where(
                    self.datadict['input_ids'][i] == self.pad_id
                )[0]
                if self.eos_id != self.pad_id:
                    non_pad_len = int(
                        pad_indices[0]
                        if len(pad_indices) > 0
                        else self.datadict['input_ids'].shape[1]
                    )
                    self.datadict['attention_mask'].append(
                        [1] * (non_pad_len)
                        + [0]
                        * (self.datadict['input_ids'].shape[1] - non_pad_len)
                    )
                else:
                    ## When the eos id is same as the pad id, get the first pad index by noting that the eos will not
                    ## have a next eos while the pad id's would be contiguous
                    if len(pad_indices) > 0:
                        pad_idx = 0
                        while (
                            pad_idx + 1 < len(pad_indices)
                            and pad_indices[pad_idx] + 1
                            < self.datadict['input_ids'].shape[1]
                            and self.datadict['input_ids'][i][
                                pad_indices[pad_idx] + 1
                            ]
                            != self.pad_id
                        ):
                            pad_idx = pad_idx + 1
                        if pad_idx == len(pad_indices) - 1:
                            ## All eos no pad
                            non_pad_len = self.datadict['input_ids'].shape[1]
                        else:
                            ## the last eos just before pad if present would be chopped off from the input ids
                            non_pad_len = pad_indices[pad_idx]
                    else:
                        non_pad_len = self.datadict['input_ids'].shape[1]
                    self.datadict['attention_mask'].append(
                        [1] * (non_pad_len)
                        + [0]
                        * (self.datadict['input_ids'].shape[1] - non_pad_len)
                    )

        # This should modify the dict attr
        self.datadict['images_bitmap'] = np.zeros(
            self.datadict['input_ids'].shape
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
        self.datadict['input_strings'] = np.full(
            self.datadict['input_ids'].shape, '', dtype='<U20'
        )
        self.datadict['label_strings'] = np.full(
            self.datadict['labels'].shape, '', dtype='<U20'
        )

    def load_data(self):
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

            elif h5_file.get('data'):
                return np.array(h5_file['data']), np.array([]), np.array([])

            return np.array(h5_file['data_data']), np.array([]), np.array([])

    def get_top_tokens(self):
        # This needs on the fly calculations to get top 5 frequent tokens.
        if self.top_tokens:
            return self.top_tokens
        token_counts = {}
        for input_string in self.datadict['input_ids']:
            for token in input_string:
                if not token_counts.get(token):
                    token_counts[token] = 0
                token_counts[token] += 1
        top_tokens = dict(
            sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        top_tokens = {
            self.tokenizer.decode(token): val
            for token, val in top_tokens.items()
        }
        top_tokens = {
            token.strip(): val
            for token, val in top_tokens.items()
            if token.strip()
        }
        for key in top_tokens.keys():
            top_tokens[key] = int(top_tokens[key])
        self.top_tokens = {
            key: value
            for i, (key, value) in enumerate(top_tokens.items())
            if i < 5
        }
        return self.top_tokens

    def get_stats(self):
        stats = self.data_params['processing']
        stats.update(self.data_params['post-process'])
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
        response['input_ids'] = self.datadict['input_ids'][sequence]
        response['labels'] = self.datadict['labels'][sequence]
        response['input_strings'] = self.tokenizer.convert_ids_to_tokens(
            response['input_ids']
        )
        response['label_strings'] = self.tokenizer.convert_ids_to_tokens(
            response['labels']
        )
        response['images_bitmap'] = self.datadict['images_bitmap'][sequence]

        response['image_paths'] = []
        if self.data_params['dataset'].get('is_multimodal', False):
            response['image_paths'] = self.datadict['image_paths'][sequence]
        if 'loss_mask' in self.datadict:
            response['loss_mask'] = self.datadict['loss_mask'][sequence]
        response['attention_mask'] = self.datadict['attention_mask'][sequence]
        # Store back the decoded strings for faster call next time
        self.datadict['input_strings'][sequence] = response['input_strings']
        self.datadict['label_strings'][sequence] = response['label_strings']

        response.update(
            {'top_tokens': self.get_top_tokens(), 'stats': self.get_stats()}
        )
        for key, val in response.items():
            if isinstance(val, np.ndarray):
                response[key] = val.tolist()
        return response


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
        return f"File not Found or error in processing: {e}", 400


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
        return "Error in loading data_params json!", 400
    if os.path.isdir(args.output_dir):
        files = [
            os.path.join(args.output_dir, f)
            for f in os.listdir(args.output_dir)
            if f.endswith('.h5')
        ]
    elif os.path.isfile(args.files) and args.files.endswith('.h5'):
        files = [args.output_dir]
    if not files:
        return "File not Found or error", 404
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
            data_params['dataset']['image_dir'],
        ),
        filename,
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
