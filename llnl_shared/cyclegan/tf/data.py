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

import numpy as np
import tensorflow as tf


def _normalize(data):
    mn = np.min(data)
    data = data - mn
    mx = np.max(data)
    if mx != 0:
        data = data / mx
    return data


def _parse_sample(sample, mixed_precision=True):
    """Casts images to float16 if mixed_precision is True"""
    image = tf.cast(
        sample['image'], tf.float16 if mixed_precision else tf.float32,
    )
    output = tf.cast(
        sample['output'], tf.float16 if mixed_precision else tf.float32,
    )
    return image, output


def input_fn(params):
    """Generates the input_fn. Uses either the training set or the \
       eval set depending on params['mode']"""
    iparams = params['input']
    data_dir = iparams['data_directory']
    images = np.load(os.path.join(data_dir, iparams['image_filename']))
    scalars = np.load(os.path.join(data_dir, iparams['scalar_filename']))
    input_params = np.load(os.path.join(data_dir, iparams['input_filename']))
    embeddings = np.load(os.path.join(data_dir, iparams['embeddings_filename']))

    np.random.seed(iparams['random_seed'])
    inds = np.random.choice(
        scalars.shape[0], int(scalars.shape[0] * 0.8), replace=False
    )

    input_dict = {}
    if 'mode' in params['runconfig']:
        if params['runconfig']['mode'] == tf.estimator.ModeKeys.EVAL:
            test_inds = list(set(range(scalars.shape[0])) - set(inds))
            ### limit the number of test images based on params
            if params['runconfig']['max_steps'] < len(test_inds):
                test_inds = test_inds[0 : params['runconfig']['max_steps']]
            np.random.shuffle(test_inds)
            inds = test_inds
    num_images = len(inds)

    # brought in the changes by Jason Wolfe for SW-27170 - peterhu
    input_dict['inputs'] = _normalize(input_params[inds, :])
    input_dict['scalars'] = _normalize(scalars[inds, :])
    input_dict['images'] = _normalize(images[inds, :])
    input_dict['latent_tensors'] = _normalize(embeddings[inds, :])

    input_dict['adv_labels'] = np.zeros((len(inds), 1))

    inputs = [input_dict[input_name] for input_name in iparams['input_names']]
    inputs = np.concatenate(inputs, axis=1)
    labels = [
        input_dict[output_name] for output_name in iparams['output_names']
    ]
    labels = np.concatenate(labels, axis=1)

    ds = tf.data.Dataset.from_tensor_slices({'image': inputs, 'output': labels})

    ds = (
        ds.shuffle(num_images)
        .batch(iparams['batch_size'], drop_remainder=True)
        .prefetch(iparams['prefetch_size'])
    )
    if 'mode' in params:
        if params['mode'] == tf.estimator.ModeKeys.TRAIN:
            ds = ds.repeat()
    ds = ds.map(
        lambda x: _parse_sample(x, iparams['mixed_precision']),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return ds
