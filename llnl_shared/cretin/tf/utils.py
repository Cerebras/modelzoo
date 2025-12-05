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
Helper functions
"""

import os

import numpy as np
import tensorflow as tf
import yaml

_curdir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PARAMS_FILE = os.path.join(_curdir, "params.yaml")


def get_params(params_file=DEFAULT_PARAMS_FILE, config="base"):
    """
    Return params dict from yaml file.
    """
    with open(params_file, "r") as stream:
        params = yaml.safe_load(stream)
    if config:
        params = params[config]
    return params


def update_params(params, new_params):
    """
    Update params in the nested yaml file
    """

    def change_key(d, tgt_k, tgt_v):
        for k, v in d.items():
            if isinstance(v, dict):
                change_key(v, tgt_k, tgt_v)
            if k == tgt_k:
                d[k] = tgt_v

    for tgt_k, tgt_v in new_params.items():
        change_key(params, tgt_k, tgt_v)
    return params


def gen_gauss(mn, sigma, N):
    '''
    Generate a gaussian curve
    Inputs:
    mn: mean
    sigma: standard deviation
    N: length of the output
    '''
    r = range(N)
    v = [
        1
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-float(x - mn) ** 2 / (2 * sigma ** 2))
        for x in r
    ]
    v = np.array(v)
    return v


def conv(y1, y2, mode="same"):
    z = np.convolve(y1, y2, mode=mode)
    return z


def gen_inp_vec(N_gauss=3, N_samples=40):
    '''
    Generate a 1-D vector that is the sum of multiple gaussian curves.
    Inputs:
    N_samples: vector length
    N_gauss: Specifies the number of gaussian curves to summate
    '''
    mns = []
    sigmas = []
    amps = []
    gs = []
    vec = np.zeros(N_samples)
    for _ in range(N_gauss):
        mn = int(40.0 * np.random.random_sample(1)[0])
        sigma = 5.0 * np.random.random_sample(1)[0]
        amp = 3.0 * np.random.random_sample(1)[0] + 2.0
        noise = 0.01 * amp * np.random.random_sample(40)
        g = gen_gauss(mn, sigma, N_samples)
        vec += amp * g
        vec += noise
        mns.append(mn)
        sigmas.append(sigma)
        amps.append(amp)
        gs.append(g)
    return vec, mns, sigmas, amps, gs


def gen_sample(input_shape, N_gauss=3):
    '''
    Generates an 1-D input vector that is the summation of
    multiple gaussian curves.
    Returns a 1-D output vector which is the result of convolving
    the input vector with a gaussian filter (std. dev. = 2.0)
    Inputs:
    input_shape: [batchsize, input vector length]
    N_gauss: Specifies the number of gaussian curves to summate
    '''
    out = {"v_out": [], "v_in": [], "input_params": [], "amp": [], "sigma": []}
    batchsize = input_shape[0]
    input_length = input_shape[1]
    for _ in range(batchsize):
        v_in, m, s, a, g = gen_inp_vec(N_gauss=N_gauss, N_samples=input_length,)
        ag = 1.0
        sg = 2.0
        gauss_filt = ag * gen_gauss(5, sg, 10)
        v_in = v_in / (np.sqrt(np.sum([_x ** 2 for _x in v_in])))
        v_out = conv(v_in, gauss_filt)
        out["v_out"].append(v_out)
        out["v_in"].append(v_in)
        out["input_params"].append([m, s, a])
        out["amp"].append(ag)
        out["sigma"].append(sg)
    for k in out.keys():
        out[k] = np.array(out[k], dtype=np.float32)
    return out


def read_tfrecord(serialized_example, mixed_precision, mode):
    feature_description = {
        "input": tf.io.FixedLenFeature((), tf.string),
        "input_params": tf.io.FixedLenFeature((), tf.string),
    }
    if mode != tf.estimator.ModeKeys.PREDICT:
        feature_description["label"] = tf.io.FixedLenFeature((), tf.string)

    example = tf.io.parse_single_example(
        serialized_example, feature_description
    )

    inp = tf.ensure_shape(
        tf.io.parse_tensor(example["input"], out_type=tf.float32), (40,)
    )
    inp_params = tf.ensure_shape(
        tf.io.parse_tensor(example["input_params"], out_type=tf.float32), (2,)
    )
    if mixed_precision:
        inp = tf.cast(inp, tf.float16)
        inp_params = tf.cast(inp_params, tf.float16)
    if mode != tf.estimator.ModeKeys.PREDICT:
        label = tf.ensure_shape(
            tf.io.parse_tensor(example["label"], out_type=tf.float32), (40,)
        )
        if mixed_precision:
            label = tf.cast(label, tf.float16)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return {"input": inp, "input_params": inp_params}
    else:
        return {"input": inp, "input_params": inp_params}, label


def write_tfrecord(num_samples, output_dir, mode="train"):
    np.random.seed(seed=1)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_fn = os.path.join(output_dir, "%s_data.tfrecord" % mode)
    with tf.io.TFRecordWriter(output_fn) as writer:
        for _idx in range(num_samples):
            if _idx % 100 == 0:
                print(_idx)
            sample = gen_sample((1, 40), N_gauss=2)
            inp = sample["v_in"][0]
            label = sample["v_out"][0]
            inp_params = np.array([sample["amp"], sample["sigma"]])[:, 0]
            inp_features = tf.io.serialize_tensor(inp)
            inp_params = tf.io.serialize_tensor(inp_params)
            label_features = tf.io.serialize_tensor(label)

            feature = {
                "input": _bytes_feature(inp_features),
                "input_params": _bytes_feature(inp_params),
                "label": _bytes_feature(label_features),
            }
            example = tf.train.Example(
                features=tf.train.Features(feature=feature)
            )
            example = example.SerializeToString()
            writer.write(example)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
