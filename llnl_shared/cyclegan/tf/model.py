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

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.mixed_precision.experimental import Policy


def build_model(features, labels, mode, params):
    """ Build CycleGAN subgraph based on layer dimensions specified \
        in params file """
    if params['model']['mixed_precision']:
        dtype = Policy('mixed_float16', loss_scale=None)
        tf.keras.mixed_precision.experimental.set_policy(dtype)
    else:
        dtype = tf.float32
    with tf.compat.v1.name_scope(params['model']['model_name']):
        hidden_sizes = params['model']['hidden_sizes'] + [
            params['model']['output_size']
        ]
        layer_num = 1
        x = features
        for sz, act in zip(hidden_sizes, params['model']['activations']):
            dense_layer = Dense(
                sz, activation=act, dtype=dtype, name='FC_%d' % layer_num
            )
            x = dense_layer(x)
            layer_num += 1
        if ('use_fp32_loss' in params['model'].keys()) and params['model'][
            'use_fp32_loss'
        ]:
            labels = tf.cast(labels, tf.float32)
            x = tf.cast(x, tf.float32)
        mse = tf.compat.v1.keras.losses.MeanSquaredError()
        loss = mse(y_true=labels, y_pred=x)
        tf.compat.v1.summary.scalar('Loss', loss)
    return x, loss, labels


def model_fn(features, labels, mode, params):
    output, loss, labels = build_model(features, labels, mode, params)
    train_op = None
    oparams = params['optimizer']
    if mode == tf.estimator.ModeKeys.TRAIN:
        ## Construct optimizer
        _OPTIMIZERS = {
            "adam": [
                tf.compat.v1.train.AdamOptimizer,
                {"learning_rate": oparams['learning_rate'], "name": 'Adam',},
            ],
            "sgd": [
                tf.compat.v1.train.GradientDescentOptimizer,
                {"learning_rate": oparams['learning_rate'], "name": 'SGD',},
            ],
        }
        opt_name = oparams["optimizer"]
        assert opt_name in _OPTIMIZERS, f"Unsupported optimizer {opt_name}"
        opt = _OPTIMIZERS[opt_name][0](**_OPTIMIZERS[opt_name][1])

        train_op = opt.minimize(
            loss=loss, global_step=tf.compat.v1.train.get_global_step()
        )
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,)
