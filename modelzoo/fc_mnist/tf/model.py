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
Simple FC MNIST model to be used with Estimator
"""
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, Dropout, LeakyReLU
from tensorflow.keras.mixed_precision.experimental import Policy

from modelzoo.common.tf.estimator.cs_estimator_spec import CSEstimatorSpec

NUM_CLASSES = 10


def build_model(features, labels, mode, params):
    dtype = Policy('mixed_float16', loss_scale=None)
    tf.keras.mixed_precision.experimental.set_policy(dtype)
    tf.keras.backend.set_floatx('float16')
    if "model" in params:
        model_params = params['model']
    else:
        model_params = params

    dropout_layer = Dropout(model_params['dropout'], dtype=dtype)
    # Set the default or None
    if "hidden_sizes" in model_params:
        # Depth is len(hidden_sizes)
        model_params["depth"] = len(model_params["hidden_sizes"])
    else:
        # same hidden size across dense layers
        model_params["hidden_sizes"] = [
            model_params["hidden_size"]
        ] * model_params["depth"]

    x = features
    for hidden_size in model_params['hidden_sizes']:
        with tf.name_scope("km_disable_scope"):
            dense_layer = Dense(hidden_size, dtype=dtype)
        act_layer = Activation(model_params['activation_fn'], dtype=dtype)
        if isinstance(model_params['activation_fn'], LeakyReLU):
            # workaround due to TF bug
            # so that the leaky_relu operation can be float16
            act_layer = tf.nn.leaky_relu
        with tf.name_scope("km_disable_scope"):
            x = dense_layer(x)
        x = act_layer(x)
        x = dropout_layer(x, training=(mode == tf.estimator.ModeKeys.TRAIN))
    with tf.name_scope("km_disable_scope"):
        # Model has len(hidden_sizes) + 1 Dense layers
        output_dense_layer = Dense(NUM_CLASSES, dtype=dtype)
        logits = output_dense_layer(x)
    losses_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits,
    )
    loss = tf.reduce_mean(input_tensor=losses_per_sample)
    tf.compat.v1.summary.scalar('loss', loss)
    return loss, logits


def model_fn(features, labels, mode, params):
    loss, logits = build_model(features, labels, mode, params['model'])

    train_op = None
    host_call = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer_params = params["optimizer"]

        optimizer_type = optimizer_params.get("optimizer_type", None)
        if optimizer_type is None or optimizer_type.lower() == "adam":
            opt = tf.compat.v1.train.AdamOptimizer(
                learning_rate=optimizer_params['learning_rate'],
                beta1=optimizer_params['beta1'],
                beta2=optimizer_params['beta2'],
                epsilon=optimizer_params['epsilon'],
            )
        elif optimizer_type.lower() == "sgd":
            opt = tf.compat.v1.train.GradientDescentOptimizer(
                learning_rate=optimizer_params["learning_rate"]
            )
        else:
            raise ValueError(f'Unsupported optimizer {optimizer_type}')

        train_op = opt.minimize(
            loss=loss,
            global_step=tf.compat.v1.train.get_or_create_global_step(),
        )
    elif mode == tf.estimator.ModeKeys.EVAL:

        def build_eval_metric_ops(logits, labels):
            return {
                "accuracy": tf.compat.v1.metrics.accuracy(
                    labels=labels, predictions=tf.argmax(input=logits, axis=1),
                ),
            }

        host_call = (build_eval_metric_ops, [logits, labels])
    else:
        raise ValueError("Only TRAIN and EVAL modes supported")

    espec = CSEstimatorSpec(
        mode=mode, loss=loss, train_op=train_op, host_call=host_call,
    )

    return espec
