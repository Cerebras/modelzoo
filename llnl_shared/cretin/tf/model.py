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
Model functions for LLNL proxy models.
Toy CRETIN model
Owner: jason@cerebras.net
"""

import sys

import tensorflow as tf
from tensorflow.keras.layers import Dense

from cerebras.tf.tf_helper import summary_layer
from modelzoo.common.tf.optimizers.Trainer import Trainer
from modelzoo.common.tf.TFBaseModel import TFBaseModel

from tensorflow.keras.activations import (  # noqa
    elu,
    relu,
    selu,
    sigmoid,
    softplus,
    softsign,
)

ENCODER_OUT_SIZE = 2
DJINN_OUT_SIZE = 4
DECODER_OUT_SIZE = 40
LOG10 = 2.302585092994046


class CretinModel(TFBaseModel):
    def __init__(self, params):
        super(CretinModel, self).__init__(
            mixed_precision=params["model"]["mixed_precision"]
        )
        self.params = params
        # Model trainer
        self.trainer = Trainer(
            params=self.params["optimizer"],
            tf_summary=params["model"]["tf_summary"],
            mixed_precision=params["model"]["mixed_precision"],
        )

    def build_model(self, features):
        """
        Constructs the graph for the CRETIN autoencoder

        :param dict features: Dictionary with keys ``x``,
            ``y``, ``m``, and ``t``
        :param dict labels: Not used but part of Estimator interface
        :param tf.estimator.ModeKeys mode:
        :param dict params: Dictionary of params
        :returns: EstimatorSpec
        """
        iparams = self.params["train_input"]
        mparams = self.params["model"]
        tf_dump = mparams["tf_summary"]

        input_len = iparams["input_len"]
        input_x = features["input"]
        input_scalars = features["input_params"]

        encoder_output_sizes = mparams["encoder_output_sizes"]
        djinn_output_sizes = mparams["djinn_output_sizes"]
        decoder_output_sizes = mparams["decoder_output_sizes"]
        encoder_activation = None
        djinn_activation = None
        decoder_activation = None
        if mparams["encoder_activation"]:
            encoder_activation = getattr(
                sys.modules[__name__], mparams["encoder_activation"]
            )
        else:
            encoder_activation = None
        if mparams["djinn_activation"]:
            djinn_activation = getattr(
                sys.modules[__name__], mparams["djinn_activation"]
            )
        else:
            djinn_activation = None
        if mparams["decoder_activation"]:
            decoder_activation = getattr(
                sys.modules[__name__], mparams["decoder_activation"]
            )
        else:
            decoder_activation = None

        ## Construct layers
        encoder_layers = []
        djinn_layers = []
        decoder_layers = []
        x = input_x
        x_out = []
        # y_out = []
        n_towers = mparams.get("n_towers", 1)  # currently n_towers>1 not tested
        in_parallel = mparams.get("in_parallel", False)
        for n in range(n_towers):
            encoder_layers.append([])
            djinn_layers.append([])
            decoder_layers.append([])
            for i_layer, out_size in enumerate(encoder_output_sizes):
                if i_layer == (len(encoder_output_sizes) - 1):
                    activation = None
                else:
                    activation = encoder_activation
                if i_layer == 0:
                    encoder_layers[n].append(
                        Dense(
                            out_size,
                            activation=activation,
                            input_shape=(input_len,),
                            dtype=self.policy,
                        )
                    )
                else:
                    encoder_layers[n].append(
                        Dense(
                            out_size, activation=activation, dtype=self.policy
                        )
                    )
            for i_layer, out_size in enumerate(djinn_output_sizes):
                if i_layer == (len(djinn_output_sizes) - 1):
                    activation = None
                else:
                    activation = djinn_activation
                if i_layer == 0:
                    djinn_layers[n].append(
                        Dense(
                            out_size,
                            activation=activation,
                            input_shape=(encoder_output_sizes[-1],),
                            dtype=self.policy,
                        )
                    )
                else:
                    djinn_layers[n].append(
                        Dense(
                            out_size, activation=activation, dtype=self.policy
                        )
                    )
            for i_layer, out_size in enumerate(decoder_output_sizes):
                if i_layer == (len(decoder_output_sizes) - 1):
                    activation = None
                else:
                    activation = decoder_activation
                if i_layer == 0:
                    decoder_layers[n].append(
                        Dense(
                            out_size,
                            activation=activation,
                            input_shape=(DJINN_OUT_SIZE,),
                            dtype=self.policy,
                        )
                    )
                else:
                    decoder_layers[n].append(
                        Dense(
                            out_size, activation=activation, dtype=self.policy
                        )
                    )

        for n in range(n_towers):
            with tf.compat.v1.variable_scope("tower_%d" % n):
                ## Construct model
                if in_parallel and (n_towers > 1):
                    x = input_x
                ### evaluate_encoder
                for layer in encoder_layers[n]:
                    x = layer(x)
                    if tf_dump:
                        x = summary_layer(x)

                ### concat encoder output with m and t input features
                x = tf.concat([x, input_scalars], axis=1)

                ### evaluate_djinn
                for layer in djinn_layers[n]:
                    x = layer(x)
                    if tf_dump:
                        x = summary_layer(x)

                ### evaluate_decoder
                for layer in decoder_layers[n]:
                    x = layer(x)
                    if tf_dump:
                        x = summary_layer(x)
                if in_parallel and (n_towers > 1):
                    x_out.append(x)

        if in_parallel and (n_towers > 1):
            x = tf.concat(x_out, axis=-1)
        return x

    def build_total_loss(self, outputs, labels, in_parallel=False, n_towers=1):
        if in_parallel and (n_towers > 1):
            all_labels = [labels for ii in range(n_towers)]
            labels = tf.concat(all_labels, axis=-1)
        ## Construct loss
        mse = tf.compat.v1.keras.losses.MeanSquaredError()
        loss = mse(y_true=labels, y_pred=outputs)
        loss = tf.cast(
            loss,
            tf.float16
            if self.params['model']["mixed_precision"]
            else tf.float32,
        )
        return loss

    def build_train_ops(self, total_loss):
        """
        Setup optimizer and build train ops.
        """
        return self.trainer.build_train_ops(total_loss)

    def build_eval_metric_ops(self, model_outputs, labels):
        """
        Compute metrics and return
        """
        return None


def model_fn(features, labels, mode, params):
    """
    Model function for CRETIN autoencoder

    :param dict features: Dictionary with keys ``x``,
        ``y``, ``m``, and ``t``
    :param dict labels: Not used but part of Estimator interface
    :param tf.estimator.ModeKeys mode:
    :param dict params: Dictionary of params
    :returns: EstimatorSpec
    """

    mparams = params["model"]
    oparams = params["optimizer"]

    cretin = CretinModel(params)
    outputs = cretin.build_model(features)

    total_loss = None
    train_op = None
    preds = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        total_loss = cretin.build_total_loss(
            outputs,
            labels,
            in_parallel=mparams.get('in_parallel', False),
            n_towers=mparams.get('n_towers', 1),
        )
        train_op = cretin.build_train_ops(total_loss)
    elif mode == tf.estimator.ModeKeys.EVAL:
        total_loss = cretin.build_total_loss(
            outputs,
            labels,
            in_parallel=mparams.get('in_parallel', False),
            n_towers=mparams.get('n_towers', 1),
        )
    elif mode == tf.estimator.ModeKeys.PREDICT:
        preds = outputs
    else:
        raise ValueError(f"Mode {mode} not supported.")

    return tf.estimator.EstimatorSpec(
        mode, loss=total_loss, predictions=preds, train_op=train_op,
    )


B = tf.keras.backend


def gelu(x):
    return 0.5 * x * (1 + B.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))
