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
Base class for TensorFlow models.
"""
from abc import ABC, abstractmethod

from tensorflow.keras.mixed_precision.experimental import Policy


class TFBaseModel(ABC):
    """
    Base class for TensorFlow models.
    Provides a general model API, consisting of the
    following methods that must be implemented by child classes:
        `build_model`: builds the model
        `build_total_loss`: builds total loss, given model
            outputs returned by build_model
        `build_train_ops`: sets up an optimizer and returns asscoiated train ops
        `build_eval_metric_ops`: build evaluation metric ops

    The `__call__` function wraps around `build_model`.

    All TF models must inherit from TFBaseModel and implement
    `__init__` (containing the call to TFBaseModel's __init__),
    `build_model`, `build_total_loss`, `build_train_ops`, and
    `build_eval_metric_ops` methods.

    :param bool mixed_precision: Enable mixed precision, if True.
    """

    def __init__(self, mixed_precision=False):
        self.policy = (
            Policy("mixed_float16", loss_scale=None)
            if mixed_precision
            else Policy("float32", loss_scale=None)
        )

    def __call__(self, features, mode):
        return self.build_model(features, mode)

    @abstractmethod
    def build_model(self, features, mode):
        """
        Build model.

        :param features: Input features.
        :param tf.estimator.ModeKeys mode: Mode (TRAIN, EVAL).
        :returns: Model outputs
        """
        raise NotImplementedError(
            "build_model must be implemented in a child class!"
        )

    @abstractmethod
    def build_total_loss(self, model_outputs, features, labels, mode):
        """
        Build loss given model outputs.

        :param model_outputs: model outputs. returned by build_model
        :param features: Input features.
        :param labels: Labels.
        :param tf.estimator.ModeKeys mode: Mode (TRAIN, EVAL).
        :returns: Total loss tensor.
        """
        raise NotImplementedError(
            "build_total_loss must be implemented in a child class!"
        )

    @abstractmethod
    def build_train_ops(self, total_loss):
        """
        Setup optimizer and build train ops.

        :param Tensor total_loss: The total loss return by `__call__`
        :return: Train ops
        """
        raise NotImplementedError(
            "build_train_ops must be implemented in a child class!"
        )

    @abstractmethod
    def build_eval_metric_ops(self, model_outputs, labels, features=None):
        """
        Build eval metric ops.

        :param model_outputs: model outputs. returned by build_model
        :param labels: Labels.
        :param features: Input features, optional
        :returns: Eval ops.
        """
        raise NotImplementedError(
            "build_eval_metrics must be implemented in a child class!"
        )
