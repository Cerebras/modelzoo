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
Model Function for Spacetimeformer model
"""
import tensorflow as tf
from singularity_dao_shared.spacetimeformer.tf.Spacetimeformer import (
    Spacetimeformer,
)
from tensorflow.compat.v1.layers import flatten

from modelzoo.common.tf.estimator.cs_estimator_spec import CSEstimatorSpec
from modelzoo.common.tf.hooks.grad_accum_hooks import get_grad_accum_hooks


def model_fn(features, labels, mode, params):
    """The model function to be used with TF estimator API."""
    model = Spacetimeformer(params)
    outputs = model(features, mode)
    loss = model.build_total_loss(outputs, features, labels, mode)

    train_op = None
    host_call = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = model.build_train_ops(loss)
    elif mode == tf.estimator.ModeKeys.EVAL:
        if not params["eval_input"].get("disable_eval_metrics"):

            features_for_metrics = {
                'scales': features['scales'],
                'means': features['means'],
            }

            host_call = (
                model.build_eval_metric_ops,
                [outputs, flatten(labels), features_for_metrics],
            )
    else:
        raise ValueError(f"Mode {mode} not supported.")

    hooks = None
    if model.trainer.is_grad_accum():
        hooks = get_grad_accum_hooks(
            model.trainer,
            runconfig_params=params["runconfig"],
            summary_dict={"train/cost": loss},
            logging_dict={"loss": loss},
        )

    espec = CSEstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        training_hooks=hooks,
        host_call=host_call,
    )

    return espec
