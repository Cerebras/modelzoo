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
UNet Model function to be used with TF Estimator API
"""
import tensorflow as tf

from modelzoo.common.tf.estimator.cs_estimator_spec import CSEstimatorSpec
from modelzoo.unet.tf.UNetModel import UNetModel


def model_fn(features, labels, mode, params):
    """
    Model function to be used with TF Estimator API
    """
    model = UNetModel(params)
    logits = model(features, mode)
    loss = model.build_total_loss(logits, features, labels, mode)

    train_op = None
    host_call = None
    evaluation_hooks = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = model.build_train_ops(loss)
    elif mode == tf.estimator.ModeKeys.EVAL:
        evaluation_hooks = model.get_evaluation_hooks(logits, labels, features)
        host_call = (model.build_eval_metric_ops, [logits, labels, features])
    else:
        raise ValueError(f"Mode {mode} not supported.")

    espec = CSEstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        evaluation_hooks=evaluation_hooks,
        host_call=host_call,
    )

    return espec
