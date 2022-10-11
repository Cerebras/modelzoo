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

""" Model configuration for T5. """

import tensorflow as tf

from modelzoo.common.tf.estimator.cs_estimator_spec import CSEstimatorSpec
from modelzoo.common.tf.hooks.grad_accum_hooks import get_grad_accum_hooks
from modelzoo.transformers.tf.t5.T5Model import T5Model


def model_fn(features, labels, mode, params):
    """
    The model function to be used with TF estimator API.
    """
    t5 = T5Model(params)
    outputs = t5(features, mode)
    total_loss = t5.build_total_loss(outputs, features, labels, mode)

    train_op = None
    host_call = None
    eval_metrics = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = t5.build_train_ops(total_loss)
    elif mode == tf.estimator.ModeKeys.EVAL:
        if not params["eval_input"].get("disable_eval_metrics"):
            # for pipeline mode, we build inputs to eval metrics on the fabric
            # to minimize work done on the host.
            # END_CEREBRAS_ONLY
            eval_metric_inputs = t5.build_eval_metric_inputs(
                outputs, labels, features
            )
            host_call = (
                t5.build_eval_metric_ops,
                [eval_metric_inputs, labels, features],
            )
    else:
        raise ValueError(f"Mode {mode} not supported.")

    hooks = None
    if t5.trainer.is_grad_accum():
        hooks = get_grad_accum_hooks(
            t5.trainer,
            runconfig_params=params["runconfig"],
            summary_dict={"loss/total_loss": total_loss},
            logging_dict={"loss": total_loss},
        )

    espec = CSEstimatorSpec(
        mode=mode,
        loss=total_loss,
        train_op=train_op,
        training_hooks=hooks,
        host_call=host_call,
        eval_metric_ops=eval_metrics,
    )

    return espec
