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

"""Model Function for GPT-2 model variants.
"""
import tensorflow as tf

from modelzoo.common.tf.estimator.cs_estimator_spec import CSEstimatorSpec
from modelzoo.common.tf.hooks.grad_accum_hooks import get_grad_accum_hooks
from modelzoo.common.tf.run_utils import ExecutionMode, get_execution_mode
from modelzoo.transformers.tf.gpt2.Gpt2Model import Gpt2Model


def model_fn(features, labels, mode, params):
    """The model function to be used with TF estimator API."""
    gpt2 = Gpt2Model(params)
    outputs = gpt2(features, mode)
    loss = gpt2.build_total_loss(outputs, features, labels, mode)

    train_op = None
    host_call = None
    eval_metrics = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = gpt2.build_train_ops(loss)
    elif mode == tf.estimator.ModeKeys.EVAL:
        # build inputs to eval metrics on the fabric to minimize work done
        # on the host.
        if get_execution_mode() == ExecutionMode.Pipeline:
            eval_metric_inputs = gpt2.build_eval_metric_inputs(
                outputs, labels, features
            )
            host_call = (
                gpt2.build_eval_metric_ops,
                [eval_metric_inputs, labels, features],
            )
        else:
            eval_metrics = gpt2.build_eval_metric_ops(outputs, labels, features)
    else:
        raise ValueError(f"Mode {mode} not supported.")

    hooks = []
    if gpt2.trainer.is_grad_accum():
        hooks.extend(
            get_grad_accum_hooks(
                gpt2.trainer,
                runconfig_params=params["runconfig"],
                summary_dict={"train/lm_cost": loss},
                logging_dict={"loss": loss},
            )
        )

    espec = CSEstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        training_hooks=hooks,
        host_call=host_call,
        eval_metric_ops=eval_metrics,
    )

    return espec
