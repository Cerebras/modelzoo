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

from modelzoo.common.tf.estimator.cs_estimator_spec import CSEstimatorSpec
from modelzoo.common.tf.hooks.grad_accum_hooks import get_grad_accum_hooks
from modelzoo.transformers.tf.bert.BertModel import BertModel


def model_fn(features, labels, mode, params):
    """
    The model function to be used with TF estimator API
    """

    bert = BertModel(params)
    outputs = bert(features, mode)
    total_loss = bert.build_total_loss(outputs, features, labels, mode)

    train_op = None
    host_call = None
    eval_metrics = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = bert.build_train_ops(total_loss)
    elif mode == tf.estimator.ModeKeys.EVAL:
        if not params["eval_input"].get("disable_eval_metrics"):
            # for pipeline mode, we build inputs to eval metrics on the fabric
            # to minimize work done on the host.
            eval_metric_inputs = bert.build_eval_metric_inputs(
                outputs, labels, features
            )
            host_call = (
                bert.build_eval_metric_ops,
                [eval_metric_inputs, labels, features],
            )
    else:
        raise ValueError(f"Mode {mode} not supported.")

    hooks = None
    if bert.trainer.is_grad_accum():
        summary_dict = {
            "train/cost": total_loss,
            "train/cost_masked_lm": bert.mlm_loss,
        }

        if not params["model"]["disable_nsp"]:
            summary_dict["train/cost_cls"] = bert.nsp_loss

        hooks = get_grad_accum_hooks(
            bert.trainer,
            runconfig_params=params["runconfig"],
            summary_dict=summary_dict,
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
