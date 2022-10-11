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
from modelzoo.transformers.tf.linformer.LinformerModel import LinformerModel


def model_fn(features, labels, mode, params):
    """
    The model function to be used with TF estimator API
    """

    linformer = LinformerModel(params)
    outputs = linformer(features, mode)

    total_loss = linformer.build_total_loss(outputs, features, labels, mode)
    host_call = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = linformer.build_train_ops(total_loss)
    elif mode == tf.estimator.ModeKeys.EVAL:
        # Note that we are doing argmax on the fabric, not on the host.
        # This is somewhat inconsistent with metrics being processed on the host,
        # but is done primarily to improve performance.
        # build inputs to eval metrics on the fabric to minimize work done on the host.
        eval_metric_inputs = linformer.build_eval_metric_inputs(
            outputs, labels, features
        )
        host_call = (
            linformer.build_eval_metric_ops,
            [eval_metric_inputs, labels, features],
        )
        train_op = None
    else:
        raise ValueError(f"Mode {mode} not supported.")

    hooks = None
    if linformer.trainer.is_grad_accum():
        summary_dict = {
            "train/cost": total_loss,
            "train/cost_masked_lm": linformer.mlm_loss,
        }

        if not params["model"]["disable_nsp"]:
            summary_dict["train/cost_cls"] = linformer.nsp_loss

        hooks = get_grad_accum_hooks(
            linformer.trainer,
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
    )

    return espec
