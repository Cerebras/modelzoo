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
from genentech_shared.bert.tf.fine_tuning.token_classifier.GenentechBertTokenClassifierModel import (
    GenentechBertTokenClassifierModel,
)

from modelzoo.common.tf.estimator.cs_estimator_spec import CSEstimatorSpec
from modelzoo.common.tf.hooks.grad_accum_hooks import get_grad_accum_hooks


def model_fn(features, labels, mode, params):
    """
    The model function to be used with TF estimator API
    """
    bert_token_classifier = GenentechBertTokenClassifierModel(params)

    # `outputs` shape: [bsz, max_seq_len, num_classes]
    outputs = bert_token_classifier(features, mode)

    total_loss = None
    train_op = None
    host_call = None
    predictions = None

    if mode != tf.estimator.ModeKeys.PREDICT:
        total_loss = bert_token_classifier.build_total_loss(
            outputs, features, labels, mode
        )
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = bert_token_classifier.build_train_ops(total_loss)
    elif mode == tf.estimator.ModeKeys.EVAL:
        host_call = (
            bert_token_classifier.build_eval_metric_ops,
            [outputs, labels, features],
        )
    elif mode == tf.estimator.ModeKeys.PREDICT:
        predictions = tf.argmax(outputs, axis=-1, output_type=tf.int32)
        predictions = tf.cast(predictions, tf.float16)

    hooks = None
    if bert_token_classifier.trainer.is_grad_accum():
        hooks = get_grad_accum_hooks(
            bert_token_classifier.trainer,
            runconfig_params=params["runconfig"],
            summary_dict={"train/cost_cls": total_loss},
            logging_dict={"loss": total_loss},
        )

    espec = CSEstimatorSpec(
        mode=mode,
        loss=total_loss,
        train_op=train_op,
        training_hooks=hooks,
        predictions=predictions,
        host_call=host_call,
    )

    return espec
