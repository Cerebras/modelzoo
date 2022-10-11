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
from modelzoo.transformers.tf.bert.fine_tuning.extractive_summarization.BertSummarizationModel import (
    BertSummarizationModel,
)


def model_fn(features, labels, mode, params):
    """
    The model function to be used with TF estimator API
    """
    bert = BertSummarizationModel(params)
    outputs = bert(features, mode)

    total_loss = None
    train_op = None
    host_call = None
    predictions = None

    if mode != tf.estimator.ModeKeys.PREDICT:
        total_loss = bert.build_total_loss(outputs, features, labels, mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = bert.build_train_ops(total_loss)
    elif mode == tf.estimator.ModeKeys.EVAL:
        if not params["eval_input"].get("disable_eval_metrics"):
            host_call = (
                bert.build_eval_metric_ops,
                [outputs, labels, features],
            )
    elif mode == tf.estimator.ModeKeys.PREDICT:
        predictions = outputs

    espec = CSEstimatorSpec(
        mode=mode,
        loss=total_loss,
        train_op=train_op,
        predictions=predictions,
        host_call=host_call,
    )

    return espec
