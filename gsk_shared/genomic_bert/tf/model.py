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
from gsk_shared.genomic_bert.tf.GenomicBertModel import GenomicBertModel


def model_fn(features, labels, mode, params):
    """
    The model function to be used with TF estimator API
    """
    bert = GenomicBertModel(params)
    outputs = bert(features, mode)
    total_loss = bert.build_total_loss(outputs, features, labels, mode)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = bert.build_train_ops(total_loss)
    elif mode == tf.estimator.ModeKeys.EVAL:
        train_op = None
    else:
        raise ValueError(f"Mode {mode} not supported.")

    espec = tf.estimator.EstimatorSpec(
        mode=mode, loss=total_loss, train_op=train_op
    )
    if mode == tf.estimator.ModeKeys.EVAL:
        espec.host_call = (
            bert.build_eval_metric_ops,
            [
                [
                    tf.transpose(outputs[0], perm=[0, 2, 1]),
                    tf.transpose(outputs[1], perm=[0, 2, 1]),
                ],
                {
                    "masked_lm_weights_dna": features["masked_lm_weights_dna"],
                    "masked_lm_weights_ideas": features[
                        "masked_lm_weights_ideas"
                    ],
                    "masked_lm_ids_dna": features["masked_lm_ids_dna"],
                    "masked_lm_ids_ideas": features["masked_lm_ids_ideas"],
                },
                labels,
            ],
        )

    return espec
