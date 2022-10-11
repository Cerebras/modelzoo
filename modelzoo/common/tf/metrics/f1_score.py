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

from modelzoo.common.tf.metrics.fbeta_score import fbeta_score_metric


def f1_score_metric(
    labels,
    predictions,
    weights=None,
    metrics_collections=None,
    updates_collections=None,
    name=None,
):

    """
    Calculate f1 score for binary classification
    
    f1 score is defined as follows:
        f1 = 2 * (precision *recall)/(precsion + recall)
        
    Uses fbeta metric implementation underneath.
    
    The predictions are accumulated in a confusion matrix, weighted by 
    weights, and f1 score is then calculated from it.
    For estimation of the metric over a stream of data, the function creates an
    update_op operation that updates these variables and returns the f1.
    If weights is None, weights default to 1. Use weights of 0 to 
    mask values.
    
    Returns:
        f1: A scalar representing f1.
        update_op: An operation that increments the confusion matrix.
    Raises:
        ValueError: If predictions and labels have mismatched shapes, or if
        weights is not None and its shape doesn't match predictions, or if
        either metrics_collections or updates_collections are not a list or
        tuple.
        RuntimeError: If eager execution is enabled.
        InvalidArgumentError: If labels and predictions are not binary 
        i.e. values not in [0, 1]

    Parameters:
    ----------
    :param Tensor labels: A Tensor of binary ground truth labels with shape [batch size] 
        and of type int32 or int64. 
        The tensor will be flattened if its rank > 1.
    :param Tensor predictions: A Tensor of binary prediction results for semantic labels,
        whose shape is [batch size] and type int32 or int64. The tensor will be
        flattened if its rank > 1.
    :param Tensor weights: Optional Tensor whose rank is either 0, 
        or the same rank as labels, and must be broadcastable 
        to labels (i.e., all dimensions must be either 1, 
        or the same as the corresponding labels dimension).
    :param List metrics_collections: An optional list of collections that
        f1_metric should be added to.
    :param List updates_collections: An optional list of collections update_op
        should be added to.
    :param string name: An optional variable_scope name.
    """

    f1_score, update_op = fbeta_score_metric(
        labels=labels,
        predictions=predictions,
        num_classes=2,
        ignore_labels=[0],
        beta=1,
        average_type="micro",
        weights=weights,
        metrics_collections=metrics_collections,
        updates_collections=updates_collections,
        name=name,
    )

    return f1_score, update_op
