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

from modelzoo.common.tf.metrics.utils import (
    aggregate_across_replicas,
    streaming_confusion_matrix,
)


def fbeta_score_metric(
    labels,
    predictions,
    num_classes,
    ignore_labels=None,
    beta=1,
    weights=None,
    average_type="micro",
    metrics_collections=None,
    updates_collections=None,
    name=None,
):

    """
    Calculate fbeta for multi-class scenario.
    
    fbeta is defined as follows:
        fbeta = (1+beta**2) * (precision *recall)/((beta**2 * precsion) + recall)

    The predictions are accumulated in a confusion matrix, weighted by 
    weights, and fbeta is then calculated from it.
    For estimation of the metric over a stream of data, the function creates an
    update_op operation that updates these variables and returns the fbeta.
    If weights is None, weights default to 1. Use weights of 0 to 
    mask values.

    Returns:
        fbeta: A scalar representing fbeta.
        update_op: An operation that increments the confusion matrix.
    Raises:
        ValueError: If predictions and labels have mismatched shapes, or if
        weights is not None and its shape doesn"t match predictions, or if
        either metrics_collections or updates_collections are not a list or
        tuple.
        ValueError: If average_type is not either "micro" or "macro"
        RuntimeError: If eager execution is enabled.

    :param Tensor labels: A Tensor of ground truth labels with shape [batch size] 
        and of type int32 or int64. 
        The tensor will be flattened if its rank > 1.
    :param Tensor predictions: A Tensor of prediction results for semantic labels,
        whose shape is [batch size] and type int32 or int64. The tensor will be
        flattened if its rank > 1.
    :param int num_classes: The possible number of labels the prediction 
        task can have. This value must be provided, since a 
        confusion matrix of dimension = [num_classes, num_classes] 
        will be allocated.
    :param Tensor ignore_labels: Optional Tensor which specifies the labels to 
        be considered when computing metric
    :param int beta: Optional param for beta parameter
    :param Tensor weights: Optional Tensor whose rank is either 0, 
        or the same rank as labels, and must be broadcastable 
        to labels (i.e., all dimensions must be either 1, 
        or the same as the corresponding labels dimension).
    :param str average_type: Optional string specifying the type of 
        averaging on data
        "micro": Calculate metrics globally by counting the total 
            true positives, false negatives and false positives.
        "macro": Calculate metrics for each label, and find their unweighted mean. 
            This does not take label imbalance into account.
    :param List metrics_collections: An optional list of collections that
        fbeta_metric should be added to.
    :param List updates_collections: An optional list of collections update_op
        should be added to.
    :param string name: An optional variable_scope name.

    Example:
     y_true = [0,0,0,0,0,0,0,1,1,1,1,2,2]
     y_pred = [0,0,1,1,1,2,2,0,0,1,2,0,2]
     confusion_matrix(y_true, y_pred) -> rows id = true label, column id = predicted label
        array([[2, 3, 2],
               [2, 1, 1],
               [1, 0, 1]])

    class_id       = [0, 1, 2]
    ----------------------------
    True_Positives = [2, 1, 1]
    Predicted_Pos =  [5, 4, 4]
    Actual_Pos =     [7, 4, 2]

                        precision recall  f1-score   

                0       0.40      0.29      0.33  
                1       0.25      0.25      0.25         
                2       0.25      0.50      0.33         
    
        macro avg       0.30      0.35      0.31 

        accuracy = 4/13 = 0.307
     

    Scenarios:
    -----------
    A. average_type = "micro", ignore_labels = None
    precision = (2+1+1)/(5+4+4) = accuracy
    recall = (2+1+1)/(7+4+2)

    B. average_type = "macro", ignore_labels = None
    per_class precision: [(2/5), (1/4), (1/4)]
    precision_macro = mean(per_class_precision) 
    per_class recall = [(2/7), (1/4), (1/2)]
    recall = mean(per_class_recall)
    fb = mean([fb_0, fb_1, fb_2])

    C. average_type = "micro", ignore_labels = [1]
    precision = (2+1)/(5+4)
    recall = (2+1)/(7+2)

    D. average_type = "macro", ignore_labels = [1]
    precision = mean([(2/5) , (1/4)])
    recall = mean([(2/7),  (1/2)])
    fb = mean([fb_0, fb_2])
    
    """

    if tf.executing_eagerly():
        raise RuntimeError(
            "fbeta metric is not supported when eager execution is enabled."
        )

    with tf.compat.v1.variable_scope(
        name, "fbeta", (labels, predictions, weights, ignore_labels)
    ):
        # Check if shape is compatible.
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        total_cm, update_op = streaming_confusion_matrix(
            labels, predictions, num_classes, weights
        )

        def _compute_metric(_, total_cm, ignore_labels, beta, average_type):
            fbeta = _compute_fbeta(
                total_cm, ignore_labels, beta=beta, average_type=average_type
            )
            return fbeta

        fbeta = aggregate_across_replicas(
            metrics_collections,
            _compute_metric,
            total_cm,
            ignore_labels,
            beta,
            average_type,
        )

        if updates_collections:
            tf.compat.v1.add_to_collection.add_to_collections(
                updates_collections, update_op
            )

        return fbeta, update_op


def _compute_fbeta(total_cm, ignore_labels=None, beta=1, average_type="micro"):
    """
    Calculate precision, recall and fbeta from confusion matrix 
    for labels (excluding "ignore_labels")

    :param total_cm: Confusion matrix of size [num_classes, num_classes]
    :param ignore_labels: labels to ignore when computing metrics. 
    :param beta: beta value to use when computing 
        fbeta = (1+beta**2) * (precision *recall)/((beta**2 * precision) + recall)
    :param average_type: One of "micro" or "macro"
            "micro" : Calculate metrics globally by counting the total 
                true positives, false negatives and false positives 
                for labels not in "ignore_labels"
            "macro" : Calculate metrics for each label in labels to consider, 
                and find their unweighted mean. This does not take label
                imbalance into account.
            
    """
    true_pos = tf.cast(tf.linalg.diag_part(total_cm), tf.float32)
    predicted_per_class = tf.cast(tf.reduce_sum(total_cm, 0), tf.float32)
    actual_per_class = tf.cast(tf.reduce_sum(total_cm, 1), tf.float32)

    num_classes = tf.shape(total_cm)[0]

    if ignore_labels:
        all_labels = tf.expand_dims(tf.range(num_classes), 1)
        mask = tf.reduce_all(tf.not_equal(all_labels, ignore_labels), axis=1)
    else:
        mask = tf.fill([num_classes,], value=True)

    mask = tf.cast(mask, tf.float32)
    num_labels_to_consider = tf.reduce_sum(mask)

    if average_type == "micro":
        precision = tf.math.divide_no_nan(
            tf.reduce_sum(true_pos * mask),
            tf.reduce_sum(predicted_per_class * mask),
        )
        recall = tf.math.divide_no_nan(
            tf.reduce_sum(true_pos * mask),
            tf.reduce_sum(actual_per_class * mask),
        )
        fbeta = tf.math.divide_no_nan(
            (1.0 + beta ** 2) * precision * recall,
            beta ** 2 * precision + recall,
        )
    elif average_type == "macro":
        precision_per_class = tf.math.divide_no_nan(
            true_pos, predicted_per_class
        )
        recall_per_class = tf.math.divide_no_nan(true_pos, actual_per_class)
        fbeta_per_class = tf.math.divide_no_nan(
            (1.0 + beta ** 2) * precision_per_class * recall_per_class,
            beta ** 2 * precision_per_class + recall_per_class,
        )
        precision = (
            tf.reduce_sum((precision_per_class * mask)) / num_labels_to_consider
        )
        recall = (
            tf.reduce_sum((recall_per_class * mask)) / num_labels_to_consider
        )
        fbeta = tf.reduce_sum((fbeta_per_class * mask)) / num_labels_to_consider
    else:
        raise ValueError(
            f"Incorrect argument {average_type} for average_type. Should be \"micro\" or \"macro\""
        )

    return fbeta
