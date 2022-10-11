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

import string
from collections import Counter

import numpy as np
import tensorflow as tf
from nltk import ngrams

from modelzoo.common.tf.metrics.utils import (
    aggregate_across_replicas,
    metric_variable,
)


def streaming_rouge_matrix(hypothesis, references, max_n=1):
    """
    Calculate a streaming rouge matrix.
    The num_matched_ngrams, num_reference_ngrams, num_hypothesis_ngrams
    are accumulated in a rouge matrix.
    Calculates a rouge matrix. For estimation over a stream of data,
    the function creates an  `update_op` operation.
    :param Tensor hypothesis: A `Tensor` of predicted summarization tokens with shape
        [batch size, max_sequence_length] and of type `tf.string`.
    :param Tensor references: A `Tensor` of tokens from target summarization, whose shape is
        [batch size, max_sequence_length] and type `tf.string`.
    :param int max_n: Optional maximum size of n-grams to consider. Default is 1.
    :returns:
        total_rm: A `Tensor` representing the rouge matrix of shape (3, ), which stores
        num_matched_ngrams, num_reference_ngrams, num_hypothesis_ngrams.
        update_op: An operation that increments the rouge matrix.
    """

    def _preprocess_before_rouge(words, max_n):
        special_words = {"[pad]", "[cls]", "[sep]"}
        punctuation_words = set(string.punctuation)
        words_to_ignore = punctuation_words | special_words

        words = [word.decode().lower() for word in words]
        words = list(filter(lambda word: word not in words_to_ignore, words))

        text_ngrams = ngrams(words, max_n)
        return text_ngrams

    def _calculate_rouge_matrix(hypothesis, references, max_n):
        # We store 3 items: num_matched_ngrams, num_reference_ngrams
        # and num_hypothesis_ngrams.
        rouge_matrix = np.zeros((3,), dtype=np.float64)

        hypothesis_ngrams = _preprocess_before_rouge(hypothesis, max_n)
        references_ngrams = _preprocess_before_rouge(references, max_n)

        hypothesis_freq = Counter(hypothesis_ngrams)
        references_freq = Counter(references_ngrams)

        num_matched_ngrams = 0
        for word, count in references_freq.items():
            num_matched_ngrams += min(hypothesis_freq[word], count)

        num_reference_ngrams = sum(references_freq.values())
        num_hypothesis_ngrams = sum(hypothesis_freq.values())

        rouge_matrix[0] = num_matched_ngrams
        rouge_matrix[1] = num_reference_ngrams
        rouge_matrix[2] = num_hypothesis_ngrams
        return rouge_matrix

    # Local variable to accumulate the predictions in the confusion matrix.
    total_rm = metric_variable([3,], tf.float64, name='total_rouge_matrix')

    # Flatten the input.
    hypothesis = tf.reshape(hypothesis, [-1])
    references = tf.reshape(references, [-1])

    # Accumulate the prediction to current confusion matrix.
    current_cm = tf.numpy_function(
        _calculate_rouge_matrix, [hypothesis, references, max_n], tf.float64
    )
    update_op = tf.compat.v1.assign_add(total_rm, current_cm)
    return total_rm, update_op


def rouge_score_metric(
    hypothesis,
    references,
    max_n=1,
    metrics_collections=None,
    updates_collections=None,
    name=None,
):
    """
    Custom TF evaluation metric for calculating rouge score when performing
    text summarization.

    ROUGE (Recall-Oriented Understudy for Gisting Evaluation).

    * the fraction of n-grams from abstracts included in the summarization.

    \begin{equation}
    ROUGE-n(s) = \frac{\sum_{r \in R}\sum_{w} [w \in s][w \in r]}{\sum_{r \in R} \sum_{w} [w \in r]}
    \end{equation}

    *   $r \in R$ -- set of abstracts, written by humans.
    *   $s$ -- abstract, built by the system.
    *   higher the better -- for all metrics of ROUGE family.
    *   $n$ -- order of n-gram:
          * $n=1$ -- unigrams, $n=2$ -- bigrams, etc.
          * with increase of $n$, you achieve more accurate results.
          * with $n =$ len_of_abstract, we require full match of predicted
          text and the one written by humans.

    Usage: Pass to Estimator through ``eval_metric_ops``, TF will accumulate
    ``loss`` over the entire validation set and use that value to calculate
    rouge score.

    The num_matched_ngrams, num_reference_ngrams, num_hypothesis_ngrams
    are accumulated in a rouge matrix, and rouge score (f1, precision, recall)
    is then calculated from it.

    For estimation of the metric over a stream of data, the function creates an
    update_op operation that updates these variables and returns the f1, precision and recall.

    :param Tensor hypothesis: A `Tensor` of predicted summarization tokens with shape
        [batch size, max_sequence_length] and of type `tf.string`.

    :param Tensor references: A `Tensor` of tokens from target summarization, whose shape is
        [batch size, max_sequence_length] and type `tf.string`.

    :param int max_n: Optional maximum size of n-grams to consider. Default is 1.

    :param List metrics_collections: An optional list of collections that
        `rouge_score` should be added to.

    :param List updates_collections: An optional list of collections `update_op`
        should be added to.

    :param string name: An optional variable_scope name.
    :returns tuple:
        rouge_score: A dict with `Tensors` representing f1, precision and recall for rouge-score.
        update_op: An operation that increments the rouge matrix.
    """

    if tf.executing_eagerly():
        raise RuntimeError(
            "rouge_score metric is not supported when eager execution is enabled."
        )

    with tf.compat.v1.variable_scope(
        name, 'rouge_score_metric', (hypothesis, references)
    ):
        # Check if shape is compatible.
        hypothesis.get_shape().assert_is_compatible_with(references.get_shape())

        total_rm, update_op = streaming_rouge_matrix(
            hypothesis, references, max_n
        )

        def _compute_rouge_scores(_, total_rm):
            """Compute the f1, precision, recall score via the rouge matrix."""
            num_matched_ngrams = total_rm[0]
            num_reference_ngrams = total_rm[1]
            num_hypothesis_ngrams = total_rm[2]

            precision = tf.math.divide_no_nan(
                num_matched_ngrams, num_hypothesis_ngrams
            )
            recall = tf.math.divide_no_nan(
                num_matched_ngrams, num_reference_ngrams
            )

            f1_score = tf.math.divide_no_nan(
                recall * precision, (recall + precision) / 2
            )
            return {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
            }

        rouge_scores = aggregate_across_replicas(
            metrics_collections, _compute_rouge_scores, total_rm
        )

        if updates_collections:
            tf.compat.v1.add_to_collection.add_to_collections(
                updates_collections, update_op
            )

        return rouge_scores, update_op
