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

"""
Classification accuracy script, simple relative of cross-entropy loss.
"""

import tensorflow as tf


def calculate_accuracy(logits, labels, seq_lens=None):
    """
    Calculate accuracy, simple relative of cross-entropy loss that
    counts how many argmax predictions of logits is correct (top-1 accuracy)

    :param Tensor logits: has the size ``[batch_size, output_vocab_size]`` or
        ``[batch_size, max_seq_len, output_vocab_size]``
    :param Tensor labels: has the size ``[batch_size]`` or
        ``[batch_size, max_seq_len]``
    :param Tensor seq_lens: Defaults to None.
        If not none, represents lengths for sentences
        represented by ``logits`` and has size ``[batch_size]``.
        If logits has seq dim, this Tensor being None means we assume
        all sequences have max sequence length.
    :returns: integer top-1 accuracy
    """
    batch_size = tf.shape(logits)[0]
    output_vocab_size = tf.shape(logits)[-1]
    prediction = tf.math.argmax(logits, axis=-1)
    if len(logits.shape.as_list()) == 3:
        max_seq_len = tf.shape(logits)[1]
        if seq_lens is None:
            # Assume all sentences have max sequence length
            num_preds = max_seq_len * batch_size
        else:
            num_preds = tf.reduce_sum(seq_lens)
            # Mask out values after end of sentence, don't contribute to loss
            prediction *= tf.sequence_mask(
                seq_lens, maxlen=max_seq_len, dtype=prediction.dtype
            )
    else:
        num_preds = batch_size

    prediction_one_hot = tf.one_hot(prediction, output_vocab_size)
    labels_one_hot = tf.one_hot(labels, output_vocab_size)
    num_correct = tf.reduce_sum(
        tf.math.multiply(prediction_one_hot, labels_one_hot),
    )
    return num_correct / tf.cast(num_preds, num_correct.dtype)
