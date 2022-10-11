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
Rouge Score metric for PyTorch.
"""

import string
from collections import Counter
from typing import Optional

import numpy as np
import torch
from nltk import ngrams

from modelzoo.common.pytorch.metrics.cb_metric import CBMetric
from modelzoo.transformers.data_processing.Tokenization import FullTokenizer


class RougeScoreMetric(CBMetric):
    """Custom evaluation metric for calculating rouge score when performing
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

    The num_matched_ngrams, num_references_ngrams, num_hypothesis_ngrams
    are accumulated in a rouge matrix, and rouge score (f1, precision, recall)
    is then calculated from it.
    """

    def __init__(
        self, vocab_file: str, max_n: int = 1, name: Optional[str] = None
    ):
        """
        Args:
            vocab_file: Path to the vocab file.
            max_n: Optional maximum size of n-grams to consider. Default is 1.
            name: Name of the metric.
        """
        self.max_n = max_n
        self.vocab_file = vocab_file
        self.tokenizer = FullTokenizer(self.vocab_file)
        super().__init__(name=name)

    def init_state(self):
        self.reset_state()

    def reset_state(self):
        # We store 3 items: num_matched_ngrams, num_references_ngrams
        # and num_hypothesis_ngrams.
        self.rouge_matrix = np.zeros((3,), dtype=np.float64)

    def update_on_host(
        self, labels, predictions, cls_indices, cls_weights, input_ids
    ):
        """
        Compute and aggregate rouge_matrix every iteration.
        Each computation comprises of:
            1. Convert labels to references.
            2. Convert predictions to hypotheses.
            3. Convert hypotheses and references to ngrams.
            4. Calculate rouge matrix.
        """

        def _preprocess_before_rouge(sentences):
            def _preprocess_sentence_before_rouge(sentence):
                special_words = {"[pad]", "[cls]", "[sep]"}
                punctuation_words = set(string.punctuation)
                words_to_ignore = punctuation_words | special_words
                words_in_sentence = [word.lower() for word in sentence]
                words_in_sentence = list(
                    filter(
                        lambda word: word not in words_to_ignore,
                        words_in_sentence,
                    )
                )

                return " ".join(words_in_sentence)

            words = np.array(
                [
                    _preprocess_sentence_before_rouge(sentence)
                    for sentence in sentences
                ]
            )
            return words

        predictions = predictions.detach()
        labels = labels.detach()
        cls_indices = cls_indices.detach()
        cls_weights = cls_weights.detach()
        input_ids = input_ids.detach()
        hypotheses = extract_text_words_given_cls_indices(
            predictions, cls_indices, cls_weights, input_ids, self.tokenizer
        )
        references = extract_text_words_given_cls_indices(
            labels, cls_indices, cls_weights, input_ids, self.tokenizer
        )

        hypotheses = _preprocess_before_rouge(hypotheses)
        references = _preprocess_before_rouge(references)

        current_rouge_matrix = np.zeros((3,), dtype=np.float64)

        hypotheses = [x.split(" ") for x in hypotheses]
        references = [x.split(" ") for x in references]

        hypotheses_ngrams = [
            ngrams(sentence, self.max_n) for sentence in hypotheses
        ]
        references_ngrams = [
            ngrams(sentence, self.max_n) for sentence in references
        ]

        hypotheses_freq = [
            Counter(hypotheses_sentence_ngrams)
            for hypotheses_sentence_ngrams in hypotheses_ngrams
        ]
        references_freq = [
            Counter(references_sentence_ngrams)
            for references_sentence_ngrams in references_ngrams
        ]

        matched_ngrams_freq = []
        num_matched_ngrams = 0
        num_references_ngrams = 0
        num_hypotheses_ngrams = 0

        # For each sentence, compute the number of matched n-grams, and
        # total n-grams in hypotheses and references.
        for sent_idx in range(len(hypotheses_freq)):
            matched_ngrams_freq.append(
                hypotheses_freq[sent_idx] & references_freq[sent_idx]
            )
            num_matched_ngrams += sum(matched_ngrams_freq[-1].values())
            num_references_ngrams += sum(references_freq[sent_idx].values())
            num_hypotheses_ngrams += sum(hypotheses_freq[sent_idx].values())

        current_rouge_matrix[0] = num_matched_ngrams
        current_rouge_matrix[1] = num_references_ngrams
        current_rouge_matrix[2] = num_hypotheses_ngrams

        self.rouge_matrix += current_rouge_matrix

    def compute(self):
        """
        Compute the f1, precision, recall score via the rouge matrix.
        """
        num_matched_ngrams = self.rouge_matrix[0]
        num_references_ngrams = self.rouge_matrix[1]
        num_hypotheses_ngrams = self.rouge_matrix[2]

        precision = (
            num_matched_ngrams if num_hypotheses_ngrams else 0.0
        ) / num_hypotheses_ngrams

        recall = (
            num_matched_ngrams if num_references_ngrams else 0.0
        ) / num_references_ngrams

        f1_score = (2 * recall * precision if recall and precision else 0.0) / (
            recall + precision
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }


def extract_text_words_given_cls_indices(
    labels, cls_indices, cls_weights, input_ids, tokenizer
):
    """Extract text words that belongs to segments which CLS tokens
    have labels equal to 1.

    Example:
        [[CLS, label=1] Dogs, like, cats, [CLS, label=0], Cats, like, dogs]
        -> [Dogs, like, cats].

    Args:
        labels: Tensor of shape (batch_size, max_cls_tokens).
        cls_indices: Tensor of shape (batch_size, max_cls_tokens).
        cls_weights: Tensor of shape (batch_size, max_cls_tokens).
        input_ids: Tensor of shape (batch_size, max_sequence_length).
        tokenizer: Tokenizer to be used.
    
    Returns:
        extracted_words: Tensor with extracted words.
    """
    batch_size = labels.shape[0]

    # token_ids = []
    extracted_words_batch = []
    max_sequence_length = input_ids.shape[-1]
    for i in range(batch_size):
        extracted_token_ids = extract_text_tokens_given_cls_indices(
            labels[i], cls_indices[i], cls_weights[i], input_ids[i]
        )
        extracted_words = extract_text_words_by_token_ids(
            extracted_token_ids, tokenizer, max_sequence_length
        )
        extracted_words_batch.append(extracted_words)

    return np.array(extracted_words_batch)


def extract_text_tokens_given_cls_indices(
    labels, cls_indices, cls_weights, input_ids
):
    """
    Extract text tokens that belongs to segments which CLS tokens
    have labels equal to 1.
    Example:
        [[CLS, label=1] Dogs, like, cats, [CLS, label=0], Cats, like, dogs]
        -> [Dogs, like, cats].
    
    Args:
        labels: Numpy array of shape (max_cls_tokens,).
        cls_indices: Numpy array of shape (max_cls_tokens).
        cls_weights: Numpy array of shape (max_cls_tokens,).
        input_ids: Numpy array of shape (max_sequence_length,).

    Returns:
        extracted_input_ids: Numpy array with extracted input ids.
    """

    # Extract only useful tokens with cls weights
    # and labels not eq 0.
    mask = (labels * cls_weights) != 0
    starts = cls_indices[mask]
    ends = torch.roll(cls_indices, shifts=-1)[mask]
    # In case we reached end of sequence, the end
    # index should be max seq length.
    if len(ends) > 0 and ends[-1] == 0:
        ends[-1] = input_ids.shape[0]

    extracted_token_ids = []
    for start_index, end_index in zip(starts, ends):
        extracted_token_ids.extend(input_ids[start_index:end_index].numpy())

    return np.array(extracted_token_ids, dtype=np.int64)


def _pad_input_sequence(input_sequence, max_sequence_length):
    input_sequence = input_sequence + ["[pad]"] * (
        max_sequence_length - len(input_sequence)
    )
    return np.array(input_sequence, dtype=object)


def extract_text_words_by_token_ids(input_ids, tokenizer, max_sequence_length):
    """
    Takes input ids of tokens and convert them to a tensor with words.

    Args:
        input_ids: Numpy array of shape (max_sequence_length,).
        tokenizer: Tokenizer object which contains functions to
            convert words to token and vice versa.
        max_sequence_length: int, maximum length of the sequence.
    Returns: 
        words_padded: Numpy array with computed words padded 
        to max seq length.
    """
    extracted_text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    # There is no decode method provided in FullTokenizer
    # which is usually used with bert. Thus, we join word pieces and remove `##`
    # symbols that define boundaries between word pieces to form words from token
    # indices.
    text = " ".join(extracted_text_tokens)
    text = text.replace(" ##", "")
    words = text.split(" ")
    words_padded = _pad_input_sequence(words, max_sequence_length)
    return words_padded
