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

import os

import numpy as np
import tensorflow as tf
import yaml

from modelzoo.transformers.data_processing.Tokenization import FullTokenizer
from modelzoo.transformers.tf.bert.utils import load_pretrain_model_params
from modelzoo.transformers.tf.bert.utils import (
    set_defaults as set_bert_defaults,
)


def get_params(params_file):
    """
    Reads in params from yaml, fills in bert pretrain params if provided,
    uses defaults from bert's utils.py.

    :param  params_file: path to yaml with classifier params
    :returns: params dict
    """
    # Load yaml into params
    with open(params_file, "r") as stream:
        params = yaml.safe_load(stream)

    set_defaults(params)

    return params


def set_defaults(params):
    if "pretrain_params_path" in params["model"]:
        load_pretrain_model_params(
            params, os.path.dirname(os.path.abspath(__file__))
        )

    set_bert_defaults(params)

    params["model_type"] = "BertSummarizationModel"

    params["model"]["loss_weight"] = params["model"].get("loss_weight", 1.0)
    # use eval_input params as defaults for predict_input
    predict_input_params = params["eval_input"].copy()
    if "predict_input" in params:
        predict_input_params.update(params["predict_input"])
    params["predict_input"] = predict_input_params


def extract_text_words_given_cls_indices_per_seq_in_batch(x, tokenizer):
    """
    This function follows the same logic as
    `extract_text_words_given_cls_indices` but working with
    each object in the batch.
    """

    def _pad_input_sequence(input_sequence, max_sequence_length):
        input_sequence = input_sequence + ["[pad]"] * (
            max_sequence_length - len(input_sequence)
        )
        return np.array(input_sequence, dtype=object)

    def _extract_text_words_by_token_ids(
        input_ids, tokenizer, max_sequence_length
    ):
        """
        Takes input ids of tokens and convert them to a tensor with words.
        :param input_ids: Numpy array of shape (max_sequence_length,).
        :param tokenizer: Tokenizer object which contains functions to
            convert words to token and vice versa.
        :param max_sequence_length: int, maximum length of the sequence.
        :returns: words_padded: Numpy array with computed words padded
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

    def _extract_text_tokens_given_cls_indices(
        labels, cls_indices, cls_weights, input_ids
    ):
        """
        Extract text tokens that belongs to segments which CLS tokens
        have labels equal to 1.
        Example:
            [[CLS, label=1] Dogs, like, cats, [CLS, label=0], Cats, like, dogs]
            -> [Dogs, like, cats].
        :param labels: Numpy array of shape (max_cls_tokens,).
        :param cls_indices: Numpy array of shape (max_cls_tokens).
        :param cls_weights: Numpy array of shape (max_cls_tokens,).
        :param input_ids: Numpy array of shape (max_sequence_length,).
        :returns: extracted_input_ids: Numpy array with extracted input ids.
        """

        # Extract only useful tokens with cls weights
        # and labels not eq 0.
        mask = (labels * cls_weights) != 0
        starts = cls_indices[mask]
        ends = np.roll(cls_indices, shift=-1)[mask]

        # In case we reached end of sequence, the end
        # index should be max seq length.
        if len(ends) > 0 and ends[-1] == 0:
            ends[-1] = input_ids.shape[0]

        extracted_token_ids = []
        for start_index, end_index in zip(starts, ends):
            extracted_token_ids.extend(input_ids[start_index:end_index])

        return np.array(extracted_token_ids, dtype=np.int64)

    input_ids, labels, cls_indices, cls_weights = x

    token_ids = tf.numpy_function(
        _extract_text_tokens_given_cls_indices,
        [labels, cls_indices, cls_weights, input_ids],
        tf.int64,
    )

    max_sequence_length = input_ids.shape[-1]
    words = tf.numpy_function(
        lambda x: _extract_text_words_by_token_ids(
            x, tokenizer, max_sequence_length
        ),
        [token_ids],
        tf.string,
    )
    return words


def extract_text_words_given_cls_indices(
    labels, cls_indices, cls_weights, input_ids, vocab_file
):
    """
    Extract text words that belongs to segments which CLS tokens
    have labels equal to 1. Compared to `extract_text_tokens_given_cls_indices`
    this function extracts entire words instead of word pieces.

    Example:
        [[CLS, label=1] Dogs, like, cats, [CLS, label=0], Cats, like, dogs]
        -> [Dogs, like, cats].

    :param labels: Tensor of shape (batch_size, max_cls_tokens).
    :param cls_indices: Tensor of shape (batch_size, max_cls_tokens).
    :param cls_weights: Tensor of shape (batch_size, max_cls_tokens).
    :param input_ids: Tensor of shape (batch_size, max_sequence_length).
    :param vocab_file: Path to vocabulary file.
    :returns: extracted_words: Tensor with extracted words.
    """
    tokenizer = FullTokenizer(vocab_file, do_lower_case=True)

    # Obtain words from extracted input ids.
    extracted_words = tf.nest.map_structure(
        tf.stop_gradient,
        tf.map_fn(
            lambda x: extract_text_words_given_cls_indices_per_seq_in_batch(
                x, tokenizer
            ),
            [input_ids, labels, cls_indices, cls_weights],
            tf.string,
        ),
    )
    return extracted_words
