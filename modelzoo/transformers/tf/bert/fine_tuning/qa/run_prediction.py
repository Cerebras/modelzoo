# This code is adapted from
# https://github.com/google-research/bert/blob/master/run_squad.py
#
# coding=utf-8
#
# Copyright 2022 Cerebras Systems.
#
# Copyright 2018 The Google AI Language Team Authors.
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

from __future__ import absolute_import, division, print_function

import collections
import json
import math
import os
import sys

import six
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../.."))
from modelzoo.transformers.data_processing.qa_utils import (
    convert_examples_to_features,
    read_squad_examples,
)
from modelzoo.transformers.data_processing.Tokenization import (
    BaseTokenizer,
    FullTokenizer,
)
from modelzoo.transformers.tf.bert.fine_tuning.qa.data import predict_input_fn
from modelzoo.transformers.tf.bert.fine_tuning.qa.input.utils import (
    FeatureWriter,
)
from modelzoo.transformers.tf.bert.fine_tuning.qa.model import model_fn
from modelzoo.transformers.tf.bert.fine_tuning.qa.utils import get_params

flags = tf.compat.v1.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string("params", None, "This specifies the model architecture.")

flags.DEFINE_string(
    "output_dir",
    None,
    "The output directory where the data and predictions will be written.",
)

flags.DEFINE_string(
    "predict_file",
    None,
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json",
)

## Other parameters
flags.DEFINE_string(
    "checkpoint_path",
    None,
    "Initial checkpoint (usually from a pre-trained BERT model).",
)

flags.DEFINE_bool(
    "do_lower_case",
    True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.",
)

flags.DEFINE_integer(
    "max_seq_length",
    384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.",
)

flags.DEFINE_integer(
    "doc_stride",
    128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.",
)

flags.DEFINE_integer(
    "max_query_length",
    64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.",
)

flags.DEFINE_integer(
    "n_best_size",
    1,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.",
)

flags.DEFINE_integer(
    "max_answer_length",
    30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.",
)

flags.DEFINE_bool(
    "verbose_logging",
    False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.",
)

flags.DEFINE_bool(
    "version_2_with_negative",
    False,
    "If true, the SQuAD examples contain some that do not have an answer.",
)

flags.DEFINE_float(
    "null_score_diff_threshold",
    0.0,
    "If null_score - best_non_null is greater than the threshold predict null.",
)


RawResult = collections.namedtuple(
    "RawResult", ["unique_id", "start_logits", "end_logits"]
)


def write_predictions(
    all_examples,
    all_features,
    all_results,
    n_best_size,
    max_answer_length,
    do_lower_case,
    output_prediction_file,
    output_nbest_file,
    output_null_log_odds_file,
    vocab_file,
):
    """Write final predictions to the json file and log-odds of null if needed."""
    tf.compat.v1.logging.info(
        "Writing predictions to: %s" % (output_prediction_file)
    )
    tf.compat.v1.logging.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(
        "PrelimPrediction",
        [
            "feature_index",
            "start_index",
            "end_index",
            "start_logit",
            "end_logit",
        ],
    )

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for example_index, example in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min mull score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for feature_index, feature in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if FLAGS.version_2_with_negative:
                feature_null_score = (
                    result.start_logits[0] + result.end_logits[0]
                )
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                        )
                    )

        if FLAGS.version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                )
            )
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True,
        )

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"]
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[
                    pred.start_index : (pred.end_index + 1)
                ]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[
                    orig_doc_start : (orig_doc_end + 1)
                ]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(
                    tok_text, orig_text, vocab_file, do_lower_case
                )
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                )
            )

        # if we didn't inlude the empty option in the n-best, inlcude it
        if FLAGS.version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit,
                    )
                )
        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0)
            )

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not FLAGS.version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = (
                score_null
                - best_non_null_entry.start_logit
                - (best_non_null_entry.end_logit)
            )
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > FLAGS.null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json

    with tf.compat.v1.gfile.GFile(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with tf.compat.v1.gfile.GFile(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if FLAGS.version_2_with_negative:
        with tf.compat.v1.gfile.GFile(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")


def get_final_text(pred_text, orig_text, vocab_file, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BaseTokenizer(vocab_file, do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if FLAGS.verbose_logging:
            tf.compat.v1.logging.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text)
            )
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if FLAGS.verbose_logging:
            tf.compat.v1.logging.info(
                "Length not equal after stripping spaces: '%s' vs '%s'",
                orig_ns_text,
                tok_ns_text,
            )
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if FLAGS.verbose_logging:
            tf.compat.v1.logging.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if FLAGS.verbose_logging:
            tf.compat.v1.logging.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position : (orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True
    )

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    data_filename = os.path.join(FLAGS.output_dir, "eval.tf_record")

    params = get_params(FLAGS.params)
    params["predict_input"]["data_file"] = data_filename
    if params["predict_input"]["batch_size"] != 1:
        tf.compat.v1.logging.warn(
            "To avoid dropping examples, batch_size 1 will be used."
        )
        params["predict_input"]["batch_size"] = 1
    vocab_file = params["predict_input"]["vocab_file"]

    tf.compat.v1.gfile.MakeDirs(FLAGS.output_dir)

    tokenizer = FullTokenizer(
        vocab_file=vocab_file, do_lower_case=FLAGS.do_lower_case
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=FLAGS.output_dir, params=params
    )

    eval_examples = read_squad_examples(
        input_file=FLAGS.predict_file,
        is_training=False,
        version_2_with_negative=FLAGS.version_2_with_negative,
    )

    eval_writer = FeatureWriter(filename=data_filename, is_training=False,)
    eval_features = []

    def append_feature(feature):
        eval_features.append(feature)
        eval_writer.process_feature(feature)

    convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        is_training=False,
        output_fn=append_feature,
    )
    eval_writer.close()

    tf.compat.v1.logging.info("***** Running predictions *****")
    tf.compat.v1.logging.info("  Num orig examples = %d", len(eval_examples))
    tf.compat.v1.logging.info("  Num split examples = %d", len(eval_features))
    tf.compat.v1.logging.info(
        "  Batch size = %d", params["predict_input"]["batch_size"]
    )

    all_results = []
    for result in estimator.predict(
        predict_input_fn,
        checkpoint_path=FLAGS.checkpoint_path,
        yield_single_examples=True,
    ):
        if len(all_results) % 1000 == 0:
            tf.compat.v1.logging.info(
                "Processing example: %d" % (len(all_results))
            )
        unique_id = int(result["unique_ids"])
        start_logits = [float(x) for x in result["start_logits"].flat]
        end_logits = [float(x) for x in result["end_logits"].flat]
        all_results.append(
            RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits,
            )
        )

    output_prediction_file = os.path.join(FLAGS.output_dir, "predictions.json")
    output_nbest_file = os.path.join(FLAGS.output_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(FLAGS.output_dir, "null_odds.json")

    write_predictions(
        eval_examples,
        eval_features,
        all_results,
        FLAGS.n_best_size,
        FLAGS.max_answer_length,
        FLAGS.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        vocab_file,
    )


if __name__ == "__main__":
    flags.mark_flag_as_required("params")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
