# This code is adapted from
# https://github.com/google-research/bert/blob/master/run_squad.py
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

import argparse
import collections
import json
import math
import os
import sys

import six
import torch
import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../.."))
from modelzoo.common.input.utils import save_params
from modelzoo.common.pytorch.utils import get_params
from modelzoo.transformers.data_processing.qa_utils import (
    convert_examples_to_features_and_write,
    read_squad_examples,
)
from modelzoo.transformers.data_processing.Tokenization import (
    BaseTokenizer,
    FullTokenizer,
)
from modelzoo.transformers.pytorch.bert.fine_tuning.qa.data import (
    predict_input_dataloader,
)
from modelzoo.transformers.pytorch.bert.fine_tuning.qa.model import (
    BertForQuestionAnsweringModel as Model,
)


def parse_args():
    parser = argparse.ArgumentParser()
    # required args
    parser.add_argument(
        "--params", required=True, help="Path to yaml configuration.",
    )
    parser.add_argument(
        "--predict_file",
        required=True,
        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json",
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        help="Path to checkpoint of a fine-tuned BERT model.",
    )
    # optional args
    parser.add_argument(
        "--output_dir",
        required=False,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "prediction_output"
        ),
        help="The output directory where the data and predictions will be written.",
    )
    parser.add_argument(
        "--do_lower_case",
        required=False,
        action="store_true",
        help="Whether to convert tokens to lowercase",
    )
    parser.add_argument(
        "--max_seq_length",
        required=False,
        type=int,
        default=384,
        help="The maximum total input sequence length after WordPiece tokenization.",
    )
    parser.add_argument(
        "--doc_stride",
        required=False,
        type=int,
        default=128,
        help="When splitting up a long document into chunks, how much stride to "
        "take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        required=False,
        type=int,
        default=64,
        help="The maximum number of tokens for the question. Questions longer than "
        "this will be truncated to this length.",
    )
    parser.add_argument(
        "--n_best_size",
        required=False,
        type=int,
        default=1,
        help="The total number of n-best predictions to generate in the "
        "nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        required=False,
        type=int,
        default=30,
        help="The maximum length of an answer that can be generated. This is needed "
        "because the start and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        required=False,
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--version_2_with_negative",
        required=False,
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        required=False,
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold "
        "predict null.",
    )
    args = parser.parse_args()
    return args


RawResult = collections.namedtuple(
    "RawResult", ["unique_id", "start_logits", "end_logits"]
)


def main():
    args = parse_args()

    params = get_params(args.params)
    if params["predict_input"]["batch_size"] != 1:
        print("To avoid dropping examples, batch_size 1 will be used.")
        params["predict_input"]["batch_size"] = 1

    vocab_file = params["predict_input"]["vocab_file"]

    os.makedirs(args.output_dir)

    tokenizer = FullTokenizer(
        vocab_file=vocab_file, do_lower_case=args.do_lower_case
    )

    print("Reading squad examples...")
    eval_examples = read_squad_examples(
        input_file=args.predict_file,
        is_training=False,
        version_2_with_negative=args.version_2_with_negative,
    )

    file_prefix = "eval"
    num_output_files = 1

    print("Writing tokenized examples to csv...")
    (
        num_examples_written,
        meta_data,
        eval_features,
    ) = convert_examples_to_features_and_write(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        output_dir=args.output_dir,
        file_prefix=file_prefix,
        num_output_files=num_output_files,
        is_training=False,
        return_features=True,
    )

    meta_file = os.path.join(args.output_dir, "meta.dat")
    with open(meta_file, "w") as fout:
        for output_file, num_lines in meta_data.items():
            fout.write("%s %s\n" % (output_file, num_lines))

    # Write args passed and number of examples
    args_dict = vars(args)
    args_dict["num_examples"] = num_examples_written
    save_params(args_dict, model_dir=args.output_dir)

    num_read_examples = len(eval_examples)
    num_features = len(eval_features)
    print("Num examples read = %i" % (num_read_examples))
    print("Num features = %i" % (num_features))
    print("Num examples written = %i" % (num_examples_written))

    assert params["predict_input"]["batch_size"] == 1
    params["predict_input"]["data_dir"] = args.output_dir
    params["runconfig"]["mode"] = "eval"

    data_loader = predict_input_dataloader(params)

    num_total_examples = len(data_loader)
    print("Num examples in dataloader = %i" % (num_total_examples))
    print("Batch size = %i" % (params["predict_input"]["batch_size"]))

    model = Model(params)
    model.eval()

    checkpoint_path = args.checkpoint_path
    if checkpoint_path:
        print("Loading checkpoint_path = %s" % checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.model.load_state_dict(state_dict["model"])

    print("Running predictions...")
    all_results = []
    with torch.no_grad():
        for example_index, features in tqdm.tqdm(
            enumerate(data_loader), total=num_total_examples
        ):
            loss = model(features)
            unique_id = features["unique_ids"]
            start_logits = model.outputs["start_logits"].squeeze().tolist()
            end_logits = model.outputs["end_logits"].squeeze().tolist()
            unique_id = int(unique_id)
            all_results.append(
                RawResult(
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits,
                )
            )

    num_results = len(all_results)
    print("Num results = %i" % (num_results))

    output_prediction_file = os.path.join(args.output_dir, "predictions.json")
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(args.output_dir, "null_odds.json")

    print("Writing predictions...")
    write_predictions(
        eval_examples,
        eval_features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        vocab_file,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        args.verbose_logging,
    )

    print("Done.")


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
    version_2_with_negative=False,
    null_score_diff_threshold=0.0,
    verbose_logging=False,
):
    """
    Write final predictions to the json file and log-odds of null if needed.
    """
    print("Writing predictions to: %s" % (output_prediction_file))
    print("Writing nbest to: %s" % (output_nbest_file))

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

        print("DEBUG: len(features) = ", len(features), flush=True)
        for feature_index, feature in enumerate(features):
            print("DEBUG: feature_index = ", feature_index, flush=True)
            print("DEBUG: feature = ", feature, flush=True)
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)

            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = (
                    result.start_logits[0] + result.end_logits[0]
                )
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]

            print("DEBUG: start_indexes = ", start_indexes, flush=True)
            print("DEBUG: end_indexes = ", end_indexes, flush=True)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    print(
                        "DEBUG: start_index, end_index = ",
                        start_index,
                        end_index,
                        flush=True,
                    )
                    if start_index >= len(feature.tokens):
                        print(
                            "DEBUG: start_index >= len(feature.tokens)",
                            start_index,
                            len(feature.tokens),
                            flush=True,
                        )
                        continue
                    if end_index >= len(feature.tokens):
                        print(
                            "DEBUG: end_index >= len(feature.tokens)",
                            end_index,
                            len(feature.tokens),
                            flush=True,
                        )
                        continue
                    if start_index not in feature.token_to_orig_map:
                        print(
                            "DEBUG: start_index not in feature.token_to_orig_map",
                            start_index,
                            flush=True,
                        )
                        continue
                    if end_index not in feature.token_to_orig_map:
                        print(
                            "DEBUG: end_index not in feature.token_to_orig_map",
                            end_index,
                            flush=True,
                        )
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        print(
                            "DEBUG: not feature.token_is_max_context",
                            start_index,
                            feature.token_is_max_context.get(
                                start_index, False
                            ),
                            flush=True,
                        )
                        continue
                    if end_index < start_index:
                        print(
                            "DEBUG: end_index < start_index",
                            end_index,
                            start_index,
                            flush=True,
                        )
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        print(
                            "DEBUG: length > max_answer_length",
                            length,
                            max_answer_length,
                            flush=True,
                        )
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

        if version_2_with_negative:
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
                    tok_text,
                    orig_text,
                    vocab_file,
                    do_lower_case,
                    verbose_logging=verbose_logging,
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
        if version_2_with_negative:
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
            print("DEBUG: empty", flush=True)
            print(
                "DEBUG: prelim_predictions = ", prelim_predictions, flush=True
            )
            print("DEBUG: features = ", features, flush=True)
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

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = (
                score_null
                - best_non_null_entry.start_logit
                - (best_non_null_entry.end_logit)
            )
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")


def get_final_text(
    pred_text, orig_text, vocab_file, do_lower_case, verbose_logging=False
):
    """
    Project the tokenized prediction back to the original text.
    """
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
        if verbose_logging:
            print("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            print(
                "Length not equal after stripping spaces: '%s' vs '%s'"
                % (orig_ns_text, tok_ns_text)
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
        if verbose_logging:
            print("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            print("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position : (orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """
    Get the n-best logits from a list.
    """
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
    """
    Compute softmax probability over raw logits.
    """
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


if __name__ == "__main__":
    main()
