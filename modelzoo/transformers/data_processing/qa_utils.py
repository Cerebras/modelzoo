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


import collections
import csv
import json
import os

import six
import tqdm

from modelzoo.transformers.data_processing.utils import (
    convert_to_unicode,
    whitespace_tokenize,
)


class SquadExample(object):
    """
    A single training/test example for simple sequence classification.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        doc_tokens,
        orig_answer_text=None,
        start_position=None,
        end_position=None,
        is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (convert_to_unicode(self.qas_id))
        s += ", question_text: %s" % (convert_to_unicode(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        if self.start_position:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


# A single sample of features of data
InputFeatures = collections.namedtuple(
    "InputFeatures",
    [
        "unique_id",
        "example_index",
        "doc_span_index",
        "tokens",
        "token_to_orig_map",
        "token_is_max_context",
        "input_ids",
        "input_mask",
        "segment_ids",
        "start_position",
        "end_position",
        "is_impossible",
    ],
)


def read_squad_examples(input_file, is_training, version_2_with_negative):
    """
    Read a SQuAD json file into a list of SquadExample.
    """
    with open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:

                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[
                            answer_offset + answer_length - 1
                        ]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(
                            doc_tokens[start_position : (end_position + 1)]
                        )
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text)
                        )
                        if actual_text.find(cleaned_answer_text) == -1:
                            print(
                                "Warning: Could not find answer: '%s' vs. '%s'"
                                % (actual_text, cleaned_answer_text,)
                            )
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible,
                )
                examples.append(example)

    return examples


def convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    output_fn,
):
    """
    Loads a data file into a list of `InputBatch`s.
    """

    num_samples = 0
    unique_id = 1000000000
    total_examples = len(examples)

    for (example_index, example) in tqdm.tqdm(
        enumerate(examples), total=total_examples
    ):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = (
                    orig_to_tok_index[example.end_position + 1] - 1
                )
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens,
                tok_start_position,
                tok_end_position,
                tokenizer,
                example.orig_answer_text,
            )

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"]
        )
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[
                    split_token_index
                ]

                is_max_context = _check_is_max_context(
                    doc_spans, doc_span_index, split_token_index
                )
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            if is_training and not example.is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (
                    tok_start_position >= doc_start
                    and tok_end_position <= doc_end
                ):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0

            if example_index < 3:
                print("*** Example ***")
                print("unique_id: %s" % (unique_id))
                print("example_index: %s" % (example_index))
                print("doc_span_index: %s" % (doc_span_index))
                print(
                    "tokens: %s"
                    % " ".join([convert_to_unicode(x) for x in tokens])
                )
                print(
                    "token_to_orig_map: %s"
                    % " ".join(
                        [
                            "%d:%d" % (x, y)
                            for (x, y) in six.iteritems(token_to_orig_map)
                        ]
                    )
                )
                print(
                    "token_is_max_context: %s"
                    % " ".join(
                        [
                            "%d:%s" % (x, y)
                            for (x, y) in six.iteritems(token_is_max_context)
                        ]
                    )
                )
                print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                print(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids])
                )
                if is_training and example.is_impossible:
                    print("impossible example")
                if is_training and not example.is_impossible:
                    answer_text = " ".join(
                        tokens[start_position : (end_position + 1)]
                    )
                    print("start_position: %d" % (start_position))
                    print("end_position: %d" % (end_position))
                    print("answer: %s" % (convert_to_unicode(answer_text)))

            features = InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_position=start_position,
                end_position=end_position,
                is_impossible=example.is_impossible,
            )

            # Run callback
            output_fn(features)

            unique_id += 1
            num_samples += 1

    return num_samples


def convert_examples_to_features_and_write(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    output_dir,
    file_prefix,
    num_output_files,
    is_training=True,
    return_features=False,
):
    meta_data = collections.defaultdict(int)
    total_num_samples = 0

    num_output_files = max(num_output_files, 1)
    output_files = [
        os.path.join(output_dir, "%s-%04i.csv" % (file_prefix, fidx + 1))
        for fidx in range(num_output_files)
    ]
    divided_examples = _divide_list(examples, num_output_files)

    all_features = list()

    for _examples, _output_file in zip(divided_examples, output_files):
        with open(_output_file, "w") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=InputFeatures._fields,
                quoting=csv.QUOTE_MINIMAL,
            )
            writer.writeheader()

            def write_fn(features):
                features_dict = features._asdict()
                writer.writerow(features_dict)
                if return_features:
                    all_features.append(features)

            num_samples = convert_examples_to_features(
                examples=_examples,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                doc_stride=doc_stride,
                max_query_length=max_query_length,
                is_training=is_training,
                output_fn=write_fn,
            )

            output_file = os.path.basename(_output_file)
            meta_data[output_file] += num_samples
            total_num_samples += num_samples

    if return_features:
        return total_num_samples, meta_data, all_features
    else:
        return total_num_samples, meta_data


def _divide_list(li, n):
    """
    Yields n successive lists of equal size, 
    modulo the remainder.

    Example:

        >>> a = list(range(10))
        >>> list(_divide_list(a, 3))
        [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
    """
    start = 0
    for i in range(n):
        stop = start + len(li[i::n])
        yield li[start:stop]
        start = stop


def _improve_answer_span(
    doc_tokens, input_start, input_end, tokenizer, orig_answer_text
):
    """
    Returns tokenized answer spans that better match the annotated answer.
    """
    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """
    Check if this is the 'max context' doc span for the token.
    """
    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = (
            min(num_left_context, num_right_context) + 0.01 * doc_span.length
        )
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index
