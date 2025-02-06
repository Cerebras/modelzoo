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

# This code taken from with a few modifications:
# https://github.com/nlpyang/BertSum/blob/master/src/prepro/data_builder.py
#
# coding=utf-8
#
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
Common pre-processing functions for BERTSUM data processing
"""

import argparse
import glob
import hashlib
import json
import logging
import os
import re
import subprocess
from multiprocessing import Pool

from nltk import ngrams

from cerebras.modelzoo.data_preparation.nlp.tokenizers.Tokenization import (
    FullTokenizer,
)

logging.basicConfig(level=logging.INFO)


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class Tokenizer:
    def __init__(self, params):
        """
        Tokenizes files from the input path into output path.
        Stanford CoreNLP is used for tokenization.
        :param params: dict params: Tokenizer configuration parameters.
        """
        self.input_path = os.path.abspath(params.input_path)
        self.output_path = os.path.abspath(params.output_path)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def process(self):
        logging.info(
            f"Preparing to tokenize {self.input_path} to {self.output_path}."
        )

        story_files = glob.iglob(os.path.join(self.input_path, "*.story"))
        story_count = 0
        with open("mapping_for_corenlp.txt", "w") as fout:
            for file in story_files:
                story_count += 1
                fout.write(f"{file}\n")

        command = [
            "java",
            "edu.stanford.nlp.pipeline.StanfordCoreNLP",
            "-annotators",
            "tokenize,ssplit",
            "-ssplit.newlineIsSentenceBreak",
            "always",
            "-filelist",
            "mapping_for_corenlp.txt",
            "-outputFormat",
            "json",
            "-outputDirectory",
            self.output_path,
        ]
        logging.info(
            f"Tokenizing {story_count} files in {self.input_path} "
            f"and saving in {self.output_path}."
        )

        subprocess.call(command)
        logging.info("Stanford CoreNLP Tokenizer has finished.")
        os.remove("mapping_for_corenlp.txt")

        check_output(self.input_path, self.output_path)
        logging.info(
            f"Successfully finished tokenizing {self.input_path} to {self.output_path}.\n"
        )


class JsonConverter:
    def __init__(self, params):
        """
        JsonConverter simplifies the input and convert it into json files format
        with source and target (summarized) texts.
        Splits input into `train`, `test` and `valid` parts
        based on the `map_path`.
        :param params: dict params: JsonConverter configuration parameters.
        """
        self.map_path = os.path.abspath(params.map_path)
        self.input_path = os.path.abspath(params.input_path)
        self.output_path = os.path.abspath(params.output_path)
        self.n_cpu = params.n_cpu
        self.shard_size = params.shard_size
        self.lower_case = params.lower_case

        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

    @staticmethod
    def _hashhex(string):
        h = hashlib.sha1()
        h.update(string.encode("utf-8"))
        return h.hexdigest()

    @staticmethod
    def _clean(string):
        re_map = {
            "-lrb-": "(",
            "-rrb-": ")",
            "-lcb-": "{",
            "-rcb-": "}",
            "-lsb-": "[",
            "-rsb-": "]",
            "``": '"',
            "''": '"',
        }
        return re.sub(
            r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
            lambda x: re_map.get(x.group()),
            string,
        )

    def _load_json_file(self, file_name):
        source = []
        target = []
        is_target = False
        with open(file_name, "r") as fin:
            for sentence in json.load(fin)["sentences"]:
                tokens = [token["word"] for token in sentence["tokens"]]

                if self.lower_case:
                    tokens = [token.lower() for token in tokens]

                # In the input format after the symbol `@highlight`
                # starts the target text (summarized text).
                if tokens[0] == "@highlight":
                    is_target = True
                    continue

                if is_target:
                    target.append(tokens)
                    is_target = False
                else:
                    source.append(tokens)

        # Removing special symbols that were generated byproduct of
        # downloading html pages.
        source = [
            self._clean(" ".join(sentence)).split() for sentence in source
        ]
        target = [
            self._clean(" ".join(sentence)).split() for sentence in target
        ]

        return {"src": source, "tgt": target}

    def process(self):
        logging.info(
            f"Preparing to convert to json files {self.input_path} to {self.output_path}."
        )

        corpus_mapping = {}
        for corpus_type in {"valid", "test", "train"}:
            urls = []
            with open(
                os.path.join(self.map_path, f"mapping_{corpus_type}.txt"), "r"
            ) as fin:
                for line in fin.readlines():
                    # Gets hash from the url.
                    urls.append(self._hashhex(line.strip()))
            corpus_mapping[corpus_type] = set([key.strip() for key in urls])

        train_files, valid_files, test_files = [], [], []

        for file_name in glob.iglob(os.path.join(self.input_path, "*.json")):
            real_name = os.path.basename(file_name).split(".")[0]

            if real_name in corpus_mapping["valid"]:
                valid_files.append(file_name)

            elif real_name in corpus_mapping["test"]:
                test_files.append(file_name)

            elif real_name in corpus_mapping["train"]:
                train_files.append(file_name)

            else:
                logging.info(
                    f"File {file_name} is not found in any"
                    f" corpus types (`train`, `test` or `valid`)."
                )

        corpora = {
            "train": train_files,
            "valid": valid_files,
            "test": test_files,
        }

        for corpus_type in ["valid", "test", "train"]:
            pool = Pool(self.n_cpu)

            dataset = []
            count_output_files = 0
            # Convert input into json format with source and target
            # (summarized) input texts.
            for d in pool.imap_unordered(
                self._load_json_file, corpora[corpus_type]
            ):
                dataset.append(d)

                if len(dataset) > self.shard_size:
                    out_fname = os.path.join(
                        self.output_path,
                        f"{corpus_type}-{count_output_files}.json",
                    )
                    with open(out_fname, "w") as fout:
                        fout.write(json.dumps(dataset))
                        count_output_files += 1
                        dataset = []

            pool.close()
            pool.join()

            if len(dataset) > 0:
                out_fname = os.path.join(
                    self.output_path,
                    f"{corpus_type}-{count_output_files}.json",
                )
                with open(out_fname, "w") as fout:
                    fout.write(json.dumps(dataset))

        check_output(self.input_path, self.output_path)
        logging.info(
            f"Successfully finished converting to json files {self.input_path} to {self.output_path}.\n"
        )


class BertData:
    def __init__(self, params):
        """
        Converts input into bert format.
        :param params: dict params: BertData configuration parameters.
        """
        self.min_tokens_per_sentence = params.min_tokens_per_sentence
        self.max_tokens_per_sentence = params.max_tokens_per_sentence
        self.min_sentences_per_sequence = params.min_sentences_per_sequence
        self.max_sentences_per_sequence = params.max_sentences_per_sequence
        self.max_sequence_length = params.max_sequence_length

        self.tokenizer = FullTokenizer(
            params.vocab_file, do_lower_case=params.lower_case
        )
        self.cls_token, self.sep_token = "[CLS]", "[SEP]"
        self.cls_id, self.sep_id = self.tokenizer.convert_tokens_to_ids(
            [self.cls_token, self.sep_token]
        )
        self.pad_id = self.tokenizer.convert_tokens_to_ids(["[PAD]"])[0]

    def _get_sentences_ids(self, sentences):
        sentences_tokenized = [sentence.split() for sentence in sentences]

        # Filter out sentences with less than min_tokens_per_sentence.
        # Cut each sentence to leave only max_tokens_per_sentence.
        sentences_tokenized_filtered = [
            tokens[: self.max_tokens_per_sentence]
            for tokens in sentences_tokenized
            if len(tokens) > self.min_tokens_per_sentence
        ]

        # Filter out sequences with less than min_sentences_per_sequence.
        # Leave only max_sentences_per_sequence.
        if len(sentences_tokenized_filtered) < self.min_sentences_per_sequence:
            return []
        sentences_tokenized_filtered = sentences_tokenized_filtered[
            : self.max_sentences_per_sequence
        ]

        # Tokenize sequence of sentences with FullTokenizer.
        # Augment with [SEP], [CLS] special tokens to separate sentences.
        sentences_tokenized = []
        for i, sentence in enumerate(sentences_tokenized_filtered):
            sentence = self.tokenizer.tokenize(" ".join(sentence))
            sentences_tokenized.append(sentence)
            if i + 1 != len(sentences_tokenized_filtered):
                sentences_tokenized[-1].append(self.sep_token)
                sentences_tokenized[-1].append(self.cls_token)

        sentences_ids = self.tokenizer.convert_tokens_to_ids(
            sentences_tokenized
        )
        # This stage needs truncation by MSL which is hardcoded in the
        # originl source code as 510. Most likely all the sequences
        # will be greater than MSL unless MSL is really large so need to
        # add checks for the same. Need to subtract 2 from MSL for
        # cls_id and sep_id
        acceptable_sequence_length = self.max_sequence_length - 2
        if len(sentences_ids) > acceptable_sequence_length:
            sentences_ids = sentences_ids[:acceptable_sequence_length]

        # Augment sequence of sentences with padding and [CLS] token in the beginning, and
        # [SEP] token at the end of the sequence.
        sentences_ids = [self.cls_id] + sentences_ids + [self.sep_id]
        return sentences_ids

    def _get_segment_ids(self, sentences_ids):
        sep_ids = [
            index
            for index, token_id in enumerate(sentences_ids)
            if token_id == self.sep_id
        ]

        # Obtain relative indices of segments to form interval segment embeddings.
        sep_ids.insert(0, -1)
        cur_sentence_lengths = [
            sep_ids[i] - sep_ids[i - 1] for i in range(1, len(sep_ids))
        ]
        # Segment embedding is 0 when index of the SEP token is even,
        # Otherwise embedding is 1.
        segment_ids = []
        for index, cur_sentence_length in enumerate(cur_sentence_lengths):
            segment_ids += cur_sentence_length * [index % 2]

        return segment_ids

    def _get_cls_ids(self, sentences_ids):
        cls_ids = [
            index
            for index, token_id in enumerate(sentences_ids)
            if token_id == self.cls_id
        ]
        return cls_ids

    def _get_labels(self, source, oracle_ids):
        labels = [0] * len(source)

        # oracle_ids specifies which sentences
        # will be present in the final summarization,
        # hence will have a label equal to 1.
        for index in oracle_ids:
            labels[index] = 1

        labels = [
            labels[index]
            for index, tokens in enumerate(source)
            if len(tokens) > self.min_tokens_per_sentence
        ][: self.max_sentences_per_sequence]
        return labels

    def process(self, source, target, oracle_ids):
        sentences = [" ".join(tokens) for tokens in source]

        sentences_ids = self._get_sentences_ids(sentences)
        labels = self._get_labels(source, oracle_ids)

        if len(sentences_ids) == 0 or len(labels) == 0:
            return None

        segment_ids = self._get_segment_ids(sentences_ids)
        cls_ids = self._get_cls_ids(sentences_ids)

        target_text = "<q>".join([" ".join(tokens) for tokens in target])
        source_text = "<q>".join(sentences)

        return (
            sentences_ids,
            labels,
            segment_ids,
            cls_ids,
            source_text,
            target_text,
        )


class RougeBasedLabelsFormatter:
    def __init__(self):
        """
        Based on the reference n-grams, `RougeBasedLabelsFormatter`
        selects sentences from the input with the highest rouge-score
        calculated between them and the reference. This is needed since we
        solve extractive summarization task, where target summarization is
        the subset of the input sentences in contrast to abstractive summarization,
        where summarized text is generated by the system without
        relying on the input text.
        """

    @staticmethod
    def _format_rouge_output(output):
        return re.sub(r"[^a-zA-Z0-9 ]", "", output)

    @staticmethod
    def _calculate_rouge(evaluated_ngrams, reference_ngrams):
        """
        ROUGE (Recall-Oriented Understudy for Gisting Evaluation).
        * the fraction of n-grams from abstracts included in the summarization.

        \begin{equation}
        ROUGE-n(s) = \frac{\sum_{r \in R}\sum_{w} [w \in s][w \in r]}{\sum_{r \in R} \sum_{w} [w \in r]}
        \end{equation}
        *   $r \in R$ -- set of reference n-grams, written by humans.
        *   $s$ -- evaluated n-grams, built by the system.
        *   higher the better -- for all metrics of ROUGE family.
        *   $n$ -- order of n-gram:
              * $n=1$ -- unigrams, $n=2$ -- bigrams, etc.
              * with increase of $n$, you achieve more accurate results.
              * with $n =$ len_of_abstract, we require full match of predicted
              text and the one written by humans.
        """
        reference_count = len(reference_ngrams)
        evaluated_count = len(evaluated_ngrams)

        overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
        overlapping_count = len(overlapping_ngrams)

        if evaluated_count == 0:
            precision = 0.0
        else:
            precision = overlapping_count / evaluated_count

        if reference_count == 0:
            recall = 0.0
        else:
            recall = overlapping_count / reference_count

        f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
        return f1_score

    def process(self, document_sentences, abstract_sentences, summary_size):
        sentences = [
            self._format_rouge_output(" ".join(sentence)).split()
            for sentence in document_sentences
        ]
        abstract = sum(abstract_sentences, [])
        abstract = self._format_rouge_output(" ".join(abstract)).split()

        evaluated_1grams = [set(ngrams(sentence, 1)) for sentence in sentences]
        reference_1grams = set(ngrams(abstract, 1))

        evaluated_2grams = [set(ngrams(sentence, 2)) for sentence in sentences]
        reference_2grams = set(ngrams(abstract, 2))

        max_rouge = 0.0
        selected_sentences = []
        selected_sentences_set = set()
        for _ in range(summary_size):
            cur_max_rouge = max_rouge
            cur_id = -1
            for sentence_index, sentence in enumerate(sentences):
                if sentence_index in selected_sentences_set:
                    continue

                summary = selected_sentences + [sentence_index]
                candidates_1 = [evaluated_1grams[index] for index in summary]
                candidates_1 = set.union(*map(set, candidates_1))

                candidates_2 = [evaluated_2grams[index] for index in summary]
                candidates_2 = set.union(*map(set, candidates_2))

                # The decision whether to output the sentence into target
                # summarization is taken based on the sum of rouge-1 (1-grams)
                # and rouge-2 (2-grams).
                rouge_1 = self._calculate_rouge(candidates_1, reference_1grams)
                rouge_2 = self._calculate_rouge(candidates_2, reference_2grams)

                rouge_score = rouge_1 + rouge_2

                if rouge_score > cur_max_rouge:
                    cur_max_rouge = rouge_score
                    cur_id = sentence_index

            if cur_id == -1:
                return selected_sentences

            selected_sentences.append(cur_id)
            selected_sentences_set.add(cur_id)
            max_rouge = cur_max_rouge

        return sorted(selected_sentences)


def check_output(input_path, output_path):
    input_files = os.listdir(input_path)
    output_files = os.listdir(output_path)

    if len(input_files) != len(output_files):
        raise Exception(
            f"The output directory {output_path} contains "
            f"{len(output_files)} files, but it should contain the same"
            f" number as {input_path} (which has {len(input_files)} files)."
            f" Was there an error during data creation?"
        )


def tokenize(params):
    """
    Split sentences and perform tokenization.
    Takes params.input_path, tokenize it
    and store it under params.output_path.
    """
    Tokenizer(params).process()


def convert_to_json_files(params):
    """
    Format input tokenized files into simpler json files.
    Takes params.input_path, convert it to json
    format and store it under params.output_path.
    """
    JsonConverter(params).process()


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        default="tokenize",
        type=str,
        choices=[
            "tokenize",
            "convert_to_json_files",
            "convert_to_bert_format_files",
        ],
        help="Supports three different modes: "
        "1) `tokenize`: split the sentences and runs tokenization; "
        "2) `convert_to_json_files`: format input files into simpler json files; "
        "3) `convert_to_bert_format_files`: format json input into bert format."
        "All modes should be run in sequential order (1->2->3).",
    )
    parser.add_argument(
        "--map_path",
        default="urls/",
        type=str,
        help="Path where urls of articles are stored."
        " which provide split into training, testing and validation.",
    )

    parser.add_argument(
        "--input_path", type=str, help="Path where to take input files."
    )
    parser.add_argument(
        "--output_path", type=str, help="Path where to store output files."
    )

    parser.add_argument(
        "--min_sentences_per_sequence",
        default=3,
        type=int,
        help="Minimum sentences per sequence allowed to consider"
        " a blob of text as an object",
    )

    parser.add_argument(
        "--max_sentences_per_sequence",
        default=100,
        type=int,
        help="Maximum sentences per sequence allowed."
        " Otherwise sentences in the object will be cut.",
    )

    parser.add_argument(
        "--min_tokens_per_sentence",
        default=5,
        type=int,
        help="Minimum number of tokens per sentence allowed."
        " to consider a sentence within a sequence.",
    )

    parser.add_argument(
        "--max_tokens_per_sentence",
        default=200,
        type=int,
        help="Maximum number of tokens per sentence allowed."
        " Otherwise tokens in the sentence will be cut.",
    )

    parser.add_argument(
        "--shard_size",
        default=2000,
        type=int,
        help="Maximum number of objects that each thread can process.",
    )

    parser.add_argument("--n_cpu", default=2, type=int)

    parser.add_argument(
        "--vocab_file",
        type=str,
    )
    parser.add_argument(
        "--lower_case",
        default=True,
        type=_str2bool,
        nargs="?",
        const=True,
        help="Specifies whether to convert to lower case for data.",
    )

    parser.add_argument(
        "--max_cls_tokens",
        default=50,
        type=int,
        help="Specifies the maximum number of cls tokens in one sequence.",
    )
    parser.add_argument(
        "--max_sequence_length",
        default=512,
        type=int,
        help="Specifies the maximum sequence length.",
    )

    return parser
