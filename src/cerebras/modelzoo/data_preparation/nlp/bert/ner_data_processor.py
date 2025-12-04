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
Common pre-processing functions taken from:
https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/run_ner.py 
with minor modifications 
"""

import argparse
import json
import os
import pickle

from cerebras.modelzoo.data_preparation.utils import convert_to_unicode


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class NERProcessor:
    def __init__(self, split) -> None:
        self.split = split
        self.file_name = ".".join([split, "tsv"])

    def get_split_examples(self, data_dir):
        input_file = os.path.join(data_dir, self.file_name)
        print(f"*** Processing {self.split} exmples: {input_file} ***")
        return self._create_example(self._read_data(input_file))

    def get_labels(self):
        # NOTE:[PAD] should always be first inorder to have an id=0
        return ["[PAD]", "B", "I", "O", "X", "[CLS]", "[SEP]"]

    def _create_example(self, lines):
        examples = []
        for i, (label, text) in enumerate(lines):
            guid = f"{self.split}-{i}"
            text = convert_to_unicode(text)
            label = convert_to_unicode(label)

            if "-DOCSTART-" in text:
                # JNLPBA dataset has some entries which have this demarcation
                # print("Text ignore:{}".format(text))
                continue

            examples.append(InputExample(guid, text, label))
        return examples

    def _read_data(self, input_file):
        """
        Read 'B', 'I', 'O' data.
        """
        if os.path.exists(input_file):
            with open(input_file, "r") as f:
                lines, words, labels = [], [], []
                for line in f:
                    contents = line.strip()
                    if len(contents) == 0:
                        assert len(words) == len(labels)
                        while len(words) > 30:
                            # split the sentence if it is longer than 30
                            tmplabel = labels[:30]
                            for _ in range(len(tmplabel)):
                                if tmplabel.pop() == 'O':
                                    break
                            l = " ".join(
                                [
                                    label
                                    for label in labels[: len(tmplabel) + 1]
                                    if len(label) > 0
                                ]
                            )
                            w = " ".join(
                                [
                                    word
                                    for word in words[: len(tmplabel) + 1]
                                    if len(word) > 0
                                ]
                            )
                            lines.append([l, w])
                            words = words[len(tmplabel) + 1 :]
                            labels = labels[len(tmplabel) + 1 :]
                        if len(words) == 0:
                            continue
                        l = " ".join(
                            [label for label in labels if len(label) > 0]
                        )
                        w = " ".join([word for word in words if len(word) > 0])
                        lines.append([l, w])
                        words = []
                        labels = []
                        continue
                    word = line.strip().split()[0]
                    label = line.strip().split()[-1]
                    words.append(word)
                    labels.append(label)
                return lines
        else:
            return []


class NERProcessor:
    def get_train_examples(self, data_dir, file_name="train.tsv"):
        print(
            f"**** Processing train examples: {os.path.join(data_dir, file_name)}"
        )
        return self._create_example(
            self._read_data(os.path.join(data_dir, file_name)), "train"
        )

    def get_dev_examples(self, data_dir, file_name="dev.tsv"):
        print(
            f"**** Processing dev examples: {os.path.join(data_dir, file_name)}"
        )
        return self._create_example(
            self._read_data(os.path.join(data_dir, file_name)), "dev"
        )

    def get_test_examples(self, data_dir, file_name="test.tsv"):
        print(
            f"**** Processing test examples: {os.path.join(data_dir, file_name)}"
        )
        return self._create_example(
            self._read_data(os.path.join(data_dir, file_name)), "test"
        )

    def get_labels(self, data_split_type=None):
        # NOTE:[PAD] should always be first inorder to have an id=0
        return ["[PAD]", "B", "I", "O", "X", "[CLS]", "[SEP]"]

    def _create_example(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = convert_to_unicode(line[1])
            label = convert_to_unicode(line[0])

            if "-DOCSTART-" in text:
                # JNLPBA dataset has some entries which have this demarcation
                # print("Text ignore:{}".format(text))
                continue

            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        if os.path.exists(input_file):
            with open(input_file, "r") as f:
                lines = []
                words = []
                labels = []
                for line in f:
                    contends = line.strip()
                    if len(contends) == 0:
                        assert len(words) == len(labels)
                        if len(words) > 30:
                            # split if the sentence is longer than 30
                            while len(words) > 30:
                                tmplabel = labels[:30]
                                for iidx in range(len(tmplabel)):
                                    if tmplabel.pop() == 'O':
                                        break
                                l = ' '.join(
                                    [
                                        label
                                        for label in labels[: len(tmplabel) + 1]
                                        if len(label) > 0
                                    ]
                                )
                                w = ' '.join(
                                    [
                                        word
                                        for word in words[: len(tmplabel) + 1]
                                        if len(word) > 0
                                    ]
                                )
                                lines.append([l, w])
                                words = words[len(tmplabel) + 1 :]
                                labels = labels[len(tmplabel) + 1 :]

                        if len(words) == 0:
                            continue
                        l = ' '.join(
                            [label for label in labels if len(label) > 0]
                        )
                        w = ' '.join([word for word in words if len(word) > 0])
                        lines.append([l, w])
                        words = []
                        labels = []
                        continue

                    word = line.strip().split()[0]
                    label = line.strip().split()[-1]
                    words.append(word)
                    labels.append(label)
                return lines
        else:
            return []


def write_label_map_files(label_list, out_dir):
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

    label2id_file = os.path.join(out_dir, 'label2id.pkl')
    if not os.path.exists(label2id_file):
        with open(label2id_file, 'wb') as w:
            pickle.dump(label_map, w)

    label2id_json_file = os.path.join(out_dir, 'label2id.json')
    if not os.path.exists(label2id_json_file):
        with open(label2id_json_file, 'w') as w:
            json.dump(label_map, w)

    return label_map


def get_tokens_and_labels(example, tokenizer, max_seq_length):
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')

    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            # If a word is split into sub-words during tokenization,
            # then the first sub-word gets the label of the word and
            # the remaining words are marked with labels "X"
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")

    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0 : (max_seq_length - 2)]
        labels = labels[0 : (max_seq_length - 2)]

    return tokens, labels


def create_parser():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_split_type",
        choices=["train", "dev", "test", "all"],
        default="all",
        help="Dataset split, choose from 'train', 'test', 'dev' or 'all'.",
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing train.tsv, test.tsv, dev.tsv",
    )
    parser.add_argument(
        "--vocab_file",
        required=True,
        help="The vocabulary file that the BERT Pretrained model was trained on.",
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
        default=128,
        help="The maximum total input sequence length after WordPiece tokenization.",
    )

    return parser
