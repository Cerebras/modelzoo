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

import collections
import json

import numpy as np
import six
from keras_preprocessing.text import text_to_word_sequence

from cerebras.modelzoo.common.model_utils.count_lines import count_lines


def convert_to_unicode(text):
    """
    Converts `text` to unicode, assuming utf-8 input
    Returns text encoded in a way suitable for print or `tf.compat.v1.logging`
    """

    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError(f"Unsupported string type: {type(text)}")
    else:
        raise ValueError(f"Not running Python3")


def count_total_documents(metadata_files):
    """
    Counts total number of documents
    in metadata_files.
    :param str or list[str] metadata_files: Path or list of paths
        to metadata files.
    :returns: Number of documents whose paths are contained
        in the metadata files.
    """
    total_documents = 0
    if isinstance(metadata_files, str):
        metadata_files = [metadata_files]
    for _file in metadata_files:
        total_documents += count_lines(_file)
    return total_documents


def whitespace_tokenize(text, lower=False):
    """
    Splits a piece of text based on whitespace characters \t\r\n
    """
    return text_to_word_sequence(text, filters='\t\n\r', lower=lower)


def get_output_type_shapes(
    max_seq_length, max_predictions_per_seq, mlm_only=False
):
    # process for output shapes and types
    output = {
        "input_ids": {
            "output_type": "int32",
            "shape": [max_seq_length],
        },
        "input_mask": {
            "output_type": "int32",
            "shape": [max_seq_length],
        },
        "masked_lm_positions": {
            "output_type": "int32",
            "shape": [max_predictions_per_seq],
        },
        "masked_lm_ids": {
            "output_type": "int32",
            "shape": [max_predictions_per_seq],
        },
        "masked_lm_weights": {
            "output_type": "float32",
            "shape": [max_predictions_per_seq],
        },
    }

    if not mlm_only:
        output["segment_ids"] = {
            "output_type": "int32",
            "shape": [max_seq_length],
        }

    return output


def pad_instance_to_max_seq_length(
    instance,
    mlm_only,
    tokenizer,
    max_seq_length,
    max_predictions_per_seq,
    output_type_shapes,
    inverted_mask,
):

    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)

    # initial assert to ensure wrong instances are not being
    # generated from the function call
    assert len(input_ids) <= max_seq_length

    # extend above lists with length difference
    length_diff = max_seq_length - len(input_ids)
    extended_list = [0] * length_diff
    input_ids.extend(extended_list)
    input_mask.extend(extended_list)

    # assertions to ensure correct output shapes
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length

    if not mlm_only:
        segment_ids = list(instance.segment_ids)
        segment_ids.extend(extended_list)
        assert len(segment_ids) == max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    # initial assert to ensure wrong instances are not being
    # generated from the function call
    assert len(masked_lm_positions) <= max_predictions_per_seq

    # extend above lists with length difference
    length_diff = max_predictions_per_seq - len(masked_lm_positions)
    extended_list = [0] * length_diff
    masked_lm_positions.extend(extended_list)
    masked_lm_ids.extend(extended_list)
    masked_lm_weights.extend(extended_list)

    # assertions to ensure correct output shapes
    assert len(masked_lm_positions) == max_predictions_per_seq
    assert len(masked_lm_ids) == max_predictions_per_seq
    assert len(masked_lm_weights) == max_predictions_per_seq

    # create feature dict
    features = dict()
    features["input_ids"] = input_ids
    features["input_mask"] = input_mask
    features["masked_lm_positions"] = masked_lm_positions
    features["masked_lm_ids"] = masked_lm_ids
    features["masked_lm_weights"] = masked_lm_weights

    if not mlm_only:
        features["segment_ids"] = segment_ids

    # get associated numpy types and convert to
    # np.dtype using output_type_shapes
    feature = {
        k: getattr(np, output_type_shapes[k]["output_type"])(v)
        for k, v in features.items()
    }
    # handling input mask switch
    if inverted_mask:
        feature["input_mask"] = np.equal(feature["input_mask"], 0).astype(
            feature["input_mask"].dtype
        )

    if not mlm_only:
        # get label for function
        next_sentence_label = 1 if instance.is_random_next else 0
        # int32 label always
        label = np.int32(next_sentence_label)
    else:
        # Currently labels=None is not supported.
        label = np.int32(np.empty(1)[0])

    return feature, label


def text_to_tokenized_documents(
    data,
    tokenizer,
    multiple_docs_in_single_file,
    multiple_docs_separator,
    single_sentence_per_line,
    spacy_nlp,
):
    """
    Convert the input data into tokens
    :param str data: Contains data read from a text file
    :param tokenizer: Tokenizer object which contains functions to
        convert words to tokens
    :param bool multiple_docs_in_single_file: Indicates whether there
        are multiple documents in the given data string
    :param str multiple_docs_separator: String used to separate documents
        if there are multiple documents in data.
        Separator can be anything. It can be a new blank line or
        some special string like "-----" etc.
        There can only be one separator string for all the documents.
    :param bool single_sentence_per_line: Indicates whether the data contains
         one sentence in each line
    :param spacy_nlp: spaCy nlp module loaded with spacy.load()
        Used in segmenting a string into sentences
    :return List[List[List]] documents: Contains the tokens corresponding to
         sentences in documents.
         List of List of Lists [[[],[]], [[],[],[]]]
         documents[i][j] -> List of tokens in document i and sentence j
    """

    if "\\n" in multiple_docs_separator:
        multiple_docs_separator = multiple_docs_separator.replace("\\n", "\n")

    get_length = lambda input: sum([len(x) for x in input])

    documents = []
    num_tokens = 0
    if multiple_docs_in_single_file:
        # "\n" is added since seperator is always in newline
        # <doc1>
        # multiple_docs_separator
        # <doc2>
        data = data.split("\n" + multiple_docs_separator)
        data = [x for x in data if x]  # data[i] -> document i
    else:
        data = [data]

    if single_sentence_per_line:
        # The document has already been into sentences and each sentence is in a newline
        for doc in data:
            documents.append([])
            # Get sentences by splitting on newline, since each new sentence is in a newline
            lines = doc.split("\n")
            for line in lines:
                if line:
                    tokens = tokenizer.tokenize(
                        line.strip()
                    )  # tokens : list of tokens
                    if tokens:
                        documents[-1].append(tokens)
                        num_tokens += len(tokens)
    else:
        # The document should be segmented into sentences with a spacy_model
        for doc in data:
            processed_doc = spacy_nlp(convert_to_unicode(doc.replace('\n', '')))
            sentences = [
                tokenizer.tokenize(s.text) for s in list(processed_doc.sents)
            ]
            sentences = [
                s for s in sentences if s
            ]  # sentences[i][j] -> token j of sentence i
            documents.append(sentences)
            num_tokens += get_length(sentences)

    # documents[i][j] -> list of tokens of sentence j in  document i
    # Remove empty documents if any
    documents = [x for x in documents if x]
    return documents, num_tokens


maskedLmInstance = collections.namedtuple(
    "maskedLmInstance", ["index", "label"]
)


def create_masked_lm_predictions(
    tokens,
    vocab_words,
    mask_whole_word,
    max_predictions_per_seq,
    masked_lm_prob,
    rng,
    exclude_from_masking=None,
):
    """
    Creates the predictions for the masked LM objective
    :param list tokens: List of tokens to process
    :param list vocab_words: List of all words present in the vocabulary
    :param bool mask_whole_word: If true, mask all the subtokens of a word
    :param int max_predictions_per_seq: Maximum number of masked LM predictions per sequence
    :param float masked_lm_prob: Masked LM probability
    :param rng: random.Random object with shuffle function
    :param Optional[list] exclude_from_masking: List of tokens to exclude from masking. Defaults to ["[CLS]", "[SEP]"]
    :returns: tuple of tokens which include masked tokens,
    the corresponding positions for the masked tokens
    and also the corresponding labels for training
    """

    if exclude_from_masking is not None:
        if not isinstance(exclude_from_masking, list):
            exclude_from_masking = list(exclude_from_masking)
    else:
        exclude_from_masking = ["[CLS]", "[SEP]"]

    cand_indexes = []
    for i, token in enumerate(tokens):
        if token in exclude_from_masking:
            continue

        # Whole word masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split
        # into WordPieces, the first token does not have any marker and
        # any subsequences tokens are prefixed with ##. So whenever we see
        # the ## token, we append it to the previous set of word indexes.

        # Note that whole word masking does not change the training code
        # at all -- we still predict each WordPiece independently,
        # softmaxed over the entire vocabulary
        if (
            mask_whole_word
            and len(cand_indexes) >= 1
            and token.startswith("##")
        ):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    rng.shuffle(cand_indexes)
    output_tokens = list(tokens)

    # get number of tokens to mask and predict
    num_to_predict = min(
        max_predictions_per_seq,
        max(1, int(round(len(tokens) * masked_lm_prob))),
    )

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break

        # if adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue

        # Check if any index is covered already.
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue

        for index in index_set:
            covered_indexes.add(index)
            # splits comes from
            # google-research/bert/create_pretraining_data.py
            masked_token = None
            random_value = rng.random()
            if random_value < 0.8:
                # 80% of times, replace with [MASK]
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep the original token
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10 % of times, replace with random word
                else:
                    masked_token = vocab_words[
                        rng.randint(0, len(vocab_words) - 1)
                    ]

            output_tokens[index] = masked_token
            masked_lms.append(
                maskedLmInstance(index=index, label=tokens[index])
            )

    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []

    # create final masked_lm_positions, masked_lm_labels
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def get_label_id_map(label_vocab_file):
    """
    Load the label-id mapping: Mapping between output labels and id
    :param str label_vocab_file: Path to the label vocab file
    """
    label_map = None
    if label_vocab_file is not None:
        with open(label_vocab_file, 'r') as fh:
            label_map = json.load(fh)

    return label_map


def convert_str_to_int_list(s):
    """
    Converts a string (e.g. from parsing CSV) of the form
        "[1, 5, 7, 2]"
    to a list of integers.
    """
    assert s.startswith("[")
    assert s.endswith("]")
    x = s.strip("[]")
    x = x.split(",")
    return [int(y.strip()) for y in x]


def pad_input_sequence(input_sequence, padding=0, max_sequence_length=512):
    input_sequence_array = padding * np.ones(
        max_sequence_length, dtype=np.int32
    )
    end_idx = min(max_sequence_length, len(input_sequence))
    input_sequence_array[:end_idx] = list(input_sequence[:end_idx])
    return input_sequence_array


def get_files_in_metadata(metadata_filepaths):
    """
    Function to read the files in metadata file
    provided as input to data generation scripts.

    :param metadata_filepaths: path/s to metadata files
    :returns List input_files: Contents of
        metadata files.
    """

    if isinstance(metadata_filepaths, str):
        metadata_filepaths = [metadata_filepaths]

    input_files = []
    for _file in metadata_filepaths:
        with open(_file, "r") as _fin:
            input_files.extend(_fin.readlines())
    input_files = [x.strip() for x in input_files if x]
    return input_files


def split_list(l, n):
    """
    Splits list/string into n sized chunks.

    :param List[str] l: List or string to split.
    :param int n: Number of chunks to split to.
    :returns List[List]: List of lists
        containing split list/string.
    """
    return [l[i : i + n] for i in range(0, len(l), n)]


def get_vocab(vocab_file_path, do_lower):
    """
    Function to generate vocab from provided
    vocab_file_path.

    :param str vocab_file_path: Path to vocab file
    :param bool do_lower: If True, convert vocab words to
        lower case.
    :returns List[str]: list containing vocab words.
    """
    vocab = []
    with open(vocab_file_path, 'r') as reader:
        for line in reader:
            token = convert_to_unicode(line)
            if not token:
                break
            token = token.strip()
            vocab.append(token)
    vocab = list(map(lambda token: token.lower(), vocab)) if do_lower else vocab
    return vocab
