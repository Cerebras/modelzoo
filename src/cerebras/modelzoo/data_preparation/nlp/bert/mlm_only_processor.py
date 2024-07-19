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
import random
from functools import reduce

import spacy
from tqdm import tqdm

from cerebras.modelzoo.data_preparation.nlp.tokenizers.Tokenization import (
    FullTokenizer,
)
from cerebras.modelzoo.data_preparation.utils import (
    convert_to_unicode,
    create_masked_lm_predictions,
    pad_instance_to_max_seq_length,
    text_to_tokenized_documents,
)


class MLMOnlyInstance:
    """
    A single training MLMOnly instance.
    :param list tokens: List of tokens for MLM example
    :param list masked_lm_positions: List of masked lm positions for sentence
    pair
    :param list masked_lm_labels: List of masked lm labels for example
    """

    def __init__(
        self,
        tokens,
        masked_lm_positions,
        masked_lm_labels,
    ):
        self.tokens = tokens
        self.masked_lm_labels = masked_lm_labels
        self.masked_lm_positions = masked_lm_positions

    def __str__(self):
        tokens = " ".join([convert_to_unicode(x) for x in self.tokens])
        mlm_positions = " ".join([str(x) for x in self.masked_lm_positions])
        mlm_labels = " ".join(
            [convert_to_unicode(x) for x in self.masked_lm_labels]
        )

        s = f"MLMOnlyInstance: \n"
        s += f"tokens: {tokens}\n"
        s += f"masked_lm_positions: {mlm_positions}\n"
        s += f"masked_lm_labels: {mlm_labels}\n"
        s += "\n"

        return s

    def __repr__(self):
        return self.__str__()


def data_generator(
    metadata_files,
    vocab_file,
    do_lower,
    disable_masking,
    mask_whole_word,
    max_seq_length,
    max_predictions_per_seq,
    masked_lm_prob,
    dupe_factor,
    output_type_shapes,
    multiple_docs_in_single_file=False,
    multiple_docs_separator="\n",
    single_sentence_per_line=False,
    buffer_size=1e6,
    min_short_seq_length=None,
    overlap_size=None,
    short_seq_prob=0,
    spacy_model="en_core_web_sm",
    inverted_mask=False,
    allow_cross_document_examples=True,
    document_separator_token="[SEP]",
    seed=None,
    input_files_prefix="",
):
    """
    Generator function used to create input dataset
    for MLM only dataset.

    1. Generate raw examples with tokens based on "overlap_size",
    "max_sequence_length", "allow_cross_document_examples"
    and "document_separator_token" and using a sliding window approach.
    The exact steps are detailed in "_create_examples_from_document" function
    2. Mask the raw examples based on "max_predictions_per_seq"
    3. Pad the masked example to "max_sequence_length" if less that msl

    :param str or list[str] metadata_files: A string or strings list each
        pointing to a metadata file. A metadata file contains file paths for
        flat text cleaned documents. It has one file path per line.
    :param str vocab_file: Vocabulary file, to build tokenization from
    :param bool do_lower: Boolean value indicating if words should be
        converted to lowercase or not
    :param bool disable_masking: whether masking should be disabled
    :param bool mask_whole_word: If True, all subtokens corresponding to a word
        will be masked.
    :param int max_seq_length: Maximum length of the sequence to generate
    :param int max_predictions_per_seq: Maximum number of Masked tokens
        in a sequence
    :param float masked_lm_prob: Proportion of tokens to be masked
    :param int dupe_factor: Number of times to duplicate the dataset
        with different static masks
    :param dict output_type_shapes: Dictionary indicating the shapes of
        different outputs
    :param bool multiple_docs_in_single_file: True, when a single text
        file contains multiple documents separated by <multiple_docs_separator>
    :param str multiple_docs_separator: String which separates multiple
        documents in a single text file.
    :param single_sentence_per_line: True,when the document is already split
        into sentences with one sentence in each line and there is no
        requirement for further sentence segmentation of a document
    :param int buffer_size: Number of tokens to be processed at a time
    :param int min_short_seq_length: When short_seq_prob > 0, this number
        indicates the least number of tokens that each example should have i.e
        the num_tokens (excluding pad) would be in the range
        [min_short_seq_length, MSL]
    :param int overlap_size: Number of tokens that overlap with previous example
        when processing buffer with a sliding window approach.
        If None, defaults to overlap to max_seq_len/4.
    :param int short_seq_prob: Probability of a short sequence. Defaults to 0.
        Sometimes we want to use shorter sequences to minimize the mismatch
        between pre-training and fine-tuning.
    :param spacy_model: spaCy model to load, i.e. shortcut
        link, package name or path. Used to segment text into sentences.
    :param bool inverted_mask: If set to False, has 0's on padded positions and
        1's elsewhere. Otherwise, "inverts" the mask, so that 1's are on padded
        positions and 0's elsewhere.
    :param bool allow_cross_document_examples: If True, the sequences can
        contain tokens from the next document.
    :param str document_separator_token: String to separate tokens from
        one document and the next when sequences span documents
    :param int seed: Random seed.
    :param str input_file_prefix: Prefix to be added to paths of the input files.

    :returns: yields training examples (feature, [])
    """
    ## Set defaults if values passed are None
    if overlap_size is None:
        overlap_size = int(max_seq_length / 4)
        print(
            f"--- Setting overlap_size to {overlap_size} since None value passed"
        )

    if not allow_cross_document_examples and document_separator_token:
        print(
            f"--- Since example cannot span documents "
            f"(allow_cross_document_examples: {allow_cross_document_examples}),"
            f" document_separator_token: {document_separator_token} will be ignored"
        )

    if min_short_seq_length is None:
        min_short_seq_length = 2 + overlap_size

    elif (min_short_seq_length < (2 + overlap_size)) or (
        min_short_seq_length > max_seq_length - 2
    ):

        raise ValueError(
            f"The min_short_seq_len param {min_short_seq_length} is invalid. \n"
            f"Allowed values are [{2 + overlap_size}, {max_seq_length - 2})"
        )

    # define tokenizer
    tokenizer = FullTokenizer(vocab_file, do_lower)
    vocab_words = tokenizer.get_vocab_words()

    if do_lower:
        document_separator_token = document_separator_token.lower()

    assert (
        document_separator_token in vocab_words
    ), f" document_separator_token: {document_separator_token} not present in vocab file"

    rng = random.Random(seed)

    # get all text files by reading metadata files
    if isinstance(metadata_files, str):
        metadata_files = [metadata_files]

    input_files = []
    for _file in metadata_files:
        with open(_file, "r") as _fin:
            input_files.extend(_fin.readlines())
    input_files = [x.strip() for x in input_files if x]
    num_input_files = len(input_files)
    rng.shuffle(input_files)

    def _generate_train_feature(example):
        if disable_masking:
            return example
        else:
            return create_masked_lm_features(
                example,
                vocab_words,
                max_seq_length,
                mask_whole_word,
                max_predictions_per_seq,
                masked_lm_prob,
                document_separator_token,
                rng,
                tokenizer,
                output_type_shapes,
                inverted_mask,
            )

    current_buffer_length = 0
    buffer_documents = []
    # to speed up processing load spacy module once here
    # disable the ununsed pipeline stages to speed up processing
    nlp = spacy.load(spacy_model, disable=['tagger', 'ner'])
    for _ in tqdm(range(dupe_factor)):
        # Reset buffers
        prev_tokens = []

        for _file_num, _file in enumerate(input_files):
            _fin_path = os.path.abspath(os.path.join(input_files_prefix, _file))
            with open(_fin_path, "r") as _fin:
                _fin_data = _fin.read()
            processed_doc, num_tokens = text_to_tokenized_documents(
                _fin_data,
                tokenizer,
                multiple_docs_in_single_file,
                multiple_docs_separator,
                single_sentence_per_line,
                nlp,
            )

            # Flatten one level
            buffer_documents.extend(
                [
                    reduce(lambda x, y: x + y, doc_list)
                    for doc_list in processed_doc
                ]
            )
            current_buffer_length += num_tokens

            # Continue if we don't have enough tokens
            if (
                current_buffer_length < buffer_size
                and _file_num < num_input_files - 1
            ):
                continue

            rng.shuffle(buffer_documents)

            # When enough tokens available, yield examples
            for document_index, document in enumerate(buffer_documents):
                _example_generator = _create_examples_from_document(
                    document,
                    allow_cross_document_examples,
                    document_separator_token,
                    overlap_size,
                    prev_tokens,
                    max_seq_length,
                    short_seq_prob,
                    min_short_seq_length,
                    rng,
                )
                for example, prev_tokens in _example_generator:
                    if example:
                        yield _generate_train_feature(example)

            # Fix buffer lengths, buffer etc
            buffer_documents = []
            current_buffer_length = 0

        # Last few tokens remaining after processing all input_files
        if prev_tokens:
            yield _generate_train_feature(["[CLS]"] + prev_tokens + ["[SEP]"])


def create_masked_lm_features(
    example,
    vocab_words,
    max_seq_length,
    mask_whole_word,
    max_predictions_per_seq,
    masked_lm_prob,
    document_separator_token,
    rng,
    tokenizer,
    output_type_shapes,
    inverted_mask,
):

    exclude_from_masking = list(
        set(["[CLS]", "[SEP]", document_separator_token])
    )
    (
        masked_example_tokens,
        masked_lm_positions,
        masked_lm_labels,
    ) = create_masked_lm_predictions(
        example,
        vocab_words,
        mask_whole_word,
        max_predictions_per_seq,
        masked_lm_prob,
        rng,
        exclude_from_masking,
    )

    masked_lm_instance = MLMOnlyInstance(
        tokens=masked_example_tokens,
        masked_lm_positions=masked_lm_positions,
        masked_lm_labels=masked_lm_labels,
    )

    # pad to MSL
    feature, label = pad_instance_to_max_seq_length(
        instance=masked_lm_instance,
        mlm_only=True,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        max_predictions_per_seq=max_predictions_per_seq,
        output_type_shapes=output_type_shapes,
        inverted_mask=inverted_mask,
    )

    return feature, label


def _create_examples_from_document(
    document,
    allow_cross_document_examples,
    document_separator_token,
    overlap_size,
    prev_tokens,
    max_seq_length,
    short_seq_prob,
    min_short_seq_length,
    rng,
):
    # Process for generating an example
    # 1. The text from metadata files is read and accumulated in a buffer
    # until the buffer_size limit is hit. Note that documents are always
    # read entirely before the buffer_size limit is checked
    # 2. Next, reading one document at a time from "buffer",
    # we slide a window of size "max_sequence_length" and construct an example.
    # 3. If "overlap_size" is set, then when generating the next example,
    # the window is slided back by "overlap_size" and the next example is constructed
    # 4. If an example can span multiple documents,
    # i.e "allow_cross_document_examples" is set to True,
    # then we use "document_separator_token" to separate tokens
    # from the two documents
    # i.e [CLS] <tokens-doc1> <document_separator_token><tokens-doc2>[SEP]
    # 5. The last remaining tokens are used to contruct the final example
    # and this example most of the times will have tokens less than "max_sequence_length"
    # and would be padded to "max_sequence_length"

    max_num_tokens = max_seq_length - 2
    start_idx = 0
    # We usually want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally not
    # needed. However, we sometimes (i.e., short_seq_prob = 0.1 == 10% of
    # the time) want to use shorter sequences to minimize the mismatch
    # between pre-training and fine-tuning. The `target_seq_len` is just a
    # rough target however, whereas `max_seq_length` is a hard limit
    target_seq_len = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_len = rng.randint(min_short_seq_length, max_num_tokens)

    assert (
        len(prev_tokens) <= max_seq_length
    ), "Number of leftover tokens i.e len(prev_tokens) > max_seq_length"

    # NOTE: prev_tokens cannot be more than MSL.
    # Basically, we do a windowing to construct examples and "overlap_size"
    # refers to the amount we push the window back.
    # "buffer_size", on the other hand enables us to process more than one
    # document at once if the document contains fewer tokens.
    # So, at a single time we can process "buffer_size" number of tokens

    if prev_tokens:
        document = prev_tokens + [document_separator_token] + document
        prev_tokens = []

    start_idx = 0  # inclusive of this element
    end_idx = start_idx + target_seq_len - 1  # inclusive of this element

    while end_idx < len(document):
        example = document[
            start_idx : end_idx + 1
        ]  # All elements from start_idx to end_idx (inclusive)

        # add special token for input start and end
        assert (
            len(example) > overlap_size
        ), f"Length of example {len(example)} less than overlap_size {overlap_size}"
        assert (
            len(example) <= max_num_tokens
        ), f"Length of example greater than max_num_tokens {max_num_tokens}"

        example.insert(0, "[CLS]")
        example.append("[SEP]")
        yield example, prev_tokens

        start_idx = end_idx - overlap_size + 1

        # Recalculate target_seq_len,
        if rng.random() < short_seq_prob:
            target_seq_len = rng.randint(min_short_seq_length, max_num_tokens)

        end_idx = start_idx + target_seq_len - 1

        assert (
            end_idx > 0
        ), f" When generating example, end_idx {end_idx} is less than zero."
        assert (
            start_idx >= 0
        ), f" When generating example, start_idx {start_idx} is less than zero."

    if allow_cross_document_examples:
        example = []
        prev_tokens = document[start_idx:]
    else:
        example = ["[CLS]"] + document[start_idx:] + ["[SEP]"]
        prev_tokens = []

    yield example, prev_tokens
