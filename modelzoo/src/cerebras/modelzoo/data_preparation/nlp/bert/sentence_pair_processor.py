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


class SentencePairInstance:
    """
    A single training (sentence-pair) instance.
    :param list tokens: List of tokens for sentence pair
    :param list segment_ids: List of segment ids for sentence pair
    :param list masked_lm_positions: List of masked lm positions for sentence
    pair
    :param list masked_lm_labels: List of masked lm labels for sentence pair
    :param bool is_random_next: Specifies whether the second element in the
    pair is random
    """

    def __init__(
        self,
        tokens,
        segment_ids,
        masked_lm_positions,
        masked_lm_labels,
        is_random_next,
    ):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.masked_lm_labels = masked_lm_labels
        self.masked_lm_positions = masked_lm_positions
        self.is_random_next = is_random_next

    def __str__(self):
        tokens = " ".join([convert_to_unicode(x) for x in self.tokens])
        segment_ids = " ".join([str(x) for x in self.segment_ids])
        mlm_positions = " ".join([str(x) for x in self.masked_lm_positions])
        mlm_labels = " ".join(
            [convert_to_unicode(x) for x in self.masked_lm_labels]
        )

        s = ""
        s += f"tokens: {tokens}\n"
        s += f"segment_ids: {segment_ids}\n"
        s += f"is_random_next: {self.is_random_next}\n"
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
    split_num,
    max_seq_length,
    short_seq_prob,
    mask_whole_word,
    max_predictions_per_seq,
    masked_lm_prob,
    dupe_factor,
    output_type_shapes,
    min_short_seq_length=None,
    multiple_docs_in_single_file=False,
    multiple_docs_separator="\n",
    single_sentence_per_line=False,
    inverted_mask=False,
    seed=None,
    spacy_model="en_core_web_sm",
    input_files_prefix="",
    sop_labels=False,
):
    """
    Generator function used to create input dataset
    for MLM + NSP dataset.

    1. Generate raw examples by concatenating two parts
    'tokens-a' and 'tokens-b' as follows:
    [CLS] <tokens-a> [SEP] <tokens-b> [SEP]
    where :
        tokens-a: list of tokens taken from the
        current document and of random length (less than msl).

        tokens-b: list of tokens chosen based on the
        randomly set "next_sentence_labels" and of
        length msl-len(<tokens-a>)- 3 (to account for [CLS] and [SEP] tokens)

    If "next_sentence_labels" is 1, (set to 1 with 0.5 probability),
        tokens-b are list of tokens from sentences chosen randomly
        from different document
    else,
        tokens-b are list of tokens taken from the same document
        and is a continuation of tokens-a in the document
    The number of raw tokens depends on "short_sequence_prob" as well
    2. Mask the raw examples based on "max_predictions_per_seq"
    3. Pad the masked example to "max_sequence_length" if less that msl

    :param str or list[str] metadata_files: A string or strings list each
        pointing to a metadata file. A metadata file contains file paths for
        flat text cleaned documents. It has one file path per line.
    :param str vocab_file: Vocabulary file, to build tokenization from
    :param bool do_lower: Boolean value indicating if words should be
        converted to lowercase or not
    :param int split_num: Number of input files to read at a given
        time for processing.
    :param int max_seq_length: Maximum length of the sequence to generate
    :param int short_seq_prob: Probability of a short sequence. Defaults to 0.
        Sometimes we want to use shorter sequences to minimize the mismatch
        between pre-training and fine-tuning.
    :param bool mask_whole_word: If True, all subtokens corresponding to a word
        will be masked.
    :param int max_predictions_per_seq: Maximum number of Masked tokens
        in a sequence
    :param float masked_lm_prob: Proportion of tokens to be masked
    :param int dupe_factor: Number of times to duplicate the dataset
        with different static masks
    :param int min_short_seq_length: When short_seq_prob > 0, this number
        indicates the least number of tokens that each example should have i.e
        the num_tokens (excluding pad) would be in the range
        [min_short_seq_length, MSL]
    :param dict output_type_shapes: Dictionary indicating the shapes of
        different outputs
    :param bool multiple_docs_in_single_file: True, when a single text file
        contains multiple documents separated by <multiple_docs_separator>
    :param str multiple_docs_separator: String which separates
    multiple documents in a single text file.
    :param single_sentence_per_line: True,when the document is already
        split into sentences with one sentence in each line and there is
        no requirement for further sentence segmentation of a document
    :param bool inverted_mask: If set to False, has 0's on padded positions and
        1's elsewhere. Otherwise, "inverts" the mask, so that 1's are on padded
        positions and 0's elsewhere.
    :param int seed: Random seed.
    :param spacy_model: spaCy model to load, i.e. shortcut
        link, package name or path. Used to segment text into sentences.
    :param str input_file_prefix: Prefix to be added to paths of the input files.
    :param bool sop_labels: If true, negative examples of the dataset will be two
        consecutive sentences in reversed order. Otherwise, uses regular (NSP)
        labels (where negative examples are from different documents).

    :returns: yields training examples (feature, label)
    where label refers to the next_sentence_prediction label

    """
    if min_short_seq_length is None:
        min_short_seq_length = 2

    elif (min_short_seq_length < 2) or (
        min_short_seq_length > max_seq_length - 3
    ):
        raise ValueError(
            f"The min_short_seq_len param {min_short_seq_length} is invalid.\n"
            f"Allowed values are [2, {max_seq_length - 3})"
        )

    # define tokenizer
    vocab_file = os.path.abspath(vocab_file)
    tokenizer = FullTokenizer(vocab_file, do_lower)
    vocab_words = tokenizer.get_vocab_words()

    rng = random.Random(seed)

    # get all text files by reading metadata files
    if isinstance(metadata_files, str):
        metadata_files = [metadata_files]

    input_files = []
    for _file in metadata_files:
        with open(_file, "r") as _fin:
            input_files.extend(_fin.readlines())
    input_files = [x.strip() for x in input_files if x]
    rng.shuffle(input_files)

    split_num = len(input_files) if split_num <= 0 else split_num

    # for better performance load spacy model once here
    nlp = spacy.load(spacy_model)
    for i in range(0, len(input_files), split_num):
        current_input_files = input_files[i : i + split_num]

        all_documents = []
        for _file in tqdm(current_input_files):
            _fin_path = os.path.abspath(os.path.join(input_files_prefix, _file))
            with open(_fin_path, "r") as _fin:
                _fin_data = _fin.read()
            processed_doc, _ = text_to_tokenized_documents(
                _fin_data,
                tokenizer,
                multiple_docs_in_single_file,
                multiple_docs_separator,
                single_sentence_per_line,
                nlp,
            )
            all_documents.extend(processed_doc)

        rng.shuffle(all_documents)

        # create a set of instance to process further
        # repeat this process `dupe_factor` times
        # get a list of SentencePairInstances
        instances = []
        for _ in range(dupe_factor):
            for document_index in range(len(all_documents)):
                instances.extend(
                    _create_sentence_instances_from_document(
                        all_documents,
                        document_index,
                        vocab_words,
                        max_seq_length,
                        short_seq_prob,
                        min_short_seq_length,
                        mask_whole_word,
                        max_predictions_per_seq,
                        masked_lm_prob,
                        rng,
                        sop_labels,
                    )
                )
        rng.shuffle(instances)

        for instance in instances:
            feature, label = pad_instance_to_max_seq_length(
                instance=instance,
                mlm_only=False,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                max_predictions_per_seq=max_predictions_per_seq,
                output_type_shapes=output_type_shapes,
                inverted_mask=inverted_mask,
            )

            yield (feature, label)


def _create_sentence_instances_from_document(
    all_documents,
    document_index,
    vocab_words,
    max_seq_length,
    short_seq_prob,
    min_short_seq_length,
    mask_whole_word,
    max_predictions_per_seq,
    masked_lm_prob,
    rng,
    sop_labels=False,
):
    """
    Create instances from documents.
    :param list all_documents: List of lists which contains tokenized
    senteneces from each document
    :param int document_index: Index of document to process currently
    :param list vocab_words: List of all words present in the vocabulary
    :param bool sop_labels: If true, negative examples of the dataset will be two
        consecutive sentences in reversed order. Otherwise, uses regular (NSP)
        labels (where negative examples are from different documents).
    :returns: List of SentencePairInstance objects
    """

    # get document with document_index
    # Example:
    # [
    #   [line1], [line2], [line3]
    # ]
    # where each line = [tokens]
    document = all_documents[document_index]

    # account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We usually want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally not
    # needed. However, we sometimes (i.e., short_seq_prob = 0.1 == 10% of
    # the time) want to use shorter sequences to minimize the mismatch
    # between pre-training and fine-tuning. The `target_seq_len` is just a
    # rough target however, whereas `max_seq_length` is a hard limit
    target_seq_len = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_len = rng.randint(min_short_seq_length, max_num_tokens)

    # We don't just concatenate all of the tokens from a line into a long
    # sequence and choose an arbitrary split point because this would make
    # the NSP task too easy. Instead, we split the input into segments
    # `A` and `B` based on the actual "sentences" provided by the user
    # input
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    # lambda function for fast internal calls. called multiple times
    flatten = lambda l: [item for sublist in l for item in sublist]

    while i < len(document):
        # a line is a list of tokens [includes words / punctuations /
        # special characters / wordpieces]
        # we initially read an entire line - but also ensure that if we
        # meet the target seq_len with the current line - we cut it off
        # remove the unused `segments` and put them back in circulation for
        # input creation
        line = document[i]
        current_chunk.append(line)
        current_length += len(line)

        if i == len(document) - 1 or current_length >= target_seq_len:
            if current_chunk:
                # generate a sentence pair instance for NSP loss

                # `a_end` is how many segments from `current_chunk` go into
                # `A` (first sentence)
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                tokens_a.extend(flatten(current_chunk[0:a_end]))

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or (
                    not sop_labels and rng.random() < 0.5
                ):
                    is_random_next = True
                    target_b_length = target_seq_len - len(tokens_a)

                    # this should rarely go for more than one iteration
                    # for large corpora. However, just to be careful, we
                    # try to make sure that the random document is
                    # not the same as the document we are processing
                    for _ in range(10):
                        random_document_index = rng.randint(
                            0, len(all_documents) - 1
                        )
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)

                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break

                    # We don't actually use these segments [peices of line]
                    # so we "put them back" so they do not to waste for
                    # later computations
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                elif sop_labels and rng.random() < 0.5:
                    is_random_next = True
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                    tokens_a, tokens_b = tokens_b, tokens_a
                else:
                    # Actual next
                    is_random_next = False
                    tokens_b.extend(
                        flatten(current_chunk[a_end : len(current_chunk)])
                    )

                # When using SOP, with prob 0.5, the sentence ordering should be
                # swapped forming the negative samples.
                if sop_labels and (
                    len(current_chunk) == 1 or rng.random() < 0.5
                ):
                    tokens_a, tokens_b = tokens_b, tokens_a
                    is_random_next = True

                # truncate seq pair tokens to max_num_tokens
                _truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                # create actual input instance
                tokens = []
                segment_ids = []

                # add special token for input start
                tokens.append("[CLS]")
                segment_ids.append(0)

                # append input `A`
                extend_list = [0] * len(tokens_a)
                segment_ids.extend(extend_list)
                tokens.extend(tokens_a)

                # add special token for input separation
                tokens.append("[SEP]")
                segment_ids.append(0)

                # append input `B`
                extend_list = [1] * len(tokens_b)
                segment_ids.extend(extend_list)
                tokens.extend(tokens_b)

                # add special token for input separation
                tokens.append("[SEP]")
                segment_ids.append(1)

                (
                    tokens,
                    masked_lm_positions,
                    masked_lm_labels,
                ) = create_masked_lm_predictions(
                    tokens,
                    vocab_words,
                    mask_whole_word,
                    max_predictions_per_seq,
                    masked_lm_prob,
                    rng,
                )

                instance = SentencePairInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels,
                )
                instances.append(instance)
            # reset buffers
            current_chunk = []
            current_length = 0
        # move on to next segment
        i += 1

    return instances


def _truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """
    Truncate a pair of tokens so that their total length is lesser than
    defined maximum number of tokens
    :param list tokens_a: First list of tokens in sequence pair
    :param list tokens_b: Second list of tokens in sequence pair
    :param int max_num_tokens: Maximum number of tokens for the length of
    sequence pair tokens
    """

    total_length = len(tokens_a) + len(tokens_b)
    while total_length > max_num_tokens:
        # find the correct list to truncate this iteration of the loop
        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert (len(trunc_tokens)) >= 1

        # check whether to remove from front or rear
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

        # recompute lengths again after deletion of token
        total_length = len(tokens_a) + len(tokens_b)
