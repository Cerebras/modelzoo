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

import random

import numpy as np
import os

from Bio import SeqIO
from modelzoo.transformers.data_processing.Tokenization import BaseTokenizer


def training_data_generator(
    data_dir,
    vocab_file,
    max_sequence_length,
    n_gram,
    skip_large_MSL=False,
    inverted_mask=False,
    add_special_tokens=True,
    eos_token="<|endoftext|>",
    pad_token="<|endoftext|>",
    input_ids_dtype="int32",
    input_mask_dtype="int32",
    labels_dtype="int32",
):
    """
    Generator function used to create input genomic dataset
    for GPT2Model.

    :param str data_dir: Directory path that contains individual fasta files,
        each with one or multiple genomic sequences (samples).
    :param str vocab_file: Vocabulary file, to build tokenization from.
    :param int max_sequence_length: Maximum length of the sequence to generate.
    :param bool inverted_mask: If set to False, has 0's on padded positions and
        1's elsewhere. Otherwise, "inverts" the mask, so that 1's are on padded
        positions and 0's elsewhere.
    :param str eos_token: End of sequence token. Defaults to "<|endoftext|>".
    :param str pad_token: Pad token. Defaults to "<|endoftext|>".
    :param str input_ids_dtype: Type of input ids. Defaults to "int32".
    :param str input_mask_dtype: Type of mask. Defaults to "int32".
    :param str labels_dtype: Type of labels. Defaults to "int32".

    :returns: yields training examples (feature, label)
    """

    def _generate_train_example(token_ids):
        return _create_features_labels(
            token_ids,
            max_sequence_length,
            inverted_mask,
            add_special_tokens,
            pad_id,
            eos_id,
            input_ids_dtype,
            input_mask_dtype,
            labels_dtype,
        )

    tokenizer = GenomicTokenizer(vocab_file=vocab_file, n_gram=n_gram)

    eos_id = tokenizer.convert_tokens_to_ids([eos_token])[0]
    pad_id = tokenizer.convert_tokens_to_ids([pad_token])[0]

    # if add_special_tokens = True, pad sequence beginning and end
    new_msl = max_sequence_length + 2 if add_special_tokens else max_sequence_length

    file_names = os.listdir(data_dir)

    for _file_name in file_names:
        file_path = os.path.join(data_dir, _file_name)
        for seq in SeqIO.parse(file_path, "fasta"):
            sample = str(seq.seq)
            sample = tokenizer.tokenize(sample)
            sample = tokenizer.convert_tokens_to_ids(sample)
            if len(sample) > (new_msl + 1):
                if skip_large_MSL:
                    continue
            yield _generate_train_example(sample)

    

def _create_features_labels(
    token_ids,
    max_sequence_length,
    inverted_mask=False,
    add_special_tokens=True,
    pad_id=0,
    eos_id=0,
    input_ids_dtype="int32",
    input_mask_dtype="int32",
    labels_dtype="int32",
):
    """
    Given a list of token_ids, generate input sequence
    and labels.
    """

    if add_special_tokens:
        new_token_ids = [eos_id,] + token_ids[:(max_sequence_length-1)] + [eos_id,]
    else:
        new_token_ids = token_ids[:(max_sequence_length+1)]

    input_ids = new_token_ids[:-1]
    labels = new_token_ids[1:]
    input_mask = [1] * len(input_ids)

    # padding
    num_pad = max_sequence_length - len(input_ids)
    padding = [pad_id] * num_pad

    input_ids.extend(padding)
    labels.extend(padding)
    input_mask.extend([0] * num_pad)

    # assertions to ensure correct output shapes
    assert (
        len(input_ids) == max_sequence_length
        and len(labels) == max_sequence_length
        and len(input_mask) == max_sequence_length
    ), "Wrong sequence length"

    # create feature dict
    features = dict()
    features["input_ids"] = getattr(np, input_ids_dtype)(input_ids)
    features["attention_mask"] = getattr(np, input_mask_dtype)(input_mask)

    if inverted_mask:
        features['attention_mask'] = np.equal(
            features['attention_mask'], 0
        ).astype(features['attention_mask'].dtype)
    labels = getattr(np, labels_dtype)(labels)
    features['labels'] = labels

    return features

class GenomicTokenizer(BaseTokenizer):
    
    def __init__(self, vocab_file, n_gram = 3):
        """
        :param str vocab_file: File containing vocabulary, each token in new line
        :param int n_gram: specifies tokenization level for the sequence 
        """
        super(GenomicTokenizer, self).__init__(vocab_file, do_lower_case=False)
        self.n_gram = n_gram

    def tokenize(self, sequence):
        """
        Tokenizes a sequence every three base pairs, without overlaps. 
        Assuming the sequence length is a multiple of n_gram.
        Returns a list of tokens. Does not convert to IDs.
        """
        tokens = [sequence[i:(i+self.n_gram)] for i in range(0, len(sequence), self.n_gram)]
        return tokens

    def convert_tokens_to_ids(self, text):
        """
        Converts a list of tokens to a list of ids.
        We shift all outputs by 1 because the dictionary formed by keras 'Tokenizer' starts
        with index 1 instead of 0.
        """
        tknzd_seq = self.tokenizer.texts_to_sequences(text)
        tknzd_seq = np.concatenate(tknzd_seq).tolist() if tknzd_seq else []
        return list(map(lambda x: x - 1, tknzd_seq))
    

    
