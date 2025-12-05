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

import yaml

from modelzoo.transformers.tf.bert.utils import set_defaults


def get_pfam_vocab():
    """
    Returns the vocabulary used by the pfam pipe as a dictionary
    along with the min/max amino acid ids, such that a random amino
    acid can be generated using:
    
        random.randint(min_aa_id, max_aa_id)

    Letter codes standard for IUPAC. 
    See here: https://www.bioinformatics.org/sms2/iupac.html
    """
    vocab = {
        "[PAD]": 0,
        "[MASK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "A": 4,
        "B": 5,
        "C": 6,
        "D": 7,
        "E": 8,
        "F": 9,
        "G": 10,
        "H": 11,
        "I": 12,
        "K": 13,
        "L": 14,
        "M": 15,
        "N": 16,
        "O": 17,
        "P": 18,
        "Q": 19,
        "R": 20,
        "S": 21,
        "T": 22,
        "U": 23,
        "V": 24,
        "W": 25,
        "X": 26,
        "Y": 27,
        "Z": 28,
    }
    min_aa_id = vocab["A"]
    max_aa_id = vocab["Z"]
    return vocab, min_aa_id, max_aa_id


def get_params(params_file, mode=None):

    # Load yaml into params.
    with open(params_file, "r") as stream:
        params = yaml.safe_load(stream)

    vocab_size = len(get_pfam_vocab()[0])
    params["train_input"]["vocab_size"] = vocab_size
    if "eval_input" in params:
        params["eval_input"]["vocab_size"] = vocab_size

    set_defaults(params, mode=mode)

    return params
