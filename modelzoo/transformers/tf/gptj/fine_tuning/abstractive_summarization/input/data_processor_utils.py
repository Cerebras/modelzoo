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

# Copyright 2016/2017 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
# Modifications Copyright 2022 Cerebras Systems
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
#
# Specifically, lines 76-85 are based on processing code from:
# https://github.com/abisee/pointer-generator/blob/master/[data.py, batcher.py]

import struct

import numpy as np
import tensorflow as tf
from tensorflow.core.example import example_pb2

from modelzoo.transformers.data_processing.BPETokenizer import BPETokenizer


def training_data_generator(
    data_file,
    vocab_file,
    encoder_file,
    max_sequence_length,
    inverted_mask=False,
    input_ids_dtype="int32",
    input_mask_dtype="int32",
    labels_dtype="int32",
    truncate_article=False,
):
    """
    Generator function used to create input dataset
    for GPTJ finetuning model.

    :param str data_file: bin file
    :param str vocab_file: Vocabulary file, to build tokenization from
    :param str encoder_file: Encoder file, map from bpe tokens to
         token IDs for tokenization
    :param int max_sequence_length: Maximum length of the sequence to generate
    :param bool inverted_mask: If set to False, has 0's on padded positions and
        1's elsewhere. Otherwise, "inverts" the mask, so that 1's are on padded
        positions and 0's elsewhere.
    :param str input_ids_dtype: Type of input ids. Defaults to "int32".
    :param str input_mask_dtype: Type of mask. Defaults to "int32".
    :param str labels_dtype: Type of labels. Defaults to "int32".
    :param bool truncate: Truncate longer articles to help fit within MSL.

    :returns: yields training examples (feature, label)
    """
    tokenizer = BPETokenizer(
        vocab_file, encoder_file, special_tokens=["<|sep|>", "<|endoftext|>"]
    )

    # use get_token_id directly as encode() will split the input token
    EOS = PAD = tokenizer.get_token_id("<|endoftext|>")
    SEP = tokenizer.get_token_id("<|sep|>")
    assert EOS is not None and SEP is not None

    lengths = []

    with open(data_file, 'rb') as reader:
        while True:
            len_bytes = reader.read(8)
            if not len_bytes:
                break  # finished reading this file
            str_len = struct.unpack('q', len_bytes)[0]
            example_str = struct.unpack(f'{str_len}s', reader.read(str_len))[0]
            example = example_pb2.Example.FromString(example_str)
            article = example.features.feature['article'].bytes_list.value[0]
            abstract = example.features.feature['abstract'].bytes_list.value[0]

            # skip empty documents
            if len(article) == 0 or len(abstract) == 0:
                tf.compat.v1.logging.info(
                    "Skipping [1] example as either"
                    " article or abstract is empty"
                )
                continue

            # skip unrealistic examples where abstract is longer than article
            if len(article) < len(abstract):
                tf.compat.v1.logging.info(
                    "Skipping [1] example as abstract is longer than article"
                )
                continue

            article_token_ids = tokenizer.encode(article.decode('utf-8'))
            abstract_token_ids = tokenizer.encode(abstract.decode('utf-8'))

            # + 2 for seperator and EOS token
            total_len = len(article_token_ids) + len(abstract_token_ids) + 2

            if total_len > max_sequence_length:
                if not truncate_article:
                    tf.compat.v1.logging.info(
                        "Skipping [1] example: article and abstract"
                        " do not fit within MSL without truncation"
                    )
                    continue
                else:
                    # truncate article instead of abstract
                    if len(abstract_token_ids) > max_sequence_length - 2:
                        tf.compat.v1.logging.info(
                            "Skipping[1] example: cannot fit entire abstract."
                        )
                        continue
                    # -2 for seperator and EOS token
                    end_idx = max_sequence_length - len(abstract_token_ids) - 2
                    article_token_ids = article_token_ids[0:end_idx]

            input_ids = [EOS] + article_token_ids + [SEP] + abstract_token_ids
            labels = article_token_ids + [EOS] + abstract_token_ids + [EOS]
            input_mask = [0] * (1 + len(article_token_ids)) + [1] * (
                1 + len(abstract_token_ids)
            )

            lengths.append(total_len)

            # add padding
            num_pad = max_sequence_length - len(input_ids)
            padding = [PAD] * num_pad

            input_ids.extend(padding)
            labels.extend(padding)
            input_mask.extend([0] * num_pad)

            # ensure correct output shapes
            assert (
                len(input_ids) == max_sequence_length
                and len(labels) == max_sequence_length
                and len(input_mask) == max_sequence_length
            ), "Wrong sequence length"

            # create feature dict
            features = dict()
            features["input_ids"] = getattr(np, input_ids_dtype)(input_ids)
            features["input_mask"] = getattr(np, input_mask_dtype)(input_mask)

            if inverted_mask:
                features['input_mask'] = np.equal(
                    features['input_mask'], 0
                ).astype(features['input_mask'].dtype)
            labels = getattr(np, labels_dtype)(labels)

            yield features, labels
