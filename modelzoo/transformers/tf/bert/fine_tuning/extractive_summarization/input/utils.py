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

import gc
import glob
import json
import logging
import os
from multiprocessing import Pool

import tensorflow as tf

from modelzoo.transformers.data_processing.bertsum_data_processor import (
    BertData,
    RougeBasedLabelsFormatter,
    check_output,
)
from modelzoo.transformers.data_processing.utils import pad_input_sequence

logging.basicConfig(level=logging.INFO)


class BertFormatter:
    def __init__(self, params):
        """
        Converts input into tf bert format, set extractive summarization
        targets based on the rouge score between references and
        input sentences.
        :param params: dict params: BertData configuration parameters.
        """
        self.bert_data = BertData(params)
        self.labels_formatter = RougeBasedLabelsFormatter()
        self.max_sequence_length = params.max_sequence_length
        self.max_cls_tokens = params.max_cls_tokens
        self.input_path = os.path.abspath(params.input_path)
        self.output_path = os.path.abspath(params.output_path)
        self.n_cpu = params.n_cpu

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _create_example(
        self,
        input_ids,
        segment_ids,
        cls_indices,
        labels,
        max_sequence_length=50,
        max_cls_tokens=50,
    ):
        input_ids = pad_input_sequence(
            input_ids, self.bert_data.pad_id, max_sequence_length
        )
        labels = pad_input_sequence(
            labels, self.bert_data.pad_id, max_cls_tokens
        )
        input_mask = tf.math.equal(input_ids, self.bert_data.pad_id)
        segment_ids = pad_input_sequence(
            segment_ids, self.bert_data.pad_id, max_sequence_length
        )
        cls_indices = pad_input_sequence(
            cls_indices, self.bert_data.pad_id, max_cls_tokens
        )
        cls_weights = tf.math.not_equal(cls_indices, self.bert_data.pad_id)

        feature = {
            "input_ids": self._int64_feature(input_ids),
            "labels": self._int64_feature(labels),
            "input_mask": self._int64_feature(input_mask),
            "segment_ids": self._int64_feature(segment_ids),
            "cls_indices": self._int64_feature(cls_indices),
            "cls_weights": self._float_feature(cls_weights),
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))

    def _format_to_bert(self, params):
        json_file, output_file_pattern = params

        with open(json_file, "r") as fin:
            for i, d in enumerate(json.load(fin)):
                output_file = f"{'/'.join(output_file_pattern.split('/')[:-1])}/@{i}{output_file_pattern.split('/')[-1]}"

                if os.path.exists(output_file):
                    logging.info(f"Ignoring file: {output_file}.")
                    continue

                source, target = d["src"], d["tgt"]
                # Get sentences which are present in the summarization.
                oracle_ids = self.labels_formatter.process(source, target, 3)
                # Convert input into bert tf format.
                data = self.bert_data.process(source, target, oracle_ids)

                if not data:
                    logging.info(
                        f"Skipping file: {output_file}. Source or target field is empty."
                    )
                    continue

                (
                    indexed_tokens,
                    labels,
                    segments_ids,
                    cls_ids,
                    src_txt,
                    tgt_txt,
                ) = data

                example = self._create_example(
                    indexed_tokens,
                    segments_ids,
                    cls_ids,
                    labels,
                    max_sequence_length=self.max_sequence_length,
                    max_cls_tokens=self.max_cls_tokens,
                )

                logging.info(f"Saving file: {output_file}")
                with tf.io.TFRecordWriter(output_file) as writer:
                    writer.write(example.SerializeToString())

        gc.collect()

    def process(self):
        logging.info(
            f"Preparing to convert to bert format {self.input_path} to {self.output_path}."
        )
        for corpus_type in ["valid", "test", "train"]:
            data = []

            for file_name in glob.iglob(
                os.path.join(self.input_path, f"{corpus_type}-*.json")
            ):
                real_name = os.path.basename(file_name)
                data.append(
                    (
                        file_name,
                        os.path.join(
                            self.output_path,
                            real_name.replace(".json", "bert.tfrecord"),
                        ),
                    )
                )

            pool = Pool(self.n_cpu)

            for _ in pool.imap(self._format_to_bert, data):
                pass

            pool.close()
            pool.join()

        check_output(self.input_path, self.output_path)
        logging.info(
            f"Successfully finished converting to bert format {self.input_path} to {self.output_path}.\n"
        )
