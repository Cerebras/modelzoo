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

import csv
import glob
import json
import logging
import os
from collections import defaultdict, namedtuple

from cerebras.modelzoo.data_preparation.nlp.bert.bertsum_data_processor import (
    BertData,
    RougeBasedLabelsFormatter,
)

logging.basicConfig(level=logging.INFO)


BertInputFeatures = namedtuple(
    "BertInputFeatures", ["input_token_ids", "labels", "segment_ids", "cls_ids"]
)


class BertCSVFormatter:
    def __init__(self, params):
        """
        Converts input into bert format, sets extractive summarization
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

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def _json_to_csv(self, json_input_file, csv_output_file, meta_data):
        with (
            open(json_input_file, "r") as fin,
            open(csv_output_file, "w", newline="") as fout,
        ):
            csv_writer = csv.DictWriter(
                fout,
                fieldnames=BertInputFeatures._fields,
                quoting=csv.QUOTE_MINIMAL,
            )
            csv_writer.writeheader()

            logging.info(
                f"Converting {json_input_file} to CSV and saving in {csv_output_file}"
            )
            for i, data in enumerate(json.load(fin)):
                source, target = data["src"], data["tgt"]

                # Get sentences which are present in the summarization.
                oracle_ids = self.labels_formatter.process(source, target, 3)
                # Convert input into bert tf format.
                bert_data = self.bert_data.process(source, target, oracle_ids)
                if not bert_data:
                    logging.info(
                        f"Skipping index: {i} in {json_input_file}. Source or "
                        f"target field is empty."
                    )
                    continue

                input_tokens, labels, segment_ids, cls_ids, _, _ = bert_data
                bert_features = BertInputFeatures(
                    input_tokens, labels, segment_ids, cls_ids
                )
                csv_writer.writerow(bert_features._asdict())
                meta_data[os.path.basename(csv_output_file)] += 1

    def process(self):
        logging.info(
            f"Preparing to convert to bert format {self.input_path} to "
            f"{self.output_path}."
        )
        for corpus_type in ["valid", "test", "train"]:
            output_path = os.path.join(self.output_path, corpus_type)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            input_files = glob.iglob(
                os.path.join(self.input_path, f"{corpus_type}-*.json")
            )
            meta_data = defaultdict(int)

            for input_file in input_files:
                output_file = os.path.join(
                    output_path,
                    os.path.basename(input_file).replace("json", "csv"),
                )
                self._json_to_csv(input_file, output_file, meta_data)
            logging.info(
                f"Converted simplified JSON to CSV for {corpus_type} set. "
                f"Writing metadata file."
            )
            meta_file = os.path.join(output_path, "meta.dat")
            with open(meta_file, "w") as fout:
                for output_file, num_lines in meta_data.items():
                    fout.write(f"{output_file} {num_lines}\n")
        logging.info(
            f"Done converting to CSV, files saved to {self.output_path}."
        )
