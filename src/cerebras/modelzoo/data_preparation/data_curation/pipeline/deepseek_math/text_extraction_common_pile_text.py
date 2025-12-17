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

import argparse
import os
import sys

import yaml
from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from filters import MathFastTextFilter


class FastTextScoreFilter(BaseFilter):
    """Keep docs with metadata['fasttext_math_score'] > threshold."""

    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = float(threshold)

    def filter(self, doc: Document) -> bool:
        md = doc.get("metadata") or {}
        score = float(md.get("fasttext_math_score", 0.0))
        return score > self.threshold


def parse_args():
    p = argparse.ArgumentParser(
        description="Minimal math filtering pipeline (local only)"
    )
    p.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the YAML configuration file',
    )
    return p.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    required = {
        'pipeline': [
            'input_jsonl_folder',
            'output_path',
            'logging_dir',
            'cache_dir',
        ],
        'models': ['math_fasttext_model_path'],
        'executor': ['type', 'local'],
    }
    for section, fields in required.items():
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in config")
        if isinstance(fields, list):
            for field in fields:
                if field not in config[section]:
                    raise ValueError(
                        f"Missing required field '{field}' in {section} section"
                    )

    if config['executor']['type'] != 'local':
        raise ValueError(
            "This script only supports local execution. Set executor.type = 'local'."
        )

    return config


def main():
    args = parse_args()
    config = load_config(args.config)

    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    )

    from glob import glob

    input_folder = config['pipeline']['input_jsonl_folder']
    input_files = glob(os.path.join(input_folder, "*.jsonl"))

    tasks = len(input_files)
    config['executor']['local']['tasks'] = tasks
    print(f"[INFO] Found {tasks} input files — setting tasks = {tasks}")

    pipeline = [
        JsonlReader(
            data_folder=config['pipeline']['input_jsonl_folder'],
            compression=None,
            glob_pattern="*.jsonl",
            default_metadata={},
            text_key="text_by_page_src",  # <-- key fix so docs aren't skipped
        ),
        MathFastTextFilter(
            model_path=config['models']['math_fasttext_model_path'],
            math_threshold=config['filters']['math_threshold'],
            math_class_name=config['filters']['labels']['math'],
            exclusion_writer=None,
        ),
        FastTextScoreFilter(
            threshold=config['filters']['fasttext_score_threshold']
        ),
        JsonlWriter(
            output_folder=config['pipeline']['output_path'],
            output_filename="${rank}.json.gz",
            compression="gzip",
        ),
    ]

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        logging_dir=config['pipeline']['logging_dir'],
        tasks=config['executor']['local']['tasks'],
        workers=config['executor']['local']['workers'],
    )
    executor.run()


if __name__ == "__main__":
    main()
