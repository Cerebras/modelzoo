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
import time

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Text extraction pipeline configuration"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the YAML configuration file',
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required_fields = {
        'pipeline': [
            'input_warc_folder',
            'extracted_english_path',
            'extracted_math_path',
            'logging_dir',
            'cache_dir',
        ],
        'models': ['english_fasttext_model_path', 'math_fasttext_model_path'],
    }

    for section, fields in required_fields.items():
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in config")
        for field in fields:
            if not config[section].get(field):
                raise ValueError(
                    f"Missing required field '{field}' in {section} section"
                )

    # Validate executor configuration
    if 'executor' not in config:
        raise ValueError("Missing required section 'executor' in config")

    if config['executor']['type'] == 'slurm' and not config['slurm'].get(
        'partition'
    ):
        raise ValueError(
            "Slurm partition must be provided when using slurm executor"
        )

    return config


def main():
    start = time.time()
    args = parse_args()
    config = load_config(args.config)

    # Set environment variables
    os.environ["XDG_CACHE_HOME"] = config['pipeline']['cache_dir']
    os.environ["HF_HOME"] = config['pipeline']['cache_dir']

    import sys

    sys.path.append("/modelzoo/data_preparation/")

    from pathlib import Path

    from data_curation.pipeline.deepseek_math.deduplication.minhash_deduplication import (
        MinHashDeduplication,
    )
    from data_curation.pipeline.deepseek_math.deduplication.url_deduplication import (
        CustomURLDedupPipelineStep,
    )
    from data_curation.pipeline.deepseek_math.extractors import (
        PIIRedactingExtractor,
        ResiliparseExtractor,
        TrafilaturaExtractor,
    )
    from data_curation.pipeline.deepseek_math.filters import (
        EnglishFastTextFilter,
        FastTextQualityFilter,
        InclusiveLatexSymbolFilter,
        MathFastTextFilter,
        PerplexityScorer,
    )
    from data_curation.pipeline.deepseek_math.readers import SafeWarcReader
    from data_curation.pipeline.deepseek_math.writers import (
        ConditionalJsonlWriter,
    )
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.filters import URLFilter
    from datatrove.pipeline.readers.base import BaseDiskReader
    from datatrove.pipeline.writers import JsonlWriter
    from executors.slurm.slurm import SlurmPipelineExecutor

    def warc_adapter(self, data: dict, path: str, record_idx):
        doc_dict = BaseDiskReader._default_adapter(self, data, path, record_idx)
        fname = Path(path).name
        clean = fname.split(".", 1)[0]
        doc_dict["metadata"]["cc_path"] = clean
        return doc_dict

    pipeline = [
        SafeWarcReader(
            data_folder=config['pipeline']['input_warc_folder'],
            adapter=warc_adapter,
            default_metadata={"dump": config['pipeline']['dump']},
        ),
        URLFilter(),
        ResiliparseExtractor(),
        EnglishFastTextFilter(
            model_path=config['models']['english_fasttext_model_path'],
            eng_threshold=config['filters']['english_threshold'],
            eng_class_name=config['filters']['labels']['english'],
            exclusion_writer=None,
        ),
        TrafilaturaExtractor(
            favour_precision=False,  # Include symbol-heavy content
            include_tables=True,  # Math in tables
            include_formatting=True,  # Preserve structure
            prune_unwanted_sections=True,  # Remove ads/navigation
            deduplicate=True,  # ← Remove duplicate sections
            include_links=False,
        ),
        PIIRedactingExtractor(),
        FastTextQualityFilter(
            model_path=config['models']['text_quality_fasttext_model_path']
        ),
        JsonlWriter(
            output_folder=config['pipeline']['extracted_english_path'],
            output_filename="${cc_path}.jsonl.gz",
        ),
        InclusiveLatexSymbolFilter(),
        MathFastTextFilter(
            model_path=config['models']['math_fasttext_model_path'],
            math_threshold=config['filters']['math_threshold'],
            math_class_name=config['filters']['labels']['math'],
            exclusion_writer=None,
        ),
        ConditionalJsonlWriter(
            base_output_folder=config['pipeline']['extracted_math_path'],
            filename_template="${cc_path}.jsonl.gz",
        ),
    ]

    perplexity_model_path = config["models"].get("perplexity_model_path")
    if perplexity_model_path:
        pipeline.append(PerplexityScorer(model_path=perplexity_model_path))

    if config['executor']['type'] == 'local':
        executor = LocalPipelineExecutor(
            pipeline=pipeline,
            logging_dir=config['pipeline']['logging_dir'],
            tasks=config['local']['tasks'],
            workers=config['local']['workers'],
        )
        executor.run()

        math_output_path = config['pipeline']['extracted_math_path']
        url_dedup_math = CustomURLDedupPipelineStep(
            input_folder=math_output_path
        )
        url_deduped_path = url_dedup_math.run()

        minhash_deduplication = MinHashDeduplication(
            input_folder=url_deduped_path
        )
        final_output_folder = minhash_deduplication.run()

        print(f"The final output is present at: {final_output_folder}")
    else:  # slurm
        executor = SlurmPipelineExecutor(pipeline=pipeline, **config['slurm'])
        executor.run()

    end = time.time()
    print(f"It took {end - start} seconds to finish.")


if __name__ == "__main__":
    main()
