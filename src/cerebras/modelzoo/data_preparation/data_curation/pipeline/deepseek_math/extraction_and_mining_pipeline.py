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
import time

import yaml

sys.path.append("/modelzoo/data_preparation/")


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
    return config


def main():
    start = time.time()
    args = parse_args()
    config = load_config(args.config)

    # Set environment variables
    os.environ["XDG_CACHE_HOME"] = config['extraction_pipeline']['cache_dir']
    os.environ["HF_HOME"] = config['extraction_pipeline']['cache_dir']

    from pathlib import Path

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
    )
    from data_curation.pipeline.deepseek_math.readers import (
        MathDomainReader,
        SafeJsonlReader,
        SafeWarcReader,
    )
    from data_curation.pipeline.deepseek_math.writers import (
        ConditionalJsonlWriter,
        Mine_Save_URL,
        WarcDownloader,
    )
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.filters import URLFilter
    from datatrove.pipeline.readers.base import BaseDiskReader
    from datatrove.pipeline.writers import JsonlWriter
    from executors.slurm.slurm import SlurmPipelineExecutor

    # Phase 1: Text Extraction Pipeline
    def warc_adapter(self, data: dict, path: str, record_idx):
        doc_dict = BaseDiskReader._default_adapter(self, data, path, record_idx)
        fname = Path(path).name
        clean = fname.split(".", 1)[0]
        doc_dict["metadata"]["cc_path"] = clean
        return doc_dict

    print("Starting extraction pipeline ...")
    extraction_pipeline = [
        SafeWarcReader(
            data_folder=config['extraction_pipeline']['input_warc_folder'],
            adapter=warc_adapter,
            default_metadata={"dump": config['extraction_pipeline']['dump']},
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
            output_folder=config['extraction_pipeline'][
                'extracted_english_path'
            ],
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
            base_output_folder=config['extraction_pipeline'][
                'extracted_math_path'
            ],
            filename_template="${cc_path}.jsonl.gz",
        ),
    ]

    if config['extraction_pipeline']['executor']['type'] == 'local':
        print("Running extraction pipeline in local mode")
        extraction_executor = LocalPipelineExecutor(
            pipeline=extraction_pipeline,
            logging_dir=config['extraction_pipeline']['logging_dir'],
            tasks=config['extraction_pipeline']['local']['tasks'],
            workers=config['extraction_pipeline']['local']['workers'],
        )

    else:  # slurm
        print("Running extraction pipeline in slurm mode")
        extraction_executor = SlurmPipelineExecutor(
            pipeline=extraction_pipeline,
            **config['extraction_pipeline']['slurm'],
        )
    extraction_executor.run()

    # Phase 2: URL Mining Pipeline
    print("Starting URL mining pipeline...")
    math_url_mining_pipeline = [
        MathDomainReader(
            data_folder=config['url_pipeline']['input_dir'],
        ),
        Mine_Save_URL(
            output_folder=config['url_pipeline']['output_dir'],
            headers=config['url_pipeline']['headers'],
            delay=config['url_pipeline']['delay'],
            max_urls_per_sitemap=config['url_pipeline']['max_urls_per_sitemap'],
        ),
    ]
    if config["url_pipeline"]["executor"]["type"] == "local":
        math_url_mining_executor = LocalPipelineExecutor(
            pipeline=math_url_mining_pipeline,
            logging_dir=config['url_pipeline']['logging_dir'],
            tasks=config["url_pipeline"]['local']['tasks'],
            workers=config["url_pipeline"]['local']['workers'],
        )
    else:
        math_url_mining_executor = SlurmPipelineExecutor(
            pipeline=math_url_mining_pipeline,
            **config["url_pipeline"]['slurm'],
            depends=extraction_executor,
        )

    math_url_mining_executor.run()
    print("URL mining pipeline completed successfully!")

    # Phase 3: Downloading Pipeline (starts only after URL mining completes)
    print("Starting downloading pipeline...")
    downloading_pipeline = [
        SafeJsonlReader(
            data_folder=os.path.join(
                config['url_pipeline']['output_dir'], "urls"
            )
        ),
        WarcDownloader(
            output_dir=config['downloading_pipeline'].get(
                'warc_output_dir',
                config['url_pipeline']['output_dir'] + '/warc_files',
            ),
            max_warc_size=config['downloading_pipeline'].get(
                'max_warc_size', 1 * 1024**3
            ),  # 1GB default
            max_workers=config['downloading_pipeline'].get('max_workers', 32),
            timeout=config['downloading_pipeline'].get('timeout', 15),
            headers=config['downloading_pipeline']['headers'],
        ),
    ]

    if config["downloading_pipeline"]["executor"]["type"] == "local":
        downloading_executor = LocalPipelineExecutor(
            pipeline=downloading_pipeline,
            logging_dir=config['downloading_pipeline']['logging_dir'],
            tasks=config['downloading_pipeline']['local']['tasks'],
            workers=config['downloading_pipeline']['local']['workers'],
        )
    else:
        downloading_executor = SlurmPipelineExecutor(
            pipeline=downloading_pipeline,
            **config["downloading_pipeline"]['slurm'],
            depends=math_url_mining_executor,
        )

    downloading_executor.run()
    print("Downloading pipeline completed successfully!")

    mined_urls_extraction_pipeline = [
        SafeWarcReader(
            data_folder=config['downloading_pipeline'].get(
                'warc_output_dir',
                config['url_pipeline']['output_dir'] + '/warc_files',
            ),
            adapter=warc_adapter,
            default_metadata={"dump": config['extraction_pipeline']['dump']},
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
            output_folder=config['mined_urls_extraction_pipeline'][
                'extracted_english_path'
            ],
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
            base_output_folder=config['mined_urls_extraction_pipeline'][
                'extracted_math_path'
            ],
            filename_template="${cc_path}.jsonl.gz",
        ),
    ]

    if config["mined_urls_extraction_pipeline"]["executor"]["type"] == "local":
        mined_urls_extraction_executor = LocalPipelineExecutor(
            pipeline=mined_urls_extraction_pipeline,
            logging_dir=config['mined_urls_extraction_pipeline']['logging_dir'],
            tasks=config['mined_urls_extraction_pipeline']['local']['tasks'],
            workers=config['mined_urls_extraction_pipeline']['local'][
                'workers'
            ],
        )
    else:
        mined_urls_extraction_executor = SlurmPipelineExecutor(
            pipeline=mined_urls_extraction_pipeline,
            **config['mined_urls_extraction_pipeline']['slurm'],
            depends=downloading_executor,
        )
    mined_urls_extraction_executor.run()

    end = time.time()
    print(f"Total time taken: {end - start} seconds")


if __name__ == "__main__":
    main()
