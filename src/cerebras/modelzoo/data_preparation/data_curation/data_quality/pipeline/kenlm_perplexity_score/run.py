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

#!/usr/bin/env python3
"""
KenLM Pipeline Runner

This script provides a unified interface to run the complete KenLM pipeline:
1. Reader: Reads input data from various sources
2. Preprocessor: Processes text data for KenLM training
3. Trainer: Trains the KenLM model
4. Evaluator: Evaluates the model using CCPerplexityNet

The pipeline can be run either directly or as part of a larger datatrove pipeline.
"""

import argparse

import yaml
from datatrove.utils.logging import logger

from cerebras.modelzoo.data_preparation.data_curation.data_quality.pipeline.kenlm_perplexity_score.config import (
    DataConfig,
    KenLMPipelineConfig,
    KenLMTrainerConfig,
    PerplexityStatsConfig,
    RunConfig,
    SentencePieceTrainerConfig,
)
from cerebras.modelzoo.data_preparation.data_curation.data_quality.pipeline.kenlm_perplexity_score.perplexity import (
    CCNetPerplexityStats,
    CollatePerplexityStats,
)
from cerebras.modelzoo.data_preparation.data_curation.data_quality.pipeline.kenlm_perplexity_score.preprocess import (
    KenLMPreprocessor,
)
from cerebras.modelzoo.data_preparation.data_curation.data_quality.pipeline.kenlm_perplexity_score.train import (
    KenLMTrainer,
    SentencePieceTrainer,
)


def load_config(config_path):
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def main():
    parser = argparse.ArgumentParser(description='Run KenLM pipeline')
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args()

    params = load_config(args.config)
    config = KenLMPipelineConfig(**params)

    # Create and run preprocessing pipeline
    depends = None
    if "KenLMPreprocessor" in config.pipelines:
        logger.info("Starting preprocessing pipeline")
        preprocessor = KenLMPreprocessor(
            DataConfig(**vars(config.data_config)),
            RunConfig(**vars(config.run_config)),
        )
        preprocessor.run()
        depends = preprocessor.executor

    # Create and run training pipeline
    trainer = None
    if "KenLMTrainer" in config.pipelines:
        logger.info("Starting training pipeline")
        trainer = KenLMTrainer(
            KenLMTrainerConfig(**config.pipelines["KenLMTrainer"]),
            RunConfig(**vars(config.run_config)),
        )
        trainer.run(depends)
        depends = trainer.executor

    if "SentencePieceTrainer" in config.pipelines:
        logger.info("Starting training pipeline")
        trainer = SentencePieceTrainer(
            SentencePieceTrainerConfig(
                **config.pipelines["SentencePieceTrainer"]
            ),
            RunConfig(**vars(config.run_config)),
        )
        trainer.run(depends)
        depends = trainer.executor

    # Create and run perplexity pipeline
    if "CCPerplexityStats" in config.pipelines:
        logger.info("Starting perplexity evaluation pipeline")
        perplexity_config = PerplexityStatsConfig(
            **config.pipelines["CCPerplexityStats"]
        )
        if trainer:
            config.data_config.input_format = config.data_config.output_format
            config.data_config.input_folder = (
                perplexity_config.input_folder
                or config.data_config.output_folder
            )
        config.data_config.output_folder = perplexity_config.output_folder

        # Remove output_folder from perplexity_config before passing to pipeline
        perplexity_config_dict = vars(perplexity_config)
        perplexity_config_dict.pop('output_folder', None)
        perplexity_config_dict.pop('input_folder', None)

        # Create and run perplexity evaluation pipeline
        perplexity = CCNetPerplexityStats(
            **perplexity_config_dict,
            run_config=RunConfig(**vars(config.run_config)),
            data_config=DataConfig(**vars(config.data_config)),
        )
        perplexity.run(depends)
        depends = perplexity.executor

        # Create and run collation pipeline with dependency on perplexity
        logger.info("Starting perplexity statistics collation")
        collator = CollatePerplexityStats(
            run_config=RunConfig(**vars(config.run_config)),
            data_config=DataConfig(**vars(config.data_config)),
        )
        collator.run(depends=depends)


if __name__ == '__main__':
    main()
