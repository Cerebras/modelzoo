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
import shutil
import subprocess
from pathlib import Path
from typing import ClassVar

from datatrove.data import DocumentsPipeline
from datatrove.utils.logging import logger

from cerebras.modelzoo.data_preparation.data_curation.data_quality.pipeline.kenlm_perplexity_score.config import (
    KenLMTrainerConfig,
    RunConfig,
    SentencePieceTrainerConfig,
)
from cerebras.modelzoo.data_preparation.data_curation.data_quality.pipeline.kenlm_perplexity_score.pipeline import (
    BasePipeline,
)


class KenLMTrainer(BasePipeline):
    """
    A pipeline for training KenLM and SentencePiece models.
    This pipeline runs sequentially with no parallelism.
    """

    name: ClassVar[str] = "KenLM Trainer"
    type: ClassVar[str] = "TRAIN"
    config: KenLMTrainerConfig = None
    kenlm_dir: str = None
    output_path: str = None
    temp_dir: str = None
    input_path: str = None
    backoff: int = 0

    def __init__(self, config: KenLMTrainerConfig, run_config: RunConfig):
        """
        Initialize the KenLM Trainer pipeline.

        Args:
            config: Configuration object containing all parameters
        """
        if run_config.executor_type == "local":
            run_config.num_workers = 1
        else:
            run_config.num_tasks = 1
            run_config.cpus_per_task = 1

        init_params = {
            **vars(run_config),
            "input_folder": config.input_folder,
            "output_folder": config.output_path,
        }
        super().__init__(**init_params)

        # Override parallel processing settings for sequential execution
        self.config = config
        self.kenlm_dir = self.config.kenlm_path or os.path.join(
            self.config.output_path, "kenlm"
        )
        self.output_path = self.config.output_path
        self.temp_dir = os.path.join(self.output_path, self.config.temp_prefix)
        self.input_path = self.config.input_folder
        self.backoff = (
            self.config.quantize
            if self.config.quantize > 0
            else self.config.backoff
        )

        # Create output directories
        os.makedirs(self.config.output_path, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        # Create training pipeline
        pipeline = []
        pipeline.append(self._train_kenlm())
        self.add_pipelines((pipeline, self.__name__()))

    def _clone_kenlm(self):
        """Clone the kenlm repository if it doesn't exist and no path is provided."""
        if self.config.kenlm_path:
            return

        if not os.path.exists(self.kenlm_dir):
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/kpu/kenlm.git",
                    self.kenlm_dir,
                ],
                check=True,
            )

    def _build_kenlm(self):
        """Build the kenlm binaries if no pre-built path is provided."""
        if self.config.kenlm_path:
            return

        if not os.path.exists(os.path.join(self.kenlm_dir, "bin", "lmplz")):
            subprocess.run(
                ["cd", self.kenlm_dir, "&&", "./bjam", "-j4"],
                shell=True,
                check=True,
            )

    def _train_kenlm(self):
        """Create a training step for KenLM model."""

        def train(
            data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1
        ) -> str:
            # Clone and build kenlm only if no pre-built path is provided
            if not self.config.kenlm_path:
                self._clone_kenlm()
                self._build_kenlm()

            arpa_path = os.path.join(self.output_path, "model.arpa")

            # Get all input files
            input_files = list(Path(self.input_path).glob("*.txt"))
            if not input_files:
                raise ValueError(f"No .txt files found in {self.input_path}")

            # Create filenames.txt with absolute paths
            filenames_path = os.path.join(self.input_path, "filenames.txt")
            with open(filenames_path, 'w', encoding='utf-8') as f:
                for file in input_files:
                    f.write(str(file.absolute()) + '\n')
            logger.info(f"Filenames path: {filenames_path}")

            # Build the single command with explicit pipe
            cmd = f"xargs -a {filenames_path} cat | {os.path.join(self.kenlm_dir, 'bin', 'lmplz')} -o{self.config.order} -S{self.config.memory} --vocab_estimate={self.config.vocab_estimate} --temp_prefix={self.temp_dir}"

            # Add intermediate and verbose flags
            if self.config.intermediate:
                cmd += " --intermediate"
            if self.config.verbose:
                cmd += " --verbose"

            # Add output redirection
            cmd += f" > {arpa_path}"

            # Run the command
            subprocess.run(cmd, shell=True, check=True)

            # Quantize if enabled
            if self.config.quantize > 0:
                binary_path = (
                    arpa_path.split('.')[0]
                    + f"_quantized_{str(self.config.quantize)}B.binary"
                )

                cmd = [
                    os.path.join(self.kenlm_dir, "bin", "build_binary"),
                    f"-q {self.config.quantize}",
                    f"-b {self.backoff}",
                    f"-a {self.config.offsets}",
                    self.config.model_type,
                    arpa_path,
                    binary_path,
                ]

                subprocess.run(cmd, check=True)
                model_path = binary_path
            else:
                model_path = arpa_path

            # Clean up temporary files
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)

            return model_path

        return train

    def __name__(self):
        return 'KenLMTrainer'


class SentencePieceTrainer(BasePipeline):
    """
    A trainer for SentencePiece models.
    """

    name: ClassVar[str] = "SentencePiece Trainer"
    type: ClassVar[str] = "TRAIN"
    config: SentencePieceTrainerConfig = None

    def __init__(
        self, config: SentencePieceTrainerConfig, run_config: RunConfig
    ):
        """
        Initialize the KenLM Trainer pipeline.

        Args:
            config: Configuration object containing all parameters
        """
        if run_config.executor_type == "local":
            run_config.num_workers = 1
        else:
            run_config.num_tasks = 1
            run_config.cpus_per_task = 1

        init_params = {
            **vars(run_config),
            "input_folder": config.input_folder,
            "output_folder": config.output_path,
        }
        super().__init__(**init_params)
        self.config = config

        # Create training pipeline
        pipeline = []
        pipeline.append(self._train_sentencepiece())
        self.add_pipelines((pipeline, self.__name__()))

    def _train_sentencepiece(self):
        """Create a training step for SentencePiece model."""

        def train(
            data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1
        ) -> str:
            import sentencepiece

            input_files = ",".join(
                [
                    os.path.join(self.config.input_folder, f)
                    for f in os.listdir(self.config.input_folder)
                    if f.startswith("preprocessed")
                ]
            )

            sentencepiece.SentencePieceTrainer.train(
                input=input_files,
                model_prefix=os.path.join(
                    self.config.output_path, self.config.model_prefix
                ),
                vocab_size=self.config.vocab_size,
                model_type=self.config.model_type,
                character_coverage=self.config.character_coverage,
            )

            return os.path.join(
                self.config.output_path, f"{self.config.model_prefix}.model"
            )

        return train

    def __name__(self):
        return 'SentencePieceTrainer'
