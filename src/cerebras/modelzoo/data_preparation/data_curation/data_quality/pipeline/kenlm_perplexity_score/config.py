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

from dataclasses import field
from typing import Dict, Optional

from datatrove.utils.typeshelper import Languages
from pydantic.dataclasses import dataclass


@dataclass
class RunConfig:
    num_workers: int = 1
    num_tasks: int = 1
    cpus_per_task: int = 1
    executor_type: str = 'local'
    start_method: str = 'fork'
    limit: int = -1


@dataclass
class DataConfig:
    input_folder: str
    output_folder: str
    input_format: str = "parquet"
    output_format: str = "text"
    text_column: str = "text"
    batch_size: int = 1024
    max_documents: Optional[int] = None


@dataclass
class PerplexityStatsConfig:
    # language specific parameters
    output_folder: str
    model_dataset: str = None
    language: str = Languages.english

    # domain specific parameters
    model_path: str = None
    tokenizer_path: str = None
    domain: str = None

    input_folder: str = None


@dataclass
class KenLMTrainerConfig:
    input_folder: str
    output_path: str = "./kenlm_models"
    kenlm_path: Optional[str] = None  # Path to pre-built kenlm repository
    order: int = 3  # Order of the language model
    memory: str = "40%"  # Memory usage limit
    temp_prefix: str = "kenlm_temp"
    vocab_estimate: int = 1000000  # Estimated vocabulary size
    model_type: str = "trie"
    intermediate: bool = False  # Write intermediate files
    verbose: bool = False  # Verbose output
    # Quantization options
    quantize: int = 0  # Quantization level (0 = no quantization)
    backoff: int = 0  # Backoff value, defaults to 0
    offsets: int = 255  # Maximum number of offsets


@dataclass
class SentencePieceTrainerConfig:
    input_folder: str
    output_path: str
    model_prefix: str
    vocab_size: int = 32000
    model_type: str = "bpe"
    character_coverage: float = 0.9995


@dataclass
class KenLMPipelineConfig:
    run_config: RunConfig
    data_config: DataConfig
    pipelines: Dict[str, Dict] = field(default_factory=dict)
