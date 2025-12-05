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

import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import ClassVar, Dict, List

import numpy as np
from datatrove.data import Document, DocumentsPipeline
from datatrove.io import cached_asset_path_or_download, get_datafolder
from datatrove.utils.logging import logger
from datatrove.utils.stats import MetricStatsDict
from datatrove.utils.text import TextNormConfig, simplify_text
from datatrove.utils.typeshelper import Languages
from huggingface_hub import hf_hub_url

from cerebras.modelzoo.data_preparation.data_curation.data_quality.pipeline.kenlm_perplexity_score.config import (
    DataConfig,
    RunConfig,
)
from cerebras.modelzoo.data_preparation.data_curation.data_quality.pipeline.kenlm_perplexity_score.pipeline import (
    BasePipeline,
)

MODEL_REPO = "edugp/kenlm"


class SentencePiece:
    def __init__(
        self,
        model_dataset: str = None,
        model_name: str = None,
        tokenizer_path: str = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.model_dataset = model_dataset
        self.tokenizer_path = tokenizer_path
        self._model = None

    @property
    def model(self):
        import sentencepiece

        # Check if either model_dataset and model_name or tokenizer_path are provided
        if not (
            (self.model_dataset is not None and self.model_name is not None)
            or (self.tokenizer_path is not None)
        ):
            raise ValueError(
                "Either (model_dataset and model_name) or (tokenizer_path) should be provided"
            )

        if self._model is None:
            # custom perpelexity for domains
            if self.tokenizer_path is not None:
                self._model = sentencepiece.SentencePieceProcessor(
                    self.tokenizer_path
                )
            # pre-trained perpelexity based on language
            else:
                path = cached_asset_path_or_download(
                    hf_hub_url(
                        MODEL_REPO,
                        str(
                            Path(
                                self.model_dataset,
                                f"{self.model_name}.sp.model",
                            )
                        ),
                    )
                )
                self._model = sentencepiece.SentencePieceProcessor()
                self._model.load(path)
        return self._model

    def tokenize(self, text: dict) -> dict:
        tokenized = self.model.encode_as_pieces(text)
        return " ".join(tokenized)


class KenlmModel:
    digit_re: re.Pattern = re.compile(r"\d")
    unicode_punct: Dict[str, str] = {
        "，": ",",
        "。": ".",
        "、": ",",
        "„": '"',
        "”": '"',
        "“": '"',
        "«": '"',
        "»": '"',
        "１": '"',
        "」": '"',
        "「": '"',
        "《": '"',
        "》": '"',
        "´": "'",
        "∶": ":",
        "：": ":",
        "？": "?",
        "！": "!",
        "（": "(",
        "）": ")",
        "；": ";",
        "–": "-",
        "—": " - ",
        "．": ". ",
        "～": "~",
        "’": "'",
        "…": "...",
        "━": "-",
        "〈": "<",
        "〉": ">",
        "【": "[",
        "】": "]",
        "％": "%",
        "►": "-",
    }
    unicode_punct_re = re.compile(f"[{''.join(unicode_punct.keys())}]")
    non_printing_chars_re = re.compile(
        f"[{''.join(map(chr, list(range(0,32)) + list(range(127,160))))}]"
    )

    def __init__(
        self,
        model_dataset: str = None,
        language: str = None,
        model_path: str = None,
        tokenizer_path: str = None,
    ):
        self.model_dataset = model_dataset
        self.language = language
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        import kenlm

        # Check if either model_dataset and language or model_path and tokenizer_path are provided
        if not (
            (self.model_dataset is not None and self.language is not None)
            or (self.model_path is not None and self.tokenizer_path is not None)
        ):
            raise ValueError(
                "Either (model_dataset and language) or (model_path and tokenizer_path) should be provided"
            )

        if self._model is None:
            # custom perpelexity for domains
            if self.model_path is not None and self.tokenizer_path is not None:
                self._model = kenlm.LanguageModel(self.model_path)

            # pre-trained perpelexity based on language
            else:
                model_path = Path(
                    self.model_dataset, f"{self.language}.arpa.bin"
                )
                path = cached_asset_path_or_download(
                    hf_hub_url(MODEL_REPO, str(model_path))
                )
                self._model = kenlm.Model(path)
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            if self.model_path is not None and self.tokenizer_path is not None:
                self._tokenizer = SentencePiece(
                    tokenizer_path=self.tokenizer_path
                )
            else:
                self._tokenizer = SentencePiece(
                    model_dataset=self.model_dataset, model_name=self.language
                )
        return self._tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_dataset: str,
        language: str,
    ):
        return cls(
            model_dataset,
            language,
        )

    def pp(self, log_score, length):
        return 10.0 ** (-log_score / length)

    def get_perplexity(self, doc: str, normalize_cc_net: bool = True):
        if normalize_cc_net:
            doc = self.normalize(
                doc,
            )
        # Tokenize (after normalizing): See https://github.com/facebookresearch/cc_net/blob/bda555bd1cf1ee2e0b925363e62a61cd46c8b60d/cc_net/mine.py#L352 for full pipeline
        doc = self.tokenizer.tokenize(doc)
        doc_log_score, doc_length = 0, 0
        for line in doc.split("\n"):
            log_score = self.model.score(line)
            length = len(line.split()) + 1
            doc_log_score += log_score
            doc_length += length
        return round(self.pp(doc_log_score, doc_length), 1)

    def normalize(
        self,
        text: str,
    ) -> str:
        text = simplify_text(
            text,
            config=TextNormConfig(
                lowercase=True,
                norm_numbers=True,
                norm_whitespace=False,
                remove_punctuation=False,
                norm_unicode_diacritics=True,
            ),
        )
        # TODO: integrate these options to simplify_text
        text = self.replace_unicode_punct(text)
        text = self.remove_non_printing_char(text)
        return text

    def replace_unicode_punct(self, text: str) -> str:
        return "".join(self.unicode_punct.get(c, c) for c in text)

    def remove_non_printing_char(self, text: str) -> str:
        return self.non_printing_chars_re.sub("", text)


class CCNetPerplexityStats(BasePipeline):
    """
    A pipeline for evaluating model perplexity using CCPerplexityNet.
    This pipeline runs in parallel to evaluate the model.
    """

    name: ClassVar[str] = "CCNet perplexity stats"
    _requires_dependencies: ClassVar[List[str]] = ["kenlm"]
    domain: str = "text"
    model: KenlmModel = None

    def __init__(
        self,
        run_config: RunConfig,
        data_config: DataConfig,
        model_dataset: str = None,
        language: str = Languages.english,
        model_path: str = None,
        tokenizer_path: str = None,
        domain: str = None,
    ) -> None:
        init_params = {**vars(run_config), **vars(data_config)}
        super().__init__(**init_params)

        self.domain = domain
        if not (
            (model_path is not None and tokenizer_path is not None)
            or (model_dataset is not None and language is not None)
        ):
            raise ValueError(
                "Either (model_path and tokenizer_path) or (model_dataset and language) must be provided"
            )

        logger.info(f"Initializing custom domain model from {model_path}")
        self.model = KenlmModel(
            model_path=model_path, tokenizer_path=tokenizer_path
        )

        # Create perplexity evaluation pipeline
        pipeline = []
        paths_file = None
        if os.path.exists(
            os.path.join(data_config.input_folder, "filenames.txt")
        ):
            paths_file = os.path.join(data_config.input_folder, "filenames.txt")

        pipeline.append(
            self._input_reader(
                data_config.input_folder,
                paths_file=paths_file,
                limit=self.limit,
                doc_progress=True,
                glob_pattern='*.*',
            )
        )
        pipeline.append(self._evaluate_perplexity())
        self.add_pipelines((pipeline, self.__name__()))

    def _evaluate_perplexity(self):
        """Create a perplexity evaluation step."""

        def evaluate(
            data: DocumentsPipeline, rank: int = 0, world_size: int = 1
        ) -> DocumentsPipeline:
            # Initialize stats dictionary
            groups_dicts = {
                "histogram": defaultdict(MetricStatsDict),
                "summary": defaultdict(MetricStatsDict),
            }

            for doc in data:
                if not isinstance(doc, Document):
                    logger.warning(f"Skipping non-Document object: {type(doc)}")
                    continue
                if not doc.text or not isinstance(doc.text, str):
                    logger.warning(
                        f"Skipping document with invalid text: {doc.id}"
                    )
                    continue

                try:
                    if (
                        self.model.model_path is not None
                        and self.model.tokenizer_path is not None
                    ):
                        if self.domain is None:
                            self.domain = "domain_specific"
                        perplexity = self.model.get_perplexity(doc.text)
                        doc.metadata[f"ccnet_perplexity_{self.domain}"] = (
                            perplexity
                        )
                    else:
                        perplexity = self.model.get_perplexity(doc.text)
                        doc.metadata[
                            f"ccnet_perplexity_{self.model.model_dataset}_{self.model.language}"
                        ] = perplexity

                    # Add to histogram stats
                    rounded_perplexity = round(perplexity, 3)
                    groups_dicts["histogram"]["perplexity"][
                        str(rounded_perplexity)
                    ] += 1

                    # Add to summary stats
                    groups_dicts["summary"]["perplexity_summary"][
                        "summary"
                    ] += perplexity

                    yield doc
                except Exception as e:
                    logger.error(
                        f"Error evaluating perplexity for document {doc.id}: {e}"
                    )
                    continue

            # Write stats to disk
            output_folder = get_datafolder(self.output_folder)
            for group, stats_dict in groups_dicts.items():
                for stat_name, stat_values in stats_dict.items():
                    with output_folder.open(
                        f"{group}/{stat_name}/{rank:05d}.json", "wt"
                    ) as f:
                        json.dump(stat_values.to_dict(), f)

        return evaluate

    def __name__(self):
        return 'CCNetPerplexityStats'


class CollatePerplexityStats(BasePipeline):
    """
    A pipeline for collating perplexity statistics from all ranks.
    This pipeline should be run after CCNetPerplexityStats has completed.
    """

    name: ClassVar[str] = "Collate perplexity stats"
    _requires_dependencies: ClassVar[List[str]] = []

    def __init__(
        self,
        run_config: RunConfig,
        data_config: DataConfig,
    ) -> None:
        # Set up for single task execution
        if run_config.executor_type == "local":
            run_config.num_workers = 1
        else:
            run_config.num_tasks = 1
            run_config.cpus_per_task = 1

        init_params = {**vars(run_config), **vars(data_config)}
        super().__init__(**init_params)

        # Create collation pipeline
        pipeline = []
        pipeline.append(self._collate_stats())
        self.add_pipelines((pipeline, self.__name__()))

    def _collate_stats(self):
        """Create a collation step."""

        def collate(
            data: DocumentsPipeline, rank: int = 0, world_size: int = 1
        ) -> None:
            # Only run on rank 0
            if rank != 0:
                yield
                return

            # Find all perplexity summary files
            pattern = f"{self.output_folder}/summary/perplexity_summary/*.json"
            stat_files = glob.glob(pattern)

            if not stat_files:
                logger.warning("No perplexity summary files found")
                yield
                return

            # Initialize aggregated stats
            total_perplexity = 0
            total_count = 0
            global_min = float('inf')
            global_max = float('-inf')

            # For Welford's online algorithm
            M2 = 0  # Sum of squares of differences from mean
            mean = 0

            # Aggregate statistics from all files
            for stat_file in stat_files:
                with open(stat_file, 'r') as f:
                    stat_data = json.load(f)
                    if "summary" in stat_data:
                        summary = stat_data["summary"]
                        file_total = summary.get("total", 0)
                        file_count = summary.get("n", 0)
                        file_mean = summary.get("mean", 0)
                        file_min = summary.get("min", float('inf'))
                        file_max = summary.get("max", float('-inf'))

                        # Update global min/max
                        global_min = min(global_min, file_min)
                        global_max = max(global_max, file_max)

                        # Update total and count
                        total_perplexity += file_total
                        total_count += file_count

                        # Update mean and M2 using Welford's online algorithm
                        delta = file_mean - mean
                        mean += delta * file_count / total_count
                        M2 += (
                            summary.get("variance", 0) * file_count
                            + delta
                            * delta
                            * file_count
                            * (total_count - file_count)
                            / total_count
                        )

            # Calculate final statistics
            if total_count > 0:
                variance = M2 / total_count if total_count > 1 else 0
                std_dev = np.sqrt(variance)

                final_stats = {
                    "summary": {
                        "total": total_perplexity,
                        "n": total_count,
                        "mean": total_perplexity / total_count,
                        "min": global_min,
                        "max": global_max,
                        "variance": variance,
                        "std_dev": std_dev,
                        "unit": "task",
                    }
                }

                # Write merged summary stats
                output_path = f"{self.output_folder}/summary/perplexity_summary/perplexity_summary_merged.json"
                with open(output_path, 'w') as f:
                    json.dump(final_stats, f)

                logger.info(f"Total perplexity: {total_perplexity:.2f}")
                logger.info(f"Total documents: {total_count}")
                logger.info(
                    f"Average perplexity: {total_perplexity/total_count:.2f}"
                )
                logger.info(f"Min perplexity: {global_min:.2f}")
                logger.info(f"Max perplexity: {global_max:.2f}")
                logger.info(f"Standard deviation: {std_dev:.2f}")

            yield

        return collate

    def __name__(self):
        return 'CollatePerplexityStats'
