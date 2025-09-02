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

import logging

import kenlm
from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep

logger = logging.getLogger(__name__)


class PerplexityScorer(PipelineStep):
    """
    Computes KenLM perplexity for each document and stores it in metadata["perplexity"].
    If model_path is not provided, this step is skipped with a warning.
    """

    name = "PerplexityScorer"

    def __init__(self, model_path: str = None, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.model = None
        self.warned_once = False  # To avoid repeated log spam

    def __getstate__(self):
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _ensure_model_loaded(self):
        if self.model is None and self.model_path:
            self.model = kenlm.Model(self.model_path)

    def compute_perplexity(self, text: str) -> float:
        text = text.strip().replace("\n", " ")
        if not text:
            return 1e9
        log_prob = self.model.score(text, bos=True, eos=True)
        length = len(text.split())
        return 10 ** (-log_prob / length) if length > 0 else 1e9

    def process(self, doc: Document) -> Document:
        if not self.model_path:
            if not self.warned_once:
                logger.warning(
                    "Skipping PerplexityScorer: no KenLM model_path provided."
                )
                self.warned_once = True
            return doc

        self._ensure_model_loaded()
        score = self.compute_perplexity(doc.text)
        doc.metadata["perplexity"] = float(score)
        return doc
