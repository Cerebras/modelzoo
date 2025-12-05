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

import copy

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from fasttext import load_model

from .math_fasttext_filter import preprocess_for_fasttext


class FastTextQualityFilter(BaseFilter):
    """
    Filter documents by content quality using a FastText classifier.
    """

    name = "FastTextQualityFilter"

    def __init__(
        self,
        model_path: str,
        quality_threshold: float = 0.0001,
        quality_label: str = "__label__hq",
        quality_key: str = "fasttext_quality_score",
        exclusion_writer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        #
        # We can use a FastText model to filter documents based on their quality.
        # refer to huggingface fastext model:
        # https://huggingface.co/mlfoundations/fasttext-oh-eli5/blob/main/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin
        # The model predicts whether a document is high quality or not.
        #
        self.model_path = model_path
        self.model = None
        self.quality_threshold = quality_threshold
        self.quality_label = quality_label
        self.quality_key = quality_key
        self.exclusion_writer = exclusion_writer

    def __getstate__(self):
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _ensure_model(self):
        if self.model is None:
            self.model = load_model(str(self.model_path))

    def filter(self, doc: Document) -> bool:
        if not doc.text:
            return False

        self._ensure_model()
        self.stat_update("quality_input")

        processed = preprocess_for_fasttext(copy.deepcopy(doc.text))
        labels, probs = self.model.predict(processed, k=-1)

        score = next(
            (p for lbl, p in zip(labels, probs) if lbl == self.quality_label),
            0.0,
        )
        doc.metadata[self.quality_key] = float(score)
        retain = score >= self.quality_threshold

        if retain:
            self.stat_update("text_quality_retained")
            return True
        else:
            self.stat_update("text_quality_filtered")
            if self.exclusion_writer:
                self.exclusion_writer.write(doc)
            return False
