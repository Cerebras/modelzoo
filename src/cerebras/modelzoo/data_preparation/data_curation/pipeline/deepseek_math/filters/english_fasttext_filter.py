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

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from fasttext import load_model


class EnglishFastTextFilter(BaseFilter):
    """
    Filter documents by eng content classification using FastText.
    """

    name = "EnglishFastTextFilter"

    def __init__(
        self,
        model_path: str,
        eng_threshold: float,
        eng_class_name: str,
        exclusion_writer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Don’t load the model yet—just remember where to find it
        self.model_path = model_path
        self.model = None

        self.eng_threshold = eng_threshold
        self.eng_class_name = eng_class_name
        self.exclusion_writer = exclusion_writer

    def __getstate__(self):
        # strip out the unpicklable model before sending to worker
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, state):
        # restore everything (model stays None)
        self.__dict__.update(state)

    def _ensure_model(self):
        if self.model is None:
            # each worker loads its own copy locally
            self.model = load_model(self.model_path)

    def filter(self, doc: Document) -> bool:
        # lazy‑load the model in this process
        self._ensure_model()

        # record that we saw a doc
        # self.stat_update("eng_input")
        # predict
        text = doc.text.replace("\n", " ")
        labels, probs = self.model.predict(text, k=-1)
        score = next(
            (p for lbl, p in zip(labels, probs) if lbl == self.eng_class_name),
            0.0,
        )
        doc.metadata["english_score"] = float(score)
        retain = score >= self.eng_threshold

        if retain:
            self.stat_update("english_retained")
        else:
            self.stat_update("english_filtered")
            if self.exclusion_writer:
                self.exclusion_writer.write(doc)

        return retain
