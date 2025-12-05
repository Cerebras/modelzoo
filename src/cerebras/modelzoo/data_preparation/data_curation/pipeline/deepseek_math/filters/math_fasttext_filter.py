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

import re
import unicodedata

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from fasttext import load_model
from nltk.tokenize import wordpunct_tokenize


def normalization(text: str) -> str:
    tokens = wordpunct_tokenize(text)
    processed_tokens = []
    for token in tokens:
        token = token.lower()
        if token.isdigit():
            processed_tokens.append("<NUM>")
        elif len(token) <= 100:
            processed_tokens.append(token)
    preprocessed_text = " ".join(processed_tokens)
    preprocessed_text = re.sub(r"[\n\r]+", " ", preprocessed_text)
    preprocessed_text = re.sub(r"[-_]+", " ", preprocessed_text)
    preprocessed_text = re.sub(r"[^a-zA-Z0-9\s<NUM>]", "", preprocessed_text)
    preprocessed_text = re.sub(r"\s+", " ", preprocessed_text).strip()
    return preprocessed_text


def preprocess_for_fasttext(text: str) -> str:
    if isinstance(text, (bytes, bytearray)):
        text = text.decode("utf-8")
    # Normalize unicode chars
    text = unicodedata.normalize("NFKC", text)
    # Collapse whitespace
    text = re.sub(r"\s", " ", text)
    # Mark end-of-sentence
    text = text.replace("\n", " <EOS> ")
    text = re.sub(r"\s+", " ", text)
    # Token-level normalization
    text = normalization(text)
    # Ensure no overly long segments
    MAX_LINE_SIZE = 1024
    lines = text.split("<EOS>")
    processed_lines = []
    for line in lines:
        tokens = line.split()
        if len(tokens) > MAX_LINE_SIZE:
            processed_lines.extend(
                [
                    " ".join(tokens[i : i + MAX_LINE_SIZE])
                    for i in range(0, len(tokens), MAX_LINE_SIZE)
                ]
            )
        else:
            processed_lines.append(line)
    return " <EOS> ".join(processed_lines).strip()


class MathFastTextFilter(BaseFilter):
    """
    Filter documents by math content classification using FastText.
    """

    name = "MathFastTextFilter"

    def __init__(
        self,
        model_path: str,
        math_threshold: float,
        math_class_name: str,
        exclusion_writer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Don’t load the model yet—just remember where to find it
        self.model_path = model_path
        self.model = None

        self.math_threshold = math_threshold
        self.math_class_name = math_class_name
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
        self.stat_update("math_input")

        # predict
        processed = preprocess_for_fasttext(doc.text)
        labels, probs = self.model.predict(processed, k=-1)
        score = next(
            (p for lbl, p in zip(labels, probs) if lbl == self.math_class_name),
            0.0,
        )
        doc.metadata["fasttext_math_score"] = float(score)
        retain = score >= self.math_threshold
        doc.metadata["math_fasttext"] = 1 if retain else 0
        if retain:
            self.stat_update("math_fasttext_retained")
        else:
            self.stat_update("math_fasttext_filtered")
            if self.exclusion_writer:
                self.exclusion_writer.write(doc)

        return True
