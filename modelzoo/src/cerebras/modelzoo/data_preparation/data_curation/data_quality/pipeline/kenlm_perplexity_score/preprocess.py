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
import os
from typing import ClassVar, List, Optional

import nltk
from datatrove.data import Document, DocumentsPipeline
from datatrove.utils.logging import logger

from cerebras.modelzoo.data_preparation.data_curation.data_quality.pipeline.kenlm_perplexity_score.config import (
    DataConfig,
    RunConfig,
)
from cerebras.modelzoo.data_preparation.data_curation.data_quality.pipeline.kenlm_perplexity_score.pipeline import (
    BasePipeline,
)


class KenLMPreprocessor(BasePipeline):
    """
    A Pipeline for preprocessing input data for KenLM training.
    This pipeline takes Document objects from the reader, preprocesses them, and writes to txt files.
    """

    name: ClassVar[str] = "KenLM Preprocessor"
    type: ClassVar[str] = "PREPROCESS"
    _requires_dependencies: ClassVar[List[str]] = ["nltk"]

    # Instance attributes
    output_dir: str
    batch_size: int
    max_documents: Optional[int]

    def __init__(self, config: DataConfig, run_config: RunConfig):
        """
        Initialize the KenLM Preprocessor.

        Args:
            config: Configuration object containing all parameters
        """
        output_folder = config.output_folder or os.path.join(
            os.path.dirname(config.input_folder), "preprocessed"
        )
        init_params = {
            **vars(config),
            **vars(run_config),
            "input_folder": config.input_folder,
            "output_folder": output_folder,
            "output_dir": output_folder,  # Add output_dir to init params
            "batch_size": config.batch_size,
            "max_documents": config.max_documents,
        }
        super().__init__(**init_params)

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(
            f"Initializing preprocessor with output directory: {self.output_dir}"
        )
        # Make sure NLTK tokenizer models are downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt')

        # Create preprocessing pipeline
        pipeline = []
        pipeline.append(
            self._input_reader(
                self.input_folder,
                limit=self.limit,
                doc_progress=True,
                glob_pattern='*.*',
            )
        )
        pipeline.append(self._preprocess_documents())
        self.add_pipelines((pipeline, self.__name__()))

    def _preprocess_documents(self):
        """Create a preprocessing step that processes documents and writes them to files."""

        def preprocess(
            data: DocumentsPipeline, rank: int = 0, world_size: int = 1
        ) -> DocumentsPipeline:
            if not data:
                logger.warning("No input data provided")
                return (doc for doc in [])
            logger.info(f"Worker {rank}/{world_size} starting preprocessing")
            # Create output file for this worker
            output_path = os.path.join(
                self.output_dir, f'preprocessed_{rank:05d}.txt'
            )
            documents_processed = 0
            logger.info(f"Output path: {output_path}")
            with open(output_path, 'w', encoding='utf-8') as out_f:
                for i, doc in enumerate(data):
                    if not isinstance(doc, Document):
                        logger.warning(
                            f"Skipping non-Document object: {type(doc)}"
                        )
                        continue
                    if not doc.text or not isinstance(doc.text, str):
                        logger.warning(
                            f"Skipping document with invalid text: {doc.id}"
                        )
                        continue
                    # Process the document
                    try:
                        # Tokenize into sentences and words
                        for sent in nltk.sent_tokenize(doc.text):
                            tokens = nltk.word_tokenize(sent)
                            # Write preprocessed text to file
                            out_f.write(' '.join(tokens).lower() + '\n')

                        documents_processed += 1
                        # Create a new Document with preprocessed text
                        processed_doc = Document(
                            id=doc.id,
                            text='\n'.join(
                                nltk.sent_tokenize(doc.text)
                            ),  # Keep original sentences
                            metadata={
                                **doc.metadata,
                                "preprocessed": True,
                                "worker": rank,
                            },
                        )
                        yield processed_doc
                    except Exception as e:
                        logger.error(f"Error processing document {doc.id}: {e}")
                        continue
            logger.info(
                f"Worker {rank}/{world_size} completed processing {documents_processed} documents"
            )

        return preprocess

    def __name__(self):
        return 'KenLMPreprocessor'
