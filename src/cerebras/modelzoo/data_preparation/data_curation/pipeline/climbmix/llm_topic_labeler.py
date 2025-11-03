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

import json
import os
import time
from collections import defaultdict
from typing import Dict, List

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import logger
from openai import OpenAI


class LLMTopicLabeler(PipelineStep):
    name = "🏷 LLM Topic Labeling"
    type = "🏷 - Cluster Labeling"

    _requires_dependencies = ["openai"]

    def __init__(
        self,
        model: str,
        api_key: str,
        representatives_path: str,
        prompt_template: str,
        base_url: str = "https://api.cerebras.ai/v1",
        text_key: str = "text",
        label_key: str = "topic_label",
        cluster_key: str = "cluster_id",
        max_len_per_doc: int = 500,
    ):
        super().__init__()
        self.model = model
        self.prompt_template = prompt_template
        self.representatives_path = representatives_path
        self.base_url = base_url
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.text_key = text_key
        self.label_key = label_key
        self.cluster_key = cluster_key
        self.max_len_per_doc = max_len_per_doc
        self.client = None
        self.cluster_labels = {}

    def _load_representatives(self) -> Dict[int, List[Dict]]:
        """Load cluster representatives from a JSONL file and group them by cluster_id.

        Representatives are expected to be pre-sorted by rank in the JSONL file.
        """
        if not self.representatives_path or not os.path.exists(
            self.representatives_path
        ):
            logger.warning(
                f"Representatives file not found: {self.representatives_path}"
            )
            return {}

        # Group representatives by cluster_id (maintaining their order)
        cluster_representatives = defaultdict(list)

        try:
            with open(self.representatives_path, 'r', encoding='utf-8') as f:
                for line in f:
                    rep = json.loads(line)
                    cluster_id = rep.get("cluster_id")
                    if cluster_id is not None:
                        cluster_representatives[cluster_id].append(rep)

            logger.info(
                f"Loaded representatives for {len(cluster_representatives)} clusters from {self.representatives_path}"
            )
            return cluster_representatives

        except Exception as e:
            logger.error(f"Error loading representatives file: {e}")
            return {}

    def _label_clusters(
        self, cluster_representatives: Dict[int, List[Dict]]
    ) -> Dict[int, str]:
        """Generate topic labels for each cluster based on its representative documents.

        Issues one API call per cluster with rate limiting to avoid API limits.
        """
        if not cluster_representatives:
            return {}

        if self.client is None:
            if not self.api_key:
                raise ValueError("OpenAI API key is required")
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        cluster_labels = {}

        # Process each cluster
        for cluster_id, reps in cluster_representatives.items():
            try:
                # Combine text from multiple representatives
                combined_text = "\n\n---\n\n".join(
                    [
                        rep.get(self.text_key, "")[: self.max_len_per_doc]
                        for rep in reps
                    ]
                )

                # Create prompt for this cluster
                prompt = self.prompt_template.format(text=combined_text)

                # System prompt for cluster labeling
                system_prompt = (
                    "You are a topic classification assistant. Based on multiple representative "
                    "documents from a cluster, assign a single concise topic label (1-3 words) "
                    "that best captures the overall subject. Focus on subject matter, not document type."
                    "Respond with only the label."
                )

                # Make API call for this cluster
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )

                topic_label = response.choices[0].message.content.strip()

                if topic_label:
                    cluster_labels[cluster_id] = topic_label
                    logger.info(
                        f"Labeled cluster {cluster_id} with '{topic_label}'"
                    )
                    self.stat_update("labeled_clusters")
                else:
                    self.stat_update("skipped_clusters")

                # Rate limiting - add delay between requests
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error labeling cluster {cluster_id}: {e}")
                self.stat_update("failed_clusters")

        logger.info(f"Generated labels for {len(cluster_labels)} clusters")
        return cluster_labels

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        """Process documents and assign topic labels based on cluster membership.

        Uses pre-computed representative documents to label entire clusters at once,
        then applies those labels to all documents in the cluster.
        """
        # Load cluster representatives
        cluster_representatives = self._load_representatives()

        # Check if representatives are available
        if not cluster_representatives:
            logger.error(
                "No cluster representatives found. Please provide a valid representatives_path."
            )
            # Pass documents through unchanged
            yield from data
            return

        # Generate labels for each cluster
        self.cluster_labels = self._label_clusters(cluster_representatives)

        # Apply cluster labels to documents
        for doc in data:
            cluster_id = doc.metadata.get(self.cluster_key)
            if cluster_id is not None and cluster_id in self.cluster_labels:
                doc.metadata[self.label_key] = self.cluster_labels[cluster_id]
                self.stat_update("labeled_documents")
            else:
                self.stat_update("skipped_documents")
            yield doc
