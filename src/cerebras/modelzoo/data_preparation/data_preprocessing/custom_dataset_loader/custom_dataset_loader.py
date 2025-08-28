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

from datasets import (
    BuilderConfig,
    DatasetInfo,
    Features,
    GeneratorBasedBuilder,
    Split,
    SplitGenerator,
    Value,
)

logger = logging.getLogger("custom_dataset_loader")
logger.setLevel(logging.INFO)


class DatasetLoaderConfig(BuilderConfig):
    def __init__(self, selected_features=[], num_nodes=1, rank=0, **kwargs):
        super().__init__(**kwargs)
        self.selected_features = selected_features
        self.num_nodes = num_nodes
        self.rank = rank


class DatasetLoader(GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = DatasetLoaderConfig
    BUILDER_CONFIGS = [
        DatasetLoaderConfig(
            name="custom",
            description="Custom dataset loader",
            selected_features=[],
            num_nodes=1,
            rank=0,
        )
    ]

    def _info(self):
        selected_features = getattr(self.config, "selected_features", None)
        return DatasetInfo(
            features=Features({k: Value("string") for k in selected_features})
        )

    def _split_generators(self, dl_manager):
        data_files = self.config.data_files
        if isinstance(data_files, dict):
            data_files = data_files.get("train", [])

        downloaded_files = dl_manager.download(data_files)
        return [
            SplitGenerator(
                name=Split.TRAIN, gen_kwargs={"filepaths": downloaded_files}
            ),
        ]

    def _generate_examples(self, filepaths):
        selected_features = getattr(self.config, "selected_features", None)
        num_nodes = getattr(self.config, "num_nodes", 1)
        rank = getattr(self.config, "rank", 0)

        for filepath in filepaths:
            if filepath.endswith(".fasta"):
                yield from self._parse_fasta(
                    filepath, selected_features, num_nodes, rank
                )

    def _parse_fasta(self, filepath, selected_features, num_nodes=1, rank=0):
        seq_id_counter = 0
        with open(filepath, "r", encoding="utf-8") as f:
            seq_id, seq = None, []
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if seq_id:
                        if seq_id_counter % num_nodes == rank:
                            yield seq_id, {selected_features[0]: "".join(seq)}
                        seq_id_counter += 1
                    seq_id = line[1:]
                    seq = []
                else:
                    seq.append(line)
            if seq_id:
                if seq_id_counter % num_nodes == rank:
                    yield seq_id, {selected_features[0]: "".join(seq)}
