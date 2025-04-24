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
from pathlib import Path
from typing import Literal, Optional

import yaml
from annotated_types import Ge, Le
from pydantic import (
    ConfigDict,
    PositiveInt,
    field_serializer,
    field_validator,
    model_validator,
)
from typing_extensions import Annotated

from cerebras.modelzoo.config import BaseConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataConfig(BaseConfig):
    """
    Config Class for the input data
    """

    model_config = ConfigDict(
        extra='allow',
    )

    type: Literal['huggingface', 'local']
    source: str
    split: Optional[str] = 'train'
    cache_dir: Optional[Path] = None

    @field_serializer('cache_dir', mode='plain')
    def serialize_path(self, v: Path) -> str:
        return str(v)


class CtxHookConfig(BaseConfig):
    """
    Config Class for the context hook
    """

    context_key: str


class QuestionHookConfig(BaseConfig):
    """
    Config Class for the question hook
    """

    question_key: str


class SetupConfig(BaseConfig):
    """
    Config Class for splitting up the input dataset into context and question datasets
    """

    model_config = ConfigDict(frozen=False)

    # Define the fields
    data: "DataConfig"
    context_hook: str
    question_hook: str
    output_data_format: Literal['parquet', 'jsonl']
    processes: PositiveInt
    output_dir: Optional[Path] = None
    context_hook_kwargs: "CtxHookConfig"
    question_hook_kwargs: "QuestionHookConfig"

    # Serializer to output the Path as a string
    @field_serializer('output_dir', mode='plain')
    def serialize_path(self, v: Path) -> str:
        return str(v)


class DatasetConfig(BaseConfig):
    """
    Config Class for seting up the data
    """

    setup: SetupConfig


class RaftConfig(BaseConfig):
    """
    Config Class which contains all the information required for raft
    """

    model_config = ConfigDict(
        frozen=False,
    )

    context_key: str
    question_key: str
    answer_key: str
    id_key: str
    distractor_type: Literal['hard', 'random']
    context_removal_probability: Annotated[float, Ge(0), Le(1)]
    answer_refusal_percent: Annotated[float, Ge(0), Le(1)]
    data_dir: Optional[Path] = None
    ctx_data_dir: Optional[Path] = None
    output_dir: Path
    k: Optional[PositiveInt] = None
    num_tokens: Optional[PositiveInt] = None
    ctx_embedding: Path = ''
    ques_embedding: Path = ''
    ctx_to_ctx: bool = True
    split: str = 'train'

    @model_validator(mode='after')
    def check_k_or_num_tokens(self):
        if self.k is None and self.num_tokens is None:
            raise ValueError('Either k or num_tokens must be provided')

        if self.k is not None and self.num_tokens is not None:
            raise ValueError('Only one of k or num_tokens must be provided')

        return self

    @field_serializer(
        'data_dir',
        'output_dir',
        'ctx_data_dir',
        'ctx_embedding',
        'ques_embedding',
        mode='plain',
    )
    def serialize_path(self, v: Path) -> str:
        return str(v)


class ConfigVerifier(BaseConfig):
    """
    Main Config Class which verifies the config for raft and does some post-processings on the given config files.
    """

    dataset: DatasetConfig
    raft_config: RaftConfig
    context_embeddings: Path
    ctx_to_ctx: bool = True
    question_embeddings: Optional[Path] = None

    @model_validator(mode='after')
    def check_context_question_embeddings(self):
        q_emb_path = self.question_embeddings
        ctx_to_ctx = self.ctx_to_ctx
        if q_emb_path is None and not ctx_to_ctx:
            raise ValueError(
                'question_embeddings must be provided if ctx_to_ctx is False'
            )

        return self

    @field_validator('context_embeddings', 'question_embeddings', mode='after')
    def check_path_exists_for_embeddings(cls, path, info):
        if path is not None and not Path(path).exists():
            raise ValueError(f'{info.field_name} path does not exist')

        return path

    def post_init(self, context):
        super().post_init(context)
        base_path = self.raft_config.output_dir

        self.raft_config.data_dir = base_path / 'data'
        self.raft_config.ctx_data_dir = base_path / 'context_split'
        self.dataset.setup.output_dir = base_path

        if self.ctx_to_ctx and self.question_embeddings is not None:
            logger.info(
                'ctx_to_ctx is set to True, so question_embeddings will be ignored'
            )

        ## Saving the correct path in embedding files
        if (
            self.context_embeddings.is_file()
            and self.context_embeddings.suffix == '.yaml'
        ):
            logger.info(
                "Context embeddings config file is provided to generate context embeddings"
            )
            context_embeddings_config_file = self.context_embeddings

            with open(context_embeddings_config_file, 'r+') as file:
                context_embeddings_config = yaml.safe_load(file)

            context_embeddings_config['data']['setup']['data']['source'] = str(
                base_path / 'context_split'
            )
            context_embeddings_config['data']['setup']['output_dir'] = str(
                base_path / 'ctx_h5'
            )
            context_embeddings_config['dpr']['trainer']['init']['model_dir'] = (
                str(base_path / 'ctx_model_dir')
            )
            context_embeddings_config['dpr']['trainer']['init']['model'][
                'embeddings_output_dir'
            ] = (
                context_embeddings_config['dpr']['trainer']['init']['model_dir']
                + '/embeddings'
            )

            context_embeddings_config['dpr']['trainer']['validate'][
                'val_dataloader'
            ]['data_dir'] = context_embeddings_config['data']['setup'][
                'output_dir'
            ]

            logger.warning(
                'pad_last is set to True and drop_last is set to False in Context Embeddings Config File'
            )
            context_embeddings_config['dpr']['trainer']['validate'][
                'val_dataloader'
            ]['drop_last'] = False
            context_embeddings_config['dpr']['trainer']['validate'][
                'val_dataloader'
            ]['pad_last'] = True

            with open(context_embeddings_config_file, 'w') as file:
                yaml.safe_dump(context_embeddings_config, file)

        if (
            self.question_embeddings is not None
            and self.question_embeddings.is_file()
            and self.question_embeddings.suffix == '.yaml'
        ):
            logger.info(
                "Question embeddings config file is provided to generate question embeddings"
            )
            question_embeddings_config_file = self.question_embeddings
            with open(question_embeddings_config_file, 'r') as file:
                question_embeddings_config = yaml.safe_load(file)

            question_embeddings_config['data']['setup']['data']['source'] = str(
                base_path / 'question_split'
            )
            question_embeddings_config['data']['setup']['output_dir'] = str(
                base_path / 'ques_h5'
            )
            question_embeddings_config['dpr']['trainer']['init'][
                'model_dir'
            ] = str(base_path / 'ques_model_dir')

            question_embeddings_config['dpr']['trainer']['init']['model'][
                'embeddings_output_dir'
            ] = (
                question_embeddings_config['dpr']['trainer']['init'][
                    'model_dir'
                ]
                + '/embeddings'
            )

            question_embeddings_config['dpr']['trainer']['validate'][
                'val_dataloader'
            ]['data_dir'] = question_embeddings_config['data']['setup'][
                'output_dir'
            ]

            logger.warning(
                'pad_last is set to True and drop_last is set to False in Question Embeddings Config File'
            )
            question_embeddings_config['dpr']['trainer']['validate'][
                'val_dataloader'
            ]['drop_last'] = False
            question_embeddings_config['dpr']['trainer']['validate'][
                'val_dataloader'
            ]['pad_last'] = True

            with open(question_embeddings_config_file, 'w') as file:
                yaml.safe_dump(question_embeddings_config, file)

    @field_serializer('context_embeddings', 'question_embeddings', mode='plain')
    def serialize_path(self, v: Path) -> str:
        return str(v)
