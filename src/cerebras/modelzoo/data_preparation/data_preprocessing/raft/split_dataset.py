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

import hashlib
import importlib
import logging
import os
from typing import Any, Callable, Dict, Tuple

from datasets import Dataset, load_dataset

from cerebras.modelzoo.data_preparation.data_preprocessing.raft.config_verifier import (
    DatasetConfig,
)
from cerebras.modelzoo.data_preparation.data_preprocessing.utils import (
    check_and_create_dir,
)

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def get_tokenizer(tokenizer_str: str, tokenizer_params: Dict[str, Any]) -> Any:
    """
    Load a tokenizer from the HuggingFace library.

    Args:
        tokenizer_str (str): The name of the tokenizer to load.
        tokenizer_params (Dict[str, Any]): Parameters to pass to the tokenizer.

    Returns:
        Any: The tokenizer object.
    """

    from transformers import AutoTokenizer

    tokenizer_params = {} if tokenizer_params is None else tokenizer_params
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_str,
        **tokenizer_params,
    )

    return tokenizer


class SplitDataset:
    def __init__(self, params: DatasetConfig):
        self.params = params
        self.context_hash_list = {}
        self.ctx_split_path = None
        self.ques_split_path = None

    def create_sha256_hash(self, input_string: str) -> str:
        '''
        Create SHA-256 hash of the input string
        '''

        # Create a SHA-256 hash object
        sha256_hash = hashlib.sha256()

        # Update the hash object with the bytes of the input string
        sha256_hash.update(input_string.encode('utf-8'))

        # Return the hexadecimal representation of the hash
        return sha256_hash.hexdigest()

    def update_global_unique_contexts(self, example: Dict[str, any]) -> None:
        '''
        Add the unique contexts to the context_hash_list dictionary
        '''

        query_id = example['id']
        contexts = example['context_list']

        for context_id, context in enumerate(contexts):

            id_tuple = (query_id, context_id)
            context_hash = self.create_sha256_hash(context)

            if context_hash in self.context_hash_list:
                self.context_hash_list[context_hash]['ids_list'].append(
                    id_tuple
                )
            else:
                self.context_hash_list[context_hash] = {
                    'ids_list': [id_tuple],
                    'content': context,
                }

    def get_hook_fn(self, hook_str: str) -> Callable:
        """
        Load a hook function from a given module and function name string.

        Args:
            hook_str (str): String in the format 'module:function'.

        Returns:
            Callable: The function object.
        """
        module_name, fn_name = hook_str.rsplit(':', 1)
        req_module = importlib.import_module(module_name)
        func = getattr(req_module, fn_name)
        return func

    def load_and_split(
        self,
        add_num_tokens: bool = False,
        tokenizer_str: str = None,
        tokenizer_params: Dict[str, Any] = None,
    ) -> Tuple[str, str]:
        """
        Load a dataset from HuggingFace or local source, split it into question and context datasets,
        and save them in the specified format.

        Args:
            add_num_tokens: boolean flag to check whether to add disctractor contexts wrt number of tokens or number of distractors.
            tokenizer_str: the string which defines the tokenizer required,currently only HF supported.
            tokenizer_params: extra params needed by tokenizer

        Returns:
            Tuple[str, str]: Paths to the directories containing the question and context datasets.
        """
        input_data_params = self.params.setup.copy()
        data_params = input_data_params.data

        load_type = data_params.type
        split_type = data_params.split
        source_dataset = data_params.source
        format_type = input_data_params.output_data_format

        if load_type == 'huggingface':
            cache_dir = data_params.cache_dir
            cache_dir = check_and_create_dir(cache_dir, split_type)
            data_params = {} if data_params is None else data_params

            if split_type is not None:
                dataset = load_dataset(
                    source_dataset,
                    split=split_type,
                    cache_dir=cache_dir,
                    **data_params,
                )
            else:
                raise KeyError("Split tag is not present.")
        elif load_type == 'local':
            dataset = load_dataset(
                'json' if format_type == 'jsonl' else format_type,
                data_files=source_dataset,
                split=split_type,
            )
        else:
            raise ValueError(
                "The 'type' key can only have values 'huggingface' or 'local'"
            )

        output_dir = input_data_params.output_dir

        contexts_fn_path = input_data_params.context_hook
        if contexts_fn_path is None:
            raise KeyError(
                'There is a function needed for extracting context from the dataset'
            )

        contexts_fn = self.get_hook_fn(contexts_fn_path)
        context_hook_kwargs = {
            'context_key': input_data_params.context_hook_kwargs.context_key
        }
        dataset = dataset.map(
            lambda example, id: {
                'id': id,
                'context_list': contexts_fn(example, context_hook_kwargs).get(
                    'contexts'
                ),
            },
            with_indices=True,
            num_proc=min(os.cpu_count(), 8),
        )

        file_path = os.path.join(output_dir, "data", f"dataset.{format_type}")
        if format_type == 'parquet':
            dataset.to_parquet(file_path)
        elif format_type == 'jsonl':
            dataset.to_json(file_path, orient='records', lines=True)
        else:
            raise ValueError(
                f"{format_type} is not supported by the data preprocessor."
            )
        logger.info(f"Dataset saved in {format_type} format at {file_path}\n")

        question_fn_path = input_data_params.question_hook
        if question_fn_path is None:
            raise KeyError(
                'There is a function needed for extracting question from the dataset'
            )

        question_dataset_dir = os.path.join(output_dir, 'question_split')
        question_dataset_file_path = os.path.join(
            question_dataset_dir, f'question_dataset.{format_type}'
        )

        self.ques_split_path = question_dataset_file_path

        question_hook_kwargs = {
            'question_key': input_data_params.question_hook_kwargs.question_key
        }
        question_fn = self.get_hook_fn(question_fn_path)
        question_dataset = dataset.map(
            lambda example: question_fn(example, question_hook_kwargs),
            remove_columns=dataset.column_names,
            num_proc=min(os.cpu_count(), 8),
        )

        context_dataset_dir = os.path.join(output_dir, 'context_split')
        context_dataset_file_path = os.path.join(
            context_dataset_dir, f'context_dataset.{format_type}'
        )

        self.ctx_split_path = context_dataset_file_path

        dataset.map(
            self.update_global_unique_contexts,
            remove_columns=dataset.column_names,
        )

        context_ids_list = []
        context_content_list = []

        for context_hash, context_dict in self.context_hash_list.items():
            context_ids_list.append(context_dict['ids_list'])
            context_content_list.append(context_dict['content'])

        data_dict = {
            'id': list(range(len(context_ids_list))),
            'global_ctx_id': context_ids_list,
            'context': context_content_list,
        }

        context_dataset = Dataset.from_dict(data_dict)

        if add_num_tokens:
            self.tokenizer = get_tokenizer(tokenizer_str, tokenizer_params)

            context_dataset = context_dataset.map(
                lambda x: {
                    'num_tokens': len(self.tokenizer(x['context'])['input_ids'])
                },
                num_proc=min(
                    os.cpu_count(), 8
                ),  # Adjust this value based on the number of cores available
            )
            logger.info(
                'Added the number of Tokens of Each Context to the context dataset\n'
            )

        if format_type == 'parquet':
            question_dataset.to_parquet(question_dataset_file_path)
            context_dataset.to_parquet(context_dataset_file_path)
        elif format_type == 'jsonl':
            question_dataset.to_json(question_dataset_file_path)
            context_dataset.to_json(context_dataset_file_path)
        else:
            raise ValueError(
                f"{format_type} is not supported by the data preprocessor."
            )

        logger.info(
            'Dataset split and saved as question and context datasets.\n\n'
        )
        return os.path.abspath(question_dataset_dir), os.path.abspath(
            context_dataset_dir
        )
