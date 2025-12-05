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
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def split_question_hook(
    example: Dict[str, Any], read_hook_kwargs
) -> Dict[str, Any]:
    """
    Description: This function is used to split the dataset into {question,id} dataset for each example

    return
        sda: {id: int, question: str}

    """
    question_key = read_hook_kwargs.get("question_key")
    sda = {'id': example['id'], 'question': example[question_key]}

    return sda


def split_context_hook(
    example: Dict[str, Any], read_hook_kwargs
) -> Dict[str, Any]:
    """
    return
        sda: {contexts: List[str]}
    """
    context_key = read_hook_kwargs.get("context_key")
    contexts = example[context_key]

    if isinstance(contexts, str):
        contexts = [contexts]

    sda = {'contexts': contexts}
    return sda


def chatqa_split_question_hook(
    example: Dict[str, Any], read_hook_kwargs
) -> Dict[str, Any]:
    """
    Description: This function is used to split the dataset into {question,id} dataset for each example

    return
        sda: {id: int, question: str}

    """
    sda = {'id': example['id'], 'question': example['messages'][0]['content']}

    return sda


def ques_embed_gen_hook(
    example: Dict[str, Any], **read_hook_kwargs
) -> List[Dict[str, Any]]:
    """
    Description: This function is used to generate embeddings for the question

    return
        sda: [{type: str, content: str}, {type: str, content: int}]
    """
    question_key = read_hook_kwargs.get("question_key")
    sda = [
        {"type": "embedding", "content": example[question_key]},
        {"type": "id", "content": int(example["id"])},
    ]
    return sda


def ctx_embed_gen_hook(
    example: Dict[str, Any], **read_hook_kwargs
) -> List[Dict[str, Any]]:
    """
    Description: This function is used to generate embeddings for the context

    return
        sda: [{type: str, content: str}, {type: str, content: int}]
    """
    context_key = read_hook_kwargs.get("context_key")
    sda = [
        {"type": "embedding", "content": example[context_key]},
        {"type": "id", "content": int(example["id"])},
    ]
    return sda
