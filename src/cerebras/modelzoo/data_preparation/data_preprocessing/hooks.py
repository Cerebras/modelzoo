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
import re
from typing import Any, Dict, List

from cerebras.modelzoo.data_preparation.data_preprocessing.utils import (
    SYSTEM_PROMPT_REGISTRY,
)

logger = logging.getLogger(__name__)


def finetuning_llava_hook(
    example: Dict[str, Any], **read_hook_kwargs: Any
) -> List[Dict[str, Any]]:
    """
    Transforms conversation data for finetuning LLaVA.

    Args:
        example (Dict[str, Any]): The input data containing conversation and image paths.
        **read_hook_kwargs (Any): Additional keyword arguments containing data_keys, system_prompt, image_token, multi_turn_content_key, and phase.

    Returns:
        List[Dict[str, Any]]: Transformed data suitable for finetuning LLaVA.

    Raises:
        AssertionError: If required keys are not provided in read_hook_kwargs.
        ValueError: If image_token is not provided, or if there are multiple image tokens in the user's role, or if image tokens are found in the assistant's response.
    """

    data_keys = read_hook_kwargs.get("data_keys")
    assert (
        data_keys != None
    ), "data_keys should be provided in the read_hook_kwargs section"
    multi_turn_key = data_keys.get("multi_turn_key")
    image_key = data_keys.get("image_key")

    system_prompt = read_hook_kwargs.get("system_prompt")
    image_token = read_hook_kwargs.get("image_token", None)
    multi_turn_content_key = read_hook_kwargs.get(
        "multi_turn_content_key", "value"
    )
    phase = read_hook_kwargs.get("phase")
    assert (
        phase != None
    ), "phase should be provided in the read_hook_kwargs section for llava"
    assert (
        image_token != None
    ), "image_token should be provided in the read_hook_kwargs section for llava"

    conversation_data = example.get(multi_turn_key, [])
    if conversation_data is None:
        conversation_data = []
    image_path = example.get(image_key, None)
    transformed_data = []

    if system_prompt:
        system_prompt_text = SYSTEM_PROMPT_REGISTRY.get(system_prompt, "")
        system_data = {
            "type": "system",
            "content": [{"text": system_prompt_text.strip()}],
        }
        transformed_data.append(system_data)

    if image_path and not image_token:
        raise ValueError(
            "Image token has not been provided inside read_hook_kwargs within the processing section in the config file for llava finetuning datasets."
        )

    for i, turn in enumerate(conversation_data):

        semantic_drop_mask = []
        role = "user" if i % 2 == 0 else "assistant"
        content_parts = []
        if role == "user" and image_path:
            # Check for multiple image tokens in the user's role
            if turn[multi_turn_content_key].count(image_token) > 1:
                raise ValueError(
                    "Multiple image tokens found in user's role. Only one image token is allowed."
                )
            # Assume there's only one image token in the user's role
            parts = re.split(
                re.escape(image_token), turn[multi_turn_content_key]
            )
            if len(parts) == 2:
                # Add the image part before the text
                content_parts.append({"image": image_path})
                content_parts.append(
                    {"text": parts[0].strip() + parts[1].strip()}
                )
                if phase == 1:
                    semantic_drop_mask.extend([False, True])
                else:
                    semantic_drop_mask.extend([False, False])
            else:
                # No image token found, add the text as is
                content_parts.append({"text": turn[multi_turn_content_key]})
                if phase == 1:
                    semantic_drop_mask.extend([True])
                else:
                    semantic_drop_mask.extend([False])
        elif role == "assistant":
            # Check that no image tokens are present in the assistant's response
            if image_token and image_token in turn[multi_turn_content_key]:
                raise ValueError(
                    "Image tokens are not allowed in the assistant's response."
                )
            content_parts.append({"text": turn[multi_turn_content_key]})
            semantic_drop_mask.extend([False])

        transformed_data.append(
            {
                "type": role,
                "content": content_parts,
                "semantic_drop_mask": semantic_drop_mask,
            }
        )
    return transformed_data


def pretraining_image_captions_hook(
    example: Dict[str, Any], **read_hook_kwargs: Any
) -> List[Dict[str, Any]]:
    """
    Transforms image and caption data for pretraining.

    Args:
        example (Dict[str, Any]): The input data containing image and caption information.
        **read_hook_kwargs (Any): Additional keyword arguments containing data_keys.

    Returns:
        List[Dict[str, Any]]: Transformed data suitable for pretraining.

    Raises:
        AssertionError: If required keys are not provided in read_hook_kwargs.
    """

    data_keys = read_hook_kwargs.get("data_keys")
    assert (
        data_keys != None
    ), "data_keys should be provided in the read_hook_kwargs section"
    image_key = data_keys.get('image_key', None)
    caption_key = data_keys.get('caption_key', None)
    assert (
        image_key != None
    ), "pretraining_image_captions_hook requires a image_key"

    if isinstance(example.get(image_key), dict):
        ## datasets downloaded directly from huggingface come in this format
        return [
            {
                "content": [
                    {"image": example.get(image_key).get("path")},
                    {"text": example.get(caption_key)},
                ],
            }
        ]
    else:
        return [
            {
                "content": [
                    {"image": example.get(image_key)},
                    {"text": example.get(caption_key)},
                ],
            }
        ]


def text_read_hook(
    example: Dict[str, Any], **read_hook_kwargs: Any
) -> List[Dict[str, Any]]:
    """
    Transforms text data for reading.

    Args:
        example (Dict[str, Any]): The input data containing text information.
        **read_hook_kwargs (Any): Additional keyword arguments containing data_keys.

    Returns:
        List[Dict[str, Any]]: Transformed data suitable for reading.

    Raises:
        AssertionError: If required keys are not provided in read_hook_kwargs.
    """

    data_keys = read_hook_kwargs.get("data_keys")
    assert (
        data_keys != None
    ), "data_keys should be provided in the read_hook_kwargs section"
    text_key = data_keys.get('text_key', None)
    assert text_key != None, "text_read_hook requires a text_key"
    return [
        {
            "content": [
                {"text": example.get(text_key, "")},
            ],
        }
    ]


def nlg_read_hook(
    example: Dict[str, Any], **read_hook_kwargs: Any
) -> List[Dict[str, Any]]:
    """
    Transforms natural language generation (NLG) data for reading.

    Args:
        example (Dict[str, Any]): The input data containing NLG information.
        **read_hook_kwargs (Any): Additional keyword arguments containing data_keys.

    Returns:
        List[Dict[str, Any]]: Transformed data suitable for reading.

    Raises:
        AssertionError: If required keys are not provided in read_hook_kwargs.
    """

    data_keys = read_hook_kwargs.get("data_keys")
    assert (
        data_keys != None
    ), "data_keys should be provided in the read_hook_kwargs section"

    context_key = data_keys.get('context_key', None)
    completion_key = data_keys.get('completion_key', None)

    assert (
        context_key is not None and completion_key is not None
    ), "nlg_read_hook requires a context_key and a completion_key"

    return [
        {
            "type": "context",
            "content": [
                {"text": example.get(context_key, "")},
            ],
        },
        {
            "type": "completion",
            "content": [
                {"text": example.get(completion_key, "")},
            ],
        },
    ]


def prompt_completion_text_read_hook(
    example: Dict[str, Any], **read_hook_kwargs
) -> List[Dict[str, Any]]:
    """
    Process prompt and completion text data into a semantic_data_array format.

    Args:
        example (Dict[str, Any]): The example data to process.
        **read_hook_kwargs: Additional keyword arguments for processing.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries in semantic_data_array format.
    """

    data_keys = read_hook_kwargs.get("data_keys")
    assert (
        data_keys != None
    ), "data_keys should be provided in the read_hook_kwargs section"
    prompt_key = data_keys.get('prompt_key', None)
    completion_key = data_keys.get('completion_key', None)
    assert (
        prompt_key is not None and completion_key is not None
    ), "prompt_completion_read_hook requires a prompt_key and a completion_key"
    return [
        {
            "type": "prompt",
            "content": [
                {"text": example.get(prompt_key)},
            ],
        },
        {
            "type": "completion",
            "content": [
                {"text": example.get(completion_key)},
            ],
        },
    ]


def chat_read_hook(
    example: Dict[str, Any], **read_hook_kwargs: Any
) -> List[Dict[str, Any]]:
    """
    Transforms chat data for reading.

    Args:
        example (Dict[str, Any]): The input data containing chat messages.
        **read_hook_kwargs (Any): Additional keyword arguments containing data_keys.

    Returns:
        List[Dict[str, Any]]: Transformed data into semantic data array format.

    Raises:
        AssertionError: If required keys are not provided in read_hook_kwargs.
    """

    ## This api assumes dataset is in ChatML format
    data_keys = read_hook_kwargs.get("data_keys")
    assert (
        data_keys != None
    ), "data_keys should be provided in the read_hook_kwargs section"
    multi_turn_key = data_keys.get('multi_turn_key')
    assert (
        multi_turn_key is not None
    ), "multi_turn_chat_read_hook requires a multi_turn_key"
    conversation_data = example.get(multi_turn_key, [])
    content_key = read_hook_kwargs.get('multi_turn_content_key', "content")
    has_system_prompt = read_hook_kwargs.get('has_system_prompt', False)

    semantic_data_array = []
    if has_system_prompt:
        system_prompt = conversation_data.pop(0)
        semantic_data_array.append(
            {"type": "system", "content": [{"text": system_prompt}]}
        )

    for i, turn in enumerate(conversation_data):
        role = "user" if i % 2 == 0 else "assistant"
        content = turn.get(content_key)
        if content:
            ## Some tokenizer's like LLaMa 3 when applying chat template strip the user and assistant.
            ## The semantic region content should be in sync with the string obtained after applying chat template.
            content = content.strip()
        semantic_data_array.append(
            {"type": role, "content": [{"text": content}]}
        )

    return semantic_data_array


def dpo_read_hook(
    example: Dict[str, Any],
    **read_hook_kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Transforms data for the Direct Preference Optimization (DPO) task into a semantic data array format.

    Args:
        example (Dict[str, Any]): The input example data.
        **read_hook_kwargs (Any): Additional keyword arguments containing data_keys.

    Returns:
        List[Dict[str, Any]]: Transformed data suitable for the DPO task.

    Raises:
        AssertionError: If required keys are not provided in read_hook_kwargs.
    """

    data_keys = read_hook_kwargs.get("data_keys")
    assert (
        data_keys != None
    ), "data_keys should be provided in the read_hook_kwargs section"
    prompt_key = data_keys.get("prompt_key", None)
    chosen_key = data_keys.get("chosen_key", None)
    rejected_key = data_keys.get("rejected_key", None)
    assistant_role = read_hook_kwargs.get("assistant_role", "assistant:")
    input = []
    if isinstance(example, dict) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in example.items()
    ):
        if prompt_key:
            prompt = {}
            prompt['content'] = [{"text": example.get(prompt_key, "")}]
            prompt['type'] = "prompt"
            chosen = {}
            chosen['content'] = [{"text": example.get(chosen_key, "")}]
            chosen['type'] = "chosen"
            rejected = {}
            rejected['content'] = [{"text": example.get(rejected_key, "")}]
            rejected['type'] = "rejected"
            input.append(prompt)
            input.append(chosen)
            input.append(rejected)
        else:
            chosen_str = example.get(chosen_key, "")
            rejected_str = example.get(rejected_key, "")
            last_assistant_index = chosen_str.lower().rfind(assistant_role)
            if last_assistant_index == -1:
                logger.warning(
                    f"Can't determine prompt from the chosen string. No demarcation found. Skipping this doc..."
                )
                return []
            prompt_str = chosen_str[
                : last_assistant_index + len(assistant_role)
            ]
            chosen_str = chosen_str[
                last_assistant_index + len(assistant_role) :
            ]
            rejected_str = rejected_str[
                last_assistant_index + len(assistant_role) :
            ]
            prompt = {}
            prompt['content'] = [{"text": prompt_str}]
            prompt['type'] = "prompt"
            chosen = {}
            chosen['content'] = [{"text": chosen_str}]
            chosen['type'] = "chosen"
            rejected = {}
            rejected['content'] = [{"text": rejected_str}]
            rejected['type'] = "rejected"
            input.append(prompt)
            input.append(chosen)
            input.append(rejected)
    elif isinstance(example, dict) and all(
        isinstance(k, str) and isinstance(v, list) for k, v in example.items()
    ):
        chosen_list = example.get(chosen_key, None)
        assert chosen_list, "chosen list must be provided"
        rejected_list = example.get(rejected_key, None)
        assert rejected_list, "rejected list must be provided"
        # The only dataset available with list of dict has only
        # prompt and response entries hence the size is assumed
        # to be 2
        prompt_str = chosen_list[0]['content']
        chosen_str = chosen_list[1]['content']
        rejected_str = rejected_list[1]['content']
        prompt = {}
        prompt['content'] = [{"text": prompt_str}]
        prompt['type'] = "prompt"
        chosen = {}
        chosen['content'] = [{"text": chosen_str}]
        chosen['type'] = "chosen"
        rejected = {}
        rejected['content'] = [{"text": rejected_str}]
        rejected['type'] = "rejected"
        input.append(prompt)
        input.append(chosen)
        input.append(rejected)

    return input


def prompt_completion_chat_read_hook(
    example: Dict[str, Any], **read_hook_kwargs: Any
) -> List[Dict[str, Any]]:
    """
    Process prompt and completion data from a chat into a semantic_data_array format.

    Args:
        example (Dict[str, Any]): The example data to process.
        **read_hook_kwargs: Additional keyword arguments for processing.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries in semantic_data_array format.
    """

    data_keys = read_hook_kwargs.get("data_keys")
    assert (
        data_keys != None
    ), "data_keys should be provided in the read_hook_kwargs section"
    prompt_key = data_keys.get('prompt_key', None)
    completion_key = data_keys.get('completion_key', None)
    assert (
        prompt_key is not None and completion_key is not None
    ), "prompt_completion_chat_read_hook requires a prompt_key and a completion_key"

    return [
        {
            "type": "user",
            "content": [
                {
                    "text": (
                        example.get(prompt_key).strip()
                        if example.get(prompt_key)
                        else None
                    )
                },
            ],
        },
        {
            "type": "assistant",
            "content": [
                {
                    "text": (
                        example.get(completion_key).strip()
                        if example.get(completion_key)
                        else None
                    )
                },
            ],
        },
    ]


def finetuning_image_captions_hook(
    example: Dict[str, Any], **read_hook_kwargs
) -> List[Dict[str, Any]]:
    """
    Process finetuning image captions data into a semantic_data_array format.

    Args:
        example (Dict[str, Any]): The example data to process.
        **read_hook_kwargs: Additional keyword arguments for processing.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries in semantic_data_array format.
    """

    data_keys = read_hook_kwargs.get("data_keys")
    assert (
        data_keys != None
    ), "data_keys should be provided in the read_hook_kwargs section"
    image_key = data_keys.get('image_key', None)
    caption_key = data_keys.get('caption_key', None)
    assert (
        image_key != None
    ), "pretraining_image_captions_hook requires a image_key"

    if isinstance(example.get(image_key), dict):
        ## datasets downloaded directly from huggingface come in this format
        return [
            {
                "type": "prompt",
                "content": [
                    {"image": example.get(image_key).get("path")},
                ],
            },
            {
                "type": "completion",
                "content": [
                    {"text": example.get(caption_key)},
                ],
            },
        ]
    else:
        return [
            {
                "type": "prompt",
                "content": [
                    {"image": example.get(image_key)},
                ],
            },
            {
                "type": "completion",
                "content": [
                    {"text": example.get(caption_key)},
                ],
            },
        ]


def finetuning_llava_hook_prompt_completion(
    example: Dict[str, Any], **read_hook_kwargs: Any
) -> List[Dict[str, Any]]:
    """
    Transforms conversation data for finetuning LLaVA.

    Args:
        example (Dict[str, Any]): The input data containing conversation and image paths.
        **read_hook_kwargs (Any): Additional keyword arguments containing data_keys, system_prompt, image_token, multi_turn_content_key, and phase.

    Returns:
        List[Dict[str, Any]]: Transformed data suitable for finetuning LLaVA.

    Raises:
        AssertionError: If required keys are not provided in read_hook_kwargs.
        ValueError: If image_token is not provided, or if there are multiple image tokens in the user's role, or if image tokens are found in the assistant's response.
    """

    data_keys = read_hook_kwargs.get("data_keys")
    assert (
        data_keys != None
    ), "data_keys should be provided in the read_hook_kwargs section"
    multi_turn_key = data_keys.get("multi_turn_key")
    image_key = data_keys.get("image_key")

    system_prompt = read_hook_kwargs.get("system_prompt")
    image_token = read_hook_kwargs.get("image_token", None)
    multi_turn_content_key = read_hook_kwargs.get(
        "multi_turn_content_key", "value"
    )
    phase = read_hook_kwargs.get("phase", 1)
    assert (
        image_token != None
    ), "image_token should be provided in the read_hook_kwargs section for llava"

    conversation_data = example.get(multi_turn_key, [])
    if conversation_data is None:
        conversation_data = []
    image_path = example.get(image_key, None)
    transformed_data = []

    if system_prompt:
        system_prompt_text = SYSTEM_PROMPT_REGISTRY.get(system_prompt, "")
        system_data = {
            "type": "system",
            "content": [{"text": system_prompt_text.strip()}],
        }
        transformed_data.append(system_data)

    if image_path and not image_token:
        raise ValueError(
            "Image token has not been provided inside read_hook_kwargs within the processing section in the config file for llava finetuning datasets."
        )

    for i, turn in enumerate(conversation_data):

        semantic_drop_mask = []
        role = "prompt" if i % 2 == 0 else "completion"
        content_parts = []
        if role == "prompt" and image_path:
            # Check for multiple image tokens in the user's role
            if turn[multi_turn_content_key].count(image_token) > 1:
                raise ValueError(
                    "Multiple image tokens found in user's role. Only one image token is allowed."
                )
            # Assume there's only one image token in the user's role
            parts = re.split(
                re.escape(image_token), turn[multi_turn_content_key]
            )
            if len(parts) == 2:
                # Add the image part before the text
                content_parts.append({"image": image_path})
                content_parts.append(
                    {"text": parts[0].strip() + parts[1].strip()}
                )
                if phase == 1:
                    semantic_drop_mask.extend([False, True])
                else:
                    semantic_drop_mask.extend([False, False])
            else:
                # No image token found, add the text as is
                content_parts.append({"text": turn[multi_turn_content_key]})
                if phase == 1:
                    semantic_drop_mask.extend([True])
                else:
                    semantic_drop_mask.extend([False])
        elif role == "completion":
            # Check that no image tokens are present in the assistant's response
            if image_token and image_token in turn[multi_turn_content_key]:
                raise ValueError(
                    "Image tokens are not allowed in the completion's response."
                )
            content_parts.append({"text": turn[multi_turn_content_key]})
            semantic_drop_mask.extend([False])

        transformed_data.append(
            {
                "type": role,
                "content": content_parts,
                "semantic_drop_mask": semantic_drop_mask,
            }
        )

    return transformed_data
