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

    multi_turn_key = read_hook_kwargs.get("multi_turn_key")
    image_key = read_hook_kwargs.get("image_key")
    image_token = read_hook_kwargs.get("image_token")
    phase = read_hook_kwargs.get("phase")

    if multi_turn_key is None:
        raise ValueError(
            "multi_turn_key must be provided in read_hook_kwargs for LLaVA "
        )
    if image_key is None:
        raise ValueError(
            "image_key must be provided in read_hook_kwargs for LLaVA "
        )
    if phase is None:
        raise ValueError(
            "phase must be provided in read_hook_kwargs for LLaVA "
        )
    if image_token is None:
        raise ValueError(
            "image_token must be provided in read_hook_kwargs for LLaVA"
        )

    multi_turn_role_key = read_hook_kwargs.get("multi_turn_role_key", "from")
    multi_turn_content_key = read_hook_kwargs.get(
        "multi_turn_content_key", "value"
    )

    # Get conversation data and image path
    conversation_data = example.get(multi_turn_key, [])
    if conversation_data is None:
        conversation_data = []
    image_path = example.get(image_key)
    transformed_data = []
    # Process conversation turns
    for i, turn in enumerate(conversation_data):
        if turn.get(multi_turn_role_key) in ["human", "user"]:
            role = "user"
        elif turn.get(multi_turn_role_key) in ["gpt", "assistant"]:
            role = "assistant"
        elif turn.get(multi_turn_role_key) == "system":
            role = "system"
        else:
            raise ValueError("Invalid multi_turn_role_key.")
        content_parts = []
        semantic_drop_mask = []

        if role == "system":
            system_content = turn.get(multi_turn_content_key, "").strip()
            if system_content:
                content_parts.append({"text": system_content})
                semantic_drop_mask.append(False)
        elif role == "user":
            content = turn[multi_turn_content_key]
            parts = re.split(re.escape(image_token), content)

            if len(parts) > 2:
                raise ValueError(
                    "Multiple image tokens found in user's role. Only one image token is allowed."
                )

            # Add image part before the text if image token exists
            if len(parts) == 2:
                content_parts.append({"image": image_path})
                text = parts[0].strip() + parts[1].strip()
                if text != "":
                    content_parts.append({"text": text})
                    if phase == 1:
                        semantic_drop_mask.extend([False, True])
                    else:
                        semantic_drop_mask.extend([False, False])
                else:
                    semantic_drop_mask.append(False)
            else:
                # No image token, just add the text
                content_parts.append({"text": content.strip()})
                semantic_drop_mask.append(False)

        # Handle assistant's response (no image allowed)
        elif role == "assistant":
            content = turn[multi_turn_content_key]
            if image_token in content:
                raise ValueError(
                    "Image token found in assistant's response, which is not allowed."
                )
            content_parts.append({"text": content.strip()})
            semantic_drop_mask.append(False)

        # Append the transformed data for each turn
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

    image_key = read_hook_kwargs.get('image_key', None)
    caption_key = read_hook_kwargs.get('caption_key', None)
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

    text_key = read_hook_kwargs.get('text_key', None)
    assert text_key is not None, "text_read_hook requires a text_key"

    text_value = example.get(
        text_key, ""
    ).strip()  # Remove leading and trailing spaces

    return [
        {
            "content": [
                {"text": text_value},
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

    context_key = read_hook_kwargs.get('context_key', None)
    completion_key = read_hook_kwargs.get('completion_key', None)

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

    prompt_key = read_hook_kwargs.get('prompt_key', None)
    completion_key = read_hook_kwargs.get('completion_key', None)
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

    # This API assumes dataset is in ChatML format
    multi_turn_key = read_hook_kwargs.get('multi_turn_key', None)
    multi_turn_role_key = read_hook_kwargs.get('multi_turn_role_key', None)
    multi_turn_content_key = read_hook_kwargs.get(
        'multi_turn_content_key', None
    )

    assert (
        multi_turn_key is not None
    ), "multi_turn_chat_read_hook requires a multi_turn_key"

    assert (
        multi_turn_role_key is not None
    ), "multi_turn_chat_read_hook requires a multi_turn_role_key"

    assert (
        multi_turn_content_key is not None
    ), "multi_turn_chat_read_hook requires a multi_turn_content_key"

    conversation_data = example.get(multi_turn_key, [])
    if not conversation_data:
        return []
    semantic_data_array = []
    first_role = conversation_data[0].get(multi_turn_role_key)
    if first_role == "system":
        system_prompt = conversation_data[0].get(multi_turn_content_key)
        if system_prompt:
            semantic_data_array.append(
                {"type": "system", "content": [{"text": system_prompt}]}
            )
        conversation_data = conversation_data[1:]  # Remove system prompt

    # Checks to ensure there are equal pairs.
    if len(conversation_data) % 2 != 0:
        logger.warning(
            "Every user should have a corresponding assistant, skipping..."
        )
        return []
    else:
        # Checks to ensure that we don't have two consecutive messages by the same user.

        for index in range(0, len(conversation_data), 2):
            user_turn = conversation_data[index]
            assistant_turn = conversation_data[index + 1]

            user_role = user_turn.get(multi_turn_role_key)
            assistant_role = assistant_turn.get(multi_turn_role_key)

            if user_role == assistant_role:
                logger.warning(
                    "Two consecutive messages by the same participant is not allowed, skipping..."
                )
                return []

            user_content = user_turn.get(multi_turn_content_key)
            assistant_content = assistant_turn.get(multi_turn_content_key)

            if user_content:
                user_content = user_content.strip()

            if assistant_content:
                assistant_content = assistant_content.strip()

            semantic_data_array.append(
                {
                    "type": user_turn.get(multi_turn_role_key),
                    "content": [{"text": user_content}],
                }
            )
            semantic_data_array.append(
                {
                    "type": assistant_turn.get(multi_turn_role_key),
                    "content": [{"text": assistant_content}],
                }
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

    prompt_key = read_hook_kwargs.get("prompt_key", None)
    chosen_key = read_hook_kwargs.get("chosen_key", None)
    rejected_key = read_hook_kwargs.get("rejected_key", None)
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

    prompt_key = read_hook_kwargs.get('prompt_key', None)
    completion_key = read_hook_kwargs.get('completion_key', None)
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

    image_key = read_hook_kwargs.get('image_key', None)
    caption_key = read_hook_kwargs.get('caption_key', None)
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
    Transforms conversation data for finetuning LLaVA into SDA format.

    Args:
        example (Dict[str, Any]): The input data containing conversation and image paths.
        **read_hook_kwargs (Any): Additional keyword arguments including:
            - data_keys (Dict[str, str]): Dictionary specifying keys for multi-turn and image data.
            - image_token (str): The token used for images.
            - multi_turn_content_key (str, optional): Key to extract conversation content.
            - phase (int): The current phase of processing (1 or 2).

    Returns:
        List[Dict[str, Any]]: Transformed data in the SDA format.

    Raises:
        ValueError: If required data is missing or in an incorrect format.
    """

    # Get required keys from read_hook_kwargs

    multi_turn_key = read_hook_kwargs.get("multi_turn_key")
    image_key = read_hook_kwargs.get("image_key")
    image_token = read_hook_kwargs.get("image_token")
    multi_turn_role_key = read_hook_kwargs.get("multi_turn_role_key", "from")
    multi_turn_content_key = read_hook_kwargs.get(
        "multi_turn_content_key", "value"
    )
    phase = read_hook_kwargs.get("phase")
    if multi_turn_key is None:
        raise ValueError(
            "multi_turn_key must be provided in read_hook_kwargs for LLaVA "
        )
    if image_key is None:
        raise ValueError(
            "image_key must be provided in read_hook_kwargs for LLaVA "
        )
    if phase is None:
        raise ValueError(
            "phase must be provided in read_hook_kwargs for LLaVA "
        )
    if image_token is None:
        raise ValueError(
            "image_token must be provided in read_hook_kwargs for LLaVA"
        )

    # Get conversation data and image path
    conversation_data = example.get(multi_turn_key, [])
    if conversation_data is None:
        conversation_data = []
    image_path = example.get(image_key)
    transformed_data = []

    # Ensure image path is provided if image_token is present
    if not image_path:
        raise ValueError("Image path must be provided when image_token is used")

    # Process conversation turns
    for turn in conversation_data:
        if turn.get(multi_turn_role_key) in ["human", "user"]:
            role = "prompt"
        elif turn.get(multi_turn_role_key) in ["gpt", "assistant"]:
            role = "completion"
        else:
            raise ValueError(
                f"Invalid multi_turn_role_key: {turn.get(multi_turn_role_key)}"
            )

        content_parts = []
        semantic_drop_mask = []

        if role == "prompt":
            content = turn[multi_turn_content_key]
            parts = re.split(re.escape(image_token), content)

            if len(parts) > 2:
                raise ValueError(
                    "Multiple image tokens found in user's role. Only one image token is allowed."
                )

            # Add image part before the text if image token exists
            if len(parts) == 2:
                content_parts.append({"image": image_path})
                text = parts[0].strip() + parts[1].strip()
                if text != "":
                    content_parts.append({"text": text})
                    if phase == 1:
                        semantic_drop_mask.extend([False, True])
                    else:
                        semantic_drop_mask.extend([False, False])
                else:
                    semantic_drop_mask.append(False)
            else:
                # No image token, just add the text
                content_parts.append({"text": content})
        # Handle assistant's response (no image allowed)
        elif role == "completion":
            content = turn[multi_turn_content_key]
            if image_token in content:
                raise ValueError(
                    "Image token found in assistant's response, which is not allowed."
                )
            content_parts.append({"text": content})
            semantic_drop_mask.append(False)

        # Append the transformed data for each turn
        transformed_data.append(
            {
                "type": role,
                "content": content_parts,
                "semantic_drop_mask": semantic_drop_mask,
            }
        )

    return transformed_data
