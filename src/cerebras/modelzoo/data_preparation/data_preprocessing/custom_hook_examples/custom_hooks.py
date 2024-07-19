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
import io
import logging
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from PIL import Image

from cerebras.modelzoo.data_preparation.data_preprocessing.utils import (
    SYSTEM_PROMPT_REGISTRY,
)

logger = logging.getLogger(__name__)


def create_unique_hash(input_string):
    # Create a hash object
    hash_object = hashlib.sha256()
    # Update the hash object with the bytes of the input string
    hash_object.update(input_string.encode('utf-8'))
    # Get the hexadecimal digest of the hash
    hex_dig = hash_object.hexdigest()
    # Return the first 10 characters of the hexadecimal digest
    return hex_dig[:10]


def fetch_single_image(
    image_url: str, timeout: int = 10, retries: int = 0
) -> Optional[Image.Image]:
    """
    Fetches an image from the provided URL.

    Args:
        image_url (str): The URL of the image to download.
        timeout (int, optional): The maximum time in seconds to wait for the download to complete. Default is 10 seconds.
        retries (int, optional): The number of times to retry downloading the image in case of failure. Default is 0.

    Returns:
        Optional[Image.Image]: The downloaded image as a PIL Image object, or None if the download failed.
    """
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = Image.open(io.BytesIO(req.read()))
            break
        except Exception:
            image = None
    return image


def multiple_images_llava_finetuning_hook(
    example: Dict[str, Any], **read_hook_kwargs: Any
) -> List[Dict[str, Any]]:
    """
    Transforms conversation data for finetuning LLaVA with multiple images.

    Args:
        example (Dict[str, Any]): The input data containing conversation and image paths.
        **read_hook_kwargs (Any): Additional keyword arguments containing data_keys, system_prompt, and image_token.

    Returns:
        List[Dict[str, Any]]: Transformed data suitable for finetuning LLaVA.

    Raises:
        AssertionError: If required keys are not provided in read_hook_kwargs.
        ValueError: If image_token is not provided or if there are inconsistencies in image paths and tokens.
    """

    data_keys = read_hook_kwargs.get("data_keys")
    assert (
        data_keys is not None
    ), "data_keys should be provided in the read_hook_kwargs section"
    multi_turn_key = data_keys.get("multi_turn_key")
    image_key = data_keys.get("image_key")

    system_prompt = read_hook_kwargs.get("system_prompt")
    image_token = read_hook_kwargs.get("image_token", None)
    assert (
        image_token is not None
    ), "image_token should be provided in the read_hook_kwargs section for llava"

    conversation_data = example.get(multi_turn_key, [])
    if conversation_data is None:
        conversation_data = []
    image_paths = example.get(image_key, [])

    transformed_data = []

    if system_prompt:
        system_prompt_text = SYSTEM_PROMPT_REGISTRY.get(system_prompt, "")
        system_data = {
            "type": "system",
            "content": [{"text": system_prompt_text.strip()}],
        }
        transformed_data.append(system_data)

    if image_paths and not image_token:
        raise ValueError(
            "Image token has not been provided inside read_hook_kwargs within the processing section in the config file for llava finetuning datasets."
        )

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    image_index = 0

    for i, turn in enumerate(conversation_data):
        role = "user" if i % 2 == 0 else "assistant"
        content_parts = []

        # Split the text by the image token
        parts = re.split(re.escape(image_token), turn.get("value", ""))
        for i, part in enumerate(parts):
            if i > 0 and image_index < len(image_paths):
                # Use the corresponding image path or the single image path provided
                image_path = (
                    image_paths[image_index]
                    if len(image_paths) > 1
                    else image_paths[0]
                )
                content_parts.append({"image": image_path})
                image_index += 1
            if part.strip():  # Add non-empty text parts
                content_parts.append({"text": part})

        # Remove any empty text parts
        content_parts = [
            part
            for part in content_parts
            if part.get("text") or part.get("image")
        ]
        transformed_data.append({"type": role, "content": content_parts})

    return transformed_data


def ultra_chat_common_words_mask_hook(
    example: Dict[str, Any], **read_hook_kwargs
) -> List[Dict[str, Any]]:
    """
    Process common words mask data from an Ultra Chat dataset into a semantic_data_array format.

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
    prompt_text = example.get(prompt_key)
    completion_text = example.get("messages")[1]['content']

    COMMON_WORDS = set(
        ["the", "be", "to", "of", "and", "a", "in", "that", "have", "I", "am"]
    )

    def join_current_chunk(chunks, current_chunk):
        if chunks == []:
            ## don't append left space to the first chunk
            chunks.append(" ".join(current_chunk))
        else:
            ## append left space to the remaining chunks
            chunks.append(" " + " ".join(current_chunk))
        return chunks

    def split_text_by_common_words(
        text: str,
        common_words: set,
    ) -> List[str]:
        # Split the text into words
        words = text.split()
        # List to hold the resulting chunks
        chunks = []
        # Temporary list to build chunks
        current_chunk = []
        current_chunk_type = "uncommon-words"
        loss_mask = []
        for word in words:
            if word in common_words:
                if current_chunk_type == "common-words":
                    current_chunk.append(word)
                else:
                    if len(current_chunk) > 0:
                        join_current_chunk(chunks, current_chunk)
                        loss_mask.append(1)
                    current_chunk = [word]
                    current_chunk_type = "common-words"
            else:
                if current_chunk_type == "common-words":
                    if len(current_chunk) > 0:
                        join_current_chunk(chunks, current_chunk)
                        loss_mask.append(0)
                    current_chunk = [word]
                    current_chunk_type = "uncommon-words"
                else:
                    current_chunk.append(word)

        # Append any remaining words in the current chunk
        if current_chunk:
            join_current_chunk(chunks, current_chunk)
            if current_chunk_type == "common-words":
                loss_mask.append(0)
            else:
                loss_mask.append(1)

        return chunks, loss_mask

    completion_chunks, completion_loss_mask = split_text_by_common_words(
        completion_text,
        COMMON_WORDS,
    )
    return [
        {
            "type": "prompt",
            "content": [{"text": prompt_text}],
        },
        {
            "type": "completion",
            "content": [{"text": chunk} for chunk in completion_chunks],
            "semantic_loss_weight": completion_loss_mask,
        },
    ]


def obelics_hook(
    example: Dict[str, Any], **read_hook_kwargs
) -> List[Dict[str, Any]]:
    """
    Process obelics dataset examples into a semantic_data_array format.

    Args:
        example (Dict[str, Any]): The example data to process.
        **read_hook_kwargs: Additional keyword arguments for processing.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries in semantic_data_array format.
    """

    data_keys = read_hook_kwargs.get("data_keys")
    image_dir = read_hook_kwargs.get("image_dir")
    assert (
        data_keys != None
    ), "data_keys should be provided in the read_hook_kwargs section"
    assert (
        data_keys != None
    ), "image_dir should be provided in the read_hook_kwargs section for obelics dataset"
    image_key = data_keys.get('image_key', None)
    caption_key = data_keys.get('caption_key', None)
    assert image_key != None, "obelics_hook requires a image_key"
    image_url_list = example.get(image_key)
    text_list = example.get(caption_key)
    assert len(image_url_list) == len(
        text_list
    ), "The number of image samples should match the number of text samples in the obelics data"
    content_parts = []
    for image_url, text in zip(image_url_list, text_list):
        if image_url is None:
            assert text is not None, "Both text and image can't be none"
            content_parts.append({"text": text})
        elif text is None:
            assert image_url is not None, "Both text and image can't be none"
            parsed_url = urlparse(image_url)
            file_name = os.path.basename(parsed_url.path)
            if file_name == ".png" or file_name == "":
                continue
            unique_file_name = f"{create_unique_hash(file_name)}.png"
            image_path = os.path.join(image_dir, unique_file_name)
            if not os.path.isfile(image_path):
                image = fetch_single_image(image_url)
                if image:
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    image.save(image_path)
                    content_parts.append({"image": unique_file_name})

            else:
                content_parts.append({"image": unique_file_name})

    return [{"content": content_parts}]
