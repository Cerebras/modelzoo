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

    prompt_key = read_hook_kwargs.get('prompt_key', None)
    completion_key = read_hook_kwargs.get('completion_key', None)
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

    image_dir = read_hook_kwargs.get("image_dir")
    assert (
        image_dir != None
    ), "image_dir should be provided in the read_hook_kwargs section for obelics dataset"
    image_key = read_hook_kwargs.get('image_key', None)
    caption_key = read_hook_kwargs.get('caption_key', None)
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


def llama3_1_chat_formatted_data_hook(
    example: Dict[str, Any], **read_hook_kwargs
) -> List[Dict[str, Any]]:
    """
    Extract multi-turn conversation data from text formatted with Llama 3.1 chat template and process it into a semantic_data_array format.

    Args:
        example (Dict[str, Any]): The example data to process.
        **read_hook_kwargs: Additional keyword arguments for processing.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries in semantic_data_array format.
    """

    chat_key = read_hook_kwargs.get("chat_key")
    assert (
        chat_key is not None
    ), "llama3.1_chat_formatted_data_hook requires a chat_key"
    chat_formatted_data = example.get(chat_key)
    # Pattern to match header and content blocks
    pattern = (
        r'<\|start_header_id\|>(.*?)<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>'
    )

    # Find all matches in the text
    matches = re.finditer(pattern, chat_formatted_data, re.DOTALL)

    semantic_data_array = []

    for match in matches:
        role = match.group(1).strip()
        content = match.group(2).strip()
        semantic_data_array.append(
            {"type": role, "content": [{"text": content}]}
        )

    return semantic_data_array
