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

from functools import wraps
from typing import Any, Dict, List, Tuple, Union

from transformers import AutoTokenizer


class CustomLlama3Tokenizer:
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        eos_token_id: Union[int, None] = None,
        pad_token_id: Union[int, None] = None,
        **kwargs: Any,
    ):
        """
        Custom implementation of Llama3 Tokenizer, which overrides compute_offsets
        of the HuggingFace (which is buggy - https://github.com/huggingface/tokenizers/issues/1553).

        Args:
            pretrained_model_name_or_path (str): The pretrained model name or path.
            eos_token_id (Union[int, None], optional): The ID of the end-of-sequence token. Defaults to None.
            pad_token_id (Union[int, None], optional): The ID of the padding token. Defaults to None.
            **kwargs (Any): Additional keyword arguments to be passed to AutoTokenizer.

        Attributes:
            tokenizer (AutoTokenizer): The AutoTokenizer instance for the given pretrained model.
            eos_token_id (int): The ID of the end-of-sequence token.
            pad_token_id (int): The ID of the padding token.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        self.eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else self.tokenizer.eos_token_id
        )
        self.pad_token_id = (
            pad_token_id if pad_token_id is not None else self.eos_token_id
        )

    def compute_offsets(
        self, encoded: Dict[str, Any], return_offsets_mapping: bool = False
    ) -> List[Tuple[int, int]]:
        """
        Compute offsets for the given encoded input.

        Args:
            encoded (Dict[str, Any]): The encoded input containing 'input_ids' and 'offset_mapping'.
            return_offsets_mapping (bool, optional): Whether to return the offsets mapping. Defaults to False.

        Returns:
            List[Tuple[int, int]]: A list of tuples representing the start and end offsets for each token.
        """
        input_ids = encoded['input_ids']
        offset_mapping = encoded['offset_mapping']

        for i, (input_id, (start, end)) in enumerate(
            zip(input_ids, offset_mapping)
        ):
            token = self.tokenizer.convert_ids_to_tokens(input_id)
            if input_id not in self.tokenizer.all_special_ids:
                offset_mapping[i] = (start, start + len(token))

        return offset_mapping

    def __call__(self, text: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Encode the given text into tokens and optionally return the offsets mapping.

        Args:
            text (str): The input text to tokenize.
            **kwargs (Any): Additional keyword arguments for tokenization.

        Returns:
            Dict[str, Any]: The encoded result containing 'input_ids', 'attention_mask', and optionally 'offset_mapping'.
        """
        return_offset_mapping = kwargs.get('return_offsets_mapping')
        encoded = self.tokenizer(text, **kwargs)

        if return_offset_mapping:
            fixed_offsets = self.compute_offsets(
                encoded, return_offsets_mapping=return_offset_mapping
            )
            encoded['offset_mapping'] = fixed_offsets

        return encoded

    def __getattr__(self, name: str) -> Any:
        """
        Forward attribute access to the underlying tokenizer.

        Args:
            name (str): The name of the attribute to access.

        Returns:
            Any: The attribute value.
        """
        attr = getattr(self.tokenizer, name)
        if callable(attr):

            @wraps(attr)
            def wrapper(*args, **kwargs):
                return attr(*args, **kwargs)

            return wrapper
        return attr

    def __len__(self) -> int:
        """
        Get the vocabulary size of the tokenizer.

        Returns:
            int: The vocabulary size.
        """
        return self.tokenizer.vocab_size
