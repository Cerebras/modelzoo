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

import numpy as np
import torch
from torch.utils import data
from torchvision.io import read_image
from torchvision.utils import save_image

from cerebras.modelzoo.data.nlp.t5.t5_utils import pad_t5_input_features


class VisualQuestionAnsweringBase(data.Dataset):
    def __init__(
        self,
        data_obj,
        src_max_sequence_length,
        tgt_max_sequence_length,
        tokenizer,
        prepend_sos_token=True,
        append_eos_token=True,
        image_transforms=None,
        attn_mask_pad_id=0,
        sos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        extra_token="<extra_id_0>",
    ):
        """
        :param prepend_sos_token (bool): If True, inserts sos_token at the beginning of str
            before tokenization by tokenizer. Default: True. Set to True if `tokenizer.encode`
            does NOT handle this already.
        :param append_eos_token (bool): If True, inserts eos_token at the end of str
            before tokenization by tokenizer. Default: True. Set to False if `tokenizer.encode`
            handles this already.
        """
        self.data_obj = data_obj
        self.tokenizer = tokenizer
        # TODO: These asserts assume that the tokenizer is from
        # transformers and not Tokenizers HS package
        assert (
            tokenizer.eos_token == eos_token
        ), f"EOS token do not match between tokenizer and class input arg "
        self.eos_token = eos_token

        assert (
            tokenizer.pad_token == pad_token
        ), f"PAD token do not match between tokenizer and class input arg "
        self.pad_token = pad_token
        self.pad_id = tokenizer.pad_token_id

        # We leave it to users to ensure that sos token
        # gets tokenized into a single token by tokenizer.
        # Changing tokenizer to enforce sos_token would
        # changes to tokenizer vocab and embedding etc
        # which we'd like to avoid
        self.sos_token = sos_token

        self.prepend_sos_token = prepend_sos_token
        self.append_eos_token = append_eos_token

        self.src_max_sequence_length = src_max_sequence_length
        self.tgt_max_sequence_length = tgt_max_sequence_length

        self.attn_mask_pad_id = attn_mask_pad_id
        self.extra_token = extra_token

        self.image_transforms = image_transforms

    def __getitem__(self, index):

        features = self.data_obj[index]
        img_tensor = features.image  # shape=(C, H, W)
        if img_tensor is None:
            img_tensor = read_image(features.image_path)  # shape=(C, H, W)

        # Build encoder str, convert to ids and truncate to MSL
        encoder_input = self.build_encoder_input(features)

        # Build decoder str, convert to ids and truncate to MSL
        tgt_decoder_input = self.build_decoder_input(features)
        tgt_decoder_output = tgt_decoder_input[1:]

        # pad sequence to MSL
        features = {
            "input_ids": np.array(encoder_input, np.int32),
            "decoder_input_ids": np.array(tgt_decoder_input, np.int32),
            "labels": np.array(tgt_decoder_output, np.int32),
        }

        data_features = pad_t5_input_features(
            src_max_sequence_length=self.src_max_sequence_length,
            tgt_max_sequence_length=self.tgt_max_sequence_length,
            input_pad_id=self.pad_id,
            attn_mask_pad_id=self.attn_mask_pad_id,
            labels_pad_id=self.pad_id,
            features=features,
        )
        if self.image_transforms is not None:
            img_tensor = self.image_transforms(img_tensor)

        data_features["image_data"] = img_tensor

        return data_features

    def build_encoder_input(self, features):
        raise ValueError(f"Child class should implement this method")

    def build_decoder_input(self, features):
        raise ValueError(f"Child class should implement this method")

    def __len__(self):
        return len(self.data_obj)

    def display_sample(self, data_features):

        input_tokens = self.tokenizer.convert_ids_to_tokens(
            data_features["input_ids"].tolist()
        )
        decoder_input_tokens = self.tokenizer.convert_ids_to_tokens(
            data_features["decoder_input_ids"].tolist()
        )
        decoder_output_tokens = self.tokenizer.convert_ids_to_tokens(
            data_features["labels"].tolist()
        )

        img = data_features["image_data"]
        save_image(
            img.unsqueeze(0).to(torch.float32),
            f"vqa_{self.__class__.__name__}.jpg",
            nrow=1,
            normalize=True,
        )

        print(f"input_tokens: {input_tokens} \n")
        print(f"decoder_input_tokens: {decoder_input_tokens} \n")
        print(f"decoder_output_tokens: {decoder_output_tokens} \n")


class VisualQuestionAnswering(VisualQuestionAnsweringBase):
    def __init__(
        self,
        data_obj,
        src_max_sequence_length,
        tgt_max_sequence_length,
        tokenizer,
        prepend_sos_token=True,
        append_eos_token=True,
        image_transforms=None,
        attn_mask_pad_id=0,
        sos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        extra_token="<extra_id_0>",
    ):

        super().__init__(
            data_obj,
            src_max_sequence_length,
            tgt_max_sequence_length,
            tokenizer,
            prepend_sos_token,
            append_eos_token,
            image_transforms,
            attn_mask_pad_id,
            sos_token,
            eos_token,
            pad_token,
            extra_token,
        )

        # Prompt strings
        self._prompt_str = "Answer in {}: {}"

    def build_encoder_input(self, features):
        q = features.question

        # space necessary for single token output for sos token
        if self.prepend_sos_token:
            encoder_str = self.sos_token + " "
        else:
            encoder_str = ""
        encoder_str += (
            self._prompt_str.format(q.question_language, q.question)
            + f" {self.extra_token}"
        )

        # space necessary for single token output for eos token
        if self.append_eos_token:
            encoder_str += " " + self.eos_token

        encoder_input = self.tokenizer.encode(encoder_str)
        # To preserve EOS token after truncation to MSL
        if len(encoder_input) > self.src_max_sequence_length:
            encoder_input = (
                encoder_input[: self.src_max_sequence_length - 1]
                + encoder_input[-1:]
            )
        return encoder_input

    def build_decoder_input(self, features):

        ans = features.multiple_choice_answer

        # space necessary for single token output for sos token
        if self.prepend_sos_token:
            decoder_str = self.sos_token + " "
        else:
            decoder_str = ""
        decoder_str += ans

        # space necessary for single token output for eos token
        if self.append_eos_token:
            decoder_str += " " + self.eos_token

        decoder_input = self.tokenizer.encode(decoder_str)
        if len(decoder_input) > self.tgt_max_sequence_length:
            # To preserve EOS token after truncation to MSL
            decoder_input = (
                decoder_input[: self.tgt_max_sequence_length - 1]
                + decoder_input[-1:]
            )
        return decoder_input


class VisualQuestionGeneration(VisualQuestionAnsweringBase):
    def __init__(
        self,
        data_obj,
        src_max_sequence_length,
        tgt_max_sequence_length,
        tokenizer,
        prepend_sos_token=True,
        append_eos_token=True,
        image_transforms=None,
        attn_mask_pad_id=0,
        sos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        extra_token="<extra_id_0>",
    ):

        super().__init__(
            data_obj,
            src_max_sequence_length,
            tgt_max_sequence_length,
            tokenizer,
            prepend_sos_token,
            append_eos_token,
            image_transforms,
            attn_mask_pad_id,
            sos_token,
            eos_token,
            pad_token,
            extra_token,
        )

        # Prompt strings
        self._prompt_str = "Generate a question in {} for {}:"

    def build_encoder_input(self, features):
        ans = features.multiple_choice_answer
        lang = features.multiple_choice_answer_language

        # space necessary for single token output for sos token
        if self.prepend_sos_token:
            encoder_str = self.sos_token + " "
        else:
            encoder_str = ""

        encoder_str += (
            self._prompt_str.format(lang, ans) + f" {self.extra_token}"
        )

        # space necessary for single token output for eos token
        if self.append_eos_token:
            encoder_str += " " + self.eos_token

        encoder_input = self.tokenizer.encode(encoder_str)

        if len(encoder_input) > self.src_max_sequence_length:
            # To preserve EOS token after truncation to MSL
            encoder_input = (
                encoder_input[: self.src_max_sequence_length - 1]
                + encoder_input[-1:]
            )
        return encoder_input

    def build_decoder_input(self, features):

        q = features.question

        # space necessary for single token output for sos token
        if self.prepend_sos_token:
            decoder_str = self.sos_token + " "
        else:
            decoder_str = ""

        decoder_str += q.question

        # space necessary for single token output for eos token
        if self.append_eos_token:
            decoder_str += " " + self.eos_token

        decoder_input = self.tokenizer.encode(decoder_str)
        if len(decoder_input) > self.tgt_max_sequence_length:
            # To preserve EOS token after truncation to MSL
            decoder_input = (
                decoder_input[: self.tgt_max_sequence_length - 1]
                + decoder_input[-1:]
            )
        return decoder_input


if __name__ == "__main__":
    import random

    from transformers import AutoTokenizer

    from cerebras.modelzoo.data.vision.preprocessing import (
        get_preprocess_transform,
    )
    from cerebras.modelzoo.multimodal.pytorch.input.datasets import VQAv2

    transform_specs = [
        {"name": "to_dtype", "mp_type": torch.float32},
        {
            "name": "normalize",
            "mean": [127.5, 127.5, 127.5],
            "std": [127.5, 127.5, 127.5],
        },
        {"name": "resize", "size": [224, 224]},
    ]

    params = {"mixed_precision": True}
    params["transforms"] = transform_specs
    image_transforms = get_preprocess_transform(params)

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

    #### VQAv2 Dataset ####
    vqa_obj = VQAv2(
        data_dir="/cb/ml/multimodal_datasets/vqa_v2/VQA", split="validation"
    )
    print(vqa_obj)
    idx = random.randint(0, len(vqa_obj))
    features_dict = vqa_obj[idx]
    vqa_obj.display_sample(features_dict)

    print("".join(["-"] * 100))

    kwargs = {
        "data_obj": vqa_obj,
        "src_max_sequence_length": 256,
        "tgt_max_sequence_length": 256,
        "tokenizer": tokenizer,
        "prepend_sos_token": True,
        "append_eos_token": False,
        "image_transforms": image_transforms,
        "attn_mask_pad_id": 0,
        "sos_token": "<pad>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "extra_token": "<extra_id_0>",
    }

    #### VisualQuestionAnswering ####

    vqa_dataset = VisualQuestionAnswering(**kwargs)
    features = vqa_dataset[idx]
    vqa_dataset.display_sample(features)

    for k, v in features.items():
        print(f"{k} --- shape: {v.shape}")

    print("".join(["-"] * 100))

    #### VisualQuestionGeneration ####

    vqg_dataset = VisualQuestionGeneration(**kwargs)
    features = vqg_dataset[idx]
    vqg_dataset.display_sample(features)

    for k, v in features.items():
        print(f"{k} --- shape: {v.shape}")
