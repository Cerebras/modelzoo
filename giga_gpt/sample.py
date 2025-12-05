# Copyright 2023 Cerebras Systems.
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

import argparse

import tiktoken
import torch

import cerebras_pytorch as cstorch
from model import GPTConfig, GPTModel

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", required=True)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top_k", type=int, default=200)
parser.add_argument("--max_length", type=int, default=500)
parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
args = parser.parse_args()

state_dict = cstorch.load(args.checkpoint_path)
model_config = GPTConfig(**state_dict["model_config"])
model, hf_config = GPTModel.load_ckpt_to_hf(state_dict["model"], model_config)

model.eval()
if torch.cuda.is_available():
    if torch.cuda.is_bf16_supported():
        model.bfloat16()
    model.cuda()

tokenizer = tiktoken.get_encoding("gpt2")

with torch.no_grad():
    while prompt := input("Enter a prompt (RETURN to exit): "):
        input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
        response = model.generate(
            input_ids,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            max_length=args.max_length,
            eos_token_id=tokenizer.eot_token,
            pad_token_id=tokenizer.eot_token,
        ).cpu().squeeze().tolist()
        response = tokenizer.decode(response)
        print(f"Response: {response}")
