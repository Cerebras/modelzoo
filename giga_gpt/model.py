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

import math
import re
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import GPT2Config as HFConfig
from transformers import GPT2LMHeadModel as HFModel


@dataclass
class GPTConfig:
    max_position_embeddings: int = 2048
    vocab_size: int = 50257  # CS is still fast with original vocab size, so no need to round up (see https://github.com/karpathy/nanoGPT/blob/master/model.py#L111)
    depth: int = 12
    heads: int = 12
    width: int = 768
    dropout: float = 0.0
    bias: bool = True
    init_std: float = 0.02


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=None):
        super(CausalSelfAttention, self).__init__()
        assert (
            not embed_dim % num_heads
        ), f"num_heads must divide embed_dim, got {num_heads} and {embed_dim}"
        self.num_heads = num_heads

        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.proj_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.proj_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout_layer = nn.Dropout(dropout)
        self.proj_output = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x, attn_mask):
        batch_size, seq_length, hidden_size = x.shape
        d = hidden_size // self.num_heads

        q = self.proj_q(x)
        q = q.view(batch_size, seq_length, self.num_heads, d).transpose(1, 2)
        k = self.proj_k(x)
        k = k.view(batch_size, seq_length, self.num_heads, d).transpose(1, 2)
        v = self.proj_v(x)
        v = v.view(batch_size, seq_length, self.num_heads, d).transpose(1, 2)

        q = q * torch.tensor(1 / float(d) ** 0.5, dtype=q.dtype)
        att = torch.matmul(q, k.transpose(-1, -2))
        att += attn_mask
        att = nn.functional.softmax(att.float(), dim=-1).type_as(att)
        att = self.dropout_layer(att)

        y = torch.matmul(att, v)
        y = y.transpose(1, 2).reshape(batch_size, seq_length, -1)
        y = self.proj_output(y)

        return y


class FFN(nn.Module):
    def __init__(self, h, bias, dropout):
        super(FFN, self).__init__()
        self.fc = nn.Linear(h, 4 * h, bias=bias)
        self.gelu = nn.functional.gelu
        self.proj = nn.Linear(4 * h, h, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.gelu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()

        self.attn = CausalSelfAttention(
            config.width, config.heads, config.dropout, config.bias,
        )
        self.ln_1 = nn.LayerNorm(config.width, eps=1e-5)
        self.ln_2 = nn.LayerNorm(config.width, eps=1e-5)
        self.ffn = FFN(config.width, config.bias, config.dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), attn_mask=mask)
        x = x + self.ffn(self.ln_2(x))
        return x


class GPTModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.width)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.width)
        self.drop_embd = nn.Dropout(config.dropout)
        self.decoder = nn.ModuleList(
            [Block(config) for _ in range(config.depth)]
        )
        self.ln_f = nn.LayerNorm(config.width, eps=1e-5)
        self.lm_head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.tie_weights()
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("proj_output.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=self.config.init_std / math.sqrt(2 * self.config.depth)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def tie_weights(self):
        self.lm_head.weight = self.wte.weight

    def _post_device_transfer(self):
        self.tie_weights()

    def forward(self, input_ids, labels=None):
        batch_size, sequence_length = input_ids.size()

        x = self.wte(input_ids)
        position_ids = torch.arange(
            sequence_length, device=input_ids.device
        ).expand((batch_size, -1))
        x += self.wpe(position_ids)
        x = self.drop_embd(x)

        causal_attention_mask = torch.triu(
            torch.ones(
                (sequence_length, sequence_length),
                dtype=x.dtype,
                device=x.device,
            ),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)
        causal_attention_mask *= torch.finfo(causal_attention_mask.dtype).min

        for l in self.decoder:
            x = l(x, mask=causal_attention_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if labels is None:
            return logits

        loss_fn = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fn(
            logits.view(-1, self.config.vocab_size), labels.view(-1).long()
        )
        loss = loss.sum() / (batch_size * sequence_length)
        loss = loss.to(logits.dtype)

        return loss

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    @staticmethod
    def load_ckpt_to_hf(state_dict, config):
        if not config.bias:
            raise ValueError(f"can only convert models with bias=True")

        hf_config = HFConfig(
            vocab_size=config.vocab_size,
            n_positions=config.max_position_embeddings,
            n_embd=config.width,
            n_layer=config.depth,
            n_head=config.heads,
            resid_pdrop=config.dropout,
            embd_pdrop=config.dropout,
            attn_pdrop=config.dropout,
        )

        weight_names = [k for k in state_dict]
        hf_state_dict = {}
        for name in weight_names:
            if name == "wte.weight":
                hf_state_dict["transformer.wte.weight"] = state_dict.pop(name)
            elif name == "wpe.weight":
                hf_state_dict["transformer.wpe.weight"] = state_dict.pop(name)
            elif name.startswith("ln_f"):
                hf_state_dict[f"transformer.{name}"] = state_dict.pop(name)
            elif name == "lm_head.weight":
                hf_state_dict[name] = state_dict.pop(name)
            elif re.match(r"decoder\.\d+\.attn\.proj_q\..*", name):
                i = re.findall(r"\d+", name)[0]
                t = name.split(".")[-1]
                k_name = name.replace("proj_q", "proj_k")
                v_name = name.replace("proj_q", "proj_v")
                weight = torch.cat(
                    (
                        state_dict.pop(name).t(),
                        state_dict.pop(k_name).t(),
                        state_dict.pop(v_name).t(),
                    ),
                    dim=-1,
                )
                hf_state_dict[f"transformer.h.{i}.attn.c_attn.{t}"] = weight
            elif re.match(r"decoder\.\d+\.attn\.proj_[kv]\..*", name):
                continue  # these get handled by the above condition
            elif re.match(r"decoder\.\d+\.attn\.proj_output\..*", name):
                i = re.findall(r"\d+", name)[0]
                t = name.split(".")[-1]
                hf_name = f"transformer.h.{i}.attn.c_proj.{t}"
                hf_state_dict[hf_name] = state_dict.pop(name).t()
            elif re.match(r"decoder\.\d+\.ln_[12]\..*", name):
                hf_name = name.replace("decoder", "transformer.h")
                hf_state_dict[hf_name] = state_dict.pop(name)
            elif re.match(r"decoder\.\d+\.ffn\.fc\..*", name):
                i = re.findall(r"\d+", name)[0]
                t = name.split(".")[-1]
                hf_name = f"transformer.h.{i}.mlp.c_fc.{t}"
                hf_state_dict[hf_name] = state_dict.pop(name).t()
            elif re.match(r"decoder\.\d+\.ffn\.proj\..*", name):
                i = re.findall(r"\d+", name)[0]
                t = name.split(".")[-1]
                hf_name = f"transformer.h.{i}.mlp.c_proj.{t}"
                hf_state_dict[hf_name] = state_dict.pop(name).t()
            else:
                raise RuntimeError(f"Unhandled weight {name}")

        if state_dict:
            raise RuntimeError(f"Leftover weights {state_dict.keys()}")
        hf_model = HFModel(hf_config)
        missing, unexpected = hf_model.load_state_dict(hf_state_dict, False)
        if unexpected:
            raise RuntimeError(
                f"Failed to load keys {unexpected} into hf model"
            )
        if any(
            not re.match(r"transformer\.h\.\d+\.attn\..*bias", k)
            for k in missing
        ):
            raise RuntimeError(
                f"Some necessary weights weren't supplied to the model. Found "
                f"missing keys {missing}, which contains keys that aren't "
                f"attention bias or attention masked_bias"
            )
        return hf_model, hf_config
