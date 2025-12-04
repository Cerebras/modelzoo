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
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union, get_args

import pandas as pd
import torch
from tabulate import tabulate

from cerebras.appliance.utils.units import bytes_to_human
from cerebras.modelzoo.trainer.callbacks import Callback


class CountParams(Callback):
    """Callback that runs on model setup for counting the number of parameters
    in a network.

    Along with printing the total number of parameters, it also prints out a table
    which shows the relative contribution (%) that each parameter has to the total
    count. Additionally, parameters can be grouped together to better see the
    relative contributions.

    For example, the following groups parameters across layers together using
    regex style search & replace:
    callbacks:
      - CountParams:
          search_and_replace: [["\\.layers\\.\\d+\\.", ".grouped_layers."]]

    ╒════════╤════════════════════════════════════════════════════════════════════════════════╤═══════════════╤═════════╤══════════════╤══════════╤══════════════╤═════════════╕
    │ Type   │ Name                                                                           │ Dtype         │ Shape   │ # Elements   │ Size     │   % of Total │ Trainable   │
    ╞════════╪════════════════════════════════════════════════════════════════════════════════╪═══════════════╪═════════╪══════════════╪══════════╪══════════════╪═════════════╡
    │ buffer │ perplexity_metric.total_loss                                                   │ torch.float32 │ ()      │ 1            │ 4.0B     │  1.46069e-05 │ False       │
    ├────────┼────────────────────────────────────────────────────────────────────────────────┼───────────────┼─────────┼──────────────┼──────────┼──────────────┼─────────────┤
    │ buffer │ perplexity_metric.total_num_tokens                                             │ torch.float32 │ ()      │ 1            │ 4.0B     │  1.46069e-05 │ False       │
    ├────────┼────────────────────────────────────────────────────────────────────────────────┼───────────────┼─────────┼──────────────┼──────────┼──────────────┼─────────────┤
    │ buffer │ accuracy_metric.total_correct_predictions                                      │ torch.float32 │ ()      │ 1            │ 4.0B     │  1.46069e-05 │ False       │
    ├────────┼────────────────────────────────────────────────────────────────────────────────┼───────────────┼─────────┼──────────────┼──────────┼──────────────┼─────────────┤
    │ buffer │ accuracy_metric.total_num_tokens                                               │ torch.float32 │ ()      │ 1            │ 4.0B     │  1.46069e-05 │ False       │
    ├────────┼────────────────────────────────────────────────────────────────────────────────┼───────────────┼─────────┼──────────────┼──────────┼──────────────┼─────────────┤
    │ group  │ model.embedding_layer.word_embeddings.w&b                                      │ N/A           │ N/A     │ 6,432,896    │ 24.5MiB  │ 93.9646      │ True        │
    ├────────┼────────────────────────────────────────────────────────────────────────────────┼───────────────┼─────────┼──────────────┼──────────┼──────────────┼─────────────┤
    │ group  │ model.embedding_layer.position_embeddings.embed.w&b                            │ N/A           │ N/A     │ 16,384       │ 64.0KiB  │  0.239319    │ True        │
    ├────────┼────────────────────────────────────────────────────────────────────────────────┼───────────────┼─────────┼──────────────┼──────────┼──────────────┼─────────────┤
    │ group  │ model.ln_f.w&b                                                                 │ N/A           │ N/A     │ 256          │ 1.0KiB   │  0.00373936  │ True        │
    ├────────┼────────────────────────────────────────────────────────────────────────────────┼───────────────┼─────────┼──────────────┼──────────┼──────────────┼─────────────┤
    │ group  │ model.transformer_decoder.grouped_layers.self_attn.proj_q_dense_layer.w&b      │ N/A           │ N/A     │ 33,024       │ 129.0KiB │  0.482378    │ True        │
    ├────────┼────────────────────────────────────────────────────────────────────────────────┼───────────────┼─────────┼──────────────┼──────────┼──────────────┼─────────────┤
    │ group  │ model.transformer_decoder.grouped_layers.self_attn.proj_k_dense_layer.w&b      │ N/A           │ N/A     │ 33,024       │ 129.0KiB │  0.482378    │ True        │
    ├────────┼────────────────────────────────────────────────────────────────────────────────┼───────────────┼─────────┼──────────────┼──────────┼──────────────┼─────────────┤
    │ group  │ model.transformer_decoder.grouped_layers.self_attn.proj_v_dense_layer.w&b      │ N/A           │ N/A     │ 33,024       │ 129.0KiB │  0.482378    │ True        │
    ├────────┼────────────────────────────────────────────────────────────────────────────────┼───────────────┼─────────┼──────────────┼──────────┼──────────────┼─────────────┤
    │ group  │ model.transformer_decoder.grouped_layers.self_attn.proj_output_dense_layer.w&b │ N/A           │ N/A     │ 33,024       │ 129.0KiB │  0.482378    │ True        │
    ├────────┼────────────────────────────────────────────────────────────────────────────────┼───────────────┼─────────┼──────────────┼──────────┼──────────────┼─────────────┤
    │ group  │ model.transformer_decoder.grouped_layers.norm1.w&b                             │ N/A           │ N/A     │ 512          │ 2.0KiB   │  0.00747873  │ True        │
    ├────────┼────────────────────────────────────────────────────────────────────────────────┼───────────────┼─────────┼──────────────┼──────────┼──────────────┼─────────────┤
    │ group  │ model.transformer_decoder.grouped_layers.norm3.w&b                             │ N/A           │ N/A     │ 512          │ 2.0KiB   │  0.00747873  │ True        │
    ├────────┼────────────────────────────────────────────────────────────────────────────────┼───────────────┼─────────┼──────────────┼──────────┼──────────────┼─────────────┤
    │ group  │ model.transformer_decoder.grouped_layers.ffn.ffn.0.linear_layer.w&b            │ N/A           │ N/A     │ 132,096      │ 516.0KiB │  1.92951     │ True        │
    ├────────┼────────────────────────────────────────────────────────────────────────────────┼───────────────┼─────────┼──────────────┼──────────┼──────────────┼─────────────┤
    │ group  │ model.transformer_decoder.grouped_layers.ffn.ffn.1.linear_layer.w&b            │ N/A           │ N/A     │ 131,328      │ 513.0KiB │  1.91829     │ True        │
    ╘════════╧════════════════════════════════════════════════════════════════════════════════╧═══════════════╧═════════╧══════════════╧══════════╧══════════════╧═════════════╛

    Total Params (including frozen): 6,846,084
    Total Bytes (including frozen): 26.1MiB
    Total Trainable Params: 6,846,080
    Total Trainable Bytes: 26.1MiB
    """

    _COLUMN_TYPE = Literal[
        "Type",
        "Name",
        "Dtype",
        "Shape",
        "# Elements",
        "Size",
        "% of Total",
        "Trainable",
    ]
    COLUMNS = get_args(_COLUMN_TYPE)

    def __init__(
        self,
        search_and_replace: Optional[List[Tuple[str, str]]] = None,
        order_by: Union[_COLUMN_TYPE, List[_COLUMN_TYPE], None] = None,
        descending: bool = False,
    ):
        """
        Args:
            search_and_replace: An optional list of search & replace to apply to
                parameter names. Each search & replace is a tuple containing a
                regex string for searching and a corresponding replacement string.
                For example, you can "group" parameters together across layers by
                using \.layers\.\d+\. for search and replace with "grouped_layers"
            order_by: Name of list of columns names to order the rows by.
            descending: Whether to sort in ascneding or descending order, when
                `order_by` is provided.
        """
        self._search_and_replace = search_and_replace
        self._order_by = order_by
        self._descending = descending

    def post_setup(self, trainer):
        output, df = self.get_table(trainer.model)

        trainer.log_metrics(parameter_counts=df)

        logging.info(output)

    def get_table(self, model: torch.nn.Module):
        self.model_info = self.get_model_info(model, self._search_and_replace)

        total_params = self.model_info.total_params
        table = [
            [
                param.typename,
                param.name,
                param.dtype if not isinstance(param, _GroupInfo) else "N/A",
                param.shape if not isinstance(param, _GroupInfo) else "N/A",
                f"{param.numel:,}",
                param.nbytes,
                float(param.numel) / total_params * 100,
                (
                    param.requires_grad
                    if isinstance(param.requires_grad, bool)
                    else "N/A"
                ),
            ]
            for param in self.model_info.parameters
        ]

        df = pd.DataFrame(table, columns=self.COLUMNS)
        if self._order_by:
            df.sort_values(
                self._order_by, inplace=True, ascending=not self._descending
            )
        df["Size"] = df["Size"].apply(lambda bytes: bytes_to_human(bytes))

        out = (
            "\n"
            + tabulate(
                df,
                headers="keys",
                tablefmt="fancy_grid",
                floatfmt=(",.0f", ",.0f", ".2f"),
                showindex=False,
            )
            + "\n"
        )

        out += f"\nTotal Params (including frozen): {total_params:,}"
        out += f"\nTotal Bytes (including frozen): {bytes_to_human(self.model_info.total_size)}"

        out += f"\nTotal Trainable Params: {self.model_info.total_trainable_params:,}"
        out += f"\nTotal Trainable Bytes: {bytes_to_human(self.model_info.total_trainable_size)}"

        return out, df

    def get_model_info(
        self,
        model: torch.nn.Module,
        search_and_replace: Optional[List[Tuple[str, str]]] = None,
    ) -> "_ModelInfo":
        params: Dict[str, Union[_ParameterInfo, _BufferInfo]] = {}
        for name, parameter in model.named_parameters():
            params[name] = _ParameterInfo(
                name=name,
                requires_grad=parameter.requires_grad,
                shape=tuple(parameter.shape),
                dtype=parameter.dtype,
                numel=parameter.numel(),
                nbytes=parameter.nbytes,
            )

        for name, buffer in model.named_buffers():
            params[name] = _BufferInfo(
                name=name,
                requires_grad=buffer.requires_grad,
                shape=tuple(buffer.shape),
                dtype=buffer.dtype,
                numel=buffer.numel(),
                nbytes=buffer.nbytes,
            )

        groups: Dict[str, _GroupInfo] = {}
        if search_and_replace:
            for name, param in params.items():
                substituted = 0
                group_name = param.name
                for search_regex, replace_str in search_and_replace:
                    group_name, num = re.subn(
                        search_regex, replace_str, group_name
                    )
                    substituted += num

                if substituted:
                    if group_name not in groups:
                        groups[group_name] = _GroupInfo(
                            name=group_name, parameters=[param]
                        )
                    else:
                        groups[group_name].parameters.append(param)

            for group in groups.values():
                for param in group.parameters:
                    del params[param.name]

        model_info = _ModelInfo(parameters=[])
        for param in params.values():
            model_info.parameters.append(param)
        for group in groups.values():
            model_info.parameters.append(group)

        return model_info

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass


@dataclass
class _ParameterInfo:
    name: str
    requires_grad: bool
    shape: Tuple[int]
    dtype: torch.dtype
    numel: int
    nbytes: int

    @property
    def typename(self) -> str:
        return "parameter"


@dataclass
class _BufferInfo(_ParameterInfo):
    @property
    def typename(self) -> str:
        return "buffer"


@dataclass
class _GroupInfo:
    name: str
    parameters: List[Union[_ParameterInfo, _BufferInfo]]

    @property
    def typename(self) -> str:
        return "group"

    @property
    def numel(self) -> int:
        return sum(p.numel for p in self.parameters)

    @property
    def nbytes(self) -> int:
        return sum(p.nbytes for p in self.parameters)

    @property
    def requires_grad(self) -> Optional[bool]:
        values = list({p.requires_grad for p in self.parameters})
        return values[0] if len(values) == 1 else None


@dataclass
class _ModelInfo:
    parameters: List[Union[_ParameterInfo, _BufferInfo, _GroupInfo]]

    @property
    def total_params(self) -> int:
        return sum(p.numel for p in self.parameters)

    @property
    def total_trainable_params(self) -> int:
        trainable = 0
        for p in self.parameters:
            if isinstance(p, (_ParameterInfo, _BufferInfo)):
                if p.requires_grad:
                    trainable += p.numel
            elif isinstance(p, _GroupInfo):
                for _p in p.parameters:
                    if _p.requires_grad:
                        trainable += _p.numel
            else:
                raise ValueError(f"Unknown parameter type: {type(p)}")
        return trainable

    @property
    def total_size(self) -> int:
        return sum(p.nbytes for p in self.parameters)

    @property
    def total_trainable_size(self) -> int:
        trainable = 0
        for p in self.parameters:
            if isinstance(p, (_ParameterInfo, _BufferInfo)):
                if p.requires_grad:
                    trainable += p.nbytes
            elif isinstance(p, _GroupInfo):
                for _p in p.parameters:
                    if _p.requires_grad:
                        trainable += _p.nbytes
            else:
                raise ValueError(f"Unknown parameter type: {type(p)}")
        return trainable
