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
from typing import List, Optional, Tuple

import pandas as pd
from tabulate import tabulate

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

    ╒═══════════════════════════════════════════════════════════════════════════════╤══════════════╤═══════╕
    │ Modules                                                                       │   Parameters │     % │
    ╞═══════════════════════════════════════════════════════════════════════════════╪══════════════╪═══════╡
    │ model.embedding_layer.word_embeddings.weight                                  │    6,432,896 │ 93.96 │
    ├───────────────────────────────────────────────────────────────────────────────┼──────────────┼───────┤
    │ model.embedding_layer.position_embeddings.embed.weight                        │       16,384 │  0.24 │
    ├───────────────────────────────────────────────────────────────────────────────┼──────────────┼───────┤
    │ model.ln_f.weight                                                             │          128 │  0.00 │
    ├───────────────────────────────────────────────────────────────────────────────┼──────────────┼───────┤
    │ model.ln_f.bias                                                               │          128 │  0.00 │
    ├───────────────────────────────────────────────────────────────────────────────┼──────────────┼───────┤
    │ model.transformer_decoder.all_layers.self_attn.proj_q_dense_layer.weight      │       32,768 │  0.48 │
    ├───────────────────────────────────────────────────────────────────────────────┼──────────────┼───────┤
    │ model.transformer_decoder.all_layers.self_attn.proj_q_dense_layer.bias        │          256 │  0.00 │
    ├───────────────────────────────────────────────────────────────────────────────┼──────────────┼───────┤
    │ model.transformer_decoder.all_layers.self_attn.proj_k_dense_layer.weight      │       32,768 │  0.48 │
    ├───────────────────────────────────────────────────────────────────────────────┼──────────────┼───────┤
    │ model.transformer_decoder.all_layers.self_attn.proj_k_dense_layer.bias        │          256 │  0.00 │
    ├───────────────────────────────────────────────────────────────────────────────┼──────────────┼───────┤
    │ model.transformer_decoder.all_layers.self_attn.proj_v_dense_layer.weight      │       32,768 │  0.48 │
    ├───────────────────────────────────────────────────────────────────────────────┼──────────────┼───────┤
    │ model.transformer_decoder.all_layers.self_attn.proj_v_dense_layer.bias        │          256 │  0.00 │
    ├───────────────────────────────────────────────────────────────────────────────┼──────────────┼───────┤
    │ model.transformer_decoder.all_layers.self_attn.proj_output_dense_layer.weight │       32,768 │  0.48 │
    ├───────────────────────────────────────────────────────────────────────────────┼──────────────┼───────┤
    │ model.transformer_decoder.all_layers.self_attn.proj_output_dense_layer.bias   │          256 │  0.00 │
    ├───────────────────────────────────────────────────────────────────────────────┼──────────────┼───────┤
    │ model.transformer_decoder.all_layers.norm1.weight                             │          256 │  0.00 │
    ├───────────────────────────────────────────────────────────────────────────────┼──────────────┼───────┤
    │ model.transformer_decoder.all_layers.norm1.bias                               │          256 │  0.00 │
    ├───────────────────────────────────────────────────────────────────────────────┼──────────────┼───────┤
    │ model.transformer_decoder.all_layers.norm3.weight                             │          256 │  0.00 │
    ├───────────────────────────────────────────────────────────────────────────────┼──────────────┼───────┤
    │ model.transformer_decoder.all_layers.norm3.bias                               │          256 │  0.00 │
    ├───────────────────────────────────────────────────────────────────────────────┼──────────────┼───────┤
    │ model.transformer_decoder.all_layers.ffn.ffn.0.linear_layer.weight            │      131,072 │  1.91 │
    ├───────────────────────────────────────────────────────────────────────────────┼──────────────┼───────┤
    │ model.transformer_decoder.all_layers.ffn.ffn.0.linear_layer.bias              │        1,024 │  0.01 │
    ├───────────────────────────────────────────────────────────────────────────────┼──────────────┼───────┤
    │ model.transformer_decoder.all_layers.ffn.ffn.1.linear_layer.weight            │      131,072 │  1.91 │
    ├───────────────────────────────────────────────────────────────────────────────┼──────────────┼───────┤
    │ model.transformer_decoder.all_layers.ffn.ffn.1.linear_layer.bias              │          256 │  0.00 │
    ╘═══════════════════════════════════════════════════════════════════════════════╧══════════════╧═══════╛
    """

    def __init__(
        self, search_and_replace: Optional[List[Tuple[str, str]]] = None
    ):
        """
        Args:
            search_and_replace: An optional list of search & replace to apply to
            parameter names. Each search & replace is a tuple containing a
            regex string for searching and a corresponding replacement string.
            For example, you can "group" parameters together across layers by
            using \.layers\.\d+\. for search and replace with "grouped_layers"
        """
        self._search_and_replace = search_and_replace

    def setup(self, trainer):
        output, df = self.get_table(trainer.model)

        trainer.log_metrics(parameter_counts=df)

        logging.info(output)

    def get_table(self, model):
        self.total_params, self.total_trainable_params, parameter_counts = (
            self.get_parameter_counts(model, self._search_and_replace)
        )

        header = ["Modules", "Parameters", "%"]
        table = [
            [name, float(count), float(count) / self.total_params * 100]
            for name, count in parameter_counts.items()
        ]

        df = pd.DataFrame(table, columns=header)

        out = (
            "\n"
            + tabulate(
                table,
                header,
                tablefmt="fancy_grid",
                floatfmt=(",.0f", ",.0f", ".2f"),
            )
            + "\n"
        )

        out += f"\nTotal Params (including frozen): {self.total_params:,}"
        out += f"\nTotal Trainable Params: {self.total_trainable_params:,}"

        return out, df

    def get_parameter_counts(self, model, search_and_replace=None):
        parameter_counts = {}
        total_params = 0
        total_trainable_params = 0
        for name, parameter in model.named_parameters():
            count = parameter.numel()
            total_params += count
            if not parameter.requires_grad:
                continue
            total_trainable_params += count

            group_name = name
            if search_and_replace:
                for search_regex, replace_str in search_and_replace:
                    match = re.search(search_regex, group_name)
                    if match:
                        group_name = (
                            group_name[: match.start()]
                            + replace_str
                            + group_name[match.end() :]
                        )
            if group_name not in parameter_counts:
                parameter_counts[group_name] = 0
            parameter_counts[group_name] += count

        return total_params, total_trainable_params, parameter_counts
