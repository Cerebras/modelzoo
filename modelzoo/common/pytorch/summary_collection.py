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

import functools
import os
from collections import defaultdict

import numpy as np
import torch

import modelzoo.common.pytorch.utils as utils
from modelzoo.common.pytorch import cb_model as cm


class SummaryCollection:
    def __init__(self, outdir: str, model: torch.nn.Module):
        self._outdir = outdir
        self._model = model

        os.makedirs(self._outdir, exist_ok=True)
        self._module_hooks = []
        self._buffer = defaultdict(list)

    def __enter__(self):
        self.enable_collection()
        return self

    def __exit__(self, *exc):
        self.flush()
        self.disable_collection()

    def enable_collection(self):
        def add_name(module, name):
            module.name = name
            for cname, child in module.named_children():
                add_name(child, f"{name}.{cname}")

        def recurse(top_scope, output):
            for scope, tensor in utils.visit_structure(
                output,
                select_fn=lambda struct: isinstance(struct, torch.Tensor),
                scope=top_scope,
            ):
                yield ".".join(scope), cm.to_cpu(
                    tensor.detach()
                ).clone().numpy()

        def save_output(key, module, input, output):
            name = getattr(module, "name", None)
            if not name:
                ids = unnamed_modules[module.__class__.__name__]
                m_id = id(module)
                if m_id not in ids:
                    ids.append(m_id)
                name = f"{module.__class__.__name__}_{ids.index(m_id)}"

            if key == "bwd":
                # hook args are `grad_input, grad_output`, where grad_input is
                # the _gradient_ of the module's input i.e. the output of the
                # backward pass and the more interesting value to summarize
                # This way, the summary named `module.fwd` is the output of the
                # forward pass (i.e. txact), and `module.bwd`  is the output of
                # the backward pass (i.e. txdelta) for the corresponding kernel
                output = input
            for scope, tensor in recurse([name, key], output):
                self._buffer[scope].append(tensor)

        add_name(self._model, "model")
        unnamed_modules = defaultdict(list)

        self._module_hooks.extend(
            [
                torch.nn.modules.module.register_module_forward_hook(
                    functools.partial(save_output, "fwd")
                ),
                torch.nn.modules.module.register_module_backward_hook(
                    functools.partial(save_output, "bwd")
                ),
            ]
        )

    def disable_collection(self):
        for hook in self._module_hooks:
            hook.remove()
        self._module_hooks = []

    def flush(self):
        np.savez(
            os.path.join(self._outdir, "summaries.npz"),
            **{key: np.stack(values) for key, values in self._buffer.items()},
        )
        self._buffer.clear()
