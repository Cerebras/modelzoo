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

import contextlib

try:
    try:
        import cerebras.framework.torch as cbtorch
        from cerebras.framework.torch import amp
        from cerebras.framework.torch.core import cb_model, modes, name_scope
        from cerebras.pb.stack.autogen_pb2 import AP_DISABLED
    except ImportError:
        import cerebras_pytorch as cbtorch
        from cerebras_pytorch import amp
        from cerebras_pytorch.core import cb_model, modes, name_scope
        from cerebras_pytorch.pb.stack.autogen_pb2 import AP_DISABLED
except ImportError:
    from types import SimpleNamespace

    from modelzoo.common.pytorch.utils import to_cpu

    amp = SimpleNamespace(get_init_params=lambda: {},)

    cbtorch = SimpleNamespace()

    cb_model = SimpleNamespace(
        use_cs=lambda: False,
        is_wse_device=lambda: False,
        is_master_ordinal=lambda: True,
        is_streamer=lambda: True,
        is_receiver=lambda: True,
        get_streaming_rank=lambda: 0,
        get_ordinal=lambda: 0,
        num_tasks=lambda: 1,
        num_receivers=lambda: 1,
        num_streamers=lambda: 1,
        step_closure=lambda f: f,
        to_cpu=to_cpu,
    )

    class modes:
        TRAIN = "train"
        EVAL = "eval"
        TRAIN_AND_EVAL = "train_and_eval"
        INFERENCE = "inference"

        @staticmethod
        def get_modes():
            return (
                modes.TRAIN,
                modes.EVAL,
                modes.TRAIN_AND_EVAL,
                modes.INFERENCE,
            )

        @staticmethod
        def is_valid(mode):
            return mode in modes.get_modes()

    @contextlib.contextmanager
    def name_scope(name: str):
        yield None
