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

try:
    from cerebras.tf.ws.ws_sparsify import (
        SPARSIFIER_MAP,
        get_AdamW_aware_sparsifier_class,
    )
except:

    class BaseSparsifier:
        """ This is an abstract class that can be passed to a training object
        to trigger sparsity. We use a class instead of a simple callback function
        to allow storage of specific state information

        Any particular Sparsifier should use this as a base class and implement
        the following functions.
        """

        def apply_sparsity(self, step):
            raise NotImplementedError

        def get_masked_weights(self, step, weights_dict, sparse_val):
            raise NotImplementedError

        def get_num_sparsified_values(self, v_name, sparse_val):
            raise NotImplementedError

    class DummySparsifier(BaseSparsifier):
        def __init__(self, *args, **kwargs):
            pass

        def apply_sparsity(self, step):
            return False

        def get_masked_weights(self, step, weights_dict, sparse_val):
            return weights_dict

        def get_num_sparsified_values(self, v_name, sparse_val):
            return 0

    def get_AdamW_aware_sparsifier_class(base_sparsifier):
        """
        Given a base sparsifier class, generate an associated AdamW-aware
        sparsifier class and return said class
        """
        return base_sparsifier

    class _sparse_map_class:
        def __init__(self, result):
            self.result = result

        def __getitem__(self, key):
            return self.result

        def __contains__(self, other):
            return True

    SPARSIFIER_MAP = _sparse_map_class(DummySparsifier)


__all__ = [
    "get_AdamW_aware_sparsifier_class",
    "SPARSIFIER_MAP" "BaseSparsifier",
]
