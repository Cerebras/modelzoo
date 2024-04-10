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

"""Cerebras ModelZoo Package."""
import os


def _register_paths_for_registry():
    """This loads all paths used by registry"""
    from cerebras.modelzoo.common.registry import registry

    modelzoo_path = os.path.dirname(os.path.realpath(__file__))

    registry.register_paths(
        "model_path", os.path.join(modelzoo_path, "models", "multimodal")
    )
    registry.register_paths(
        "model_path", os.path.join(modelzoo_path, "models", "nlp")
    )
    registry.register_paths(
        "model_path", os.path.join(modelzoo_path, "models", "vision")
    )
    registry.register_paths(
        "model_path", os.path.join(modelzoo_path, "models", "internal")
    )
    registry.register_paths("loss_path", os.path.join(modelzoo_path, "losses"))
    registry.register_paths(
        "datasetprocessor_path",
        os.path.join(modelzoo_path, "data", "multimodal"),
    )
    registry.register_paths(
        "datasetprocessor_path", os.path.join(modelzoo_path, "data", "nlp")
    )
    registry.register_paths(
        "datasetprocessor_path", os.path.join(modelzoo_path, "data", "vision")
    )
    registry.register_paths(
        "datasetprocessor_path", os.path.join(modelzoo_path, "data", "internal")
    )


_register_paths_for_registry()
