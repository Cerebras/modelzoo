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

import pkgutil
from collections import Counter
from functools import cached_property, lru_cache, partial
from importlib import import_module
from pathlib import Path
from typing import List, Tuple

import yaml
from pydantic import (
    BaseModel,
    StringConstraints,
    field_validator,
    model_validator,
)
from typing_extensions import Annotated

_NonEmptyString = Annotated[str, StringConstraints(min_length=1)]


class _RegistrySchema(BaseModel):
    class Config:
        validate_assignment = True

    class ModelEntry(BaseModel):
        name: _NonEmptyString
        path: _NonEmptyString
        data_processor_paths: Tuple[_NonEmptyString, ...]

        @property
        def model_path(self) -> Path:
            module, _ = self.path.rsplit(".", 1)
            return Path(pkgutil.get_loader(module).get_filename()).parent

        @model_validator(mode="after")
        def validate_data_processor_paths(self, info):
            if len(self.data_processor_paths) == 0:
                extra_info = "."
                if (
                    info.context
                    and (fp := info.context.get("registry_file")) is not None
                ):
                    extra_info = f" in: {fp}"

                raise ValueError(
                    f"Expected at least one data processor path "
                    f"for model {self.name}.\n\n"
                    f"Please add at least one data processor path "
                    f"to the registry entry of {self.name}{extra_info}"
                )
            return self

    models: Tuple[ModelEntry, ...]

    @field_validator("models", mode="after")
    @classmethod
    def check_duplicates(cls, models, info):
        extra_info = "."
        if (
            info.context
            and (fp := info.context.get("registry_file")) is not None
        ):
            extra_info = f":\n{fp}"

        duplicates = {
            name: count
            for name, count in Counter(m.name for m in models).items()
            if count > 1
        }
        if duplicates:
            raise RuntimeError(
                f"Found duplicate model names:\n{duplicates}\n"
                f"Please ensure each model has a unique name in the registry"
                f"{extra_info}"
            )

        duplicates = {
            name: count
            for name, count in Counter(m.path for m in models).items()
            if count > 1
        }
        if duplicates:
            raise RuntimeError(
                f"Found duplicate model paths:\n{duplicates}\n"
                f"Please ensure each model has a unique path in the registry"
                f"{extra_info}"
            )

        return models


class _Registry:
    @property
    def registry_file(self):
        registry_file = (
            Path(pkgutil.get_loader("cerebras.modelzoo").get_filename()).parent
            / "registry/registry.yaml"
        )
        if not registry_file.exists():
            raise FileNotFoundError(f"Registry file not found: {registry_file}")
        return registry_file

    @cached_property
    def registry(self):
        with open(self.registry_file, "r") as f:
            return _RegistrySchema.model_validate(
                yaml.safe_load(f),
                context={"registry_file": self.registry_file},
            )

    @cached_property
    def model_registry(self):
        return {entry.name: entry for entry in self.registry.models}

    def invalidate(self):
        # Not clearing `self.registry` since models may have been
        # registered dynamically since load.
        self.__dict__.pop("model_registry", None)
        self.get_model_names.cache_clear()
        self.get_model_class.cache_clear()
        self.get_data_processors.cache_clear()

    def register_model(
        self, *, name: str, path: str, data_processor_paths: Tuple[str, ...]
    ):
        """Register a model to the registry.

        Args:
            name: Name of the model.
            path: dot-separated module/symbol model path. e.g., for registering class
                `Foo` from module `a.b.c`, the path should be `a.b.c.Foo`.
            data_processor_paths: List of dot-separated module/symbol dataprocessor
                paths for this model. Format of each item is the same as `path`.
        """
        self.registry.models = tuple(
            [
                *self.registry.models,
                _RegistrySchema.ModelEntry(
                    name=name,
                    path=path,
                    data_processor_paths=data_processor_paths,
                ),
            ]
        )
        self.invalidate()

    @lru_cache
    def get_model_names(self) -> List[str]:
        """Returns list of all currently registered models."""
        return sorted(self.model_registry.keys())

    def get_model(self, model_name: str) -> _RegistrySchema.ModelEntry:
        """Returns info about the registered model."""
        if model_name not in self.model_registry:
            model_list = '\n- '.join(self.get_model_names())
            raise ValueError(
                f"Model {model_name} not found in registry.\n\n"
                f"Expected the name to be one of:"
                f"\n- {model_list}\n\n"
                f"If {model_name} is a new model, "
                f"please add it to the registry file:\n{self.registry_file}"
            )
        return self.model_registry[model_name]

    @lru_cache
    def get_model_class(self, model_name: str):
        """Returns the model class for the given model name."""
        return self._import_class(self.get_model(model_name).path, model_name)

    @lru_cache
    def get_data_processors(self, model_name: str):
        """Returns a tuple of data processors for the given model."""
        return tuple(
            map(
                partial(self._import_class, name=model_name),
                self.get_model(model_name).data_processor_paths,
            )
        )

    def get_data_processors_map(self, model_name: str):
        """Returns the dict of data processors for the given model."""
        return {
            data_processor.__name__: data_processor
            for data_processor in self.get_data_processors(model_name)
        }

    def get_data_processor_names(self, model_name: str):
        """Returns data processor names for the given model."""
        return tuple(self.get_data_processors_map(model_name).keys())

    def get_model_path(self, model_name: str):
        """Returns the module path of the model."""
        return self.get_model(model_name).model_path

    def _import_class(self, path, name):
        module, class_name = path.rsplit(".", 1)
        try:
            module = import_module(module)
        except Exception as e:
            raise ImportError(
                f"Failed to import {class_name} from {module}:\n{e}\n"
                f"Please ensure the registry entry for {name} inside "
                f"{self.registry_file} is correct "
                f"and the module is available on the python path."
            ) from e

        if cls := getattr(module, class_name, None):
            return cls

        raise RuntimeError(
            f"Class {class_name} not found in module {module}. "
            f"Please ensure the registry entry for {name} inside "
            f"{self.registry_file} is correct."
        )


registry = _Registry()
