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

from typing import ClassVar, final
from warnings import warn

from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

_GLOBAL_CONFIG = None


class BaseConfig(BaseModel):
    """
    Base class for all config classes in the ModelZoo.

    This class is a thin wrapper around Pydantic's BaseModel class that adds some
    additional functionality and constraints to the models.
    """

    __orig_class__: ClassVar = None

    model_config = ConfigDict(
        # Forbid extra attributes during config initialization
        extra="forbid",
        # Make the model faux-immutable
        frozen=True,
        # Allow arbitrary types for field types
        arbitrary_types_allowed=True,
        # Always revalidate models during validation
        revalidate_instances="always",
        # Always validate default values during validation
        validate_default=True,
        # Always validate the return value from call validators
        validate_return=True,
        # Always validate assignments
        validate_assignment=True,
        # Use attribute docstrings for field descriptions
        use_attribute_docstrings=True,
        # Allow populating the model with the field names as well as the aliases
        populate_by_name=True,
    )

    def __call__(self, **kwargs):
        """
        Construct the original class with the current config.

        By original class, we mean the class that this config class is associated with.
        """
        if self.__orig_class__ is None:
            raise TypeError(
                f"{self.__class__.__name__} has no original class associated with it. "
            )

        u = self.get_orig_class_args(**kwargs)
        try:
            return self.__orig_class__(**u)
        except:
            from rich import print as pprint

            pprint(u)
            raise

    @classmethod
    def get_orig_class(cls):
        return cls.__orig_class__

    def get_orig_class_args(self, **kwargs) -> dict:
        fields = {
            k: kwargs.get(k, getattr(self, k))
            for k, f in self.model_fields.items()
        }
        if self.model_extra and self.model_config.get("include_extra", True):
            fields.update(self.model_extra)

        return fields

    def copy(self, *, validate: bool = True, **kwargs) -> Self:
        copy = super().model_copy(**kwargs)

        if "update" in kwargs and validate:
            return self.model_validate(copy)

        return copy

    def model_copy(self, **kwargs) -> Self:
        if kwargs.get("update"):
            warn(
                "model_copy() does not validate the data before creating the new model. "
                "If you trust this data, then ignore this warning. "
                "Otherwise, please call copy() instead."
            )
        return super().model_copy(**kwargs)

    @model_validator(mode="before")
    @classmethod
    def check_for_deprecated_fields(cls, data):
        if deprecated_fields := [
            name
            for name, field in cls.model_fields.items()
            if field.deprecated and name in data
        ]:
            warn(
                f"Found deprecated fields for {cls.__name__}: "
                f"{sorted(deprecated_fields)}\n"
                f"Support for passing these fields in will be removed in the future.",
                category=FutureWarning,
            )
        return data

    def post_init(self, context):
        pass

    @final
    def model_post_init(self, context):
        if context is None:
            context = {}

        orig = self.model_config["frozen"]
        try:
            self.model_config["frozen"] = False
            self.post_init(context)
        finally:
            self.model_config["frozen"] = orig
