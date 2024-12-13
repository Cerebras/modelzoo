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

from inspect import isabstract
from typing import ClassVar, Literal, get_args, get_origin

from pydantic import BaseModel, model_validator

from .base_config import BaseConfig


class NamedConfig(BaseConfig):
    discriminator: ClassVar[str] = "name"

    @model_validator(mode="before")
    def check_literal_discriminator_field(cls, data):
        if isabstract(cls):
            return data

        # This is just a sanity check. It should always be a BaseModel subclass
        # as BaseConfig is a BaseModel subclass.
        if not issubclass(cls, BaseModel):
            raise TypeError(f"Expected Pydantic BaseModel. Got: {cls}")

        discriminator_field = cls.model_fields.get(cls.discriminator)
        if discriminator_field is None:
            raise TypeError(
                f"Expected {cls} to have a `{cls.discriminator}` field, "
                f"but found none."
            )

        if get_origin(discriminator_field.annotation) is not Literal:
            raise TypeError(
                f"Expected {cls}.{cls.discriminator} to be a Literal. "
                f"Got: {discriminator_field.annotation}"
            )

        if len(get_args(discriminator_field.annotation)) < 1:
            raise TypeError(
                f"Expected {cls}.{cls.discriminator} to have at least one valid literal value, "
                f"but found none."
            )

        # Make the first literal value the default so that users directly
        # constructing the config class via Python don't have to provide
        # the name.
        if discriminator_field.is_required() and cls.discriminator not in data:
            data.setdefault(
                cls.discriminator, get_args(discriminator_field.annotation)[0]
            )

        # It is not necessary to show the discriminator field in the repr.
        discriminator_field.repr = False

        return data

    @property
    def discriminator_value(self):
        return getattr(self, self.discriminator)
