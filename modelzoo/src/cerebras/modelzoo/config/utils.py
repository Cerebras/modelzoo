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

import inspect
from copy import deepcopy
from dataclasses import is_dataclass
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
)
from warnings import catch_warnings, simplefilter

from pydantic import create_model, model_serializer, model_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

from .base_config import BaseConfig


def create_config_class(
    cls: type,
    custom_type_mapping: Optional[Dict[type, type]] = None,
    **kwargs,
):
    """
    Dynamically construct a BaseConfig class from the signature of a class/function.

    This function will inspect the signature of the provided `cls` and use the annotations
    to create a Pydantic model.


    Args:
        cls: The class or function to create a config class for.
        custom_type_mapping: A dictionary mapping annotations to Pydantic types.
        **kwargs: Additional keyword arguments to pass to `create_model`.
    """

    custom_type_mapping = custom_type_mapping or {}

    def get_annotation(annotation, name="UNKNOWN"):
        if annotation is inspect._empty:
            return Any

        # If the annotation exists in the custom type mapping, return
        # the custom type that it maps to
        if annotation in custom_type_mapping:
            return custom_type_mapping[annotation]
        elif (annotation, name) in custom_type_mapping:
            return custom_type_mapping[(annotation, name)]

        # If the annotation has an origin, then recursively get the annotation
        # for its arguments
        origin = get_origin(annotation)
        if origin is not None:
            args = get_args(annotation)
            if origin is list:
                return List[tuple(map(get_annotation, args))]
            if origin is tuple:
                return Tuple[tuple(map(get_annotation, args))]
            if origin is dict:
                if len(args) == 0:
                    return Dict
                return Dict[tuple(map(get_annotation, args))]
            if origin in (Union, Optional):
                return origin[tuple(map(get_annotation, args))]

        # If the argument is a dataclass, we recursively generate a config class
        # so that we can properly type validate the dataclass arguments as well.
        elif is_dataclass(annotation):
            return pydantic_dataclass(
                annotation,
                config=dict(
                    extra="forbid",
                    arbitrary_types_allowed=True,
                    validate_default=True,
                    validate_assignment=True,
                    use_attribute_docstrings=True,
                    populate_by_name=True,
                ),
            )

        return annotation

    if isinstance(cls, type):
        parameters = dict(inspect.signature(cls.__init__).parameters)
        # TODO: What if the first argument is not called "self"?
        parameters.pop("self", None)
    else:
        parameters = dict(inspect.signature(cls).parameters)

    fields = {}
    config = {}

    for name, param in parameters.items():
        # Ignore variable positional arguments as config classes are required
        # to pass in keywords only
        if param.kind == param.VAR_POSITIONAL:
            continue
        # If variable keyword arguments are found, then allow the config classes
        # to arbitrarily accept and store all extra keys
        elif param.kind == param.VAR_KEYWORD:
            config["extra"] = "allow"
            continue

        # This is a limitation of pydantic as `model_config` is a reserved key
        if name == "model_config":
            raise ValueError(
                f"`model_config` is a reserved name. Please use a different "
                f"argument name in {cls}."
            )

        annotation = get_annotation(param.annotation, name)
        if annotation is not None:
            # Only include the field if the annotation is not None
            fields[name] = (
                annotation,
                (param.default if param.default is not inspect._empty else ...),
            )
        else:
            raise TypeError(
                f"Could not determine annotation for {name} in {cls}."
            )

    base_cls = kwargs.get("__base__", BaseConfig)

    # A subclass of the base config with additional validation and serialization
    # to handle the common case where the config is wrapped in a single field.
    class _BaseConfig(base_cls):
        model_config = config

        @model_validator(mode="before")
        @classmethod
        def wrap_config(cls, data):
            fields = list(cls.model_fields)

            data = deepcopy(data)

            if (
                len(fields) == 1
                and inspect.isclass(
                    annotation := cls.model_fields[fields[0]].annotation
                )
                and issubclass(annotation, BaseConfig)
                and (
                    isinstance(data, BaseConfig)
                    or (
                        isinstance(data, dict)
                        and (len(data) != 1 or fields[0] not in data)
                    )
                )
            ):
                return {fields[0]: data}

            return data

        @model_serializer(mode="wrap")
        def serialize_config(self, handler):
            serialized = handler(self)

            # Flatten the serialized config if it is a single field
            # that is a config
            fields = list(self.model_fields)
            if (
                len(fields) == 1
                and inspect.isclass(
                    annotation := self.model_fields[fields[0]].annotation
                )
                and issubclass(annotation, BaseConfig)
                and set(fields) == set(serialized)
            ):
                serialized = serialized[fields[0]]

            return serialized

    kwargs["__base__"] = _BaseConfig

    name = kwargs.pop("__name__", "{name}").format(name=cls.__name__)

    kwargs.setdefault("__module__", cls.__module__)

    try:
        with catch_warnings():
            simplefilter("ignore", category=RuntimeWarning)
            return create_model(
                name,
                __orig_class__=(ClassVar, cls),
                **fields,
                **kwargs,
            )
    except Exception as e:
        from pprint import pformat

        raise RuntimeError(
            f"Failed to create config with fields:\n"
            f"{pformat(fields, sort_dicts=False, width=1)}"
        ) from e


def parse_field_type(type_):
    from typing import get_args, get_origin

    if isinstance(type_, str):
        return f"'{type_}'"
    if get_origin(type_) == list:
        return f"List[{', '.join(map(parse_field_type, get_args(type_)))}]"
    if isinstance(type_, type):
        return type_.__name__

    return "\n".join(map(parse_field_type, get_args(type_)))


def describe_fields(cfg, show_deprecated=False):
    import re

    fields = []
    for name, field in cfg.model_fields.items():
        if field.deprecated and not show_deprecated:
            continue
        f_type = parse_field_type(field.annotation)

        if field.is_required():
            default = ""
        else:
            default = field.get_default()
            if isinstance(default, str):
                default = f"'{default}'"
            if default is None:
                default = getattr(field.default_factory, "__name__", "None")

        description = field.description
        # strip docstring hyperlinks formatting from the descriptions
        if description:
            description = re.sub(r"\[(.*)\]\(.*\)", r"\g<1>", description)

        fields.append(
            {
                "Name": name,
                "Type": f_type,
                "Default": default,
                "Description": description,
            }
        )

    return fields
