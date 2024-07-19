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

"""
Base implementation for config classes with helper modules

"""

import ast
import logging
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Callable, Optional, Union, get_args, get_type_hints

from cerebras.modelzoo.config_manager.config_validators import (
    get_origin,
    validate_field_type,
)

# Alias required as an empty object to check for values that are mandatory and not provided
required = object()


def config_field(
    default: Any = required,
    constraint: Optional[Callable[..., Any]] = None,
):
    """
    Custom field definition for config to abstract metadata usage
    Args:
        default: Default value expected for the field
        constraint: The function to be invoked to set constraints to the parameter value
    """
    metadata = {
        "constraints": constraint,
    }
    return field(default=default, metadata=metadata)


def is_union_type_hint(type_hint):
    """Check if Union type"""
    return get_origin(type_hint) is Union


def get_member_type_hints(cls):
    """
    Iterates over all members of a class and extracts their type hints.

    Args:
        cls: The class to iterate over.

    Returns:
        A dictionary mapping member names to their corresponding type hints.
    """
    class_fields = fields(cls)
    type_hints = {}
    # Extracting type information
    for class_field in class_fields:
        type_hint = class_field.type
        type_hints[class_field.name] = get_args(type_hint) or type_hint
    return type_hints


def get_class_type(config_class, parameter):
    annotations = get_type_hints(config_class)
    field_type = annotations[parameter]
    if get_origin(field_type) is Union:
        for union_type in get_args(field_type):
            if is_dataclass(union_type):
                return union_type
    elif is_dataclass(field_type):
        return field_type
    return None


def set_constraint(current_constraint, updated_constraint):
    """Set the required constraint type if not already set"""
    if current_constraint[0] is not None:
        logging.warning(
            "Trying to select constraint for config implicitly, more than one valid type exists"
        )
    current_constraint[0] = updated_constraint


@dataclass
class BaseConfig:
    """This class represents a Base Model config, inherited by sub config classes"""

    def __validate__(self):
        """Validation method that iterates over class members with validation meta attached"""
        type_hints = get_member_type_hints(self)
        class_fields = fields(self)
        # Iterate over all class attributes and call their validations
        for class_field in class_fields:
            field_name = class_field.name
            field_value = getattr(self, field_name)
            if field_name != "__validate__" and hasattr(self, field_name):
                curr_field = getattr(self, field_name)
                # Check if the field is an instance of a child class
                if isinstance(curr_field, BaseConfig):
                    # If it's an instance of a child class, recursively call its __validate__ method
                    curr_field.__validate__()
                    type_hints_child = get_member_type_hints(curr_field)
                    type_hints.update(type_hints_child)

                field_meta = class_field.metadata
                constraint = None
                # Get the implicit constraint if we have one set explicitly
                if "constraints" in field_meta:
                    constraint = field_meta['constraints']

                # Check if all the mandatory params received a value
                if field_value is required:
                    raise ValueError(
                        f"required value for {field_name}, which is mandatory and must be set"
                    )

                # Check if the field is made optional
                is_optional = is_union_type_hint(class_field.type) and type(
                    None
                ) in get_args(class_field.type)
                if not is_optional and field_value is None:
                    raise ValueError(
                        f"None value for {field_name}, which is not of optional type"
                    )
                # If there is a custom validation logic attached use that
                if constraint is not None and field_value is not None:
                    if constraint(field_value) is False:
                        raise ValueError(
                            f"value for {field_name}, does not match the constraint"
                        )
                elif field_value is not None:
                    # If its a valid value, check for type based validation
                    validate_field_type(class_field, field_value)

    def _set_class_type(self, field_name, class_type, field_value):
        """
        Set the field to class type instance
        It calls the constructor of the class object type
        Init params are the same as the dict/list we get.
        The typecase will fail in class init if the param list
        doesn`t match the class signature

        """
        try:
            if isinstance(field_value, str):
                field_dict = ast.literal_eval(field_value)
                setattr(self, field_name, class_type(**field_dict))
            elif isinstance(field_value, list):
                for i, item in enumerate(field_value):
                    if isinstance(item, dict):
                        field_value[i] = class_type(**item)
            elif isinstance(field_value, dict):
                field_dict = field_value
                setattr(self, field_name, class_type(**field_dict))
            elif not is_dataclass(field_value) and not isinstance(
                field_value, dict
            ):
                logging.warning(
                    f"We got a config class initialization with invalid type {type(field_value)}"
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to set field \"{field_name}\" with class type \"{class_type}\"."
            ) from e

    def __post_init__(self):
        """
        Post init runs through the class object and creates sub-class objects
        from dict type initializations

        """
        for curr_field in fields(self):
            field_name = curr_field.name
            field_type = curr_field.type
            field_value = getattr(self, field_name)
            # Check if the field type is a Union
            if get_origin(field_type) is Union:
                for union_type in get_args(field_type):
                    if (
                        is_dataclass(union_type)
                        and field_value is not None
                        and field_value is not required
                    ):
                        self._set_class_type(
                            field_name=field_name,
                            class_type=union_type,
                            field_value=field_value,
                        )
                        break
                    elif get_origin(union_type) is list and get_args(
                        union_type
                    ):
                        if (
                            is_dataclass(get_args(union_type)[0])
                            and field_value is not None
                            and field_value is not required
                        ):
                            self._set_class_type(
                                field_name=field_name,
                                class_type=get_args(union_type)[0],
                                field_value=field_value,
                            )
                        break
            elif is_dataclass(field_type):
                if field_value is not None and field_value is not required:
                    self._set_class_type(
                        field_name=field_name,
                        class_type=field_type,
                        field_value=field_value,
                    )
            elif get_origin(field_type) is list and get_args(field_type):
                if (
                    is_dataclass(get_args(field_type)[0])
                    and field_value is not None
                    and field_value is not required
                ):
                    self._set_class_type(
                        field_name=field_name,
                        class_type=get_args(field_type)[0],
                        field_value=field_value,
                    )
