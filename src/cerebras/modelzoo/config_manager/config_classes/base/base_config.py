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
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import List, Literal, Optional  # pylint: disable=W0611

import yaml

from cerebras.modelzoo.common.registry import registry  # no qa

# pylint: disable=wildcard-import
from cerebras.modelzoo.config_manager.config_validators import *

from typing import (  # noqa
    Any,
    Callable,
    Union,
    get_args,
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

    def to_yaml(self, file_path):
        """
        This method writes the config to a yaml file

        Args:
            file_path: The path of output yaml file
        """
        with open(file_path, "w") as file:
            yaml.dump(asdict(self), file)

    def validate(self):
        """Validation method that iterates over class members with validation meta attached"""
        type_hints = get_member_type_hints(self)
        class_fields = fields(self)
        # Iterate over all class attributes and call their validations
        for class_field in class_fields:
            field_name = class_field.name
            field_value = getattr(self, field_name)
            if field_name != "validate" and hasattr(self, field_name):
                curr_field = getattr(self, field_name)
                # Check if the field is an instance of a child class
                if isinstance(curr_field, BaseConfig):
                    # If it's an instance of a child class, recursively call its validate method
                    curr_field.validate()
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

                # Check if the field is maked optional
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

    def set_class_type(self, field_name, class_type, field_value):
        """
        Set the field to class type instance
        It calls the constructor of the class object type
        Init params are the same as the dict/list we get.
        The typecase will fail in class init if the param list
        doesnt match the class signature

        """
        if isinstance(field_value, str):
            field_dict = ast.literal_eval(field_value)
            setattr(self, field_name, class_type(**field_dict))
        elif isinstance(field_value, list):
            for value in field_value:
                setattr(self, field_name, class_type(**value))
        elif isinstance(field_value, dict):
            field_dict = field_value
            setattr(self, field_name, class_type(**field_dict))
        elif not is_dataclass(field_value) and not isinstance(
            field_value, dict
        ):
            logging.error(
                f"We got a config class initialization with invalid type {type(field_value)}"
            )

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
                    if is_dataclass(union_type) and field_value is not None:
                        self.set_class_type(
                            field_name=field_name,
                            class_type=union_type,
                            field_value=field_value,
                        )
                        break
            elif is_dataclass(field_type):
                if field_value is not None:
                    self.set_class_type(
                        field_name=field_name,
                        class_type=field_type,
                        field_value=field_value,
                    )
