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
Validators for all param level validation for config classes

"""
from dataclasses import is_dataclass
from typing import Literal, Union, get_args, get_origin


def check_valid_integer(value):
    """Check if param is a valid integer"""
    return isinstance(value, int)


def check_positive_integer(value):
    """Check if param is a positive integer"""
    return check_valid_integer(value=value) and value > 0


def check_valid_string(value):
    """Check if param is a string"""
    return isinstance(value, str)


def check_valid_bool(value):
    """Check if param is a bool"""
    return isinstance(value, bool)


def check_valid_float(value):
    """Check if param is a float"""
    return isinstance(value, (float, int))


def validate_literal(value, field_type):
    """Check if param is a valid literal"""
    literal_values = get_args(field_type)
    return value in literal_values


def check_loss_scaling_factor(value: Union[str, float]):
    """Custom check for loss scaling factor values"""
    if isinstance(value, float) and value < 0:
        return False
    if isinstance(value, str) and value != "dynamic":
        return False
    return True


type_hint_dict = {
    str: check_valid_string,
    float: check_valid_float,
    int: check_valid_integer,
    bool: check_valid_bool,
    Literal: validate_literal,
}


def get_constraint_for_type(typehint):
    """Get type of constraint to be added"""
    if typehint in type_hint_dict:
        return type_hint_dict[typehint]
    else:
        return None


def check_field_type(class_field, field_value, field_type):
    """Validate the field against the type hint"""
    # Handle Literal type
    if getattr(field_type, "__origin__", None) is Literal:
        return validate_literal(field_value, field_type)
    elif getattr(field_type, "__origin__", None) is list:
        # Handle List type
        element_type = field_type.__args__[0]
        constraint = get_constraint_for_type(element_type)
        for field_instance in field_value:
            if constraint and not constraint(field_instance):
                return False
        return True
    # Handle other types
    else:
        constraint = get_constraint_for_type(field_type)
        if constraint and not constraint(field_value):
            return False
        return True


def validate_field_type(class_field, field_value):
    """Validate the field against the type hint"""
    field_type = class_field.type
    field_name = class_field.name
    # Handle Union type
    if get_origin(field_type) is Union:
        union_types = get_args(field_type)
        for union_type in union_types:
            if union_type is type(None):
                continue  # Skip None type in Union
            if is_dataclass(union_type):
                if is_dataclass(field_value):
                    break
            else:
                if check_field_type(class_field, field_value, union_type):
                    break
        else:
            raise ValueError(
                f"Value for {field_name} is not any of types {union_types} : {field_value}"
            )
    else:
        if not check_field_type(class_field, field_value, field_type):
            raise ValueError(
                f"Value for {field_name} is not any of types {field_type} : {field_value}"
            )


# Aliases
LossScalingFactor = check_loss_scaling_factor
PositiveInteger = check_positive_integer
