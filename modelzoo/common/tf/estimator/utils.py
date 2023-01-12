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

import tensorflow as tf


def _host_call_to_eval_metric_ops(host_call):
    eval_metric_ops = host_call[0](*host_call[1])

    if isinstance(eval_metric_ops, dict):
        metric_ops_dict = eval_metric_ops
    elif isinstance(eval_metric_ops, (tuple, list)):
        metric_ops_dict = {
            f"elem_{index}": value
            for index, value in enumerate(eval_metric_ops)
        }
    else:
        raise ValueError(
            f"Invalid `eval_metric_ops` of type {type(eval_metric_ops)}"
        )

    new_eval_metric_ops = {}
    for key, value in metric_ops_dict.items():
        if (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and tf.is_tensor(value[0])
        ):
            new_eval_metric_ops[key] = value

    return new_eval_metric_ops


def _validate_host_call(host_call):
    if not host_call:
        return ()

    if not isinstance(host_call, (list, tuple)) or len(host_call) not in [2, 3]:
        raise ValueError(
            f"`host_call` must be an iterable with length 2 or 3. "
            f"Instead received {host_call}"
        )

    if not callable(host_call[0]):
        raise ValueError(
            f"Expected first item of `host_call` to be a callable. "
            f"Instead received {type(host_call[0])}."
        )

    if not isinstance(host_call[1], (list, tuple)):
        raise ValueError(
            f"Expected second item of `host_call` to be an iterable. "
            f" Instead received {type(host_call[1])}."
        )

    if len(host_call) > 2 and not isinstance(host_call[2], (list, tuple)):
        raise ValueError(
            f"Expected third item of host_call to be an iterable. "
            f"Instead received {type(host_call[2])}."
        )

    return host_call


try:
    from cerebras.tf.host_call import (
        host_call_to_eval_metric_ops,
        validate_host_call,
    )
    from cerebras.tf.summary import (
        cs1_disable_summaries as cs_disable_summaries,
    )
    from cerebras.tf.summary import cs1_enable_summaries as cs_enable_summaries
except:
    from contextlib import nullcontext

    cs_enable_summaries = nullcontext
    cs_disable_summaries = nullcontext

    host_call_to_eval_metric_ops = _host_call_to_eval_metric_ops
    validate_host_call = _validate_host_call


__all__ = [
    "cs_enable_summaries",
    "cs_disable_summaries",
    "host_call_to_eval_metric_ops",
    "validate_host_call",
]
