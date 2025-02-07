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

import logging
import random
from typing import Optional
from warnings import warn

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

import cerebras.pytorch as cstorch
import cerebras.pytorch.distributed as dist
from cerebras.pytorch.utils.data.data_executor import MbsSetting


def bucketed_batch(
    data_iterator,
    batch_size,
    buckets=None,
    element_length_fn=None,
    collate_fn=None,
    drop_last=False,
    seed=None,
):
    """
    Batch the data from an iterator such that sampels of similar length end
    up in the same batch. If `buckets` is not supplied, then this just batches
    the dataset normally.

    :param data_iterator: An iterater that yields data one sample at a time.
    :param int batch_size: The number of samples in a batch.
    :param list buckets: A list of bucket boundaries. If set to None, then no
        bucketing will happen, and data will be batched normally. If set to
        a list, then data will be grouped into `len(buckets) + 1` buckets. A
        sample `s` will go into bucket `i` if
        `buckets[i-1] <= element_length_fn(s) < buckets[i]` where 0 and inf are
        the implied lowest and highest boundaries respectively. `buckets` must
        be sorted and all elements must be non-zero.
    :param callable element_length_fn: A function that takes a single sample and
        returns an int representing the length of that sample.
    :param callable collate_fn: The function to use to collate samples into a
        batch. Defaults to PyTorch's default collate function.
    :param bool drop_last: Whether or not to drop incomplete batches at the end
        of the dataset. If using bucketing, buckets that are not completely full
        will also be dropped, even if combined there are more than `batch_size`
        samples remaining spread across multiple buckets.
    :param int seed: If using `drop_last = False`, we don't want to feed out
        leftover samples with order correlated to their lengths. The solution
        is to shuffle the leftover samples before batching and yielding them.
        This seed gives the option to make this shuffle deterministic. It is
        only used when `buckets` is not `None` and `drop_last = True`.

    :yields: Batches of samples of type returned by `collate_fn`, or batches of
        PyTorch tensors if using the default collate function.
    """
    if batch_size < 1:
        raise ValueError(f"Batch size must be at least 1. Got {batch_size}.")

    if collate_fn is None:
        collate_fn = default_collate
    rng = random.Random(seed)

    if buckets is None:
        # batch the data normally
        batch = []
        for element in data_iterator:
            batch.append(element)
            if len(batch) == batch_size:
                yield collate_fn(batch)
                batch = []
        if not drop_last and len(batch) > 0:
            yield collate_fn(batch)

    elif isinstance(buckets, list):
        if sorted(buckets) != buckets:
            raise ValueError(
                f"Bucket boundaries must be sorted. Got {buckets}."
            )
        if buckets[0] <= 0:
            raise ValueError(
                f"Bucket boundaries must be greater than zero. Got {buckets}."
            )
        if not isinstance(buckets[0], int):
            t = type(buckets[0])
            raise ValueError(
                f"Elements of `buckets` must be integers. Got {t}."
            )
        if element_length_fn is None:
            raise ValueError(
                "You must supply a length function when using bucketing."
            )

        bucket_contents = [[] for i in range(len(buckets) + 1)]
        for element in data_iterator:
            length = element_length_fn(element)
            bucket_index = np.searchsorted(buckets, length, side="right")
            bucket_contents[bucket_index].append(element)
            if len(bucket_contents[bucket_index]) == batch_size:
                yield collate_fn(bucket_contents[bucket_index])
                bucket_contents[bucket_index] = []

        if not drop_last:
            remaining_data = []
            for bucket in bucket_contents:
                remaining_data.extend(bucket)
            rng.shuffle(remaining_data)
            batch = []
            for element in remaining_data:
                batch.append(element)
                if len(batch) == batch_size:
                    yield collate_fn(batch)
                    batch = []
            if len(batch) > 0:
                yield collate_fn(batch)

    else:
        raise ValueError(
            "buckets must be None or a list of boundaries. " f"Got {buckets}."
        )


def get_streaming_batch_size(
    effective_batch_size: int, global_rank: Optional[int] = None
) -> int:
    """Returns the streaming batch size of the given task.

    In a Wafer-Scaler Cluster setup with more than 1 CS-X node, the batch size
    used in compile and specified by user is the effective batch size at
    which gradient updates are done. However, each worker node streams a local
    batch of data to a given CS-X node to enable data parallel training.

    This helper method returns the local batch size that the current task should
    use given the effective batch size.

    Args:
        effective_batch_size: The effective batch size of the model.
        global_rank: The global rank of the task to return the streaming batch size for. If None,
            it returns the streaming batch size of the current task.
    Returns:
        The local batch size to be streamed by the given task. If queried on the
        user node (used when compiling the model), this returns the original
        effective batch size as passed in the argument.
    """
    streaming_batch_size = dist.get_streaming_batch_size(
        effective_batch_size, global_rank=global_rank
    )

    msg = f"Effective batch size is {effective_batch_size}."
    if cstorch.use_cs() and dist.is_streamer():
        msg += f" Using batch size {streaming_batch_size} for streaming."
    logging.info(msg)

    return streaming_batch_size


def validate_streaming_and_micro_batch_size(
    batch_size: int,
    micro_batch_size: MbsSetting,
    num_csx: int,
) -> None:
    """Validates the streaming and micro batch sizes.

    Args:
        batch_size: The global batch size of the model.
        micro_batch_size: The micro batch size of the model.
        num_csx: The number of CSX in the cluster.
    """
    if batch_size < num_csx:
        raise ValueError(
            f"Expected batch_size ({batch_size}) to be greater than or equal to "
            f"num_csx ({num_csx})."
        )

    if micro_batch_size == "auto":
        warn(
            f"Gradient accumulation will search for a well-performing micro "
            f"batch size based on internal performance models, which can lead "
            f"to an increased compile time. Also, this search is limited to "
            f"dividers of model global batch size {batch_size}.\n"
            f"You can specify your own micro batch size to reduce compile "
            f"time or have our `Automatic Batch Exploration` tool do a "
            f"thorough exploration for a more optimal micro batch size by "
            f"doing the following in your `config.yaml` file:\n"
            f"1. Set the `micro_batch_size` option to your the desired value, "
            f"if a preferred micro batch size is already known.\n"
            "2. Set the `micro_batch_size` option to `explore`. This will "
            f"perform a longer, broad search for an optimal micro batch size "
            f"to maximize the performance of your run, regardless of current "
            f"global batch size.\n"
            f"You can find more information on the `Automatic Batch "
            f"Exploration` page on the Cerebras documentation site."
        )

    if micro_batch_size in ["explore", "auto", None]:
        # If we want to run quick batch exploration, automatically pick the best, or disable
        # completely, then the batch size and num_csx can be anything. We don't need to validate
        # anything.
        return
    elif (
        isinstance(micro_batch_size, dict)
        and isinstance(micro_batch_size.get("explore"), dict)
        and len(set(micro_batch_size["explore"]) - {"min", "max"}) == 0
    ):
        return

    if not isinstance(micro_batch_size, int):
        raise ValueError(
            f'Invalid value "{micro_batch_size}" for "micro_batch_size". Expected one of:'
            f'\n\t"auto": Automatically choose an optimal micro batch size.'
            f'\n\t"explore": Search for an optimal micro batch size and return.'
            f'\n\t{{"explore": {{"min": Optional[<positive_int>], "max": Optional[<positive_int>]}}}}:'
            f' Search for an optimal micro batch size within the min and max bounds and return.'
            f"\n\t<positive_int>: Use this micro batch size."
            f"\n\tNone: Disable micro batch tiling."
        )

    if micro_batch_size < 1 or micro_batch_size > batch_size:
        raise ValueError(
            f"Expected micro_batch_size ({micro_batch_size}) to be a positive"
            f"integer less than or equal to the batch_size ({batch_size})."
        )


class PaddingSample(torch.Tensor):
    def __new__(cls, shape, dtype, require_grad=False):
        return cls._make_subclass(
            cls, torch.empty(shape, dtype=dtype), require_grad=require_grad
        )

    @property
    def is_padding(self):
        return True
