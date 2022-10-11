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

import random

import numpy as np
from torch.utils.data._utils.collate import default_collate


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
