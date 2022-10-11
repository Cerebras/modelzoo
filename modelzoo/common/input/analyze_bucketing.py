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
Utilities for generating buckets and estimating VTS speedups.

This script can do three things
1. Give an overview of the average sequence length of a dataset and potential
    for throughput increase
2. Analyze a bucketing scheme supplied by the user for estimate throughput
    increase and data distribution within buckets
3. Generate a new set of bucket boundaries for a user's dataset such that
    approximately the same fraction of the data falls in each bucket.
All throughput estimates are approximate and assume that the deltat for a batch
is linear in the length of the longest sample in that batch. This is not true
in general, and the result is that when using `analyze` or `generate`, the
outputs are generally underestimates.

The data provided by the user is assumed to be a npy file containing a histogram
of the frequencies of each sequence length. For example, `data[100]` should
be the number of samples with length exactly 100.
"""
import argparse
import os
import sys

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action")
    parser.add_argument(
        "--action",
        required=True,
        choices=["overview", "analyze", "generate"],
        help=(
            "The desired action. Can be 'overview' to get an overview of the "
            "dataset, 'analyze' to analyze an existing bucketing scheme, or "
            "'generate' to generate a new bucketing scheme"
        ),
    )
    parser.add_argument("--buckets", type=int, nargs="+")
    parser.add_argument(
        "--msl",
        type=int,
        required=False,
        help=(
            "The maximum sequence length of the data. If the data provided "
            "has a longer max sequence length, it is assumed that long "
            "sequences are truncated to msl. If not supplied, the msl is "
            "assumed from the data."
        ),
    )
    parser.add_argument(
        "--data",
        required=True,
        help=(
            "The path to a npy file containing a histogram of sequence lengths "
            "for a particular dataset"
        ),
    )
    args = parser.parse_args(sys.argv[1:])

    if not os.path.exists(args.data):
        raise ValueError(f"Got invalid data location {args.data}")
    if args.action == "generate":
        if args.buckets is None or len(args.buckets) != 1:
            raise ValueError(
                "You must specify a single value for --buckets when using "
                f"generate mode. Got {args.buckets}."
            )
        args.buckets = args.buckets[0]
    elif args.action == "analyze":
        if args.buckets is None or not args.buckets:
            raise ValueError(
                "You must specify a list of bucket boundaries using --buckets "
                "when using analyze mode."
            )

    return args


def bucketed_cost(data, buckets):
    assert isinstance(
        buckets, list
    ), f"Got buckets {buckets} of type {type(buckets)}"
    msl = len(data)
    lower_bds = [0] + buckets
    upper_bds = buckets + [msl]

    bucketed_data = np.zeros(len(upper_bds))
    for i, lower, upper in zip(range(len(bucketed_data)), lower_bds, upper_bds):
        bucketed_data[i] = np.sum(data[lower:upper])
    return np.sum(bucketed_data * np.array(upper_bds))


def bucket_data(data, buckets):
    assert isinstance(
        buckets, list
    ), f"Got buckets {buckets} of type {type(buckets)}"
    msl = len(data)
    lower_bds = [0] + buckets
    upper_bds = buckets + [msl]

    bucketed_data = np.zeros(len(upper_bds))
    for i, lower, upper in zip(range(len(bucketed_data)), lower_bds, upper_bds):
        bucketed_data[i] = np.sum(data[lower:upper])
    return bucketed_data


def find_even_buckets(raw_data, num_buckets):
    normalized_data = np.cumsum(raw_data) / np.sum(raw_data)
    p = (np.arange(num_buckets) / num_buckets)[1:]
    buckets = list(np.searchsorted(normalized_data, p))
    frequencies = bucket_data(raw_data, buckets) / np.sum(raw_data)
    return buckets, frequencies


def main(args):
    data = np.load(args.data)
    if args.msl is None:
        args.msl = len(data)
    data = data[: args.msl]
    data[-1] += np.sum(data[args.msl :])
    total_samples = int(np.sum(data))

    if args.action == "overview":
        avg_seq_len = np.dot(np.arange(len(data)), data,) / total_samples
        speedup = args.msl / avg_seq_len
        print(f"The average sequence length is {avg_seq_len:.2f}")
        print(f"The estimated througput increase upper bound is {speedup:.2f}")
    elif args.action == "analyze":
        non_vsl_compute = args.msl * total_samples
        bucketed_compute = bucketed_cost(data, args.buckets)
        speedup = non_vsl_compute / bucketed_compute
        frequencies = bucket_data(data, args.buckets) / total_samples
        print(f"The estimated throughput increase is {speedup:.2f}")
        for i, (lower, upper, f) in enumerate(
            zip([0] + args.buckets, args.buckets + [args.msl], frequencies)
        ):
            print(
                f"bucket {i+1} has lower bound {lower}, upper bound {upper}, "
                f"and contains {f:.2f} fraction of the data"
            )
    else:
        non_vsl_compute = args.msl * total_samples
        buckets, frequencies = find_even_buckets(data, args.buckets)
        bucketed_compute = bucketed_cost(data, buckets)
        speedup = non_vsl_compute / bucketed_compute
        print(
            f"The boundaries for {args.buckets} balanced buckets are {buckets}"
        )
        for i, (lower, upper, f) in enumerate(
            zip([0] + buckets, buckets + [args.msl], frequencies)
        ):
            print(
                f"bucket {i+1} has lower bound {lower}, upper bound {upper}, "
                f"and contains {f:.2f} fraction of the data"
            )
        print(
            f"The estimated throughput increase with these buckets is {speedup:.2f}"
        )


if __name__ == "__main__":
    main(parse_args())
