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

#!/usr/bin/env python

import argparse

import numpy as np
import pandas as pd
import torch

import cerebras.pytorch as cstorch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path")
    parser.add_argument("--output_path")
    parser.add_argument("--framework", choices=["tf", "pt"], default="pt")
    parser.add_argument(
        "--output_format", choices=["text", "csv"], default="text"
    )
    return parser.parse_args()


def read_checkpoint(input_path, framework):
    assert framework in ("pt"), f"Unsupported framework {framework}"
    state_dict = cstorch.load(input_path)
    flattened_vals, spec = torch.utils._pytree.tree_flatten(state_dict)
    flattened_keys = map(".".join, cstorch.utils.nest.recurse_spec(spec))
    yield from zip(flattened_keys, flattened_vals)


def main():
    args = parse_args()

    if args.output_format == "text":
        f = open(args.output_path, "w")
        f.write(
            "Layer stats layer_id: min_abs_wt, max_abs_wt, mean_wt, stdev_wt, "
            "mean_abs_wt, stdev_abs_wt, mean_log_wt, stdev_log_wt, wt_norm\n"
        )
    weight_norm = 0.0
    model_weight_norm = 0.0
    all_data = []
    for name, weight in read_checkpoint(args.input_path, args.framework):
        nan_check = np.isnan(weight)
        if np.all(nan_check) and args.output_format == "text":
            f.write(f"Layer {name} contains ***** ALL NaNs! *****\n")
            continue
        elif np.any(nan_check) and args.output_format == "text":
            num_nans = 100 * nan_check.sum() / nan_check.size
            f.write(
                f"WARNING: Layer {name} contains {num_nans:.1f}% NaNs. "
                "Removing them...\n"
            )
            weight = np.where(nan_check, 0.0, weight)

        if isinstance(weight, int) or isinstance(weight, float):
            weight = np.array(weight)

        num_weights = weight.size
        num_zeros = num_weights - np.count_nonzero(weight)
        if num_zeros > 0 and args.output_format == "text":
            f.write(
                f"Layer {name} zero: {num_zeros} "
                f"({num_zeros * 100 / num_weights:.2f}%)\n"
            )

        weight = weight.astype(float)
        norm_sq = np.sum(np.square(weight))
        weight_norm += norm_sq
        if (
            "Adam" not in name
            and "grad_accum" not in name
            and "step" not in name
            and "scale" not in name
            and "optimizer" not in name
            and "lr_scheduler" not in name
        ):
            model_weight_norm += norm_sq

        norm = np.sqrt(norm_sq)

        mean = np.mean(weight)
        stdev = np.std(weight)

        weight = np.absolute(weight)
        minimum = np.amin(weight)
        maximum = np.amax(weight)
        absmean = np.mean(weight)
        absstdev = np.std(weight)

        weight = np.log2(weight)
        logmean = np.mean(weight)
        logstdev = np.std(weight)
        if args.output_format == "text":
            f.write(
                f"Layer stats {name}: {minimum} {maximum} {mean} {stdev} "
                f"{absmean} {absstdev} {logmean} {logstdev} {norm}\n"
            )
        else:
            all_data.append(
                [
                    name,
                    minimum,
                    maximum,
                    mean,
                    stdev,
                    absmean,
                    absstdev,
                    logmean,
                    logstdev,
                    norm,
                ]
            )
    if args.output_format == "text":
        weight_norm = np.sqrt(weight_norm)
        model_weight_norm = np.sqrt(model_weight_norm)
        f.write(f"Total weights norm: {weight_norm}\n")
        f.write(f"Non-optimizer weights norm: {model_weight_norm}\n")
        f.close()
    else:
        df = pd.DataFrame(
            all_data,
            columns=[
                "name",
                "minimum",
                "maximum",
                "mean",
                "stdev",
                "absmean",
                "absstdev",
                "logmean",
                "logstdev",
                "norm",
            ],
        )
        df.to_csv(args.output_path)


if __name__ == "__main__":
    main()
