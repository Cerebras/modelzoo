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

"""Script used to fix checkpoints taken in releases <= 1.7 and make them compatible with releases >= 2.0"""
import argparse
import re
import sys
from pathlib import Path
from types import ModuleType

import dill
import h5py as h5
import torch
from tqdm import tqdm

import cerebras_pytorch as cstorch


def fix_checkpoint(
    checkpoint_path: str,
    fixed_checkpoint_path: str = None,
    fix_inplace: bool = False,
):
    checkpoint_path = Path(checkpoint_path)

    def fix_spec(f: h5.File):
        # Dynamically patch this import so that dill can load the spec
        tensor_scalar_dict = ModuleType("tensor_scalar_dict")
        tensor_scalar_dict.TensorScalarDict = dict
        sys.modules[
            "cerebras_pytorch.utils.tensor_scalar_dict"
        ] = tensor_scalar_dict

        spec = dill.loads(bytes.fromhex(f.attrs["__spec__"]))
        state_dict = torch.utils._pytree.tree_unflatten(
            [None for i in range(spec.num_leaves)], spec
        )

        # Fix the spec to remove the TensorScalarDict class
        _, spec = torch.utils._pytree.tree_flatten(state_dict)
        f.attrs["__spec__"] = dill.dumps(spec).hex()

        sys.modules.pop("cerebras_pytorch.utils.tensor_scalar_dict")

    def fix_duplicate_weights(f: h5.File):
        # Known duplicate weights that have issues.
        aliases = {
            "embedding_layer.word_embeddings.weight": "lm_head.weight",
            "ln_f.weight": "transformer_decoder.norm.weight",
            "ln_f.bias": "transformer_decoder.norm.bias",
        }

        spec = dill.loads(bytes.fromhex(f.attrs["__spec__"]))
        for scope in cstorch.utils.nest.recurse_spec(spec):
            tensor_name = ".".join(scope)

            if tensor_name not in f.keys():
                for k1, k2 in aliases.items():
                    if tensor_name.endswith(k1):
                        prefix = tensor_name[: -len(k1)]
                        k2 = f"{prefix}{k2}"
                        assert (
                            k2 in f.keys()
                        ), f"Could not find key {k2} in checkpoint"
                        f[tensor_name] = f[k2]
                        break
                    elif tensor_name.endswith(k2):
                        prefix = tensor_name[: -len(k2)]
                        k1 = f"{prefix}{k1}"
                        assert (
                            k1 in f.keys()
                        ), f"Could not find key {k1} in checkpoint"
                        f[tensor_name] = f[k1]
                        break
                else:
                    raise KeyError(
                        f"Alias not found for missing key {tensor_name}"
                    )

    def fix_last_epoch(f: h5.File):
        if "global_step" not in f.keys():
            print(
                "Cannot fix last epoch(s) without a global_step key. Skipping..."
            )
            return

        global_step = f["global_step"][...]

        milestones = None
        if any("milestones" in k for k in f.keys() if "lr_scheduler" in k):
            milestones = [
                v
                for k, v in sorted(
                    (
                        (k, f[k][...].item())
                        for k in f.keys()
                        if "lr_scheduler" in k and "milestones" in k
                    ),
                    key=lambda kv: int(kv[0].rsplit(".", 1)[-1]),
                )
            ]

            pattern = re.compile(r"(?P<scheduler_id>[0-9]+)\.last_epoch")

        last_epoch = global_step

        for k in f.keys():
            if "lr_scheduler" not in k or "last_epoch" not in k:
                continue

            if milestones is not None and (match := pattern.search(k)):
                scheduler_id = int(match.group("scheduler_id"))
                if scheduler_id > 0:
                    last_epoch = max(
                        global_step - milestones[scheduler_id - 1], 0
                    )
                else:
                    last_epoch = global_step
            else:
                last_epoch = global_step

            f[k][...] = last_epoch

    if fix_inplace:
        confirm = input(
            "Are you sure you want to fix the checkpoint inplace? [y/N]: "
        )
        if confirm.lower() != "y":
            print("Aborting...")
            return

        with h5.File(checkpoint_path, "a") as orig:
            # Fix the spec to remove the TensorScalarDict class
            print("Fixing spec...")
            fix_spec(orig)

            print("Fixing duplicate weights...")
            fix_duplicate_weights(orig)

            print("Fixing last_epochs...")
            fix_last_epoch(orig)

            fixed_checkpoint_path = checkpoint_path
    else:
        if fixed_checkpoint_path is None:
            fixed_checkpoint_path = checkpoint_path.with_name(
                checkpoint_path.stem + "_fixed" + checkpoint_path.suffix
            )
        else:
            fixed_checkpoint_path = Path(fixed_checkpoint_path)

        with h5.File(checkpoint_path, "r") as orig:
            with h5.File(fixed_checkpoint_path, "w") as fixed:
                # copy over the attributes from the original checkpoint
                fixed.attrs.update(orig.attrs)

                # Fix the spec to remove the TensorScalarDict class
                print("Fixing spec...")
                fix_spec(fixed)

                print("Copying keys...")
                for k in tqdm(orig.keys()):
                    d = fixed.require_dataset(
                        k,
                        shape=orig[k].shape,
                        dtype=orig[k].dtype,
                        data=orig[k][...],
                    )
                    d.attrs.update(orig[k].attrs)

                print("Fixing duplicate weights...")
                fix_duplicate_weights(fixed)

                print("Fixing last_epochs...")
                fix_last_epoch(fixed)

    print(f"Fixed checkpoint saved to: {fixed_checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fix the specification of the checkpoints taken in releases < 2.0 "
            "to be functional for releases >= 2.0"
        )
    )
    parser.add_argument(
        "checkpoint_path", type=str, help="Path to the checkpoint to fix."
    )
    parser.add_argument(
        "--fixed_checkpoint_path",
        default=None,
        type=str,
        required=False,
        help=(
            "Path to save the fixed checkpoint to. "
            "If not provided, the fixed checkpoint will be saved "
            "to the same path as the checkpoint, "
            "with '_fixed' appended to the name."
        ),
    )
    parser.add_argument(
        "--fix_inplace",
        action="store_true",
        help="Fix the provided checkpoint instead of copying and saving to a new file.",
    )
    args = parser.parse_args()

    fix_checkpoint(
        args.checkpoint_path, args.fixed_checkpoint_path, args.fix_inplace
    )


if __name__ == '__main__':
    main()
