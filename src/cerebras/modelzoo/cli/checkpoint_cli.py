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

"""Cerebras ModelZoo checkpoint CLI Tool."""

import argparse
from dataclasses import fields, is_dataclass
from datetime import datetime

from cerebras.modelzoo.cli.utils import (
    MZ_CLI_NAME,
    append_source_basename,
    is_dir,
)
from cerebras.modelzoo.tools.convert_checkpoint import CheckpointConverterCLI

TENSOR_CMP_SUPPORTED_OPS = ["equal", "allclose"]


class CheckpointCLI:
    def __init__(self):
        parser = argparse.ArgumentParser()
        self.configure_parser(parser)
        args = parser.parse_args()
        args.func(args)

    @staticmethod
    def epilog():
        return (
            f"Use `{MZ_CLI_NAME} checkpoint -h` to learn how to use the checkpoint command.\n\n"
            f"See below for some basic examples.\n\n"
            f"Get information about the checkpoint (i.e. size):\n"
            f"  $ {MZ_CLI_NAME} checkpoint info /path/to/checkpoint.mdl\n\n"
            f"Delete checkpoint:\n"
            f"  $ {MZ_CLI_NAME} checkpoint delete /path/to/checkpoint.mdl\n\n"
            f"Copy checkpoint:\n"
            f"  $ {MZ_CLI_NAME} checkpoint copy /path/to/local/checkpoint.mdl s3://bucket/path/to/remote/checkpoint.mdl\n\n"
            f"Move checkpoint:\n"
            f"  $ {MZ_CLI_NAME} checkpoint move /path/to/local/checkpoint.mdl s3://bucket/path/to/remote/checkpoint.mdl\n\n"
            f"Compare two checkpoints:\n"
            f"  $ {MZ_CLI_NAME} checkpoint diff model_dir/checkpoint.mdl checkpoints_database/checkpoint.mdl\n\n"
            f"List all checkpoint converters for gpt2:\n"
            f"  $ {MZ_CLI_NAME} checkpoint list gpt2\n\n"
            f"Convert a gpt2 checkpoint from Cerebras format to HuggingFace format:\n"
            f"  $ {MZ_CLI_NAME} checkpoint convert --model gpt2 --src-fmt "
            f"cs-auto --tgt-fmt hf --config workdir/params_gpt_tiny.yaml "
            f"model_dir/checkpoint.mdl\n\n"
            f"Convert a gpt2 config from one Cerebras version to the next:\n"
            f"  $ {MZ_CLI_NAME} checkpoint convert-config --model gpt2 --src-fmt "
            f"cs-2.4 --tgt-fmt cs-2.5 workdir/params_gpt_tiny.yaml\n\n"
            f"For more information on checkpoint conversion, see: "
            f"https://docs.cerebras.net/en/latest/wsc/Model-zoo/Migration/porting-checkpoints.html"
        )

    @staticmethod
    def configure_parser(parser):
        subparsers = parser.add_subparsers(dest="cmd", required=True)

        info_parser = subparsers.add_parser(
            "info",
            add_help=True,
            help="Gives a high level summary of a checkpoint.",
        )
        info_parser.add_argument(
            "ckpt_path",
            help=(
                "Path to checkpoint. "
                "Can be local filesystem path (i.e. /path/to/local/checkpoint.mdl), "
                "or remote path (i.e. s3://bucket/path/to/remote/checkpoint.mdl)"
            ),
        )
        info_parser.add_argument(
            "-l",
            "--list_objects",
            action="store_true",
            help="If provided, list all the objects in the checkpoint.",
        )
        info_parser.add_argument(
            "-p",
            "--profile",
            default=None,
            help="The S3 profile to use.",
        )
        info_parser.set_defaults(func=CheckpointCLI.checkpoint_info)

        delete_parser = subparsers.add_parser(
            "delete",
            add_help=True,
            help="Delete a checkpoint",
        )
        delete_parser.add_argument(
            "ckpt_path",
            help=(
                "Path to checkpoint. "
                "Can be local filesystem path (i.e. /path/to/local/checkpoint.mdl), "
                "or remote path (i.e. s3://bucket/path/to/remote/checkpoint.mdl)"
            ),
        )
        delete_parser.add_argument(
            "-p",
            "--profile",
            default=None,
            help="The S3 profile to use.",
        )
        delete_parser.set_defaults(func=CheckpointCLI.checkpoint_delete)

        copy_parser = subparsers.add_parser(
            "copy",
            add_help=True,
            help="Copy a checkpoint",
        )
        copy_parser.add_argument(
            "source_path",
            help="Path of source checkpoint to be copied.",
        )
        copy_parser.add_argument(
            "dest_path",
            help=(
                "Path to copy checkpoint to. "
                "If the path ends with a '/', "
                "then the destination path will adopt the name of the source path"
            ),
        )
        copy_parser.add_argument(
            "-p",
            "--profile",
            default=None,
            help="The S3 profile to use.",
        )
        copy_parser.set_defaults(func=CheckpointCLI.checkpoint_copy)

        move_parser = subparsers.add_parser(
            "move",
            add_help=True,
            help="Move a checkpoint.",
        )
        move_parser.add_argument(
            "source_path",
            help="Path of source checkpoint to be moved.",
        )
        move_parser.add_argument(
            "dest_path",
            help=(
                "Path to move checkpoint to. "
                "If the path ends with a '/', "
                "then the destination path will adopt the name of the source path"
            ),
        )
        move_parser.add_argument(
            "-p",
            "--profile",
            default=None,
            help="The S3 profile to use.",
        )
        move_parser.set_defaults(func=CheckpointCLI.checkpoint_move)

        diff_parser = subparsers.add_parser(
            "diff", add_help=True, help="Diff two checkpoints."
        )
        diff_parser.add_argument(
            'left_checkpoint',
            type=str,
            help="Path to left checkpoint",
        )
        diff_parser.add_argument(
            'right_checkpoint',
            type=str,
            help="Path to right checkpoint",
        )
        diff_parser.add_argument(
            '--tensor_comparison_op',
            choices=TENSOR_CMP_SUPPORTED_OPS,
            default=TENSOR_CMP_SUPPORTED_OPS[0],
        )
        diff_parser.add_argument(
            "-p",
            "--profile",
            default=None,
            help="The S3 profile to use.",
        )
        diff_parser.set_defaults(func=CheckpointCLI.checkpoint_diff)

        CheckpointConverterCLI.configure_subparsers(subparsers)

    @staticmethod
    def checkpoint_info(args):
        import cerebras.pytorch as cstorch
        from cerebras.appliance.storage import H5Reader, S3Reader, StorageReader
        from cerebras.appliance.utils.units import bytes_to_human

        if S3Reader.is_valid_path(args.ckpt_path):
            if args.profile is not None:
                profile_names = [args.profile]
            else:
                import boto3

                profile_names = [
                    cstorch.backends.csx.storage.s3.profile
                ] + boto3.session.Session().available_profiles

            for profile in profile_names:
                cstorch.backends.csx.storage.s3.profile = profile
                if S3Reader.path_exists(args.ckpt_path):
                    if profile is not None:
                        print(f"AWS_PROFILE: {profile}")
                    break
            else:
                print(
                    "S3 Checkpoint path does not exist or "
                    "you do not have permission to access it. "
                    "Please check the credentials and try again. "
                )
                return False

        elif not StorageReader.path_exists(args.ckpt_path):
            print("Checkpoint path does not exist.")
            return False

        if S3Reader.is_valid_path(args.ckpt_path):
            reader = S3Reader(args.ckpt_path)
            total_objects = 0
            total_bytes = 0
            last_modified = None
            for obj in reader.s3_bucket.objects.filter(Prefix=reader.key):
                total_objects += 1
                total_bytes += obj.size

                if last_modified is None:
                    last_modified = obj.last_modified
                else:
                    last_modified = max(last_modified, obj.last_modified)

                if args.list_objects:
                    print(
                        f"{obj.last_modified:%Y-%m-%d %H:%M:%S} "
                        f"{bytes_to_human(obj.size): >10} {obj.key}"
                    )

            print(f"Total objects:            {total_objects}")
        else:
            reader = H5Reader(args.ckpt_path)
            total_bytes = reader.fstat.st_size
            last_modified = datetime.fromtimestamp(reader.fstat.st_mtime)

        cstorch_version = reader.global_metadata.get(
            cstorch.storage.constants.__CSTORCH_VERSION__, "N/A"
        )

        print(f"Total checkpoint size:    {bytes_to_human(total_bytes)}")
        print(f"Last modified:            {last_modified:%Y-%m-%d %H:%M:%S}")
        print(f"Cerebras PyTorch Version: {cstorch_version}")
        return True

    @staticmethod
    def checkpoint_delete(args):
        import cerebras.pytorch as cstorch
        from cerebras.appliance.storage import StorageDeleter

        if args.profile is not None:
            cstorch.backends.csx.storage.s3.profile = args.profile

        if not StorageDeleter.path_exists(args.ckpt_path):
            print("Checkpoint path does not exist. Cannot delete.")
            return False

        try:
            StorageDeleter.get(args.ckpt_path).delete()

            print(f"Deleted checkpoint {args.ckpt_path}")
        except:
            print(
                f"Failed to delete checkpoint {args.ckpt_path}. "
                f"Checkpoint may have been partially deleted and "
                f"is no longer valid. Please try deleting the "
                f"checkpoint again to cleanup remaining artifacts."
            )
            raise

    @staticmethod
    def checkpoint_copy(args, move=False):
        from shutil import copyfile
        from shutil import move as movefile
        from types import SimpleNamespace

        from tqdm import tqdm

        import cerebras.pytorch as cstorch
        from cerebras.appliance.storage import (
            H5Reader,
            S3Reader,
            S3Writer,
            StorageReader,
        )

        if args.profile is not None:
            cstorch.backends.csx.storage.s3.profile = args.profile

        action = "move" if move else "copy"

        if not StorageReader.path_exists(args.source_path):
            print(f"Source path does not exist. Cannot {action}.")
            return False
        if is_dir(args.source_path):
            print(f"Source path cannot be a directory. Cannot {action}.")
            return False

        # If destination is a directory, use the source's basename
        args.dest_path = append_source_basename(
            args.source_path, args.dest_path
        )
        if StorageReader.path_exists(args.dest_path):
            print(f"Destination path already exists. Cannot {action}.")
            return False

        source_type = StorageReader.get_type(args.source_path)
        dest_type = StorageReader.get_type(args.dest_path)

        if source_type is dest_type:
            # If source and destination are the same type, we can just copy/move
            # the file directly.  No need to load and save the checkpoint
            if source_type is S3Reader:
                reader = S3Reader(args.source_path)
                writer = S3Writer(args.dest_path)

                for obj in tqdm(
                    list(reader.s3_bucket.objects.filter(Prefix=reader.key))
                ):
                    copy = writer.s3_bucket.Object(
                        obj.key.replace(reader.key, writer.key)
                    )
                    copy.copy_from(
                        CopySource={"Bucket": obj.bucket_name, "Key": obj.key},
                        Metadata=obj.Object().metadata,
                    )

                    if move:
                        obj.delete()

            elif source_type is H5Reader:
                if move:
                    movefile(args.source_path, args.dest_path)
                else:
                    copyfile(args.source_path, args.dest_path)
        else:
            # If source and destination are different types, we need to load and
            # save the checkpoint
            state_dict = cstorch.load(args.source_path)
            cstorch.save(state_dict, args.dest_path)

            if move:
                CheckpointCLI.checkpoint_delete(
                    SimpleNamespace(ckpt_path=args.source_path)
                )

        action = "Moved" if move else "Copied"
        print(
            f"{action} checkpoint from {args.source_path} to {args.dest_path}"
        )

    @staticmethod
    def checkpoint_move(args):
        CheckpointCLI.checkpoint_copy(args, move=True)

    @staticmethod
    def checkpoint_diff(args):
        import cerebras.pytorch as cstorch

        if args.profile is not None:
            cstorch.backends.csx.storage.s3.profile = args.profile

        diff_checkpoints_from_file(
            args.left_checkpoint,
            args.right_checkpoint,
            tensor_comparison_op=args.tensor_comparison_op,
        )


def diff_checkpoints_from_file(
    file_left: str, file_right: str, tensor_comparison_op: str = "equal"
) -> bool:
    """
    Compare two checkpoints (left and right). Returns True if the dicts are the
    same.
    """
    import cerebras.pytorch as cstorch
    from cerebras.appliance.storage import StorageReader

    file_left_exists, file_right_exists = (
        StorageReader.path_exists(file_left),
        StorageReader.path_exists(file_right),
    )
    if not file_left_exists:
        print("No such file: {}".format(file_left))
        return False
    if not file_right_exists:
        print("No such file: {}".format(file_right))
        return False
    if file_left_exists and file_right_exists:
        import cerebras.pytorch as cstorch

        print("Loading checkpoints...")
        # Don't cache deferred tensors after comparing
        with cstorch.storage.serializers.cache_deferred_tensors(False):
            checkpoint_left = cstorch.load(file_left)
            checkpoint_right = cstorch.load(file_right)
        print("Comparing checkpoints...")
        diff_checkpoints(
            checkpoint_left,
            checkpoint_right,
            tensor_comparison_op=tensor_comparison_op,
        )


def diff_checkpoints(
    checkpoint_left: dict,
    checkpoint_right: dict,
    tensor_comparison_op: str = "equal",
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> bool:
    """
    Compare state dictionaries of two checkpoints (left and right). Returns True
    if the dicts are the same. Tensors can be compared via the "equal" or
    "allclose" operators. All other types are compared for strict equality.
    """
    import torch

    if tensor_comparison_op not in TENSOR_CMP_SUPPORTED_OPS:
        raise ValueError(
            f"{tensor_comparison_op} is not a supported tensor comparison operation. "
            f"Please select one of the following: {TENSOR_CMP_SUPPORTED_OPS}"
        )

    if tensor_comparison_op == "equal":
        compare_tensors = lambda x, y: torch.equal(x, y)
    elif tensor_comparison_op == "allclose":
        compare_tensors = lambda x, y: torch.allclose(
            x, y, rtol=rtol, atol=atol
        )

    def format_keys(key_path):
        return ".".join(
            str(getattr(k, fields(k)[0].name)) if is_dataclass(k) else str(k)
            for k in key_path
        )

    left_flattened, _ = torch.utils._pytree.tree_flatten_with_path(
        checkpoint_left
    )
    left_flattened_map = {
        format_keys(key_path): val for key_path, val in left_flattened
    }

    right_flattened, _ = torch.utils._pytree.tree_flatten_with_path(
        checkpoint_right
    )
    right_flattened_map = {
        format_keys(key_path): val for key_path, val in right_flattened
    }

    mismatches = []

    for key in sorted(set(left_flattened_map) | set(right_flattened_map)):
        if key not in left_flattened_map:
            mismatches.append(f"{key} not found in left checkpoint")
            continue

        if key not in right_flattened_map:
            mismatches.append(f"{key} not found in right checkpoint")
            continue

        left_val = left_flattened_map[key]
        right_val = right_flattened_map[key]

        if isinstance(left_val, torch.Tensor):
            if not isinstance(right_val, torch.Tensor):
                mismatches.append(
                    f"{key} has type tensor in left and type {type(right_val)} in right"
                )
            elif left_val.shape != right_val.shape:
                mismatches.append(
                    f"{key} tensor left has shape {left_val.shape} while right has shape {right_val.shape}"
                )
            elif left_val.dtype != right_val.dtype:
                mismatches.append(
                    f"{key} tensor left has dtype {left_val.dtype} while right has dtype {right_val.dtype}"
                )
            else:
                left_val_cpu = left_val.to("cpu")
                right_val_cpu = right_val.to("cpu")

                if not compare_tensors(left_val_cpu, right_val_cpu):
                    if not torch.is_floating_point(left_val_cpu):
                        left_val_cpu = left_val_cpu.to(torch.float64)
                        right_val_cpu = right_val_cpu.to(torch.float64)

                    loss = torch.nn.functional.mse_loss(
                        left_val_cpu, right_val_cpu
                    )

                    mismatches.append(
                        f"{key} left tensor is not {tensor_comparison_op} to right. "
                        f"MSE Loss: {loss.item()}"
                    )

                del left_val_cpu
                del right_val_cpu

        elif type(left_val) != type(right_val):
            mismatches.append(
                f"{key} has type {type(left_val)} in left and type {type(right_val)} in right"
            )
        elif left_val != right_val:
            mismatches.append(
                f"{key} in left {left_val!r} does not match right {right_val!r}"
            )

    if mismatches:
        print(f"Found {len(mismatches)} difference(s):")
        print("\n".join(mismatches))
        print("\nCheckpoints differ.")
        return False
    else:
        print("\nCheckpoints match.")
        return True


if __name__ == '__main__':
    CheckpointCLI()
