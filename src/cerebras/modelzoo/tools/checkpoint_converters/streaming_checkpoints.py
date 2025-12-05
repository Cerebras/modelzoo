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

import json
import logging
import math
import os
import re
from pathlib import Path
from shutil import rmtree
from typing import Union
from weakref import finalize

import safetensors.torch as safetensors_torch
import torch

import cerebras.pytorch as cstorch


def convert_file_size_to_int(size: Union[int, str]):
    """
    Converts a size expressed as a string with digits and unit to an integer.

    Args:
        size (`int` or `str`): The size to convert (e.g., `"5MB"`). Will be directly returned if
            an `int`.
    Returns:
        The size in bytes.

    Example:
    ```py
    >>> convert_file_size_to_int("10GiB")
    10737418240
    ```
    """
    from cerebras.appliance.utils.units import convert_byte_unit

    if isinstance(size, str):
        match = re.search(r'(\d+)(.*)', size)
        if not match:
            raise ValueError(
                f"size '{size}' is not in a valid format. Use an integer followed by the "
                f"unit, e.g., '10GB'."
            )
        try:
            num = int(match.group(1))
            unit = match.group(2)
            size = convert_byte_unit(num, "B", src_unit=unit)
        except:
            raise ValueError(
                f"size '{size}' is not in a valid format. Use an integer followed by the "
                f"unit, e.g., '10GB'."
            )
    return size


def dtype_byte_size(dtype: torch.dtype) -> float:
    """
    Returns the size (in bytes) occupied by one parameter of type `dtype`.

    Example:

    ```py
    >>> dtype_byte_size(torch.float32)
    4.0
    ```
    """
    if dtype == torch.bool:
        return 1 / 8
    if dtype.is_floating_point:
        return torch.finfo(dtype).bits / 8
    else:
        return torch.iinfo(dtype).bits / 8


class StreamingShardedHFReader:
    r"""Allows sharded HuggingFace checkpoints to be read in a streaming manner
    rather than loading all shards into memory all at once. The underlying
    checkpoint is read-only.

    Only one shard is stored into memory at a time. For this reason, accessing
    random keys may slow due to the switching cost (loading) between shards. For
    this reason, it is recommend that keys are accessed in the order given by
    `self.keys()` or `self.__iter__()` as keys that appear in the same shard
    are in consecutive order.

    Args:
        index_file: Path to .index.json file.

    """

    def __init__(self, index_file: str) -> None:
        self.index_dir = os.path.dirname(index_file)
        with open(index_file, "r") as f:
            index = json.load(f)
            self.weight_map = index["weight_map"]

        self.file2keys = {
            file: [] for file in sorted(set(self.weight_map.values()))
        }

        for file in self.file2keys:
            shard_path = os.path.join(self.index_dir, file)
            if not os.path.exists(shard_path):
                raise FileNotFoundError(
                    f"Detected missing checkpoint shard: {shard_path}"
                )

        for key, file in self.weight_map.items():
            self.file2keys[file].append(key)

        self.active_file_name = None
        self.active_file_data = None

    def load_shard(self, file):
        if file.endswith(".safetensors"):
            return safetensors_torch.load_file(file, device="cpu")
        else:
            return torch.load(file, map_location="cpu")

    def __len__(self):
        return len(self.weight_map)

    def __iter__(self):
        for file in self.file2keys:
            for key in self.file2keys[file]:
                yield key

    def __getitem__(self, key):
        if key not in self.weight_map:
            raise KeyError

        file = self.weight_map[key]
        if file != self.active_file_name:
            self.active_file_name = file
            if self.active_file_data is not None:
                # Drop old data *before* load.
                # Without this, peak mem usage = prev shard + new shard
                del self.active_file_data
            self.active_file_data = self.load_shard(
                os.path.join(self.index_dir, file),
            )
        return self.active_file_data[key]

    def items(self):
        for key in self.keys():
            yield key, self[key]

    def keys(self):
        return list(self.__iter__())

    def values(self):
        for key in self.keys():
            yield self[key]


class StreamingShardedHFWriter:
    r"""Writes a HuggingFace sharded checkpoint in a streaming manner rather
    than accumulating the full checkpoint into memory and then writing all
    shards at the end.

    A partial checkpoint is accumulated into memory until it reaches the shard
    size limit at which point this shard is written to disk.

    It is essential that `self.save()` is called in order to flush the last
    shard to disk and to save other required metadata.

    The StreamingShardedHFWriter class supports re-accessing and even updating
    keys that have already been written. Note that accessing existing keys
    randomly may be slow due to the switching cost (loading) between shards that
    have already been written to disk. For this reason, it is recommend that
    keys are re-accessed in the order given by `self.keys()` or
    `self.__iter__()` as keys that appear in the same shard are in consecutive
    order. Note that updating data stored in a shard may result in a shard that
    is smaller/larger than the original shard size, as StreamingShardedHFWriter
    will not intelligently split or coalesce shards during updates.

    Args:
        checkpoint_dir: Path to where a new directory will be created to store
                        the checkpoint shards.

        shard_size:     The maximum size each checkpoint shard should be. Can be
                        an integer representing the number of bytes, or a
                        formatted string (ex: "10GB").
                        See convert_file_size_to_int for valid string formats.

        export_safetensors: Whether the output shards should be saved as
                            safetensors or pickle files. Default: False. When
                            using pickle files, the checkpoint & index files
                            are saved with the 'pytorch_model` prefix while
                            they use the 'model' prefix when using safetensors.

    """

    def __init__(
        self,
        checkpoint_dir: str,
        shard_size: Union[str, int] = "10GB",
        export_safetensors=False,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.file_ext = 'safetensors' if export_safetensors else 'bin'
        self.file_prefix = "pytorch_" if not export_safetensors else ""
        os.mkdir(self.checkpoint_dir)
        self.index_file = os.path.join(
            self.checkpoint_dir,
            f"{self.file_prefix}model.{self.file_ext}.index.json",
        )
        self.weight_map = {}
        self.current_file_number = 0
        self.last_file_number = 0
        self.total_shards_finalized = 0
        self.active_file_name = self.get_filename(
            self.current_file_number, self.total_shards_finalized
        )
        self.active_file_data = {}
        self.file_size = {self.active_file_name: 0}
        self.dirty = True
        self.max_shard_size = convert_file_size_to_int(shard_size)

    def __len__(self):
        return len(self.weight_map)

    def __iter__(self):
        for key in self.weight_map:
            yield key

    def __getitem__(self, key):
        if key not in self.weight_map:
            raise KeyError

        file = self.weight_map[key]
        if file != self.active_file_name:
            self._switch_shards(file)

        return self.active_file_data[key]

    def __setitem__(self, key, value):
        if key in self.weight_map:
            # We are updating a key that has already been seen before
            file = self.weight_map[key]
            if self.active_file_name != file:
                self._switch_shards(file)

            old_value = self.active_file_data[key]
            old_weight_size = math.ceil(
                old_value.numel() * dtype_byte_size(old_value.dtype)
            )
            weight_size = math.ceil(
                value.numel() * dtype_byte_size(value.dtype)
            )
            delta_size = weight_size - old_weight_size

            if (
                self.file_size[self.active_file_name] + delta_size
                > self.max_shard_size
            ):
                logging.warning(
                    f"Updating {key} is causing shard {self.active_file_name} to be larger than "
                    f"limit."
                )

            self.active_file_data[key] = value
            self.weight_map[key] = self.active_file_name
            self.file_size[self.active_file_name] += delta_size
            self.dirty = True
        else:
            # We are adding a new key that hasn't been seen before

            weight_size = math.ceil(
                value.numel() * dtype_byte_size(value.dtype)
            )

            if self.current_file_number != self.last_file_number:
                self._switch_shards(
                    self.get_filename(
                        self.last_file_number, self.total_shards_finalized
                    )
                )

            # Create a new shard if this new weight "tips" us over the limit:
            if (
                self.file_size[self.active_file_name] + weight_size
                > self.max_shard_size
            ):
                self._flush()
                self.last_file_number += 1
                self.current_file_number = self.last_file_number

                if self.active_file_data is not None:
                    # Drop old data *before* load.
                    # Without this, peak mem usage = prev shard + new shard
                    del self.active_file_data
                self.active_file_data = {}
                self.active_file_name = self.get_filename(
                    self.current_file_number, self.total_shards_finalized
                )
                self.file_size[self.active_file_name] = 0

            self.active_file_data[key] = value
            self.weight_map[key] = self.active_file_name
            self.file_size[self.active_file_name] += weight_size
            self.dirty = True

    def get_filename(self, file_number, total_shards=0):
        return f"{self.file_prefix}model-{file_number+1:05d}-of-{total_shards:05d}.{self.file_ext}"

    def load_shard(self, file):
        if self.file_ext == "safetensors":
            return safetensors_torch.load_file(file, device="cpu")
        else:
            return torch.load(file, map_location="cpu")

    def save_shard(self, data, file):
        if self.file_ext == "safetensors":

            def materialize(value):
                if hasattr(value, "_materialize"):
                    value = value._materialize()
                if isinstance(value, torch.Tensor):
                    value = value.contiguous()
                return value

            materialized_data = {k: materialize(v) for k, v in data.items()}
            safetensors_torch.save_file(
                materialized_data, file, {"format": "pt"}
            )
        else:
            torch.save(data, file)

    def _flush(self):
        if self.dirty:
            self.save_shard(
                self.active_file_data,
                os.path.join(self.checkpoint_dir, self.active_file_name),
            )
            self.dirty = False

    def _switch_shards(self, new_file):
        self._flush()
        self.active_file_name = new_file
        if self.active_file_data is not None:
            # Drop old data *before* load.
            # Without this, peak mem usage = prev shard + new shard
            del self.active_file_data
        self.active_file_data = self.load_shard(
            os.path.join(self.checkpoint_dir, new_file),
        )

    def save(self):
        self._flush()

        total_size = sum(shard_size for shard_size in self.file_size.values())

        # Finalize total number of shards:
        new_total_shards = self.last_file_number + 1
        if self.total_shards_finalized != new_total_shards:
            # Step 1: Figure out the prev file -> new file mapping so that
            # we can rename the files / data structures
            file_renames = {
                self.get_filename(
                    i, self.total_shards_finalized
                ): self.get_filename(i, new_total_shards)
                for i in range(new_total_shards)
            }

            # Step 2: Rename the checkpoint files
            for prev_file, new_file in file_renames.items():
                os.rename(
                    os.path.join(self.checkpoint_dir, prev_file),
                    os.path.join(self.checkpoint_dir, new_file),
                )

            # Step 3: Update the weight map & file size data structures:
            self.weight_map = {
                key: file_renames[prev_file]
                for key, prev_file in self.weight_map.items()
            }

            self.file_size = {
                file_renames[prev_file]: size
                for prev_file, size in self.file_size.items()
            }

            # Step 4: Update the # of finalized shards so that future updates
            # to the writer will be able to correctly pick up the shards
            self.total_shards_finalized = new_total_shards

        with open(self.index_file, "w") as f:
            f.write(
                json.dumps(
                    {
                        "metadata": {
                            "total_size": total_size,
                        },
                        "weight_map": self.weight_map,
                    },
                    indent=4,
                )
            )

    def items(self):
        for key in self.keys():
            yield key, self[key]

    def keys(self):
        return list(self.__iter__())

    def values(self):
        for key in self.keys():
            yield self[key]


class StreamingCSLeaf:
    r"""Marks checkpoint keys that can be directly loaded from/saved to the
    H5 checkpoint. Non-leafs are accessed through StreamingCSWriterView due to
    their iterable nature.
    """

    def __init__(self, dir=None) -> None:
        from tempfile import mkstemp

        _, self.path = mkstemp(dir=dir)

        def cleanup(path):
            Path(path).unlink(missing_ok=True)

        self._f = finalize(self, cleanup, self.path)

    def read(self):
        return cstorch.load(self.path)[0]

    def write(self, val):
        Path(self.path).unlink(missing_ok=True)
        cstorch.save([val], self.path)

    def __str__(self) -> str:
        return "*"

    def __repr__(self) -> str:
        return "*"


class StreamingCSWriterView:
    r"""StreamingCSWriterView allows for checkpoints with arbitrarily nested
    dictionaries/lists to be written in a streaming (incremental) manner by
    offering a "view" into a StreamingCSWriter. For example, in a checkpoint
    with the structure {"model": {<model state>}}, we can obtain a view into the
    model state via checkpoint["model"]. This view has state <model state> and
    prefix ["model"]. The view acts like a dict (offers `__getitem__`,
    `__setitem__`, etc operations) which incrementally saves/loads from an H5
    checkpoint under the hood.

    Args:
        checkpoint_file:    Path to H5 checkpoint
        state:              (Sub)state dictionary corresponding to the current
                            view of the checkpoint.
        prefix:             Chain of keys that were accessed in the checkpoint
                            that yielded the current view

    """

    def __init__(self, checkpoint_file, state, prefix=None, dir=None) -> None:
        self.checkpoint_file = checkpoint_file
        self.state = state
        self.prefix = prefix or []

        if dir is None:
            import tempfile

            from cerebras.appliance.storage.s3_storage import S3Deleter

            if S3Deleter.is_valid_path(self.checkpoint_file):
                # Doesn't actually create a temp file, only generates a name
                self.checkpoint_dir = tempfile.mktemp(
                    dir=os.path.dirname(self.checkpoint_file)
                )
            else:
                self.checkpoint_dir = tempfile.mkdtemp(
                    dir=os.path.dirname(self.checkpoint_file)
                )

            def cleanup(path):
                if S3Deleter.is_valid_path(path):
                    S3Deleter(path).delete()
                else:
                    rmtree(path)

            self._f = finalize(self, cleanup, self.checkpoint_dir)
        else:
            self.checkpoint_dir = dir

    def __str__(self):
        return str(self.state)

    def __repr__(self):
        return f"StreamingCSWriterView: {str(self)}"

    def __iter__(self):
        if isinstance(self.state, dict):
            for key in self.keys():
                yield key
        if isinstance(self.state, (list, tuple)):
            for i in range(len(self.state)):
                yield self[i]

    def __len__(self):
        return len(self.state)

    def items(self):
        assert isinstance(self.state, dict)
        for key in self.keys():
            yield key, self[key]

    def keys(self):
        assert isinstance(self.state, dict)
        for key in self.state:
            if key in self:
                yield key

    def values(self):
        assert isinstance(self.state, dict)
        for key in self.keys():
            yield self[key]

    def __contains__(self, item):
        return item in self.state

    def __getitem__(self, key):
        value = self.state[key]

        if isinstance(value, StreamingCSLeaf):
            return value.read()

        if isinstance(value, StreamingCSWriterView):
            return value
        if isinstance(value, (dict, list, tuple)):
            subview = StreamingCSWriterView(
                self.checkpoint_file,
                value,
                self.prefix + [str(key)],
                self.checkpoint_dir,
            )
            return subview

        return value

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default

    def pop(self, key, default=None):
        if key in self:
            item = self[key]
            self.state.pop(key)
            return item
        return default

    def __setitem__(self, key, value):
        if key in self.state and not isinstance(
            self.state[key], StreamingCSLeaf
        ):
            raise ValueError(
                "StreamingCSWriter does not support updating an existing \
                     key which had a dict/list/tuple value"
            )

        if isinstance(value, (dict, list, tuple)):
            if key in self.state:
                raise ValueError(
                    "StreamingCSWriter does not support updating a key which \
                    already exists with a dict/list/tuple"
                )

            flattened, spec = torch.utils._pytree.tree_flatten(value)
            leaves = []

            for v in flattened:
                leaves.append(StreamingCSLeaf(self.checkpoint_dir))
                leaves[-1].write(v)

            self.state[key] = torch.utils._pytree.tree_unflatten(leaves, spec)
        else:
            self.state[key] = StreamingCSLeaf(self.checkpoint_dir)
            self.state[key].write(value)


class StreamingCSWriter(StreamingCSWriterView):
    r"""Writes a Cerebras H5 checkpoint in a streaming (incremental) manner
    rather than accumulating the full checkpoint into memory and then writing
    all weights at the end.

    It is essential that `self.save()` is called in order to flush the required
    metadata (state's spec). Without this call, the resulting checkpoint will
    not be able to be loaded with `cstorch.load(...)`.

    The StreamingCSWriter class supports re-accessing and even updating
    keys that have already been written. There are two restrictions:
    1.  An existing key that stores a dict/list/tuple cannot be replaced.
    2.  An existing key storing any type cannot be replaced by a dict/list/tuple

    Args:
        checkpoint_file:    Path to new H5 checkpoint. A file cannot already
                            exist at this location.

    """

    def __init__(self, checkpoint_file) -> None:
        if os.path.exists(checkpoint_file):
            raise FileExistsError(
                f"Checkpoint file \"{checkpoint_file}\" cannot be created because "
                "file already exists"
            )

        super().__init__(checkpoint_file, {})

    def save(self):
        vals, spec = torch.utils._pytree.tree_flatten(self.state)
        state_dict = torch.utils._pytree.tree_unflatten(
            (v.read() for v in vals), spec
        )

        cstorch.save(state_dict, self.checkpoint_file)

    def __str__(self):
        return f"{self.checkpoint_file}:\n{self.state}"

    def __repr__(self):
        return f"StreamingCSWriter: {str(self)}"


class OnDemandDictionaryConverter:
    r"""Wraps around an input dictionary in order to transform its values
    on-the-fly. The transformation has the following restrictions:
    1. It must maintain a 1-1 mapping (i.e. no new/dropped keys)
    2. The keys cannot change names (only values can change)
    There is error checking during object initialization and during runtime to
    ensure that this restriction holds.

    Args:
        underlying_dict:    Underlying dictionary that needs to be transformed
                            in an on-demand fashion
        converter_class:    A subclass of BaseDictionaryConverter which
                            describes the transformation of the underlying
                            dictionary
        action_fn_args:     Additional arguments that may be used in the
                            converter's action functions.

    """

    def __init__(
        self, underlying_dict, converter_class, action_fn_args=None
    ) -> None:
        super().__init__()
        self.underlying_dict = ReadOnlyDict(underlying_dict)
        self.converter_instance = converter_class()
        self.action_fn_args = action_fn_args or {}
        self.verify_converter()

    def verify_converter(self):
        # Deferred to prevent circular import:
        from cerebras.modelzoo.tools.checkpoint_converters.base_converter import (
            BaseDictionaryConverter,
        )

        assert isinstance(self.converter_instance, BaseDictionaryConverter), (
            f"{self.__class__}'s nested converter must subclass "
            f"BaseDictionaryConverter"
        )
        disallowed_fns = [
            "pre_checkpoint_convert",
            "pre_model_convert",
            "post_model_convert",
            "post_checkpoint_convert",
        ]
        for fn_name in disallowed_fns:
            assert not hasattr(self.converter_instance, fn_name), (
                f"{self.__class__} only supports converters that are 1-1 "
                f"mappings. Therefore, the nested converter cannot contain the "
                f"{fn_name} function"
            )

        for rule in self.converter_instance.rules:
            if not all(isinstance(elm, str) for elm in rule.segments):
                raise ValueError(
                    f"{self.__class__} only supports converters that are 1-1 "
                    f"mappings. Therefore, their rules can only contain regex "
                    f"strings (no EquivalentSubkey or BaseDictionaryConverter "
                    f"objects). The following conversion rule offends this "
                    f"constraint:\n{rule}"
                )

    def __len__(self):
        return len(self.underlying_dict)

    def __iter__(self):
        return self.underlying_dict.__iter__()

    def __getitem__(self, key):
        if key not in self.underlying_dict:
            raise KeyError

        new_temp_dict = {}
        from_index = 0
        match = self.converter_instance.convert_key(
            key,
            self.underlying_dict,
            new_temp_dict,
            from_index,
            action_fn_args=self.action_fn_args,
        )
        if set(new_temp_dict) != {key}:
            raise ValueError(
                f"{self.__class__}'s nested converter did not create a 1-1 "
                f"mapping."
            )
        if not match:
            raise KeyError
        return new_temp_dict[key]

    def items(self):
        for key in self.keys():
            yield key, self[key]

    def keys(self):
        return self.underlying_dict.keys()

    def values(self):
        for key in self.keys():
            yield self[key]


def _readonly(self, *args, **kwargs):
    raise RuntimeError("Cannot modify ReadOnlyDict")


class ReadOnlyDict(dict):
    """A Read-only dict.

    Note that this object doesn't guard against the values from being mutated in-place.
    """

    __setitem__ = _readonly
    __delitem__ = _readonly
    pop = _readonly
    popitem = _readonly
    clear = _readonly
    update = _readonly
    setdefault = _readonly
