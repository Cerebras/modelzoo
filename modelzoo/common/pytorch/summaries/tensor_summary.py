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
Utilities for saving tensor summaries.
"""
import dataclasses
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Union

import torch
from tensorboard.backend.event_processing.io_wrapper import (
    IsSummaryEventsFile as is_summary_events_file,
)
from torch.utils.tensorboard import SummaryWriter

from modelzoo.common.pytorch.summaries.cb_summary import (
    CBSummary,
    DeviceOutputs,
)

_LOCK = threading.Lock()
_METADATA = "__metadata__"
_VERSION_KEY = "__version__"
_VERSION = "1.0"


class TensorSummary(CBSummary):
    """A class for providing tensor summaries on CS/CPU/GPU devices.

    In constrast to other summaries, such as scalar summaries, Tensor summaries
    are not written to Tensorboard events files and are not viewable in
    Tensorboard. Instead, they are written to files and a convenience API class
    is provided to load the values (see `TensorSummaryReader` class below).

    This class still takes in a SummaryWriter, like other summary classes,
    but instead of using it to save values into events files, it identifies the
    events file path, creates a sibling directory with a similar naming scheme,
    and places summarized tensor data in that directory. Users are discouraged
    from inspecting this directory as the implementation might change. Instead,
    users are encouraged to use the API provided below for loading summarized
    tensors.
    """

    def __init__(self, name: str):
        """Constructs a `TensorSummary` instance.

        Args:
            name: Name of the summary. This is the tag that appears in
                TensorBoard.
        """
        if name == _METADATA:
            raise ValueError(
                f"{_METADATA} is a reserved name. Please use a different name "
                f"for summaries."
            )

        super().__init__(name)

        self._log_provider = _LogProvider(name)

    # pylint: disable=arguments-differ
    def run_on_device(self, tensor: torch.Tensor) -> DeviceOutputs:
        """Define the portion of the summary computation that runs on device.

        Args:
            tensor: The tensor to be summarized.
        Returns:
            An instance of `DeviceOutputs`.
        """
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                f"Expected a torch Tensor for tensor summary {self.name}, "
                f"got {type(tensor)}"
            )
        return super().run_on_device(tensor)

    # pylint: disable=arguments-differ
    def run_on_host(self, tensor: torch.Tensor) -> torch.Tensor:
        """Runs the host portion of the summary computation.

        Args:
            tensor: The tensor to be summarized.
        Returns:
            The summarized tensor.
        """
        return tensor

    def save_on_host(
        self, host_outputs: torch.Tensor, writer: SummaryWriter, step: int,
    ) -> None:
        """Saves the tensor summary to events file.

        Args:
            host_outputs: The summarized tensor to write to events file.
            writer: A writer for writing summaries to events files.
            step: The current global step.
        """
        # pylint: disable=protected-access
        self._log_provider.update(writer.file_writer.event_writer._file_name)
        curr_time = time.time_ns()
        filename = f"{step}.{curr_time}"

        filepath = self._log_provider.logdir.joinpath(filename)

        # Write the tensor data to file
        torch.save(
            TensorDescriptor(
                step=int(step),
                ns_since_epoch=curr_time,
                tensor=host_outputs.detach(),
            ).to_dict(),
            str(filepath),
        )


def tensor_summary(name: str, tensor: torch.Tensor):
    """Convenience method for creating and running tensor summaries.

    This method searches registered summaries for the given name. If one is
    found, it uses it. Otherwise, it creates a new summary and runs the tensor
    through that summary.

    Args:
        name: Name of the summary. This is the tag that appears in TensorBoard.
        tensor: The tensor to be summarized.
    """
    summary = TensorSummary(name)
    # Run the summary op
    summary(tensor)


@dataclasses.dataclass(frozen=True)
class TensorDescriptor:
    """Descriptor for a summarized tensor.

    Args:
        step: Step at which the tensor was summarized.
        ns_since_epoch: Nanoseconds since "epoch" (e.g., UNIX time).
        tensor: The summarized tensor.
    """

    step: int
    ns_since_epoch: int
    tensor: torch.Tensor

    @property
    def utctime(self) -> datetime:
        """Returns the UTC time when this tensor was saved."""
        return datetime.utcfromtimestamp(float(self.ns_since_epoch) / 1e9)

    def to_dict(self) -> dict:
        """Returns the descriptor converted to a dict."""
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(values) -> "TensorDescriptor":
        """Returns a descriptor from a dict of values."""
        return TensorDescriptor(**values)


class TensorSummaryReader:
    """Class for reading summarized tensors.

    This class works in tandem with `TensorSummary` defined above. It provides
    general convenience APIs for inspecting tensor summaries produced by a run.

    Currently this class does not do any caching. So it can be used to inspect
    a live run. As more data becomes available, calling the APIs will reload the
    latest values.
    """

    def __init__(self, path: str):
        """Constructs a `TensorSummaryReader` instance.

        Args:
            path: Path to a Tensorboard events file or a directory containing
                Tensorboard events files. Location of tensor summaries are
                inferred from these events files as there is a one-to-one
                mapping from Tensorboard events files and tensor summary
                directories.
        """
        self._path = path
        self._summary_dirs: List[Path] = []

        event_files = self._discover_event_files(self._path)
        cb_summaries = self._discover_cerebras_summary_dirs(event_files)
        self._summary_dirs = self._discover_tensor_summary_dirs(cb_summaries)

        if not self._summary_dirs:
            logging.warning(
                f"Could not find any tensor summaries in {self._path}"
            )

    def load(
        self, name: str, step: int, latest_only: bool = True
    ) -> Union[TensorDescriptor, List[TensorDescriptor], None]:
        """Loads and returns tensor(s) with given name at the given step.

        Args:
            name: Name of the tensor.
            step: Step at which the tensor was summarized.
            latest_only: If False, return all if there are multiple tensors with
                the same name and step. If True, only return the latest value.
        Returns:
            A single tensor, multiple tensors, or no tensors matching the given
                name and step.
        """
        descriptors = []
        for summary_dir in self._summary_dirs:
            for path in summary_dir.joinpath(name).glob(f"{step}.*"):
                if path.exists():
                    content = torch.load(str(path))
                    descriptors.append(TensorDescriptor.from_dict(content))

        if not descriptors:
            logging.warning(
                f"No tensor with name {name} has been summarized at step {step}"
            )
            return None

        if not latest_only:
            return descriptors
        else:
            if len(descriptors) > 1:
                logging.warning(
                    f"Multiple summarized tensors with name {name} found at "
                    f"step {step}. Returning the latest one."
                )
            descriptors.sort(key=lambda x: x.ns_since_epoch)
            return descriptors[-1]

    def names(self) -> List[str]:
        """Returns a list of available tensor names."""
        names = set()
        for summary_dir in self._summary_dirs:
            for subdir in summary_dir.glob("*"):
                if subdir == _METADATA:
                    continue
                if subdir.is_dir():
                    names.add(subdir.name)
        return sorted(names)

    @staticmethod
    def _discover_cerebras_summary_dirs(event_files: List[Path]) -> List[Path]:
        """Returns root cerebras summary dirs corresponding to given events.

        Args:
            event_files: List of event files to find matching tensor summary
                directories for.
        Returns:
            List of root cerebras summary directories.
        """
        cb_summary_dirs = []
        for events_file in event_files:
            root = _LogProvider.get_cb_summaries_root(events_file)
            add = True
            if not root.exists():
                logging.warning(
                    f"No Cerebras summary directories were found for events "
                    f"file: {events_file}"
                )
                add = False
            if not root.is_dir():
                logging.warning(
                    f"Expected {root} to be a directory containing Cerebras "
                    f"summaries, but it is a file. Skipping as it has an "
                    f"unknown format."
                )
                add = False

            version = _read_version(root)
            if version is None:
                logging.warning(
                    f"Could not detect version of Cerebras summaries at "
                    f"directory {root}. This may lead to unexpected behavior."
                )
            if version != _VERSION:
                logging.warning(
                    f"Unknown version {version} for Cerebras summaries at "
                    f"directory {root}. Skipping this directory."
                )
                add = False

            if add:
                cb_summary_dirs.append(root)

        return cb_summary_dirs

    @staticmethod
    def _discover_tensor_summary_dirs(
        cb_summary_dirs: List[Path],
    ) -> List[Path]:
        """Returns tensor summary dirs corresponding to given Cerebras rootdirs.

        Args:
            cb_summary_dirs: List of root directories containing Cerebras
                specific summaries.
        Returns:
            List of tensor summary directories.
        """
        tensor_summary_dirs = []
        for cb_summary_dir in cb_summary_dirs:
            root = _LogProvider.get_tensor_summaries_root(cb_summary_dir)
            add = True
            if not root.exists():
                logging.warning(
                    f"No tensor summaries were found in directory: "
                    f"{cb_summary_dir}"
                )
                add = False
            if not root.is_dir():
                logging.warning(
                    f"Expected {root} to be a directory containing Tensor "
                    f"summaries, but it is a file. Skipping as it has an "
                    f"unknown format."
                )
                add = False

            if add:
                tensor_summary_dirs.append(root)

        return tensor_summary_dirs

    @staticmethod
    def _discover_event_files(path: str) -> List[Path]:
        """Returns all events files in the given directory.

        Args:
            path: Path to an events file or directory containing events files.
        Returns:
            List of all events files in the given directory, or the path itself
                if it is an events file.
        """
        event_files = []

        path = Path(path)
        if path.is_file():
            if not is_summary_events_file(str(path)):
                raise ValueError(f"Path {path} is not a summary events file")
            event_files.append(path)
        elif path.is_dir():
            for subpath in path.glob("*"):
                if is_summary_events_file(str(subpath)):
                    event_files.append(subpath)
            if not event_files:
                raise ValueError(
                    f"No summary events files found in directory {path}"
                )
        else:
            raise ValueError(f"Path {path} is neither a file nor a directory")

        return sorted(event_files)


class _LogProvider:
    """Class for providing log directories for saving tensor summary data."""

    def __init__(self, name: str):
        """Constructs a `_LogProvider` instance.

        Args:
            name: The summary name.
        """
        self._events_file: str = None
        self._logdir: Path = None
        self._rel_logdir: Path = None
        self._name = name

    @property
    def logdir(self) -> Path:
        """Returns absolute path to directory for writing tensor data."""
        assert (
            self._logdir
        ), "Log provider has not been tied to a SummaryWriter yet"
        return self._logdir

    @property
    def rel_logdir(self) -> Path:
        """Returns relative path to directory for writing tensor data.

        This is relative to the directory where the corresponding tensorboard
        events file resides.
        """
        assert (
            self._logdir
        ), "Log provider has not been tied to a SummaryWriter yet"
        return self._rel_logdir

    @staticmethod
    def get_cb_summaries_root(events_file: Union[str, Path]) -> Path:
        """Returns a root directory for placing tensor summaries."""
        events_file = Path(events_file).resolve(strict=True)
        dirname = events_file.name.replace(
            "events.out.tfevents.", "events.out.cbevents."
        )
        return events_file.parent.joinpath(dirname)

    @staticmethod
    def get_tensor_summaries_root(cb_summaries_root: Union[str, Path]) -> Path:
        """Returns a root directory for placing Cerebras-specific summaries."""
        return cb_summaries_root.joinpath("tensors")

    def update(self, events_file: str) -> None:
        """Updates log paths based on the current tensorboard events file."""
        if events_file == self._events_file:
            # Same events file as before, nothing to do update
            return
        self._events_file = events_file

        events_file = Path(events_file).resolve(strict=True)

        # First create the Cerebras summaries root directory
        rootdir = self.get_cb_summaries_root(events_file)
        rootdir.mkdir(parents=True, exist_ok=True)

        # Write metadata file
        with _LOCK:
            metadata = rootdir.joinpath(_METADATA)
            if not metadata.exists():
                with metadata.open("w") as f:
                    json.dump({_VERSION_KEY: _VERSION}, f, indent=True)

        # Now create the tensor summaries directory
        summaries_dir = self.get_tensor_summaries_root(rootdir)
        summaries_dir.mkdir(parents=True, exist_ok=True)

        # Now create a directory for this specific tensor summary
        self._logdir = summaries_dir.joinpath(self._name)
        self._logdir.mkdir(parents=True, exist_ok=True)

        self._rel_logdir = self._logdir.relative_to(Path(events_file).parent)


def _read_version(rootdir: Path) -> Union[str, None]:
    """Reads the tensor summary version from summaries root dir.

    Args:
        rootdir: The root directory where tensor summaries are written to.

    Returns:
        The version string or None if it can't be found.
    """
    version_file = rootdir.joinpath(_METADATA)
    if version_file.exists():
        with version_file.open("r") as f:
            content = json.load(f)
            if _VERSION_KEY in content:
                return content[_VERSION_KEY]
    return None
