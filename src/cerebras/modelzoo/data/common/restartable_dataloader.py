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

import torch
from torch.utils.data import Dataset, IterableDataset

import cerebras.pytorch as cstorch
from cerebras.pytorch.distributed import get_worker_state

from .h5_map_dataset.readers import Mixture


class RestartableDataLoader(torch.utils.data.DataLoader):
    """
    Restartable dataloader for an `torch.utils.data.Dataset`.

    The state we care about for allowing deterministic restart of instances
    of `Dataset` is the total number of samples streamed globally,
    which gets consumed by the sampler. Accordingly each worker saves the number
    of samples that it has streamed in `state_dict()`. We aggregate these
    together via summation to save the global number of samples streamed across
    all workers, which is the same thing that is used to set the state of the
    sampler on state dict load.
    """

    __version__ = 1

    def __init__(self, dataset: Dataset, *args, **kwargs):
        """Constructs a `RestartableDataLoader` instance."""

        if not isinstance(dataset, Dataset):
            raise ValueError(
                f"\"{type(self)}\" expects a \"{Dataset}\", but "
                f"got {type(dataset)}."
            )

        if isinstance(dataset, IterableDataset):
            raise ValueError(
                f"\"{type(self)}\" expects \"{type(dataset)}\" to be a map style dataset, "
                f"but got {IterableDataset}!."
            )

        super().__init__(dataset, *args, **kwargs)

        if not isinstance(
            self.batch_sampler, cstorch.utils.data.DistributedSampler
        ):
            raise ValueError(
                f"\"{type(self)}\" expects a \"{cstorch.utils.data.DistributedSampler}\" batch_sampler, but "
                f"got {type(self.batch_sampler)}."
            )

        # keep track of how many samples were streamed in the previous portion
        # of the run so that we can track cumulative samples streamed in the
        # state_dict
        self._previous_samples_streamed = 0

    def state_dict(self):
        """Returns the state of the current dataloader."""
        worker_state = get_worker_state()
        return {
            "samples_streamed": worker_state.samples_streamed,
            "previous_samples_streamed": self._previous_samples_streamed,
        }

    def load_state_dict(self, state_dict, strict: bool = True):
        """Loads given state into the dataloader."""
        if not isinstance(state_dict, dict):
            raise ValueError(
                f"\"state_dict\" must be a dict, but got \"{type(state_dict)}\"."
            )

        def _raise_missing_key(key):
            raise KeyError(f"Dataloader state_dict is missing key \"{key}\"")

        version = state_dict.get("__version__", 0)

        if version == 0:
            if "seed" in state_dict:
                # Not all datasets will have these attributes, so we have to be careful and check.
                if (
                    getattr(self.dataset, "shuffle", False)
                    or isinstance(
                        getattr(self.dataset, "reader", None), Mixture
                    )
                ) and (
                    state_dict["seed"]
                    != getattr(self.dataset, "shuffle_seed", None)
                ):
                    raise ValueError(
                        f"shuffle seed {getattr(self.dataset, 'shuffle_seed', None)} doesn't match the seed used "
                        f"for the previous portion of the run {state_dict['seed']}"
                    )
            elif strict:
                _raise_missing_key("seed")

        elif version == 1:
            load_state_dict_fn = getattr(self.dataset, "load_state_dict", None)

            # If a dataset does not have a `state_dict`, we skip the
            # check and warn the user that there might be issues with restartability.
            if load_state_dict_fn is not None:
                if "dataset" in state_dict:
                    load_state_dict_fn(state_dict["dataset"], strict=strict)
                elif strict:
                    _raise_missing_key("dataset")
            else:
                self._show_state_dict_warning()
        else:
            raise ValueError(
                f"Invalid state_dict version: {version}. Known versions are 0 (None) and 1."
            )

        if "samples_streamed" in state_dict:
            self._previous_samples_streamed = state_dict["samples_streamed"]
            self.batch_sampler.set_state(state_dict["samples_streamed"])
        elif strict:
            _raise_missing_key("samples_streamed")

    def aggregate_state_dict(self, worker_states):
        """Aggregates states across all dataloaders into a single state."""

        state_dict = {
            "samples_streamed": worker_states[0]["previous_samples_streamed"]
            + sum(sd["samples_streamed"] for sd in worker_states),
            "__version__": self.__version__,
        }

        # If a dataset does not have a `state_dict`, we skip the
        # insertion and warn the user that there might be issues with restartability.
        dataset_state = getattr(self.dataset, "state_dict", lambda: None)()

        if dataset_state is not None:
            state_dict["dataset"] = dataset_state
        else:
            self._show_state_dict_warning()

        return state_dict

    def deaggregate_state_dict(
        self, aggregated_state_dict, strict: bool = True
    ):
        """Deaggregates state from all dataloaders."""
        return aggregated_state_dict

    def _show_state_dict_warning(self):
        logging.warning(
            "Dataset does not have the `state_dict()` method implemented.\n\n"
            "If the internal state of the dataset has changed between now and when "
            "training resumes you will not be notified! "
            "(e.g. if you change the dataset path or if the dataset is not deterministic)"
        )
