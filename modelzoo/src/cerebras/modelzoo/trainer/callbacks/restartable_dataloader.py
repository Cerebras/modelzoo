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

import cerebras.pytorch as cstorch
from cerebras.appliance.errors import ApplianceNanError
from cerebras.modelzoo.data.common.restartable_dataloader import (
    RestartableDataLoader,
)
from cerebras.modelzoo.trainer.callbacks import Callback, DataLoaderCallback
from cerebras.pytorch.utils.data.utils import infer_batch_size


class SamplesStreamedInfo(Callback):
    """
    Callback class to keep track of Samples streamed
    till the checkpoint step and the current step.
    """

    def __init__(self):
        """
        The following two attributes are needed to query
        the samples streamed at a Nan Loss generating
        step, because state_dict can only be queried at
        checkpointing steps.
        """
        self.samples_streamed_till_ckpt_step = 0
        self.samples_streamed_till_curr_step = 0

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        dataloader = trainer.get_callback(DataLoaderCallback).dataloader
        if dataloader is not None and isinstance(
            dataloader.dataloader, RestartableDataLoader
        ):
            self.samples_streamed_till_curr_step += infer_batch_size(batch)

    def on_save_checkpoint(self, trainer, state_dict):
        dataloader = trainer.get_callback(DataLoaderCallback).dataloader
        if dataloader is not None and isinstance(
            dataloader.dataloader, RestartableDataLoader
        ):
            self.samples_streamed_till_ckpt_step = dataloader.state_dict()[
                "samples_streamed"
            ]

    def get_samples_to_skip(self):
        return (
            self.samples_streamed_till_curr_step
            - self.samples_streamed_till_ckpt_step
        )

    def on_fit_exception(self, trainer, exception):
        dataloader = trainer.get_callback(DataLoaderCallback).dataloader
        if dataloader is not None and (
            isinstance(dataloader.dataloader, RestartableDataLoader)
            and isinstance(exception, ApplianceNanError)
            and 'NaN' in str(exception)
        ):
            trainer.logger.error(
                " NaN could also be a result of a bad/noisy sample. To resolve "
                "the bad/noisy sample issue, add a callback named "
                f"\"SkipSamples\" and pass {self.get_samples_to_skip()} as an "
                "argument to it. This will skip the first "
                f"{self.get_samples_to_skip()} samples."
            )

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass


class SkipSamples(Callback):
    """
    Callback class to skip n dataset samples for a training run
    when the training is resumed from the checkpoint.
    n is a user supplied argument.
    """

    def __init__(self, n: int):
        if n < 1:
            raise ValueError(
                f"Number of Samples to be skipped should be greater than 0. Received {n}."
            )
        self.n = n
        self.validate_started = False

    def on_enter_train(self, *args, **kwargs):
        self.validate_started = False

    def on_enter_validate_all(self, *args, **kwargs):
        self.validate_started = True

    def on_enter_validate(self, *args, **kwargs):
        self.validate_started = True

    def verify_dataloader(self, dataloader: cstorch.utils.data.DataLoader):
        fqcn = f"\"{RestartableDataLoader.__module__}.{RestartableDataLoader.__qualname__}\""
        if dataloader is None:
            raise ValueError(
                f"\"{type(self).__name__}\" expects a dataloader of type "
                f"{fqcn} but got \"None\"."
            )
        if not isinstance(dataloader.dataloader, RestartableDataLoader):
            raise ValueError(
                f"\"{type(self).__name__}\" expects a dataloader of type"
                f"{fqcn} but got \"{type(dataloader.dataloader)}\"."
            )

    def on_before_load_checkpoint(self, trainer, ckpt_path):
        """
        ckpt_path is None means Nan Loss generated
        before taking a checkpoint. So there is no
        state_dict which dataloader can load from.
        So create a dataloader state-dict on-the-fly
        and load dataloader state from there.
        """
        if not self.validate_started and ckpt_path is None:
            dataloader = trainer.get_callback(DataLoaderCallback).dataloader
            self.verify_dataloader(dataloader)
            dataloader.load_state_dict(
                {'samples_streamed': self.n, '__version__': 1},
                strict=False,
            )

            info = trainer.get_callback(SamplesStreamedInfo)

            info.samples_streamed_till_curr_step = self.n
            info.samples_streamed_till_ckpt_step = self.n
            trainer.logger.info(
                "Checkpoint path not provided. DataLoaders will skip "
                f"{self.n} samples from the beginning."
            )

    def preprocess_checkpoint(self, trainer, state_dict):
        """
        This function is called before the checkpoint is saved.
        We need to save the dataloader state in the checkpoint.
        """
        if not self.validate_started:
            self.verify_dataloader(
                trainer.get_callback(DataLoaderCallback).dataloader
            )
            if "dataloader" not in state_dict:
                state_dict["dataloader"] = {}
                msg = (
                    "Dataloader state not found in the checkpoint. DataLoaders "
                    f"will skip {self.n} samples from the beginning."
                )
            else:
                msg = (
                    f"Dataloader skipping {self.n} samples from the "
                    "saved checkpointed state."
                )

            state_dict["dataloader"]["samples_streamed"] = (
                state_dict["dataloader"].get("samples_streamed", 0) + self.n
            )

            info = trainer.get_callback(SamplesStreamedInfo)

            info.samples_streamed_till_ckpt_step = state_dict["dataloader"][
                "samples_streamed"
            ]
            info.samples_streamed_till_curr_step = state_dict["dataloader"][
                "samples_streamed"
            ]
            trainer.logger.info(msg)

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass
