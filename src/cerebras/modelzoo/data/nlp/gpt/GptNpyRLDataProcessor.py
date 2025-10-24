from typing import List, Literal, Optional, Union

import numpy as np
import torch
from pydantic import PositiveInt

from cerebras.modelzoo.common.input_utils import get_streaming_batch_size
from cerebras.modelzoo.config import DataConfig
from cerebras.modelzoo.config.types import ValidatedPath
from cerebras.modelzoo.data.common.input_utils import is_distributed


class NpyRLDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
    ):
        super().__init__()
        rollouts_path = data_dir + "/rollouts.npz"
        self.MSL = 131072
        self.prompt_len = 256
        try:
            with open(rollouts_path, 'rb') as f:
                samples = np.load(f)

                # For this to work, we build the .npz file exactly as we've built for training.
                # TODO: Revisit this later, to make it more robust.

                self.advantages = samples['first'] if 'first' in samples else None
                self.attention_mask = samples['second'].astype(np.int32) if 'second' in samples else None
                self.input_ids = samples['third'].astype(np.int32) if 'third' in samples else None
                self.responses = samples['fourth'].astype(np.int32) if 'fourth' in samples else None
                self.old_log_probs = samples['fifth'] if 'fifth' in samples else None
                self.position_ids = samples['sixth'].astype(np.int32) if 'sixth' in samples else None
                self.ref_log_probs = samples['seventh'] if 'seventh' in samples else None

                # If True, we're in training mode; else we're calculating old log probs.
                self._has_log_prob = self.old_log_probs is not None

                if not self._has_log_prob:
                    self.dataset_size = len(self.input_ids)
                else:
                    self.dataset_size = len(self.advantages)

                #self.prompt = samples['eighth']

                # These keys are common to both the modes i.e training as well as old-log-prob calculation.
                for name in ("attention_mask","input_ids","responses","position_ids"):
                    if getattr(self, name) is None:
                        raise KeyError(f"Missing required key for '{name}' in {rollouts_path}")

                self.loss_mask = np.zeros((self.dataset_size, self.MSL), dtype=np.float32)
                self.loss_mask[:, self.prompt_len:self.prompt_len + len(self.responses[0])] = 1.0
                self.input_ids = self.pad_length(self.input_ids)       
                self.attention_mask = self.pad_length(self.attention_mask)       
                self.position_ids = self.pad_length(self.position_ids)

                if self._has_log_prob:
                    self.advantages = self.pad_in_bw(self.advantages)
                    self.responses = self.pad_in_bw(self.responses)
                    self.old_log_probs = self.pad_in_bw(self.old_log_probs)
                    self.ref_log_probs = self.pad_in_bw(self.ref_log_probs)

        except Exception as e:
            raise RuntimeError(f"Failed to read : {rollouts_path}") from e

    def pad_length(self, batch):
        return np.array([np.pad(seq, (0, self.MSL - len(seq)), mode='constant', constant_values=0) for seq in batch])
    
    def pad_in_bw(self, batch):
        batch_size = batch.shape[0]
        insert_len = batch.shape[1]
        out = np.zeros((batch_size, self.MSL), dtype=batch.dtype)
        out[:, self.prompt_len:self.prompt_len+insert_len] = batch
        return out

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        data = {}

        if self._has_log_prob:
            data = {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "advantages": self.advantages[idx],
                "loss_mask": self.loss_mask[idx],
                "ref_log_probs": self.ref_log_probs[idx],
                "old_log_probs": self.old_log_probs[idx],
                "position_ids": self.position_ids[idx],
            }
        else:
            data = {
                "responses":      self.responses[idx],
                "input_ids":      self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "position_ids":   self.position_ids[idx],
            }

        return data

class GptNpyRLDataProcessorConfig(DataConfig):
    data_processor: Literal["GptNpyRLDataProcessor"]

    num_workers: int = 0
    """ The number of PyTorch processes used in the dataloader. """

    prefetch_factor: Optional[int] = 10
    """ The number of batches to prefetch in the dataloader. """

    persistent_workers: bool = True

    batch_size: PositiveInt = ...

    data_dir: Union[ValidatedPath, List[ValidatedPath]] = ...
    "Path to the data files to use."

    sampler: Optional[torch.utils.data.sampler.Sampler] = None


class GptNpyRLDataProcessor:
    """
    A map style dataset for GPT style models.

    Supports data saved on disk in either of the following formats:
        - `(num_tokens,)`, i.e. a set of documents tokenized and concatenated.
            We refer to this as the 'corpus' format in what follows.
        - `(num_sequences, 3, sequence_length)`, i.e. data that has already
            been preprocessed into sequences. We refer to this as the
            'sample' format in what follows.

    Args:
        config: The config used to configure the data processor.
    """

    def __init__(self, config: GptNpyRLDataProcessorConfig):
        if isinstance(config, dict):
            config = GptNpyRLDataProcessorConfig(**config)

        self.config = config

        self.dataset = NpyRLDataset(config.data_dir)
        self.batch_size = get_streaming_batch_size(config.batch_size)
        self.sampler = config.sampler

        if is_distributed():
            assert self.sampler is None, "Cannot use sampler in config with DDP"
            self.sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                shuffle=False,
                seed=1,
            )

    def create_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=None,
            batch_sampler=self.sampler,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=self.config.persistent_workers,
        )
