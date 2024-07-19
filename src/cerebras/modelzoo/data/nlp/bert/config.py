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
Config classes of T5 data Configs

"""

from dataclasses import dataclass, field
from typing import List, Optional, Union

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.config_manager.config_classes.base.base_config import (
    required,
)
from cerebras.modelzoo.config_manager.config_classes.base.data_config import (
    DataProcessorConfig,
)


@dataclass
class BertDataProcessorConfig(DataProcessorConfig):
    vocab_file: str = required
    data_dir: Union[str, List[str]] = required
    "The path to the HDF5 files."
    max_sequence_length: int = required
    shuffle_buffer: Optional[int] = None
    "Size of shuffle buffer in samples."
    do_lower: bool = False
    drop_last: bool = True
    """
        similar to the PyTorch drop_last setting
        except that samples that when set to True, samples that would
        have been dropped at the end of one epoch are yielded at the
        start of the next epoch so that there is no data loss. This is
        necessary for a data ordering that is independent of the
        distributed setup being used.
    """
    prefetch_factor: int = 10
    persistent_workers: bool = True


@registry.register_data_config("SST2DataProcessor")
@dataclass
class SST2DataProcessorConfig(BertDataProcessorConfig):
    pass


@registry.register_data_config("MNLIDataProcessor")
@dataclass
class MNLIDataProcessorConfig(BertDataProcessorConfig):
    pass


@registry.register_data_config("BertCSVDataProcessor")
@dataclass
class BertCSVDataProcessorConfig(DataProcessorConfig):
    data_dir: Union[str, List[str]] = required
    "The path to the HDF5 files."
    shuffle_buffer: Optional[int] = None
    "Size of shuffle buffer in samples."
    dynamic_mlm_scale: bool = False
    drop_last: bool = True
    """
        similar to the PyTorch drop_last setting
        except that samples that when set to True, samples that would
        have been dropped at the end of one epoch are yielded at the
        start of the next epoch so that there is no data loss. This is
        necessary for a data ordering that is independent of the
        distributed setup being used.
    """
    prefetch_factor: int = 2
    persistent_workers: int = False
    mixed_precision: bool = False
    disable_nsp: bool = False
    buckets: Optional[List[int]] = None


@registry.register_data_config("BertCSVDynamicMaskDataProcessor")
@dataclass
class BertCSVDynamicMaskDataProcessorConfig(DataProcessorConfig):
    data_dir: Union[str, List[str]] = required
    "The path to the HDF5 files."
    max_sequence_length: int = required
    max_predictions_per_seq: int = required
    vocab_file: Optional[str] = None
    shuffle_buffer: Optional[int] = None
    "Size of shuffle buffer in samples."
    mask_whole_word: bool = False
    do_lower: bool = False
    dynamic_mlm_scale: bool = False
    buckets: Optional[List[int]] = None
    drop_last: bool = True
    """
        similar to the PyTorch drop_last setting
        except that samples that when set to True, samples that would
        have been dropped at the end of one epoch are yielded at the
        start of the next epoch so that there is no data loss. This is
        necessary for a data ordering that is independent of the
        distributed setup being used.
    """
    prefetch_factor: int = 10
    persistent_workers: bool = True
    oov_token: str = "[UNK]"
    mask_token: str = "[MASK]"
    document_separator_token: str = "[SEP]"
    exclude_from_masking: List[str] = field(default_factory=list)
    masked_lm_prob: float = 0.15
    gather_mlm_labels: bool = True
    mixed_precision: bool = False
    disable_nsp: bool = False
    labels_pad_id: int = 0
    input_pad_id: int = 0
    attn_mask_pad_id: int = 0
    segment_pad_id: int = 0


@registry.register_data_config("BertSumCSVDataProcessor")
@dataclass
class BertSumCSVDataProcessorConfig(DataProcessorConfig):
    data_dir: Union[str, List[str]] = required
    "The path to the HDF5 files."
    vocab_file: str = required
    max_sequence_length: int = required
    max_cls_tokens: int = required
    shuffle_buffer: Optional[int] = None
    "Size of shuffle buffer in samples."
    mask_whole_word: bool = False
    do_lower: bool = False
    drop_last: bool = True
    """
        similar to the PyTorch drop_last setting
        except that samples that when set to True, samples that would
        have been dropped at the end of one epoch are yielded at the
        start of the next epoch so that there is no data loss. This is
        necessary for a data ordering that is independent of the
        distributed setup being used.
    """
    prefetch_factor: int = 10
    persistent_workers: bool = True
    pad_id: Optional[int] = None


@registry.register_data_config("BertTokenClassifierDataProcessor")
@dataclass
class BertTokenClassifierDataProcessorConfig(DataProcessorConfig):
    data_dir: Union[str, List[str]] = required
    "The path to the HDF5 files."
    vocab_file: str = required
    label_vocab_file: str = required
    max_sequence_length: int = required
    shuffle_buffer: Optional[int] = None
    "Size of shuffle buffer in samples."
    mask_whole_word: bool = False
    do_lower: bool = False
    drop_last: bool = True
    """
        similar to the PyTorch drop_last setting
        except that samples that when set to True, samples that would
        have been dropped at the end of one epoch are yielded at the
        start of the next epoch so that there is no data loss. This is
        necessary for a data ordering that is independent of the
        distributed setup being used.
    """
    prefetch_factor: int = 10
    persistent_workers: bool = True
    labels_pad_id: Optional[int] = None
    input_pad_id: Optional[int] = None
    attn_mask_pad_id: Optional[int] = None
