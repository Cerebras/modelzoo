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

# Script for converting the official DPR retrieval datasets from
# https://github.com/facebookresearch/DPR/tree/main repo into the HDF5 format
# in order to train/eval on CS.
#
# The script supports the same arguments as the retrieval training script:
# (https://github.com/facebookresearch/DPR/blob/main/train_dense_encoder.py).
# The notable args are:
#   train (optional): DPR model configuration (which also includes
#       tokenizer settings)
#   train_datasets: path to dataset that should be converted
#   output_dir: output path for converted HDF5 dataset
#   +num_workers: number of workers to use in the dataloader for conversion
#
# Example:
#
#   python convert_dpr_dataset.py train_datasets=retriever/webq-train.json \
#          output_dir=webq_train_hdf5 +num_workers=4
#

import logging
import os
import sys

import hydra
import torch
from dpr.options import set_seed, setup_cfg_gpu, setup_logger
from dpr.utils.model_utils import get_model_obj
from omegaconf import DictConfig, OmegaConf
from train_dense_encoder import BiEncoderTrainer

from cerebras.modelzoo.data_preparation.nlp.hdf5_preprocessing.convert_dataset_to_HDF5 import (
    convert_dataset_to_HDF5,
)

# The following imports need to be from the facebookresearch/DPR repo.
# This involves two things:
# (1) Running with the DPR repo environment
# (2) Pointing python path to the DPR repo
# For Step (1):
# Follow the instructions from the following section:
# Installation (within DPR repo)
# https://github.com/facebookresearch/DPR?tab=readme-ov-file#installation
# For Step (2):
# If you're running the script from within modelzoo, you can point python
# to the repo as follows:
# dpr_path = <path to DPR repo>
# sys.path.append(dpr_path)


logger = logging.getLogger()
setup_logger(logger)


class Fair2CS_Collator(object):
    def __init__(self, trainer) -> None:
        self.trainer = trainer
        self.biencoder = get_model_obj(trainer.biencoder)
        self.dropped_samples = 0  # Count number of samples that were dropped
        # due to missing negative contexts

    def __call__(self, batch) -> dict:
        idx = 0
        if isinstance(batch, tuple):
            batch, idx = batch

        ds_cfg = self.trainer.ds_cfg.train_datasets[idx]
        special_token = ds_cfg.special_token
        shuffle_positives = ds_cfg.shuffle_positives
        num_hard_negatives = self.trainer.cfg.train.hard_negatives
        num_other_negatives = self.trainer.cfg.train.other_negatives

        # Tokenize a batch of text:
        tokenized_batch = self.biencoder.create_biencoder_input(
            batch,
            self.trainer.tensorizer,
            True,
            num_hard_negatives,
            num_other_negatives,
            shuffle=True,
            shuffle_positives=shuffle_positives,
            query_token=special_token,
        )

        # Determine & assert the positions of the representation token:
        rep_positions = ds_cfg.selector.get_positions(
            tokenized_batch.question_ids, self.trainer.tensorizer
        )
        if isinstance(rep_positions, int):
            assert (
                rep_positions == 0
            ), "The representation token position must be index 0"
        elif isinstance(rep_positions, torch.tensor):
            assert torch.equal(
                rep_positions, torch.zeros(rep_positions.shape)
            ), "The representation token position must be index 0"
        else:
            raise TypeError(
                f"rep_positions has unexpected type: {type(rep_positions)}"
            )

        # Generate masks based on token ids:
        q_attn_mask = self.trainer.tensorizer.get_attn_mask(
            tokenized_batch.question_ids
        )
        ctx_attn_mask = self.trainer.tensorizer.get_attn_mask(
            tokenized_batch.context_ids
        )

        # Some samples may have missing negative contexts. Determine
        # which samples have the correct number of contexts vectors:
        positive_indexes = tokenized_batch.is_positive
        contexts_per_sample = [
            positive_indexes[i] - positive_indexes[i - 1]
            for i in range(1, len(positive_indexes))
        ] + [tokenized_batch.context_ids.shape[0] - positive_indexes[-1]]
        expected_contexts_per_sample = (
            1 + num_hard_negatives + num_other_negatives
        )

        # Determine the context & question indexes that have the correct number
        # of negatives:
        (
            positive_indexes_with_correct_count,
            indexes_with_correct_count,
            _,
        ) = list(
            map(
                list,
                zip(
                    *filter(
                        lambda ijc: ijc[2] == expected_contexts_per_sample,
                        zip(
                            positive_indexes,
                            range(len(positive_indexes)),
                            contexts_per_sample,
                        ),
                    )
                ),
            )
        )

        num_dropped_samples = len(batch) - len(
            positive_indexes_with_correct_count
        )

        if num_dropped_samples != 0:
            print(
                f"Dropping {num_dropped_samples} samples because they were "
                f"missing some negative contexts"
            )
            self.dropped_samples += num_dropped_samples

        # Gather question & context samples. Question tensors have shape:
        # [batch_size, length] while context tensors have shape:
        # [batch_size, num_context, length]
        questions_input_ids = tokenized_batch.question_ids[
            indexes_with_correct_count
        ]
        questions_attention_mask = q_attn_mask[indexes_with_correct_count]
        questions_token_type_ids = tokenized_batch.question_segments[
            indexes_with_correct_count
        ]

        def gather_context_per_sample(tensor):
            return torch.concatenate(
                [
                    tensor[idx : idx + expected_contexts_per_sample].unsqueeze(
                        0
                    )
                    for idx in positive_indexes_with_correct_count
                ],
                dim=0,
            )

        ctx_input_ids = gather_context_per_sample(tokenized_batch.context_ids)
        ctx_attention_mask = gather_context_per_sample(ctx_attn_mask)
        ctx_token_type_ids = gather_context_per_sample(
            tokenized_batch.ctx_segments
        )

        output = {
            "questions_input_ids": questions_input_ids,
            "questions_attention_mask": questions_attention_mask,
            "questions_token_type_ids": questions_token_type_ids,
            "ctx_input_ids": ctx_input_ids,
            "ctx_attention_mask": ctx_attention_mask,
            "ctx_token_type_ids": ctx_token_type_ids,
        }
        return output

    def print_dropped_samples(self):
        if self.dropped_samples != 0:
            print(
                f"A total of {self.dropped_samples} samples were dropped due to"
                f"missing negative contexts"
            )


@hydra.main(
    config_path=os.path.join(dpr_path, "conf"),
    config_name="biencoder_train_cfg",
)
def main(cfg: DictConfig):

    assert cfg.output_dir is not None, "Must provided output_dir"
    os.makedirs(cfg.output_dir)

    cfg = setup_cfg_gpu(cfg)
    set_seed(cfg)
    print("The following DPR config will be used for conversion:")
    print(OmegaConf.to_yaml(cfg))

    trainer = BiEncoderTrainer(cfg)

    assert (
        len(trainer.ds_cfg.train_datasets) == 1
    ), "Must provide one dataset to convert via train_datasets arg"
    dataset = trainer.ds_cfg.train_datasets[0]
    dataset.load_data()

    fairseq2cs_collator = Fair2CS_Collator(trainer)

    convert_dataset_to_HDF5(
        dataset=dataset,
        data_collator=fairseq2cs_collator,
        output_dir=cfg.output_dir,
        num_workers=cfg.num_workers if hasattr(cfg, "num_workers") else 0,
    )

    fairseq2cs_collator.print_dropped_samples()


if __name__ == "__main__":
    logger.info("Sys.argv: %s", sys.argv)
    hydra_formatted_args = []
    # convert the cli params added by torch.distributed.launch into Hydra format
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--") :])
        else:
            hydra_formatted_args.append(arg)
    # Disable hydra from changing run directory / logging
    hydra_formatted_args += [
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
        "hydra/job_logging=disabled",
        "hydra/hydra_logging=disabled",
    ]
    logger.info("Hydra formatted Sys.argv: %s", hydra_formatted_args)
    sys.argv = hydra_formatted_args

    main()
