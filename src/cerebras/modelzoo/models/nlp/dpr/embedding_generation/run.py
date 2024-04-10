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

# isort: off
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
# isort: on
import logging
import os
from functools import partial

from cerebras.modelzoo.common.utils.run.cli_pytorch import get_params_from_args


def main():
    from cerebras.modelzoo.models.nlp.dpr.embedding_generation.utils import (
        DEFAULT_EMBEDDINGS_PER_FILE,
        closest_multiple,
        extra_args_parser_fn,
    )

    params = get_params_from_args(extra_args_parser_fn=extra_args_parser_fn)
    runconfig = params["runconfig"]

    # Make sure that embedding generation is only run in eval mode:
    if runconfig["mode"] != "eval":
        raise RuntimeError(
            "DPR Embedding Generation needs to be run in eval mode"
        )

    # Figure out which encoder we want to generate embeddings for:
    selected_encoder = params["model"].get("selected_encoder", None)
    if selected_encoder != "q_encoder" and selected_encoder != "ctx_encoder":
        raise RuntimeError(
            "The 'selected_encoder' model parameter should either be "
            "'q_encoder' or 'ctx_encoder' for embedding generation."
        )

    # Determine output dir to dump embeddings:
    embed_dir = runconfig.get("embeddings_output_dir", None)
    if embed_dir is None:
        embed_dir = os.path.join(
            runconfig["model_dir"],
            f"embeddings_{selected_encoder}",
        )
    print(f"Embeddings will be saved to {embed_dir}")

    # Determine number of embeddings/batches per dumped file:
    batch_size = params["eval_input"]["batch_size"]
    embeddings_per_file = runconfig.get(
        "embeddings_per_file", DEFAULT_EMBEDDINGS_PER_FILE
    )

    if embeddings_per_file % batch_size != 0:
        embeddings_per_file = closest_multiple(embeddings_per_file, batch_size)
        logging.warning(
            f"In order to be performant, embeddings_per_file must be a "
            f"multiple of batch_size. Continuing by rounding "
            f"embeddings_per_file to {embeddings_per_file}"
        )

    batches_per_file = embeddings_per_file // batch_size

    # Run model:
    from cerebras.modelzoo.common.run_utils import main
    from cerebras.modelzoo.models.nlp.dpr.embedding_generation.data import (  # from cerebras.modelzoo.models.nlp.bert.data import (
        eval_input_dataloader,
        train_input_dataloader,
    )
    from cerebras.modelzoo.models.nlp.dpr.embedding_generation.model import (
        DPRWrapperModelForEmbeddingGeneration,
    )
    from cerebras.modelzoo.models.nlp.dpr.embedding_generation.saver import (
        DPREmbeddingSaver,
    )

    with DPREmbeddingSaver(embed_dir, batches_per_file) as embedding_saver:
        main(
            params,
            partial(
                DPRWrapperModelForEmbeddingGeneration,
                embedding_saver,
                selected_encoder,
            ),
            train_input_dataloader,
            eval_input_dataloader,
            extra_args_parser_fn=extra_args_parser_fn,
        )


if __name__ == '__main__':
    main()
