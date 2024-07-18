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
Script that generates a dataset in HDF5 format for GPT Models.
"""

import logging
import os
import sys
from multiprocessing import cpu_count
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from cerebras.modelzoo.common.utils.utils import check_and_create_output_dirs
from cerebras.modelzoo.data_preparation.nlp.hdf5_preprocessing.utils import (
    dump_args,
    dump_result,
    get_files,
    get_params,
    get_verification_args,
    multimodal_add_image_patch_start_idx,
    process_dataset,
    verify_saved_hdf5_files_mp,
)

# Custom preprocessors
from cerebras.modelzoo.data_preparation.nlp.hdf5_preprocessing.hdf5_curation_corpus_preprocessor import (  # noqa
    CurationCorpusPreprocessor,
)
from cerebras.modelzoo.data_preparation.nlp.hdf5_preprocessing.hdf5_nlg_preprocessor import (  # noqa
    NLGPreprocessor,
)

from cerebras.modelzoo.data_preparation.nlp.hdf5_preprocessing.hdf5_dataset_preprocessors import (  # noqa
    FIMDataPreprocessor,
    LlavaPhaseOnePreprocessor,
    LlavaPhaseTwoPreprocessor,
    LMDataPreprocessor,
    SummarizationPreprocessor,
    VSLLMDataPreprocessor,
    VSLSummarizationPreprocessor,
)

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def main():
    """Main function for execution."""
    params = get_params(desc="Create HDF5 dataset for language models")
    logger.warning(
        f"'create_hdf5_dataset.py script has been deprecated and will be removed in the next release version.\
                      Please use script from `data_preprocessing` folder under `cerebras.modelzoo.data_preparation.data_preprocessing`.\
                      Reach out to `developer@cerebras.net ` for any concerns or support."
    )
    output_dir = params["setup"].get("output_dir", "./data_dir/")
    if not params["processing"].get("resume_from_checkpoint", False):
        check_and_create_output_dirs(output_dir, filetype="h5")
    logger.info(f"\nWriting data to {output_dir}.")
    json_params_file = os.path.join(output_dir, "data_params.json")
    dump_args(params, json_params_file)

    metadata_files = params["setup"].pop("metadata_files", None)
    if metadata_files:
        metadata_files = metadata_files.split(",")
    input_dir = params["setup"].pop("input_dir", None)
    input_files = get_files(input_dir=input_dir, metadata_files=metadata_files)

    processes = params["setup"].pop("processes", 0)
    if processes == 0:
        processes = cpu_count()

    ds_processor = params["setup"].pop(
        "dataset_processor", "LMDataPreprocessor"
    )
    module_name = params["setup"].pop("module", None)
    dataset_processor = getattr(sys.modules[__name__], ds_processor)(params)

    unused_setup_params = [
        key for key in params["setup"].keys() if key != "output_dir"
    ]
    if unused_setup_params:
        logger.warning(
            "The following setup params are unused: "
            + ", ".join(unused_setup_params)
        )
    unused_dataset_params = [key for key in params["dataset"].keys()]
    if unused_dataset_params:
        logger.warning(
            "The following dataset params are unused: "
            + ", ".join(unused_dataset_params)
        )

    ## Set this to avoid the warning - The current process just got forked. Disabling parallelism to avoid deadlocks...
    #  To disable this warning, please explicitly set TOKENIZERS_PARALLELISM=(true | false)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    results = process_dataset(input_files, dataset_processor, processes)
    vocab_size = dataset_processor.get_vocab_size()

    logger.info(
        f"\nFinished writing data to {output_dir}."
        f" Runtime arguments and outputs can be found at {json_params_file}."
    )

    logger.info(f"Verifying the converted dataset at: {output_dir}")
    output_files = list(Path(output_dir).glob("*.h5"))
    verification_args = get_verification_args(
        processes, dataset_processor
    )  # for verify_saved_hdf5_files_mp
    dataset_stats = verify_saved_hdf5_files_mp(
        output_files, verification_args, vocab_size
    )
    logger.info("Done verifying the converted dataset.")

    dump_result(
        results,
        dataset_stats,
        json_params_file,
        dataset_processor.eos_id,
        dataset_processor.pad_id,
        vocab_size,
    )

    multimodal_add_image_patch_start_idx(
        json_params_file,
        dataset_processor,
    )


if __name__ == "__main__":
    main()
