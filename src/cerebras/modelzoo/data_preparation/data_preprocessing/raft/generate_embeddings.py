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

from typing import Dict


def generate_embeddings(config: Dict) -> str:
    """
    Generate embeddings based on the provided configuration.

    Args:
        config (Dict): Configuration parameters for data preprocessing and embedding generation.

    Returns:
        str: Path to the directory where embeddings are saved.
    """
    # Prepare H5 files that will be used for embedding generation
    from cerebras.modelzoo.data_preparation.data_preprocessing.preprocess_data import (
        preprocess_data,
    )

    preprocess_data(config['data'])

    # Generate Embeddings
    params = config['dpr']
    from cerebras.modelzoo.common.run_utils import main

    main(params)

    return params['trainer']['init']['model']['embeddings_output_dir']
