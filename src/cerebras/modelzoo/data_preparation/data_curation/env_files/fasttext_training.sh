#!/bin/bash
# This script is used to train a FastText classifier for DeepSeek Math using Singularity.
# Path to the Singularity image
IMAGE="<path-to-sif>"

# Directories for input, output, and model zoo 
OUTPUT_DIR="<path-to-output-folder-for-output-files>"
MODELZOO_DIR="<path-to-modelzoo>"

# Bind the necessary directories for Singularity container
BIND_PATHS=(
    --bind $MODELZOO_DIR:/modelzoo
    --bind $OUTPUT_DIR:/output
)


singularity exec "${BIND_PATHS[@]}" "$IMAGE" bash -c "
    
    python3 /modelzoo/data_preparation/data_curation/pipeline/deepseek_math/train_fasttext_math_classifier.py \
        --config /modelzoo/data_preparation/data_curation/pipeline/deepseek_math/fasttext_training_config.yaml \
"

