#!/bin/bash

IMAGE="<path-to-sif>"

# Directories for input, output, and model zoo
INPUT="<path-to-input-folder-containing-warc-files>"  
OUTPUT_DIR="<path-to-output-folder-for-output-files>"
MODELZOO_DIR="<path-to-modelzoo>"
MODEL_DIR="<path-to-trained-fasttext-model>"

# Bind the necessary directories for Singularity container
BIND_PATHS=(
    --bind /run/munge:/run/munge
    --bind /opt/slurm:/opt/slurm
    --bind /opt/slurm/etc:/etc/slurm
    --bind $MODELZOO_DIR:/modelzoo
    --bind $INPUT_DIR:/input
    --bind $OUTPUT_DIR:/output
)

# Note that this is relative to the binded modelzoo directory
PIPELINE_PATH="/modelzoo/data_preparation/data_curation/pipeline/deepseek_math/text_extraction_pipeline.py"
CONFIG_PATH="/modelzoo/data_preparation/data_curation/pipeline/deepseek_math/configs/text_extraction_config.yaml"

singularity exec "${BIND_PATHS[@]}" "$IMAGE" bash -c 
"
    python3 $PIPELINE_PATH --config ${CONFIG_PATH}
"