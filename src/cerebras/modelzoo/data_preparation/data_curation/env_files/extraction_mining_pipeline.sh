#!/bin/bash

# These paths need to be accessible to slurm nodes (use "realpath <path>" to get absolute paths)
IMAGE="<path-to-sif>"

# Directories for input, output, and model zoo
RAW_WARC_DIR="<path-to-input-folder-containing-warc-files>"  
EXTRACTED_DIR="<path-to-output-folder-for-output-files>"
MODELZOO_DIR="<path-to-modelzoo>"
MODEL_DIR="<path-to-trained-fasttext-model>"
# Bind the necessary directories for Singularity container
BIND_PATHS=(
    --bind /run/munge:/run/munge
    --bind /opt/slurm:/opt/slurm
    --bind /opt/slurm/etc:/etc/slurm
    --bind $MODELZOO_DIR:/modelzoo
    --bind $RAW_WARC_DIR:/raw_warc_dir
    --bind $EXTRACTED_DIR:/extracted_dir
    --bind $MODEL_DIR:/model_dir
)

# Note that this is relative to the binded modelzoo directory
PIPELINE_PATH="/modelzoo/data_preparation/data_curation/pipeline/deepseek_math/extraction_and_mining_pipeline.py"
CONFIG_PATH="/modelzoo/data_preparation/data_curation/pipeline/deepseek_math/configs/extraction_mining.yaml"
# Now run your Python command
singularity exec "${BIND_PATHS[@]}" "$IMAGE" bash -c 
"
    python3 $PIPELINE_PATH --config ${CONFIG_PATH}
"
        
