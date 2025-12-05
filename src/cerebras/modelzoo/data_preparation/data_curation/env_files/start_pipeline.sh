#!/bin/bash

# These paths need to be accessible to slurm nodes (use "realpath <path>" to get absolute paths)
SIF_PATH="<path-to-sif>"
SETUP_DIR="<root-dir>"

IMAGE=$SIF_PATH
INPUT_DIR="${SETUP_DIR}/input"
OUTPUT_DIR="${SETUP_DIR}/output"
MONOLITH="${SETUP_DIR}/monolith"
MODELZOO="${SETUP_DIR}/monolith/src/models/src/cerebras/modelzoo" 
SLURM_LOGS_DIR="${MODELZOO}/data_preparation/data_curation/slurm_logs"

# Bind the munge socket and other necessary directories
BIND_PATHS=(--bind /run/munge:/run/munge \
            --bind /opt/slurm:/opt/slurm \
            --bind /opt/slurm/etc:/etc/slurm \
            
            --bind $MONOLITH:/monolith \
            --bind $MODELZOO:/modelzoo \
            --bind $INPUT_DIR:/input \
            --bind $OUTPUT_DIR:/output \
)

# Note that this is relative to the binded modelzoo directory
PIPELINE_PATH="/monolith/tests/models/transformers/data_processing/datatrove_slurm/test.py"

singularity exec ${BIND_PATHS[@]} $IMAGE \
    python3 $PIPELINE_PATH \
            $SLURM_LOGS_DIR \
            $SIF_PATH \