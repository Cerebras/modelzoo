#!/bin/bash
# Run this script from the directory where the requirements.txt file is located, that is, from ./data_curation

set -xe

# If facing out of storage issues, manually set these to build the singularity image 
# You'll need approx. >= (2 * docker img size) free space

# export SINGULARITY_CACHEDIR=$(realpath ./)
# export SINGULARITY_TMPDIR=$(realpath ./)

IMG_PATH=$(realpath ./env_files/data_curation.img)
SIF_PATH=$(realpath ./env_files/data_curation.sif)

docker build -f ./env_files/Dockerfile -t data_curation .
docker save data_curation -o $IMG_PATH
singularity build $SIF_PATH docker-archive://$IMG_PATH

# Use this if you want to use the docker-daemon instead of docker-archive:
# singularity build ./data_curation.sif docker-daemon://data_curation:latest