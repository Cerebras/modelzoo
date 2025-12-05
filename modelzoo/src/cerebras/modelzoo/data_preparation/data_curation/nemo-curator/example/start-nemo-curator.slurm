#! /bin/bash

#SBATCH --job-name=nemo-curator
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --partition=a10-spot
#SBATCH --time=00:20:00
#SBATCH --output=./nemo-curator-jobs/slurm-%x.%j.log
#SBATCH --error=./nemo-curator-jobs/slurm-%x.%j.log


# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

# =================================================================
# Begin easy customization
# =================================================================

# Base directory for all Slurm job logs and files
# Does not affect directories referenced in your script
export REALPWD=$(realpath $(pwd))
export BASE_JOB_DIR=${REALPWD}/nemo-curator-jobs
export JOB_DIR=$BASE_JOB_DIR/$SLURM_JOB_ID

# Directory for Dask cluster communication and logging
# Must be paths inside the container that are accessible across nodes
export LOGDIR=$JOB_DIR/logs
export PROFILESDIR=$JOB_DIR/profiles
export SCHEDULER_FILE=$LOGDIR/scheduler.json
export SCHEDULER_LOG=$LOGDIR/scheduler.log
export DONE_MARKER=$LOGDIR/done.txt

# Main script to run
# In the script, Dask must connect to a cluster through the Dask scheduler
# We recommend passing the path to a Dask scheduler's file in a
# nemo_curator.utils.distributed_utils.get_client call like the examples
export DEVICE='gpu'
export SCRIPT_PATH=${REALPWD}/semdedup.py
export SCRIPT_COMMAND="python3 $SCRIPT_PATH \
    --scheduler-file $SCHEDULER_FILE \
    --device $DEVICE \
    --config-file ${REALPWD}/semdedup-config.yaml \
    --input-data-dir ${REALPWD}/fineweb-output/minhash/CC-MAIN-2025-13/deduped_output/ \
    --input-file-type jsonl \
    --input-file-extension gz \
    --input-text-field text"

# Container parameters
export CONTAINER_IMAGE=/cb/cold2/guneshwors/nemo.sif
# Make sure to mount the directories your script references
export BASE_DIR=${REALPWD}
export MOUNTS="${BASE_DIR}:${BASE_DIR}"

# Below must be path to entrypoint script on your system
export CONTAINER_ENTRYPOINT=$BASE_DIR/container-entrypoint.sh

# Network interface specific to the cluster being used
export INTERFACE=eth0
export PROTOCOL=tcp

# CPU related variables
# 0 means no memory limit
export CPU_WORKER_MEMORY_LIMIT="0"

# GPU related variables
export RAPIDS_NO_INITIALIZE="1"
export CUDF_SPILL="1"
export RMM_SCHEDULER_POOL_SIZE="1GB"
export RMM_WORKER_POOL_SIZE="1GiB"
export LIBCUDF_CUFILE_POLICY=OFF

# tmp dir for HF models
export HF_HOME="/tmp/huggingface_cache"

# =================================================================
# End easy customization
# =================================================================

# Start the container
cbrun -t gpu-a10 -- srun \
    --partition=a10-spot \
    singularity exec --nv --bind ${MOUNTS} ${CONTAINER_IMAGE} \
    ${CONTAINER_ENTRYPOINT}
