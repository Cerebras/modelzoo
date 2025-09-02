#! /bin/bash

#SBATCH --job-name=dist-clustering
#SBATCH --queue=cpu-spot
#SBATCH --nodes=16
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=./dist-clustering-jobs/slurm-%x.%j.log
#SBATCH --error=./dist-clustering-jobs/slurm-%x.%j.log


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
export BASE_JOB_DIR=${REALPWD}/dist-clustering-jobs
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
export DEVICE='cpu'
export SCRIPT_PATH=${REALPWD}/dask_ml_dbscan_pipeline.py
export SCRIPT_COMMAND="python3 $SCRIPT_PATH"

# Container parameters
export CONTAINER_IMAGE=/net/prateekgu-dev/srv/nfs/prateekgu-data/ws/monolith/src/models/src/cerebras/modelzoo/data_preparation/data_preprocessing/distributed_clustering/docker_files/distributed_clustering.sif
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
export HF_HOME="~/.cache"

# =================================================================
# End easy customization
# =================================================================

# Start the container
cbrun -- srun \
    singularity exec --bind ${MOUNTS} ${CONTAINER_IMAGE} \
    ${CONTAINER_ENTRYPOINT}