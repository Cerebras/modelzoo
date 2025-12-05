#! /bin/bash

#SBATCH --job-name=vllm
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --tasks-per-node=1
#SBATCH --partition=a10x8-192c768m
#SBATCH --time=00:10:00
#SBATCH --output=./vllm-jobs/%x.%j.log

set -e

export CONTAINER_IMAGE=<Fill path to container image>
export HF_HOME=<Fill path to HF home>
export MODEL_PATH=${HF_HOME}/hub/<Fill model path>
export TP_SIZE=<Fill tensor parallel size>
export PP_SIZE=<Fill pipeline parallel size>

# Ensure all the paths required by vllm are accessible inside the container
# You may need to add more paths here
export MOUNTS="${PWD},${HF_HOME}"

NODES=$(scontrol show hostnames "${SLURM_JOB_NODELIST}") # Getting the node names
NODES_ARRAY=(${NODES})

HEAD_NODE=${NODES_ARRAY[0]}
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname --ip-address)


PORT=6379
HEAD_NODE_ADDRESS=${HEAD_NODE_IP}:${PORT}
export HEAD_NODE_ADDRESS

echo "Starting HEAD at ${HEAD_NODE_ADDRESS}"
srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
  singularity exec --nv --bind ${MOUNTS} ${CONTAINER_IMAGE} \
  ray start --head --port=${PORT} --block &

sleep 10

# Number of nodes other than the head node
WORKER_NUM=$((${SLURM_JOB_NUM_NODES} - 1))

echo "Starting WORKERS at ${WORKER_NUM} node(s)"

srun -n ${WORKER_NUM} --nodes=${WORKER_NUM} --ntasks-per-node=1 --exclude ${HEAD_NODE} \
  singularity exec --nv --bind ${MOUNTS} ${CONTAINER_IMAGE} \
  ray start --address $HEAD_NODE_ADDRESS --block &


echo "Waiting for the cluster to start..."
sleep 30


# Set the tensor parallel size to the number of GPUs in each node,
# and the pipeline parallel size to the number of nodes
# Add other parameters as required from
# https://docs.vllm.ai/en/v0.7.2/serving/engine_args.html

# Start vLLM inside the container
singularity exec --nv --bind ${MOUNTS} ${CONTAINER_IMAGE} \
  vllm serve ${MODEL_PATH} \
  --gpu-memory-utilization 1 \
  --tensor-parallel-size ${TP_SIZE} \
  --pipeline-parallel-size ${PP_SIZE}
