# Distributed vLLM Deployment Guide


This README explains how to deploy vllm in a distributed setting using Singularity containers and slurm job scheduling. Follow these steps to:
1. Build a Singularity Image File (SIF).
2. Configure vllm server to host your model on GPU cluster via slurm.
3. And then submit a job to serve openai compatible api requests

## 1. Build a Singularity Image (SIF)

To build the container image from the definition file, run:
```bash
singularity build --fakeroot vllm.sif vllm.def
```

Note: This process may take several minutes as the resulting image is relatively large.

## 2. Configuring the vLLM Server

Edit the `run.slurm` script to customize your deployment:

- Fill in all placeholder values (container image path, model path, etc.)
- Adjust the number of nodes, time limit, and GPU partition according to your model requirements
- Configure tensor and pipeline parallelism settings based on your hardware configuration

## 3. Submitting the Job and Serving API Requests

To deploy the vLLM server on the GPU cluster, run:

```bash
cbrun -t gpu-a10 sbatch run.slurm
```

Monitor your job status with:

```bash
cbrun -t gpu-a10 squeue --me
```

When the job is running successfully, the logs will show the progress of loading the model. Once ready, you can test the service using:


```bash
curl -X POST 'http://<IP>:8000/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "messages": [
    {
      "role": "user",
      "content": "Are there earthquakes on Mercury?"
    }
  ]
}'
```

Important: Replace `<IP>` with the IP address of the head node shown in the job logs.

To cancel/stop a running job, use the `scancel` command with your job ID:

```bash
cbrun -t gpu-a10 scancel <job_id>
```

Where `<job_id>` is the ID assigned to your job by slurm.




