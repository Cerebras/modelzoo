
# Datatrove + NeMo Curator Pipeline

This repository provides a unified script to run a data processing pipeline using **Datatrove** followed by **NVIDIA NeMo Curator** on a SLURM-based cluster. The script automates job submission, monitoring, and transitions between stages.

## Prerequisites

Ensure the following are set up before running the pipeline:

- SLURM is installed and configured.
- `cbrun` is available and working for your cluster environment.
- You have access to a SLURM GPU partition (e.g., `gpu-a10`).
- `fineweb.py` contains your Datatrove data ingestion logic.
- `start-nemo-curator.slurm` is a SLURM batch script to launch NeMo Curator jobs.
- Python environment with necessary dependencies for Datatrove and NeMo Curator.

---

## Usage

Run the pipeline script with:

```bash
./full-pipeline.sh
```

This script performs the following steps:

### 1. Cancel Existing SLURM Jobs

Clears any existing pending or running jobs from the current user to avoid conflicts:

```bash
cbrun -- scancel -u $USER
rm -fr logs fineweb-output
```

### 2. Launch Datatrove

Runs `fineweb.py` to ingest or process data:

```bash
python fineweb.py
```

### 3. Live Job Monitoring

Monitors SLURM job status in real time until all jobs from the current user are completed. Pressing `Ctrl+C` will safely exit and restore the terminal state.

```bash
squeue -u "$USER" --noheader --format="%.18i %.20j %.10T %.10M"
```

### 4. Launch NeMo Curator

Once the Datatrove jobs finish, it:

- Cleans up previous output directories.
- Submits NeMo Curator SLURM jobs using a specific GPU partition.

```bash
rm -fr nemo-curator-jobs semdedup_cache semdedup-output
cbrun -t gpu-a10 -- sbatch start-nemo-curator.slurm
```

---

## Output

- **Datatrove output:** `fineweb-output/`
- **NeMo Curator output:** `semdedup-output/`
- Logs and intermediate files may appear in `logs/` and `nemo-curator-jobs/`.

---

## Notes

- If you're using a shared SLURM cluster, double-check before cancelling all jobs with `scancel -u $USER`.
- The pipeline assumes exclusive access to the job queue during execution.
- You can adjust SLURM partition (`gpu-a10`) in the script to fit your hardware setup.

---


