# Running the Distributed Pipeline

## Creating the Virtual Environment

### Step 1: Create Virtual Environment
```bash
conda create --name <env>
```

### Step 2: Add packages using pip
If any more packages are needed for the work, it can be specified here.
```bash
pip install -r <path/to/requirements.txt>
```

### Step 3: Recompile fasttext from source
Due to the limitation of current Glibc, fasttext needs to be recompiled from source and the binary is to be used.
```bash 
git clone https://github.com/facebookresearch/fastText.git
cd fastText
make clean
make
```
Copy the fasttext binary into your working directory.

## Running the distributed pipeline

### Step 1: Set up config.yaml

Set up the config file with all the requirements like the model to be used for classification, model parameters, dask parameters, embedding model etc

### Step 2: Reserve nodes on slurm.
```bash
ssh <cluster_address>
srun --partition=<partition_name> --nodes=16 --time=04:00:00 --exclusive --pty /bin/bash
```

### Step 3: Activate conda environment
```bash
conda activate dask_conda_311
export PATH="<path/to/conda/bin>:$PATH"
```

### Step 4: Run the pipeline

```bash
python dask_ml_dbscan_pipeline.py --device=<cpu/cuda>
```
