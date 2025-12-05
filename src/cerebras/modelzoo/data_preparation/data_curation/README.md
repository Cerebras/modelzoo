## Setting up the Environment - 

We will be using singularity containers to setup a minimal, reproducible environment for all the tasks within data curation.

### 1. Build the Container
If you already have a prebuilt `.sif` file, you can skip this step and proceed to [Running the Container](#running-the-container).
* The Docker file is located at `./env_files/Dockerfile`, with the dependencies specified in `./requirements.txt`

* To build the Docker image and convert it into a Singularity container (.sif file), run `./env_files/prepare_singularity_image.sh`. 

> [!NOTE]
> Don't run the above script from inside the `env_files` directory, run it from `./data_curation` where the requirements.txt is located. This is because Docker expects all files to be inside the build context.

* If you encounter an out-of-space error while building the image, you may need to change the temporary and cache directories used by Singularity. You can configure these in `./env_files/prepare_singularity_image.sh`. By default, they are set to: `~./singularity`

* By default, the resulting image file `data_curation.sif` will be saved to  `./env_files`. You can change the output location by editing the corresponding variable in `./env_files/prepare_singularity_image.sh`

### Running the Container

Once you have the `.sif` file, you can run any code inside the container using the following commands:
```
# To execute a command inside the container
singularity exec <path-to-sif> <command>

# To open an interactive shell session inside the container
singularity shell <path-to-sif>
```
For example, to verify that Python is properly installed and accessible within the container, you can run:
```
singularity exec ./env_files/data_curation.sif python --version
```
This should return the Python version if the image was built correctly.

> [!IMPORTANT]
> When using Singularity, the container is isolated from your host file system by default. This means any code or data outside of your home directory is not accessible unless you explicitly bind it.
>
> You must explicitly bind directories using the `--bind` flag to access or persist data.
> 
> Without binding, any generated files will be stored inside the container’s temporary environment and lost upon exit.

Here’s an example:

```bash
singularity exec --bind $(realpath ./my_pipeline):/pipeline \
                 --bind $(realpath ./output):/output \
                 ./env_files/data_curation.sif \
                 python /pipeline/run_pipeline.py --out_dir /output
```
This commands mounts: 

* `./my_pipeline` → `/pipeline` (code directory)

* `./output` → `/output` (for results)

## Running DataTrove Pipelines with SLURM

To run DataTrove pipelines on a SLURM cluster, some additional setup is needed to ensure the containerized environment can interact with SLURM services:

* If you're not already on a machine with SLURM installed, SSH into a SLURM cluster node to access the SLURM config from within the container.
```
# Example - ssh into the cluster 'cpu-blue'
ssh 172.31.49.1
```
You can use `cbrun clusters` to look up the correct IP address.

* Use the script `./env_files/start_pipeline.sh` to launch a pipeline. Before running it, make sure to configure the required path bindings and update the container run command as needed.

* The following system paths must be bound into the container for SLURM to function correctly:

```
--bind /run/munge:/run/munge \
--bind /opt/slurm:/opt/slurm \
--bind /opt/slurm/etc:/etc/slurm \
```

* Use our custom `SlurmExecutor` defined in `data_preparation/executors/slurm/slurm.py`. It is similar to the default DataTrove executor, with two key differences: 
    1. It requires a `container_path` argument to specify the `.sif` file for remote SLURM execution.
    2. The parameter `slurm_logs_folder` must also be an absolute path, so SLURM nodes can successfully write logs.

* We can import our custom `SlurmExecutor` like this - 
```
sys.path.append("/modelzoo/data_preparation/")

from executors.slurm.slurm import SlurmPipelineExecutor
```

* An example datatrove pipeline script is available at `/tests/models/transformers/data_processing/datatrove_slurm/test.py` 

* Use `cbrun -- squeue -u $USER` to check the status of your SLURM jobs.