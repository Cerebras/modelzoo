# Overview

**NOTE: These scripts are meant to be used with the Slurm flow in Pipeline mode. The similar run scripts for Kubernetes (k8s) workflow are available through `cerebras_appliance` wheel.**

The given scripts are meant to be used to interact with the cerebras software
stack and launch jobs on the CS system. The cerebras software product is shipped in
the form of a singularity container and it contains all the components necessary
to validate/compile your model, and  actually run your task
(train/eval/predict) on the CS system (you do not need to directly interact with
the CS system). A key piece of setup required for these scripts is the location of
this container image, along with a list of directories this container will mount (only
data inside the mounted directories are accessible to the container).

In the Cerebras execution model, we run the input pipeline on cpu nodes and stream
them into the CS system. Due to the speed of the CS system, we launch multiple tasks,
each running the input pipeline and we use slurm to manage this.


# Installation guide (for the sys admin)
* place all scripts in a common/accessible location.
    * These are bash executable scripts - so they can be placed where other executables live
* The csrun_cpu script has a few varaibles that need setting (SINGULARITY_IMAGE, MOUNT_DIRS, DEF_NODES, etc). These
  are variables whose value depends on the local installation setup (for example, location of the sif image,
  default directories to mount, etc). Please fill these in. Amoung the variables to set, are a list of default
  slurm configurations, which will be used when running jobs on the CS system. Feel free to consult with Cerebras
  when setting these slurm defaults.
* Ensure that all the scripts are executable and if not, make them executable (ie chmod +x csrun_cpu, etc)
* Ensure that the location of the scripts are in the PATH variable. If not, add it to the PATH and export
    * This will ensure the scripts can be launched from anywhere
* To sanity check if the setup is correct, try running ''csrun_cpu python'' from anywhere. This should launch
  a Python interpreter inside the Cerebras Singularity Container.

# Brief description of the scripts

## csrun\_cpu
This script launches any given user command inside the cerebras singularity environment. For example:
* ''csrun_cpu bash'' - Will launch a bash shell inside the Cerebras container
* ''csrun_cpu python'' - Will launch a python interpreter inside the Cerebras container

This is also the base script for all other scripts. Cerebras users can use this command to
launch any cpu related tasks on the Cerebras container (model compilation, validation etc).

This scripts also contains all the relevent variables that are meant to be set by
system admins when they install the system onto user cluster (see above).The variables
include the pointer to the sif image, directories to mount, default slurm settings and so on.

Note this script does not execute anything on the CS system.

### Arguments:
* <command to run>: The command to execute inside the container
* --alloc-node: (Optional) If set to True, will reserve/allocate a whole node exclusively for the job. Defaults to False.
* --mount-dirs: (Optional) String of comma seperated paths to mount alongside the default paths specified in MOUNT_DIRS variable.
  Default is empty string (only those dirs specified in MOUNT_DIRS are mounted).

## cs\_input\_analyzer
This script analyzes your input pipeline after completing a compile or validation. It generates
a report outlining the estimated input pipeline performance as well as recommended slurm configurations
to use when launching training using csrun_wse. Note that the input pipeline performance will likely be
higher than actual training performance unless the input pipeline is the bottleneck.

It takes as arguments, a command to initiate compile or validation, and an optional `nodes`
argument, signifying the number of nodes that will be used for training. Note this latter argument
is only used for estimation purposes. For example:
* ''cs_input_analyzer python run.py --mode=compile_only''
        Will run compile and then return an input performance estimate
* ''cs_input_analyzer --avail_nodes=3 python run.py --validate_only''
        Will run validation and then return an input performance estimate assumming 3 nodes are available for training.

Note this script does not execute anything on the CS system.

### Arguments:
* <command to run>: A python command to initiate a full-compile or validation only
* --available-nodes: (Optional) Set this to the number of nodes available when training. Defaults to 1 if not set.
* --mount-dirs: (Optional) String of comma-seperated paths to mount alongside the default paths specified in MOUNT_DIRS variable.
  Default to empty string (only those dirs specified in MOUNT_DIRS are mounted)


## csrun\_wse
Runs execution on the CS system (train/eval/predict). If the users specify certain arguments pertaining to the slurm cluster setting,
(nodes, tasks_per_node, cpus_per_task), those will be used. Otherwise, the default configuration
specified in the csrun_cpu file will be used. For example:
* csrun_wse python run.py --mode=train --cs_ip=0.0.0.0
* csrun_wse --nodes=3 --tasks_per_worker=5 --cpus_per_task=16 python run.py --mode=train --cs_ip=0.0.0.0

### Arguments:
* <command for cs execution>: The python command to initiate a task that will be executed on the CS system (train/eval/predict).
* --mount_dirs:  (Optional) String of comma seperated paths to mount alongside the default paths specified in MOUNT_DIRS variable.
  Default to empty string (only those dirs specified in MOUNT_DIRS are mounted).
* --nodes            (Optional) Number of nodes to execute with (passed to slurm)'
* --tasks_per_node   (Optional) Number of tasks per node to execute with (passed to slurm)'
* --cpus_per_task    (Optional) Number of cpus per task to execute with (passed to slurm)'


