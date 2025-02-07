# Copyright 2022 Cerebras Systems.
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

import os
from contextlib import contextmanager


def check_presence_of_wandb_dir(rundir):
    # Ensure that the wandb directory is not already present or empty
    wandb_dir = os.path.join(rundir, "wandb")
    if os.path.exists(wandb_dir):
        # Ensure there are no run-* folders in the wandb directory.
        run_dirs = [
            d
            for d in os.listdir(wandb_dir)
            if os.path.isdir(os.path.join(wandb_dir, d))
            and d.startswith('run-')
        ]
        if run_dirs:
            raise FileExistsError(
                f"A previous run seems to already exist in {wandb_dir}. "
                "Please specify a different 'model_dir'."
            )


@contextmanager
def mlops_run(params):
    if "wandb" not in params:
        yield None
        return

    try:
        import wandb
    except ImportError:
        raise RuntimeError(
            "wandb is an optional dependency of modelzoo. "
            "In order to use it, 'pip install wandb==0.16.2' into this venv"
        )

    runconfig = params.get("runconfig")
    wandb_config = params["wandb"]
    wandb_project = wandb_config.get("project", None)
    group = wandb_config.get("group", None)
    run_id = wandb_config.get("id", None)
    run_name = wandb_config.get("name", None)
    job_type = wandb_config.get("job_type", runconfig["mode"])
    tags = wandb_config.get("tags", None)
    resume = wandb_config.get("resume", "auto")
    rundir = runconfig.get("model_dir", "./model_dir")
    previous_run_id = None

    import glob

    run_files = glob.glob(os.path.join(rundir, "wandb", "run-*"))
    if run_files:
        previous_run_id = run_files[0].split('-')[-1]

    if resume == "never":
        if (
            run_id is not None
            and previous_run_id is not None
            and run_id == previous_run_id
        ):
            raise ValueError(
                f"The specified run_id ({run_id}) matches with a previous_run_id ({previous_run_id}) "
                "but 'never' mode requires them to be different."
            )
        check_presence_of_wandb_dir(rundir)
    elif resume in ["allow", "auto"]:
        if run_id is not None and previous_run_id is not None:
            if run_id == previous_run_id:
                # Log into this previous run as it's the same run
                pass
            else:
                ## Raise an error if a wandb run already exists inside the specified run dir.
                check_presence_of_wandb_dir(rundir)
        elif previous_run_id is not None:
            # No new run ID provided, so default to the previous run ID
            run_id = previous_run_id
    elif resume == "must" and previous_run_id:
        if run_id is not None and run_id != previous_run_id:
            raise ValueError(
                f"The specified run_id ({run_id}) does not match previous_run_id ({previous_run_id}) "
                "but resume mode 'must' requires them to be the same."
            )
        else:
            run_id = previous_run_id

    with wandb.init(
        dir=rundir,
        job_type=job_type,
        config=params,
        project=wandb_project,
        group=group,
        tags=tags,
        name=run_name,
        id=run_id,
        resume=resume,
        sync_tensorboard=True,
    ) as run:
        yield run
