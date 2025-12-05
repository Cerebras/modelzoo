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

"""Contains the WandbLogger class for logging metrics to Weights and Biases."""

from typing import List, Literal, Optional
from warnings import warn

import torch

from cerebras.modelzoo.trainer.loggers import Logger


class WandbLogger(Logger):
    """
    Logger class for logging metrics to Weights and Biases.
    """

    def __init__(
        self,
        project: Optional[str] = None,
        group: Optional[str] = None,
        run_id: Optional[str] = None,
        run_name: Optional[str] = None,
        job_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        resume: Literal["never", "allow", "auto", "must"] = "auto",
        entity: Optional[str] = None,
    ):
        """
        Args:
            project: The name of the project to which the run belongs.
            group: The name of the group to which the run belongs.
            run_id: The unique identifier for the run.
            run_name: The name of the run.
            job_type: The type of job.
            tags: List of tags to be associated with the run.
            resume: Resume mode for the run. It can be one of the following:
                - "never": Do not resume the run.
                - "allow": Allow the run to resume if a previous run exists.
                - "auto": Automatically resume the run if a previous run exists.
                - "must": Resume the run if a previous run exists.
            entity: An entity is a username or team name where you're sending runs.
                This entity must exist before you can send runs there,
                so make sure to create your account or team in the UI
                before starting to log runs.
        """
        self.project = project
        self.group = group
        self.run_id = run_id
        self.run_name = run_name
        self.job_type = job_type
        self.tags = tags
        self.resume = resume
        self.entity = entity

    def pre_setup(self, trainer):  # pylint: disable=no-self-use
        try:
            # pylint: disable=unused-import
            import wandb  # noqa
        except ImportError:
            raise RuntimeError(
                "wandb is an optional dependency of modelzoo. "
                "In order to use it, 'pip install wandb==0.16.2' into this venv"
            )

    def finalize(self):
        try:
            import wandb

            if wandb.run is not None:
                wandb.run.finish()
        finally:
            pass

    def check_presence_of_wandb_dir(  # pylint: disable=no-self-use
        self, rundir
    ):
        """Check if the wandb directory is present in the run directory.

        Args:
            rundir: The directory where the run is being stored.
        """
        # Ensure that the wandb directory is not already present or empty
        wandb_dir = rundir / "wandb"
        if wandb_dir.exists():
            # Ensure there are no run-* folders in the wandb directory.
            if any(
                dir.is_dir() and dir.name.startswith('run-')
                for dir in wandb_dir.iterdir()
            ):
                raise FileExistsError(
                    f"A previous run seems to already exist in {wandb_dir}. "
                    "Please specify a different 'model_dir'."
                )

    def setup(self, trainer):
        import wandb
        from wandb.sdk.lib import RunDisabled
        from wandb.wandb_run import Run

        rundir = trainer.model_dir
        previous_run_id = None

        run_files = list((rundir / "wandb").glob("run-*"))
        if run_files:
            previous_run_id = str(run_files[0]).split('-')[-1]

        if self.resume == "never":
            if (
                self.run_id is not None
                and previous_run_id is not None
                and self.run_id == previous_run_id
            ):
                raise ValueError(
                    f"The specified run_id ({self.run_id}) matches with a "
                    f"previous_run_id ({previous_run_id}) "
                    "but 'never' mode requires them to be different."
                )
            self.check_presence_of_wandb_dir(rundir)
        elif self.resume in ["allow", "auto"]:
            if self.run_id is not None and previous_run_id is not None:
                if self.run_id == previous_run_id:
                    # Log into this previous run as it's the same run
                    pass
                else:
                    # Raise an error if a wandb run already exists inside the specified run dir.
                    self.check_presence_of_wandb_dir(rundir)
            elif previous_run_id is not None:
                # No new run ID provided, so default to the previous run ID
                self.run_id = previous_run_id
        elif self.resume == "must" and previous_run_id:
            if self.run_id is not None and self.run_id != previous_run_id:
                raise ValueError(
                    f"The specified run_id ({self.run_id}) does not match "
                    f"previous_run_id ({previous_run_id}) "
                    "but resume mode 'must' requires them to be the same."
                )

            self.run_id = previous_run_id

        if wandb.run is None:
            # pylint: disable=all
            self.run = wandb.init(
                dir=rundir,
                job_type=self.job_type,
                # config=params,
                project=self.project,
                group=self.group,
                tags=self.tags,
                name=self.run_name,
                id=self.run_id,
                resume=self.resume,
                entity=self.entity,
            )
            # define default x-axis
            if isinstance(self.run, (Run, RunDisabled)) and getattr(
                self.run, "define_metric", None
            ):
                self.run.define_metric("global_step")
                self.run.define_metric(
                    "*", step_metric="global_step", step_sync=True
                )
        else:
            self.run = wandb.run

    def log_metrics(self, metrics, step):
        m = {"global_step": step}
        summary = {}
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    m[name] = value.item()
                else:
                    warn(
                        "Attempting to log a non-scalar tensor for {name}. "
                        "WandB Logger does not support logging non-scalar tensors."
                    )
            elif isinstance(value, (int, float)):
                m[name] = value
            elif isinstance(value, str):
                summary[name] = value
            else:
                try:
                    import pandas as pd
                    import wandb

                    if isinstance(value, pd.DataFrame):
                        m[name] = wandb.Table(dataframe=value)
                        continue
                except ImportError:
                    pass

                warn(
                    f"Attempting to log a {type(value)} for {name}. "
                    f"WandB Logger does not support logging {type(value)}"
                )

        self.run.log(m)
        self.run.summary.update(summary)

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass
