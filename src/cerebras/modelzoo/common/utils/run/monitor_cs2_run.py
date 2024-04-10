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

import argparse
import getpass
import logging
import os
import re
import subprocess
import time
from pathlib import Path

import paramiko

logging.basicConfig(
    format='%(asctime)s %(name)s: %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
)

PT_CKPT_PATTERN = r"checkpoint_\d+.mdl"
TF_CKPT_PATTERN = r"model.ckpt-\d+"
CKPT_PATTERN = f"({PT_CKPT_PATTERN})|({TF_CKPT_PATTERN})"

# don't copy a checkpoint unless it's been untouched for 2 minutes
CKPT_UNTOUCHED_THRESHOLD = 2 * 60


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir_colo")
    parser.add_argument("--remote_host")
    parser.add_argument("--model_dir_aws")
    parser.add_argument(
        "--coarse_checkpoint_steps",
        type=int,
        help=(
            "The frequency with which checkpoints are saved with the "
            "intension of long term storage and analysis. Often this interval "
            "is coarser than the frequency with which checkpoints are saved "
            "for restart purposes, see '--keep_last_n_checkpoints'"
        ),
    )
    parser.add_argument(
        "--keep_last_n_checkpoints",
        type=int,
        help=(
            "How many checkpoints to keep on remote for restarts in adddition "
            "to those kept according to `coarse_checkpoint_steps` for long "
            "term storage and analysis"
        ),
    )
    parser.add_argument(
        "--polling_interval",
        type=int,
        default=60 * 5,
        help="How often to check for new events (in seconds)",
    )
    parser.add_argument(
        "--analyze_weights",
        action="store_true",
        help="Extract summaries from weights after copying to aws",
    )
    args = parser.parse_args()
    return args


def exists_remote(remote_host, p):
    file_exists = subprocess.call(["ssh", remote_host, f"test -f {p}"]) == 0
    dir_exists = subprocess.call(["ssh", remote_host, f"test -d {p}"]) == 0
    return file_exists or dir_exists


def ckpt_name_to_step_num(name):
    if re.fullmatch(PT_CKPT_PATTERN, name):
        return int(name[len("checkpoint_") : -len(".mdl")])
    elif re.fullmatch(TF_CKPT_PATTERN, name):
        return int(name[len("model.ckpt-") :])
    else:
        raise ValueError(
            f"attempted to extract step number from invalid checkpoint {name}"
        )


def maybe_copy_checkpoint(ckpt, args):
    ckpt_path = os.path.join(args.model_dir_colo, ckpt)
    logs_dir = os.path.join(args.model_dir_aws, "logs")
    step_num = ckpt_name_to_step_num(ckpt)

    did_something = False
    modified_time = subprocess.run(
        ["ssh", args.remote_host, "stat", ckpt_path, "-c", r"%Y"],
        capture_output=True,
        text=True,
    ).stdout
    modified_time = int(modified_time)

    # get time from remote machine to remove potential consistency
    # or time zone issues
    current_time = subprocess.run(
        ["ssh", args.remote_host, "date", r"+%s"],
        capture_output=True,
        text=True,
    ).stdout
    current_time = int(current_time)

    if current_time - modified_time > CKPT_UNTOUCHED_THRESHOLD:
        # wait a few minutes before copying checkpoints to avoid
        # copying partially written files
        did_something = True
        log_file_path = os.path.join(
            logs_dir, f"logs_process_checkpoint_{step_num}.out"
        )
        cmd = [
            "cbrun",
            "--",
            "sbatch",
            "-c4",
            "-o",
            log_file_path,
            "launch_checkpoint_copy.sh",
            args.model_dir_colo,
            args.model_dir_aws,
            ckpt,
            args.remote_host,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        slurm_id = result.stdout.split()[-1]
        logging.info(
            f"Launched copy and processing of checkpoint {ckpt} with "
            f"slurm id {slurm_id}."
        )

        # Queue up weight analysis to run after checkpoint copy
        framework = "pt" if re.fullmatch(PT_CKPT_PATTERN, ckpt) else "tf"
        aws_ckpt_path = os.path.join(args.model_dir_aws, ckpt)
        cmd = [
            "cbrun",
            "--",
            "sbatch",
            "-c8",
            "--open-mode=append",
            "-o",
            log_file_path,
            "-d",
            f"afterok:{slurm_id}",
            "write_weight_summaries.py",
            "--input_path",
            aws_ckpt_path,
            "--output_path",
            aws_ckpt_path + ".wt_summary.txt",
            "--framework",
            framework,
        ]
        if args.analyze_weights:
            result = subprocess.run(cmd, capture_output=True, text=True)
            summaries_slurm_id = result.stdout.split()[-1]
            logging.info(
                f"Queued weight analysis of checkpoint {ckpt} with "
                f"slurm id {summaries_slurm_id} to start after job "
                f"{slurm_id} finishes successfully."
            )
    return did_something


def main():
    args = parse_args()

    if not os.path.exists(args.model_dir_aws):
        Path(args.model_dir_aws).mkdir(parents=True)

    params_coppied = False
    params_path = os.path.join(
        args.model_dir_colo, "train", "params_train.yaml"
    )

    logs_dir = os.path.join(args.model_dir_aws, "logs")
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)

    processed_checkpoints = set(
        ckpt_name_to_step_num(f)
        for f in os.listdir(args.model_dir_aws)
        if re.fullmatch(CKPT_PATTERN, f)
    )

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.client.AutoAddPolicy)
    ssh.connect(
        args.remote_host,
        username="lab",
        password=getpass.getpass(f"password for lab@{args.remote_host}: "),
    )
    sftp = ssh.open_sftp()

    while True:
        tick = time.time()

        # copy params files
        if not params_coppied and exists_remote(args.remote_host, params_path):
            logging.info(f"Copying params {args.remote_host}:{params_path}")
            sftp.get(
                params_path,
                os.path.join(args.model_dir_aws, "params_train.yaml"),
            )
            params_coppied = True

        # copy checkpoints
        all_ckpts = [
            f
            for f in sftp.listdir(args.model_dir_colo)
            if re.fullmatch(CKPT_PATTERN, f)
        ]
        all_ckpts.sort(key=ckpt_name_to_step_num)

        for i, ckpt in enumerate(reversed(all_ckpts)):
            step_num = ckpt_name_to_step_num(ckpt)
            ckpt_path = os.path.join(args.model_dir_colo, ckpt)
            if step_num in processed_checkpoints:
                continue
            elif step_num % args.coarse_checkpoint_steps == 0:
                success = maybe_copy_checkpoint(ckpt, args)
                if success:
                    processed_checkpoints.add(step_num)
            elif (
                args.keep_last_n_checkpoints is not None
                and i >= args.keep_last_n_checkpoints
            ):
                logging.info(f"Removing remote checkpoint {ckpt_path}")
                sftp.remove(ckpt_path)
        tock = time.time()
        elapsed = tock - tick
        time.sleep(max(args.polling_interval - elapsed, 0))


if __name__ == "__main__":
    main()
