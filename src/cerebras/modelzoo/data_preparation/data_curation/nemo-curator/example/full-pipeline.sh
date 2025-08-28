#!/bin/bash

set -e

# Cancel any prev pending jobs
cbrun -- scancel -u $USER
rm -fr logs fineweb-output

# Start datatrove pipeline
python fineweb.py


# Save current screen
tput smcup
trap "tput rmcup; exit" INT TERM EXIT

# Wait until the prev datatrove jobs are done
while true; do
    sleep 2

    # Get all active (not completed) jobs for the user
    jobs=$(squeue -u "$USER" --noheader --format="%.18i %.20j %.10T %.10M")

    clear
    echo "Live Slurm Job Monitor"
    echo "============================================="
    printf "%-18s %-20s %-10s %-10s\n" "JOBID" "NAME" "STATE" "TIME"
    echo "$jobs"

    # Exit if no jobs are left
    if [[ -z "$jobs" ]]; then
        echo -e "\nAll jobs are completed or cleared."
        break
    fi

done

# Restore screen when done
tput rmcup

# Start nemo curator jobs
rm -fr nemo-curator-jobs semdedup_cache semdedup-output
cbrun -t gpu-a10 -- sbatch start-nemo-curator.slurm
