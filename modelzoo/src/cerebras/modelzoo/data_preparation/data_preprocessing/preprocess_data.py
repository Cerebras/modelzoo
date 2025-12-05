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

"""
Script to generate an HDF5 dataset for GPT Models.
"""

# isort: off
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))
# isort: on
import copy
import json
import logging
import os
import re
import signal
import sys

from cerebras.modelzoo.data_preparation.data_preprocessing.data_preprocessor import (
    DataPreprocessor,
)
from cerebras.modelzoo.data_preparation.data_preprocessing.data_split import (
    DataSplit,
)
from cerebras.modelzoo.data_preparation.data_preprocessing.subset_split import (
    SubsetSplit,
)
from cerebras.modelzoo.data_preparation.data_preprocessing.utils import (
    MultiprocessingExitEvent,
    ProgressMonitor,
    SLURMPipe,
    YamlReader,
    get_params,
)

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

os.environ["SUBMITIT_EXECUTOR_WAIT_TIME"] = "120"


# Signal handlers
def setup_signal_handlers(event_name: str):
    """
    Setup signal handlers to trigger the exit event and return the event object.

    :param event_name: The name of the event to trigger.
    :return: The event object MultiprocessingExitEvent.
    """
    # Determine which event to use
    event = MultiprocessingExitEvent(name=event_name)

    def handle_signal(signum, frame):
        logger.info(
            f"Received signal {signum}. Setting exit event: {event_name}"
        )
        event.set()  # Set the event to signal exit
        sys.exit(0)  # Exit after handling the signal

    # Register the signal handler
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT]:
        signal.signal(sig, handle_signal)

    # Return the event object for later use
    return event


def generate_params_list(params):
    subset_class = SubsetSplit(params)
    params_list = subset_class.generate_subsets()
    final_params_list = []
    ## Perform a dataset split on each of the data subsets individually
    for subset_param in params_list:
        split_class = DataSplit(subset_param)
        if not split_class.do_split:
            ## If no split needs to be done just return with subset params list. If a split needs to be done it will be done on all subsets.
            return params_list
        split_class.prepare_splits()
        split_class.split()
        final_params_list.extend(split_class.get_params_list())

    return final_params_list


def merge_data_params(output_dir: str, num_nodes: int):
    """
    Merge per-node data_params_{rank}.json files into a single merged data_params.json.
    Some stats are summed, others are averaged, and metadata is copied from the first file.

    Args:
        output_dir (str): Directory containing data_params JSON files
        num_nodes (int): Number of nodes/files to merge
    """
    files = [
        os.path.join(output_dir, f"data_params_{rank}.json")
        for rank in range(num_nodes)
    ]

    summed_keys = {
        "discarded_files",
        "loss_valid_tokens",
        "n_examples",
        "non_pad_tokens",
        "normalized_bytes_count",
        "normalized_chars_count",
        "num_masked_tokens",
        "num_pad_tokens",
        "num_tokens",
        "processed_files",
        "raw_bytes_count",
        "raw_chars_count",
        "successful_files",
    }

    averaged_keys = {
        "average_bytes_per_sequence",
        "average_chars_per_sequence",
    }

    meta_keys = {"dataset", "features", "processing", "setup"}

    merged_sums = {}
    weighted_avgs = {}
    total_examples = 0
    meta_data = {}

    for i, fpath in enumerate(files):
        if not os.path.exists(fpath):
            logger.warning(f"Missing file {fpath}, skipping.")
            continue

        with open(fpath, 'r') as f:
            data = json.load(f)

        stats = data.get("post-process", {})
        n_examples = stats.get("n_examples", 0)

        if n_examples == 0:
            logger.warning(f"File {fpath} has zero n_examples, skipping.")
            continue

        total_examples += n_examples

        for key in summed_keys:
            merged_sums[key] = merged_sums.get(key, 0) + stats.get(key, 0)

        for key in averaged_keys:
            weighted_avgs[key] = (
                weighted_avgs.get(key, 0.0) + stats.get(key, 0.0) * n_examples
            )

        if i == 0:
            for key in meta_keys:
                if key in data:
                    meta_data[key] = data[key]

    if total_examples == 0:
        logger.error("No valid data found to merge.")
        raise RuntimeError("No valid data found to merge.")

    merged_stats = dict(merged_sums)
    for key in averaged_keys:
        merged_stats[key] = weighted_avgs.get(key, 0.0) / total_examples

    merged_data = {
        "post-process": merged_stats,
        **meta_data,
    }

    merged_file = os.path.join(output_dir, "data_params.json")
    with open(merged_file, "w") as f:
        json.dump(merged_data, f, indent=2)

    logger.info(f"Merged data_params saved to {merged_file}")


def preprocess_node(params):
    import os

    os.environ["SUBMITIT_EXECUTOR_WAIT_TIME"] = "120"
    exit_event = setup_signal_handlers("node_exit_event")
    return preprocess_data(params, exit_event)


def multi_node_preprocess_data(params_file: str):
    """
    Enhanced multi-node data preprocessing with comprehensive SLURM support.

    Args:
        params_file (str): Path to YAML parameters file
    """
    cP = YamlReader(params_file)
    slurm_config = cP.get('setup.slurm', None)

    exit_event = setup_signal_handlers("exit_event")

    # Get number of nodes
    num_nodes = cP.get('setup.num_nodes', 1)
    # Generate expected nodes directly from num_nodes
    expected_nodes = [f"{i}" for i in range(num_nodes)]

    if slurm_config is None or num_nodes == 1:
        # Fallback to local processing
        preprocess_data(cP.params, exit_event)
        return

    # Validate required SLURM parameters
    required_params = ['queue', 'cores', 'memory', 'walltime']
    missing_params = [p for p in required_params if p not in slurm_config]
    if missing_params:
        raise ValueError(
            f"Missing required SLURM parameters: {', '.join(missing_params)}"
        )

    # Parse SLURM configuration
    partition = slurm_config['queue']
    cpus_per_task = slurm_config['cores']

    # Parse memory (support various formats: 4GB, 4000MB, 4)
    memory_str = slurm_config.get('memory', '4GB')
    mem_match = re.search(r'(\d+)([GM]?B?)', memory_str.upper())
    if mem_match:
        mem_value, mem_unit = mem_match.groups()
        mem_value = int(mem_value)
        if mem_unit.startswith('M'):
            mem_gb = max(1, mem_value // 1024)  # Convert MB to GB, minimum 1GB
        else:
            mem_gb = mem_value
    else:
        mem_gb = 4  # Default fallback

    # Parse walltime
    walltime_str = slurm_config.get('walltime', '01:00:00')
    try:
        time_parts = walltime_str.split(':')
        if len(time_parts) == 3:
            h, m, s = map(int, time_parts)
        elif len(time_parts) == 2:
            h, m, s = 0, int(time_parts[0]), int(time_parts[1])
        else:
            h, m, s = 0, 0, int(time_parts[0])
        timeout_min = h * 60 + m + (1 if s > 0 else 0)
    except (ValueError, IndexError):
        timeout_min = 60  # Default 1 hour

    output_dir = cP.get("setup.output_dir", "./output/")

    # Initialize enhanced SLURMPipe
    slurm_pipe = SLURMPipe(
        partition=partition,
        cpus_per_task=cpus_per_task,
        mem_gb=mem_gb,
        timeout_min=timeout_min,
        log_dir=slurm_config.get('log_dir', "slurm_logs"),
        # Additional configurations
        ntasks_per_node=slurm_config.get('ntasks_per_node', 1),
        array_parallelism=slurm_config.get('array_parallelism'),
        max_retries=slurm_config.get('max_retries', 3),
        retry_delay=slurm_config.get('retry_delay', 60),
    )

    # Optional progress monitor settings with good defaults from slurm_config
    refresh_rate = slurm_config.get(
        'progress_refresh_rate', 2.0
    )  # 2 second refresh
    stall_threshold = slurm_config.get(
        'progress_stall_threshold', 300.0
    )  # 5 minutes

    # Simple monitoring selection: Progress Monitor by default, Job Tracking if explicitly enabled
    enable_job_tracking = slurm_config.get('enable_job_tracking', False)
    enable_progress_monitor = not enable_job_tracking  # Mutually exclusive

    dp = DataPreprocessor(copy.deepcopy(cP.params), None)

    if enable_job_tracking:
        logger.info("Monitoring mode: Job Tracking (detailed logging)")
    else:
        logger.info("Monitoring mode: Progress Monitor (visual display)")

    # Initialize progress monitor (default behavior)
    progress_monitor = None
    if enable_progress_monitor:
        try:
            progress_monitor = ProgressMonitor(
                checkpoint_dir=output_dir,
                expected_nodes=expected_nodes,
                processes_per_node=dp.writer_process_num,
                total_chunks=dp.total_chunks // num_nodes,
                refresh_rate=refresh_rate,
                stall_threshold=stall_threshold,
                logger=logger,
            )
            progress_monitor.start()
            logger.info(f"Progress monitor started for {num_nodes} nodes")
        except Exception as e:
            logger.warning(f"Failed to start progress monitor: {e}")
            progress_monitor = None
            # Fallback to job tracking
            enable_job_tracking = True
            logger.info("Falling back to job tracking mode")

    try:
        # Submit jobs to SLURM
        param_list = []
        for rank in range(num_nodes):
            cP.set('setup.rank', rank)
            params = copy.deepcopy(cP.params)
            param_list.append((params,))

        # Launch jobs in batch
        jobs = slurm_pipe.launch_batch(preprocess_node, param_list)

        # Track job completion - only if job tracking is enabled
        if enable_job_tracking:
            results = slurm_pipe.track_jobs(
                poll_interval=slurm_config.get('poll_interval', 30),
                detailed_status=slurm_config.get('detailed_status', True),
            )
        else:
            # Minimal job tracking when progress monitor is active
            logger.info(
                "Job tracking disabled - using progress monitor for status"
            )
            results = (
                slurm_pipe.wait_for_completion()
            )  # Assume this method exists for minimal waiting

        # Report final results
        completed_count = len(results['completed'])
        failed_count = len(results['failed'])

        if failed_count > 0:
            logger.warning(
                f"Warning: {failed_count} out of {num_nodes} jobs failed"
            )
            if slurm_config.get('fail_on_any_error', False):
                raise RuntimeError(f"{failed_count} jobs failed")

        logger.info(
            f"Job execution complete: {completed_count} succeeded, {failed_count} failed"
        )

        # Print final progress summary if monitor was used
        if progress_monitor:
            final_summary = progress_monitor.get_progress_summary()
            logger.info(
                f"Final progress: {final_summary['overall']['completion_percent']:.1f}% complete"
            )

    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        slurm_pipe.cancel_all_jobs()
        raise
    except Exception as e:
        logger.info(f"Execution failed: {e}")
        slurm_pipe.cancel_all_jobs()
        raise
    finally:
        # Always stop progress monitor
        if progress_monitor:
            progress_monitor.stop()
            logger.info("Progress monitor stopped")

    merge_data_params(output_dir=output_dir, num_nodes=num_nodes)


def main():
    """Main function for execution."""

    params = get_params(desc="Create HDF5 dataset for language models")
    multi_node_preprocess_data(params)


def preprocess_data(params, exit_event: MultiprocessingExitEvent = None):
    params_list = generate_params_list(params)

    for updated_params in params_list:
        dataset_processor = DataPreprocessor(updated_params, exit_event)
        dataset_processor.process_dataset()
        output_dir = dataset_processor.get_output_dir()
        json_params_file = dataset_processor.get_params_file()
        # Retrieve vocab size and log completion
        vocab_size = dataset_processor.get_vocab_size()
        logger.info(
            f"\nFinished writing data to {output_dir}."
            f" Args & outputs can be found at {json_params_file}."
        )


if __name__ == "__main__":
    main()
