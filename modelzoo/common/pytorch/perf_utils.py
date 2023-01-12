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

from __future__ import annotations

import dataclasses
import json
import os
from typing import List

from modelzoo.common.pytorch import CBTORCH_PACKAGE, CbtorchPackage
from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch import cbtorch

if CBTORCH_PACKAGE == CbtorchPackage.SRC:
    from cerebras.workflow.python.cs_perf_analyzer import (
        AssessBottlenecks,
        locate_csperf_log,
    )
elif CBTORCH_PACKAGE == CbtorchPackage.WHEEL:
    from cerebras_pytorch.workflow.python.cs_perf_analyzer import (
        AssessBottlenecks,
        locate_csperf_log,
    )
else:
    assert False, f"This file should only be imported when running on CS-X."

# XLA counter keys used to track various metrics
KEY_COMPILE_TIME = "CerebrasCompileTimeMs"
KEY_PROGRAMMING_TIME = "CerebrasProgrammingTimeNs"
KEY_SYSTEM_PERF = "CerebrasSystemEstSamples"


@dataclasses.dataclass
class PerfData:
    """Data structure for holding performance data.

    Args:
        total_samples: Total number of samples processes.
        total_time: Total time spent processing those samples.
        samples_per_sec: Throuput of processing those samples.
        compile_time: Time spent compiling the model.
        programming_time: Time spent programming the fabric.
        est_samples_per_sec: Estimated throughput based on compile and fabric.
    """

    total_samples: int = 0
    total_time: float = 0.0
    samples_per_sec: float = 0.0
    compile_time: float = 0.0
    programming_time: float = 0.0
    est_samples_per_sec: float = 0.0

    def merge(self, other: PerfData):
        """Merge another `PerfData` instance into self.

        Args:
            other: The other `PerfData` instance to merge.
        """
        self.total_samples += other.total_samples
        if self.total_time == 0.0:
            self.total_time = other.total_time
        self.compile_time = max(self.compile_time, other.compile_time)
        self.programming_time = max(
            self.programming_time, other.programming_time
        )
        if self.est_samples_per_sec == 0.0:
            self.est_samples_per_sec = other.est_samples_per_sec
        else:
            assert (
                self.est_samples_per_sec == other.est_samples_per_sec
            ), "Expected all fabric-based performance estimates to be identical"

        if self.total_time > 0:
            self.samples_per_sec = float(self.total_samples) / self.total_time
        else:
            self.samples_per_sec = 0.0

    def throughput_dict(self) -> dict:
        return {
            key: getattr(self, key)
            for key in ("total_samples", "total_time", "samples_per_sec")
        }


def _get_optional_counter(name: str, default: int = 0) -> int:
    """Return XLA counter value by name.

    Args:
        name: Name of the XLA counter.
        default: Default value to return if XLA counter with that name does not
            exist. Defaults to 0.
    Returns:
        The counter value.
    """
    counter = cm.metrics_counter_value(name)
    if counter is None:
        return default
    return counter


def collect_perf_data(tracker: cm.RateTracker):
    """Collect performance data from a run.

    Args:
        tracker: Tracker which contains performance data.
    Returns:
        A PerfData instance containing the perf data.
    """
    pd = PerfData()

    pd.compile_time = _get_optional_counter(KEY_COMPILE_TIME) * 1e-3
    pd.programming_time = _get_optional_counter(KEY_PROGRAMMING_TIME) * 1e-9
    pd.est_samples_per_sec = _get_optional_counter(KEY_SYSTEM_PERF)

    pd.total_samples = tracker._partial_count + tracker._count
    pd.samples_per_sec = tracker.global_rate()
    if pd.samples_per_sec > 0:
        pd.total_time = float(pd.total_samples) / pd.samples_per_sec
    else:
        pd.total_time = 0.0

    return pd


def _assess_input_bottlecks():
    detected_bottleneck = None

    log_location = "cerebras_logs/job_[!default]*"
    model_dir = os.path.join(cbtorch.env().service_workdir, log_location)

    perf_file = locate_csperf_log(model_dir)
    if perf_file is None:
        detected_bottleneck = "Not Applicable"
        return detected_bottleneck

    input_assessment = AssessBottlenecks(perf_file)
    detected_bottleneck = input_assessment.assess_bottleneck_main()

    return detected_bottleneck


def _aggregate_perf_data(perf_all_ordinals: List[str]):
    """Aggregate performance data from multiple workers.

    Args:
        files: List of objects containing individual worker performance data.
    Returns:
        The aggregated performance metrics.
    """
    aggregate = {}

    pd = PerfData()
    for data in perf_all_ordinals:
        worker_pd = PerfData(**json.loads(data))
        if len(perf_all_ordinals) > 1:
            aggregate.setdefault("ordinals", [])
            aggregate["ordinals"].append(worker_pd.throughput_dict())

        pd.merge(worker_pd)

    aggregate.update(dataclasses.asdict(pd))
    return aggregate


def _rendezvous(payload: str):
    """Perform a rendezvouz across workers and exchange payload.

    Args:
        payload: String data from each worker to exchange.
    Returns:
        The list of payloads passed by all ordinals, ordered by ordinal number.
    """
    payloads = cm.rendezvous("save_individual_perf_data", payload=payload)
    if not payloads:
        # no mesh service (i.e., single ordinal)
        payloads = [payload]
    return payloads


def save_perf(outdir: str):
    """Utility method for saving performance metrics from a run.

    Args:
        outdir: Output directory to write performance files to.
    """
    tracker = cbtorch.state().rate_tracker
    if tracker is None:  # No performance data to save
        return

    perf_this_ordinal = collect_perf_data(tracker)

    # Sync across ordinals and receive the perf files
    perf_all_ordinals = _rendezvous(
        json.dumps(dataclasses.asdict(perf_this_ordinal))
    )

    # Aggregate perf data in master ordinal
    if cm.is_master_ordinal():
        aggregate = _aggregate_perf_data(perf_all_ordinals)

        if not cbtorch.env().appliance:
            # Update the "input bottleneck" performance field based on csperf.csv
            aggregate[
                "suspected_input_bottleneck (beta)"
            ] = _assess_input_bottlecks()

        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "performance.json"), "w") as fp:
            json.dump(aggregate, fp, sort_keys=True, indent=4)
