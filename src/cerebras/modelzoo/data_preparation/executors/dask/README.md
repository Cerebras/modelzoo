## Datatrove Executors

The dask executor is an alternative to the default SLURM executor of Datatrove. 
Internally it uses SLURM only, but comes with a built in dashboard which allows us to monitor resources on a per worker basis.

Example Usage - 

```
from data_preparation.executors.dask.launcher import DaskPipelineExecutor

...

compute = DaskPipelineExecutor(
    pipeline=[
        JsonlReader(SOURCE, doc_progress=True, limit=-1, glob_pattern="test-shard-*.jsonl", text_key=args.text_key),
        SamplerFilter(
            rate=args.sample_rate,
        ),
        WordStats(
            output_folder=OUTPUT_FOLDER,
            top_k_config=top_k_config,
        ),
        DocStats(
            output_folder=OUTPUT_FOLDER,
            top_k_config=top_k_config,
        ),
    ],
    tasks=TOTAL_TASKS,
    job_name=f"summary-stats-{experiment_name}",
    time="24:00:00",
    partition="cpu-spot",
    logging_dir=f"{LOCAL_LOGS_FOLDER}-compute",
    qos="normal",
    mem_per_cpu_gb=2,
    cpus_per_task=2,
)

compute.run()
```