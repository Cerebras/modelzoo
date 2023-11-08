# ModelZoo Compatibility Scripts

The scripts in this directory exist to make older checkpoints compatible with the latest release versions.

## fix_checkpoint_spec.py

The checkpointing implementation changed fundamentally in the 2.0 Cerebras software release. As such,
Checkpoints taken before the 2.0 release may be incompatible with 2.0+ release software.

### Usage

```
usage: fix_checkpoint_spec.py [-h] [--fixed_checkpoint_path FIXED_CHECKPOINT_PATH] [--fix_inplace] checkpoint_path

Fix the specification of the checkpoints taken in releases < 2.0 to be functional for releases >= 2.0

positional arguments:
  checkpoint_path       Path to the checkpoint to fix.

optional arguments:
  -h, --help            show this help message and exit
  --fixed_checkpoint_path FIXED_CHECKPOINT_PATH
                        Path to save the fixed checkpoint to. If not provided, the fixed checkpoint will be saved to the same path as the checkpoint, with '_fixed' appended to the name.
  --fix_inplace         Fix the checkpoint inplace instead of saving to a new file.
```

## data_iter_state_conversion.py


Script to convert DataLoader state files saved in release 1.9 to DataLoader checkpoint format
for the new map and iterable DataLoaders in MZ in release 2.0. This is useful to provide
backwards comptability for deterministic restart on 2.0 runs from old dataloader state
files.


### Usage

```
usage: data_iter_state_conversion.py [-h] --old_checkpoint OLD_CHECKPOINT --worker_data_iter_files_dir WORKER_DATA_ITER_FILES_DIR --output_file OUTPUT_FILE [--dataloader_type {map,iterable}] [--shuffle_seed SHUFFLE_SEED]

Convert R1.9 DataLoader state to R2.0 DataLoader state.

optional arguments:
  -h, --help            show this help message and exit
  --old_checkpoint OLD_CHECKPOINT, -c OLD_CHECKPOINT
                        Path to the r1.9 checkpoint file
  --worker_data_iter_files_dir WORKER_DATA_ITER_FILES_DIR, -w WORKER_DATA_ITER_FILES_DIR
                        Path to directory containing data step file `data_iter_checkpoint_state_file_global` and worker checkpoint files of the format `data_iter_state_file_worker_*_step_*.txt`
  --output_file OUTPUT_FILE, -o OUTPUT_FILE
                        Path where the output R2.0 checkpoint file with the converted DataLoader state should be saved.
  --dataloader_type {map,iterable}, -d {map,iterable}
                        The MZ DataLoader for which state is being converted. Use `map` for the map-style dataloader and `iterable` for the iterable-style dataloader. Defaults to map-style dataloader.
  --shuffle_seed SHUFFLE_SEED, -s SHUFFLE_SEED
                        The seed value to be captured in the DataLoader state for the map-style dataloader. Note that the seed is only relevant for deterministically restarting the map-style dataloader if dataset shuffling/mixing is enabled.
```

