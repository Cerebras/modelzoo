# Shuffling Samples for HDF5 dataset of GPT Models

## Input files format

The script expects the dataset files to be in `.h5` file format. They can be located either in the `input_dir` or one of its subdirectories.

## Generating Shuffled HDF5 Dataset

```bash
bash launch_h5_shuffle.sh <path/to/input_dir> <path/to/output_dir> <num_chunks> <num_workers> <multi_modal_flag>
```

Here `num_chunks` is the number of output HDF5 files and `multi_modal_flag` denotes if the dataset is multimodal or not. This flag 
should be either empty or `--multi_modal`.

## Output files structure

The output directory will contain a bunch of `h5` files as shown below:

```bash
<path/to/output_dir>
├── logs/
├── shuf_split/
├── workers/
├── 0/
├── 1/
├── 2/
├── ⋮
├── data-00000.h5
├── data-00001.h5
├── data-00002.h5
└── ⋮
```

- The numbered directories (`0/`, `1/`, ...) are temporary directories that were created while shuffling the samples and can be removed at the end of the run. 
- `logs/` contains the logs of each worker. 
- `shuf_split/` contains `.shuf` files for each HDF5 input file to indicate it has completed reading all samples from that file.
- `workers/` contains `.txt` files for each worker to indicate that it has completed reading all samples from the HDF5 files that were assigned to it. They are used to sync between workers.
