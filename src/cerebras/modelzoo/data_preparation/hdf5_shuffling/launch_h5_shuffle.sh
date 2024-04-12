#! /bin/bash



# Input directory containing *.h5 or subdirectories containing *.h5 files
indir=$1
# Output directory in which to create the shuffled *.h5 files
# The output directory will be a flat directory with all resulting *.h5 files
outdir=$2
# Number of output *.h5 file chunks
numchunks=$3
# number of workers
workers=$4

if [[ $# -ge 5 ]]; then 
	multi_modal_flag=$5
else
	multi_modal=""
fi

# Make sure that the provided multimodal flag is valid
if [[ "$multi_modal_flag" != "" && "$multi_modal_flag" != "--multi_modal" ]]; then
    echo "Error: The multi_modal_flag argument must be either an empty string or \"--multi_modal\""
    exit 1
fi

mkdir -p $outdir/logs

# Recommend running with up to 40 worker processes
# Recommend estimating shuffle time as: (dataset chunks total size) / (2 MB/s)
# For instance, 1 GB is likely to take roughly 8 minutes of total worker time
# This process parallelizes quite well as long as the input chunks are all roughly
# the same size, so we can estimate that the time will be roughly divided by
# the number of workers. Would recommend doubling this value for --time, just in
# case there is heavy disk access that causes slowdowns.

for ((i=0; i<workers; i++))
do
	python h5_dataset_shuffle.py $indir $outdir $numchunks --num_parallel $workers --worker_id $i $multi_modal_flag > $outdir/logs/shuf-$i.txt 2>&1 &
done
