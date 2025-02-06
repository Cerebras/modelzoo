# This script downloads and preprocesses the C4 dataset for use in the T5 model.
# Leverages the version of the C4 dataset published at
# https://huggingface.co/datasets/allenai/c4/tree/main.
# The download of the train set takes roughly 56 cpu hours on a 4 core machine.
# To download to a different location, modify `dataset_root` in the script.

dataset_root=./c4 # Location to download the dataset to
script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if [ ! -d $dataset_root/raw ]; then
    mkdir -p $dataset_root/raw
fi
if [ ! -d $dataset_root/train ]; then
    mkdir -p $dataset_root/train
fi
if [ ! -d $dataset_root/validation ]; then
    mkdir -p $dataset_root/validation
fi

wd=$(pwd)
cd $dataset_root/raw

# Download and preprocess train data files. There are 1024 files total in the
# dataset, with urls generated based on the index of the file.
split="train"
for i in $(seq -f "%05g" 0 1023)
do
    name=c4-${split}.${i}-of-01024
    echo processing file $name
    wget https://huggingface.co/datasets/allenai/c4/resolve/main/en/${name}.json.gz\
         --no-verbose
    gzip -d ${name}.json.gz
    python\
         $script_dir/data_processing/preprocess_t5_dataset.py\
         --input_dir $dataset_root/raw\
         --output_dir $dataset_root/${split}\
         --file_name ${name}.json\
         --spiece_model $script_dir/input/data_processing/spiece.model
    rm $dataset_root/raw/${name}.json
done

# Download and preprocess validation data files. There are 8 files total in the
# dataset, with urls generated based on the index of the file.
split="validation"
for i in $(seq -f "%05g" 0 7)
do
    name=c4-${split}.${i}-of-01024
    echo processing file $name
    wget https://huggingface.co/datasets/allenai/c4/resolve/main/en/${name}.json.gz\
         --no-verbose
    gzip -d ${name}.json.gz
    python\
         $script_dir/data_processing/preprocess_t5_dataset.py\
         --input_dir $dataset_root/raw\
         --output_dir $dataset_root/${split}\
         --file_name ${name}.json\
         --spiece_model $script_dir/input/data_processing/spiece.model
    rm $dataset_root/raw/${name}.json
done

cd $wd
