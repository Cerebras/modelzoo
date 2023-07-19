#! /bin/bash
# Run pile dataset generation for given input directory of PILE
# settings include GPT2, NeoX Tokenizer

metadata_files=$1
input_dir=$2
processes=$3
tokenizer=$4
msl=$5
output_dir=$6
job_name=$7
seed=$8
files_per_record=$9

if [ "$tokenizer" == "GPT2Tokenizer" ]
then
    vocab_file=/cb/ml/language/datasets/pile_original/raw_data/vocab/gpt2-vocab.bpe
    encoder_file=/cb/ml/language/datasets/pile_original/raw_data/vocab/gpt2-encoder.json
    ftfy_normalizer="NFC"
elif [ "$tokenizer" == "NeoXTokenizer" ]
then
    vocab_file=None
    encoder_file=/cb/ml/language/datasets/pile_original/raw_data/vocab/neox-20B-tokenizer.json
    ftfy_normalizer=None
else
    echo "ERROR: UNKOWN dataset type: $tokenizer"
    exit
fi


DISTRIBUTED_ARGS="-p cpu \
        -C m5.4xlarge \
        -c 16 \
        --time 1-12:00:00 \
        -J $job_name \
        -o $job_name.out"


if [ "$metadata_files" != "None" ]
then
    FULL_FLAGS="python -B create_hdf5_dataset.py preprocessed_text \
            --metadata_files $metadata_files \
            --tokenizer_type $tokenizer \
            --vocab_file $vocab_file \
            --encoder_file $encoder_file \
            --max_seq_length $msl \
            --output_dir $output_dir \
            --seed $seed \
            --processes $processes \
            --ftfy \
            --ftfy_normalizer $ftfy_normalizer \
            --write_remainder \
            --display_pbar \
            --files_per_record $files_per_record"
else
    FULL_FLAGS="python -B create_hdf5_dataset.py preprocessed_text \
            --input_dir $input_dir \
            --tokenizer_type $tokenizer \
            --vocab_file $vocab_file \
            --encoder_file $encoder_file \
            --max_seq_length $msl \
            --output_dir $output_dir \
            --seed $seed \
            --processes $processes \
            --ftfy \
            --ftfy_normalizer $ftfy_normalizer \
            --write_remainder \
            --display_pbar \
            --files_per_record $files_per_record"
fi

echo "Running: "
echo "cbrun -t cpu srund -x $DISTRIBUTED_ARGS -e $FULL_FLAGS"
cbrun -t cpu srund -x "$DISTRIBUTED_ARGS" -e "$FULL_FLAGS"
