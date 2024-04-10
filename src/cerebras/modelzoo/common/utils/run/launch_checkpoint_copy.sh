#!/usr/bin/bash

set -e
set -x

model_dir_colo=$1
model_dir_aws=$2
ckpt_name=$3
host_name=$4

scp $host_name:$model_dir_colo/$ckpt_name $model_dir_aws/$ckpt_name
