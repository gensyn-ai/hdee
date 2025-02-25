#!/bin/bash
# Data domain name
DATA_DOMAIN=$1;

ROOT=$(dirname "$(pwd)");

if [ ! -d "$ROOT/hdee/dataset/" ]; then
    mkdir $ROOT/hdee/dataset/;
fi;

if [ ! -d "$ROOT/hdee/dataset/${DATA_DOMAIN}/" ]; then
    echo "Processing $DATA_DOMAIN"
    mkdir $ROOT/hdee/dataset/${DATA_DOMAIN}/;
    for data_partition in "train" "validation" "test"; do
    mkdir $ROOT/hdee/dataset/${DATA_DOMAIN}/${data_partition}/;
        pdm run torchrun \
        "$ROOT/hdee/src/gensyn_dataprep/dataprep/pretokenize_data.py" \
            --num_processes 8 \
            --dataset_name machelreid/m2d2 \
            --domain_name ${DATA_DOMAIN} \
            --partition ${data_partition} \
            --textfield_name text \
            --output_prefix $ROOT/hdee/dataset/${DATA_DOMAIN}/${data_partition} \
            --tokenizer_name <TOKEN_NAME> \
            --access_token <TOKEN> \
            --sequence_length 1024 \
            --number_of_samplers_per_file 10000;
    done;
else echo "$DATA_DOMAIN already exists."
fi;
