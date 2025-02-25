#!/bin/bash
ROOT=$(dirname "$(pwd)");

if [ ! -d "$ROOT/hdee/dataset/" ]; then
    mkdir $ROOT/hdee/dataset/;
fi;


mkdir $ROOT/hdee/dataset/openwebtext/raw/
pdm run torchrun \
"$ROOT/hdee/src/gensyn_dataprep/dataprep/pretokenize_data.py" \
    --num_processes 32 \
    --dataset_name vietgpt/openwebtext_en \
    --partition train \
    --textfield_name text \
    --output_prefix $ROOT/hdee/dataset/openwebtext/raw \
    --tokenizer_name <TOKEN_NAME> \
    --access_token <TOKEN> \
    --sequence_length 1024 \
    --number_of_samplers_per_file 10000;

