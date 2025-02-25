#!/bin/bash
ROOT=$(dirname "$(pwd)");

if [ ! -d "$ROOT/hdee/dataset/" ]; then
    mkdir $ROOT/hdee/dataset/;
fi;

# Load Caselaw dataset
mkdir $ROOT/hbtm/dataset/Caselaw_Access_Project/;
mkdir $ROOT/hbtm/dataset/Caselaw_Access_Project/raw/;
pdm run torchrun \
"$ROOT/hbtm/src/gensyn_dataprep/dataprep/pretokenize_data.py" \
    --num_processes 8 \
    --dataset_name free-law/Caselaw_Access_Project \
    --domain_name default \
    --partition train \
    --textfield_name text \
    --output_prefix $ROOT/hbtm/dataset/Caselaw_Access_Project/raw \
    --tokenizer_name <TOKEN_NAME> \
    --access_token <TOKEN> \
    --sequence_length 1024 \
    --number_of_samplers_per_file 10000

# Load TinyStories dataset
mkdir $ROOT/hbtm/dataset/TinyStories/;
mkdir $ROOT/hbtm/dataset/TinyStories/train/;
pdm run torchrun \
"$ROOT/hbtm/src/gensyn_dataprep/dataprep/pretokenize_data.py" \
    --num_processes 8 \
    --dataset_name fhswf/TinyStoriesV2_cleaned \
    --domain_name train \
    --partition train \
    --textfield_name text \
    --output_prefix $ROOT/hbtm/dataset/TinyStories/train \
    --tokenizer_name <TOKEN_NAME> \
    --access_token <TOKEN> \
    --sequence_length 1024 \
    --number_of_samplers_per_file 10000

mkdir $ROOT/hbtm/dataset/TinyStories/validation/;
pdm run torchrun \
"$ROOT/hbtm/src/gensyn_dataprep/dataprep/pretokenize_data.py" \
    --num_processes 8 \
    --dataset_name fhswf/TinyStoriesV2_cleaned \
    --domain_name train \
    --partition test \
    --textfield_name text \
    --output_prefix $ROOT/hbtm/dataset/TinyStories/validation \
    --tokenizer_name <TOKEN_NAME> \
    --access_token <TOKEN> \
    --sequence_length 1024 \
    --number_of_samplers_per_file 10000

# Load Simple Wiki dataset
mkdir $ROOT/hbtm/dataset/simple_wikipedia_LM/;
for data_partition in "train" "validation" "test"; do
    mkdir $ROOT/hbtm/dataset/simple_wikipedia_LM/${data_partition}/;
    pdm run torchrun \
    "$ROOT/hbtm/src/gensyn_dataprep/dataprep/pretokenize_data.py" \
        --num_processes 8 \
        --dataset_name pszemraj/simple_wikipedia_LM \
        --domain_name default \
        --partition ${data_partition} \
        --textfield_name text \
        --output_prefix $ROOT/hbtm/dataset/simple_wikipedia_LM/${data_partition} \
        --tokenizer_name <TOKEN_NAME> \
        --access_token <TOKEN> \
        --sequence_length 1024 \
        --number_of_samplers_per_file 10000
    done;