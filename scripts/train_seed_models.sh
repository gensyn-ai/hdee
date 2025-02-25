#!/bin/bash

ROOT=$(dirname "$(pwd)")

if [ ! -d "$ROOT/hdee/experiments/" ]; then
    mkdir $ROOT/hdee/experiments/;
fi;

mkdir $ROOT/hdee/experiments/seed_models

for config in "train_seed_expert_s.yaml" "train_seed_expert_m.yaml" "train_seed_expert_l.yaml"; do
 
    pdm run torchrun \
        --nnodes=1 \
        --nproc-per-node=8 \
        --rdzv-backend=c10d \
        --rdzv-endpoint=localhost:0 \
        --node_rank=0 \
        "$ROOT/hdee/src/hdee/train.py" \
        --config-path "$ROOT/hdee/configs/hdee_seed_models" \
        --config-name $config
    
done;

