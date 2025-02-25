#!/bin/bash

ROOT=$(dirname "$(pwd)")

for config in "eval_ensemble_mhe_iho_iter3.yaml" "eval_ensemble_mho_ihe_iter3.yaml" "eval_ensemble_mho_iho_iter3.yaml" ; do
 
    pdm run torchrun \
        --nnodes=1 \
        --nproc-per-node=1 \
        --rdzv-backend=c10d \
        --rdzv-endpoint=localhost:0 \
        --node_rank=0 \
        "$ROOT/hdee/src/hdee/eval_ensemble.py" \
        --config-path "$ROOT/hdee/configs/hdee_3_iterations/eval" \
        --config-name $config
    
done;