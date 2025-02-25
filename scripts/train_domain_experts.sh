#!/bin/bash

ROOT=$(dirname "$(pwd)")

for config in "train_iter1_expert_l_domain_math_l1.yaml" "train_iter1_expert_m_domain_Caselaw_Access_Project.yaml" "train_iter1_expert_m_domain_History_and_events.yaml" "train_iter1_expert_m_domain_math_l1.yaml" "train_iter1_expert_s_domain_Caselaw_Access_Project.yaml" \
            "train_iter2_MHC_expert_l_domain_physics_l1.yaml" "train_iter2_MHC_expert_m_domain_Human_activites.yaml" "train_iter2_MHC_expert_m_domain_physics_l1.yaml" "train_iter2_MHC_expert_m_domain_simple_wikipedia_LM.yaml" "train_iter2_MHC_expert_s_domain_simple_wikipedia_LM.yaml" \
            "train_iter3_MHC_PHS_expert_l_domain_cs_l1.yaml" "train_iter3_MHC_PHS_expert_m_domain_Philosophy_and_thinking.yaml" "train_iter3_MHC_PHS_expert_m_domain_TinyStories.yaml" "train_iter3_MHC_PHS_expert_m_domain_cs_l1.yaml" "train_iter3_MHC_PHS_expert_s_domain_TinyStories.yaml" \
            ; do
 
    pdm run torchrun \
        --nnodes=1 \
        --nproc-per-node=8 \
        --rdzv-backend=c10d \
        --rdzv-endpoint=localhost:0 \
        --node_rank=0 \
        "$ROOT/hdee/src/hdee/train_domain.py" \
        --config-path "$ROOT/hdee/configs/hdee_3_iterations/train_domains" \
        --config-name $config
    
done;
