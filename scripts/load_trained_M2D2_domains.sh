#!/bin/bash
#SET_OF_DOMAINS=$1;

ROOT=$(dirname "$(pwd)");

if [ ! -d "$ROOT/hdee/dataset/" ]; then
    mkdir $ROOT/hdee/dataset/;
fi;

#echo $SET_OF_DOMAINS

for data_domain in "math_l1" "physics_l1" "cs_l1" "History_and_events" "Philosophy_and_thinking" "Human_activites"; do
    bash ./scripts/load_single_M2D2_domain_dataset.sh $data_domain;
done;
