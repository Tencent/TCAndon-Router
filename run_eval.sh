#!/bin/bash

DATASETS=("clinc150" "hwu64" "minds14" "sgd")

for DATASET in "${DATASETS[@]}"; do
    python evaluate.py --dataset "$DATASET" --processes 8 --limit 32
done