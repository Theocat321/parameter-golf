#!/bin/bash
set -e

cd /workspace/parameter-golf

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPU(s)"

RUN_ID=${RUN_ID:-baseline_sp1024} \
  torchrun --standalone --nproc_per_node="$NUM_GPUS" train_gpt.py
