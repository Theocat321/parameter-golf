#!/bin/bash
set -e

cd /workspace/parameter-golf

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "Detected $NUM_GPUS GPU(s)"

if [ "$NUM_GPUS" -eq 0 ]; then
  echo "ERROR: No GPUs found. Check that your RunPod pod has a GPU attached."
  exit 1
fi

RUN_ID=${RUN_ID:-baseline_sp1024} \
  torchrun --standalone --nproc_per_node="$NUM_GPUS" train_gpt.py
