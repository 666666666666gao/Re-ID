#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/mmreid-trifusion/TriFusion-ReID"
RUN_ROOT="/root/mmreid-trifusion/runs/demo_rgbnt201_seed42_b32k4_tb64"
PYTHON_BIN="/root/miniconda3/envs/tri_reid/bin/python"
MAX_IDLE_GPU_MEMORY_MIB=512

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing tri_reid Python: ${PYTHON_BIN}" >&2
  exit 2
fi
cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

exec "${PYTHON_BIN}" tools/run_demo_baseline.py \
  --output-dir "${RUN_ROOT}" \
  --max-epochs 50 \
  --eval-period 1 \
  --checkpoint-period 10 \
  --batch-size 32 \
  --test-batch-size 64 \
  --num-instances 4 \
  --workers 4 \
  --seed 42 \
  --max-idle-gpu-memory-mib "${MAX_IDLE_GPU_MEMORY_MIB}"
