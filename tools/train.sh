#!/bin/bash

set -euo pipefail

export NO_ALBUMENTATIONS_UPDATE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 20000))}"
export RFDETR_DEBUG_FIRST_BATCH="${RFDETR_DEBUG_FIRST_BATCH:-0}"

# Single GPU training (commented out)
# python tools/train.py \
#     --model medium \
#     --dataset-dir datasets/coco8/images \
#     --epochs 2 \
#     --batch-size 1 \
#     --output-dir test_val_folder \
#     --no-tensorboard

# Multi GPU distributed training (default)
DATASET_DIRS=(
    "datasets/车型检测/ruqi_wuxi0728_yolo"
    # "datasets/车型检测/s17_2023_yolo"
    "datasets/车型检测/TYJT_2022_yolo"
    "datasets/车型检测/TYJT_p053_yolo"
    "datasets/车型检测/TYJT_p054_yolo"
    "datasets/车型检测/xiandao_2023_yolo"
)

DATASET_ARGS=()
for dataset_dir in "${DATASET_DIRS[@]}"; do
    DATASET_ARGS+=(--dataset-dir "$dataset_dir")
done

MODEL_NAME="small"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

uv run --no-sync python -c "from tools.train import get_model_from_factory; get_model_from_factory('${MODEL_NAME}')"

uv run --no-sync python -m torch.distributed.run \
    --nproc_per_node="${NPROC_PER_NODE}" \
    tools/train.py \
    --model "${MODEL_NAME}" \
    --dataset-file yolo \
    "${DATASET_ARGS[@]}" \
    --epochs 1 \
    --batch-size 8 \
    --num-workers 0 \
    --lr 1e-4 \
    --output-dir output/20260310_small \
    --use-ema
