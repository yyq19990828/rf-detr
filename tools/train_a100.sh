#!/bin/bash

set -euo pipefail

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM

  # Propagate termination to the whole process group so torchrun workers
  # do not linger after Ctrl+C or script exit.
  kill -- -$$ 2>/dev/null || true

  exit "${exit_code}"
}

trap cleanup EXIT INT TERM
export NO_ALBUMENTATIONS_UPDATE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 20000))}"
export RFDETR_DEBUG_FIRST_BATCH="${RFDETR_DEBUG_FIRST_BATCH:-0}"
export CLEAR_YOLO_CACHE="${CLEAR_YOLO_CACHE:-0}"

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

if [[ "${CLEAR_YOLO_CACHE}" == "1" ]]; then
  echo "Clearing YOLO label caches before training..."
  for dataset_dir in "${DATASET_DIRS[@]}"; do
    find "${dataset_dir}" -path "*/labels/.cache" -delete
  done
fi

MODEL_NAME="small"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
OUTPUT_DIR="${OUTPUT_DIR:-output/20260310_small}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
PRETRAIN_WEIGHTS="${PRETRAIN_WEIGHTS:-}"

TRAIN_MODE_ARGS=()
if [[ -n "${RESUME_CHECKPOINT}" && -n "${PRETRAIN_WEIGHTS}" ]]; then
  echo "RESUME_CHECKPOINT 和 PRETRAIN_WEIGHTS 不能同时设置" >&2
  exit 1
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  TRAIN_MODE_ARGS+=(--resume "${RESUME_CHECKPOINT}")
fi
if [[ -n "${PRETRAIN_WEIGHTS}" ]]; then
  TRAIN_MODE_ARGS+=(--pretrain-weights "${PRETRAIN_WEIGHTS}")
fi

# Recommended throughput-oriented defaults for 4x A100 on local SSD:
# - grad accumulation disabled to avoid 4x forward/backward per optimizer step
# - multi-scale disabled to reduce per-step variance and compute cost
# - EDA disabled to avoid startup synchronization overhead
# Tune BATCH_SIZE upward (e.g. 10/12/16) if memory allows.

if [[ -n "${PRETRAIN_WEIGHTS}" ]]; then
  uv run --no-sync python -c "from tools.train import get_model_from_factory; get_model_from_factory('${MODEL_NAME}', pretrain_weights='${PRETRAIN_WEIGHTS}')"
else
  uv run --no-sync python -c "from tools.train import get_model_from_factory; get_model_from_factory('${MODEL_NAME}')"
fi

uv run --no-sync python -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  tools/train.py \
  --model "${MODEL_NAME}" \
  --dataset-file yolo \
  "${DATASET_ARGS[@]}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --num-workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --output-dir "${OUTPUT_DIR}" \
  --use-ema \
  --no-run-eda \
  "${TRAIN_MODE_ARGS[@]}"
#  --eval
# --no-multi-scale \
# --no-expanded-scales
