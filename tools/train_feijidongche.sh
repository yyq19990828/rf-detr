#!/bin/bash

set -euo pipefail

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM

  kill -- -$$ 2>/dev/null || true

  exit "${exit_code}"
}

trap cleanup EXIT INT TERM
export NO_ALBUMENTATIONS_UPDATE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RFDETR_DEBUG_FIRST_BATCH="${RFDETR_DEBUG_FIRST_BATCH:-0}"
export CLEAR_YOLO_CACHE="${CLEAR_YOLO_CACHE:-0}"

DATASET_DIR="datasets/非机动车车牌/suzhou_yolo"

if [[ "${CLEAR_YOLO_CACHE}" == "1" ]]; then
  echo "Clearing YOLO label cache before training..."
  find "${DATASET_DIR}" -path "*/labels/.cache" -delete
fi

MODEL_NAME="small"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
OUTPUT_DIR="${OUTPUT_DIR:-output/20260325_feijidongche_small}"
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

if [[ -n "${PRETRAIN_WEIGHTS}" ]]; then
  uv run --no-sync python -c "from tools.train import get_model_from_factory; get_model_from_factory('${MODEL_NAME}', pretrain_weights='${PRETRAIN_WEIGHTS}')"
else
  uv run --no-sync python -c "from tools.train import get_model_from_factory; get_model_from_factory('${MODEL_NAME}')"
fi

uv run --no-sync python tools/train.py \
  --model "${MODEL_NAME}" \
  --dataset-file yolo \
  --dataset-dir "${DATASET_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --num-workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --output-dir "${OUTPUT_DIR}" \
  --use-ema \
  --no-run-eda \
  "${TRAIN_MODE_ARGS[@]}"
