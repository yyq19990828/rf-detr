#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CHECKPOINT="${CHECKPOINT:-output/checkpoint_best_total.pth}"
OUTPUT="${OUTPUT:-}"
MODEL="${MODEL:-}"
NUM_CLASSES="${NUM_CLASSES:-}"
RESOLUTION="${RESOLUTION:-}"
DEVICE="${DEVICE:-}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OPSET_VERSION="${OPSET_VERSION:-17}"
IR_VERSION="${IR_VERSION:-10}"
TRAINING_CONFIG="${TRAINING_CONFIG:-}"
DYNAMIC_BATCH="${DYNAMIC_BATCH:-0}"
SIMPLIFY="${SIMPLIFY:-0}"

args=(
  --checkpoint "${CHECKPOINT}"
  --batch-size "${BATCH_SIZE}"
  --opset-version "${OPSET_VERSION}"
  --ir-version "${IR_VERSION}"
)

if [[ -n "${OUTPUT}" ]]; then
  args+=(--output "${OUTPUT}")
fi
if [[ -n "${MODEL}" ]]; then
  args+=(--model "${MODEL}")
fi
if [[ -n "${NUM_CLASSES}" ]]; then
  args+=(--num-classes "${NUM_CLASSES}")
fi
if [[ -n "${RESOLUTION}" ]]; then
  args+=(--resolution "${RESOLUTION}")
fi
if [[ -n "${DEVICE}" ]]; then
  args+=(--device "${DEVICE}")
fi
if [[ -n "${TRAINING_CONFIG}" ]]; then
  args+=(--training-config "${TRAINING_CONFIG}")
fi
if [[ "${DYNAMIC_BATCH}" == "1" ]]; then
  args+=(--dynamic-batch)
fi
if [[ "${SIMPLIFY}" == "1" ]]; then
  args+=(--simplify)
fi

python tools/export.py "${args[@]}" "$@"
