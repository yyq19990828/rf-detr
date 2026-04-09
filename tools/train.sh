#!/usr/bin/env bash

set -euo pipefail

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM

  # 退出或 Ctrl+C 时清理本脚本直接拉起的子进程，避免 DDP worker 残留。
  # 不 kill 整个进程组，防止正常退出时误伤调用方 shell。
  local child_pids
  child_pids="$(jobs -pr)"
  if [[ -n "${child_pids}" ]]; then
    kill ${child_pids} 2>/dev/null || true
  fi

  exit "${exit_code}"
}

trap cleanup EXIT INT TERM

export NO_ALBUMENTATIONS_UPDATE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 20000))}"
export RFDETR_DEBUG_FIRST_BATCH="${RFDETR_DEBUG_FIRST_BATCH:-0}"
export CLEAR_YOLO_CACHE="${CLEAR_YOLO_CACHE:-0}"

# 多数据集用逗号分隔；脚本会展开成多个 --dataset-dir。
# 例：DATASET_DIRS="datasets/a_yolo,datasets/b_yolo" tools/train.sh
DATASET_DIRS="${DATASET_DIRS:-datasets/车型检测/ruqi_wuxi0728_yolo,datasets/上电/shangdian_yolo}"
IFS=',' read -r -a DATASET_DIR_ARRAY <<< "${DATASET_DIRS}"

# 去掉每个目录前后的空白，避免 "a, b" 这类写法产生带空格路径。
DATASET_ARGS=()
for dataset_dir in "${DATASET_DIR_ARRAY[@]}"; do
  dataset_dir="${dataset_dir#"${dataset_dir%%[![:space:]]*}"}"
  dataset_dir="${dataset_dir%"${dataset_dir##*[![:space:]]}"}"
  if [[ -n "${dataset_dir}" ]]; then
    DATASET_ARGS+=(--dataset-dir "${dataset_dir}")
  fi
done

if [[ "${#DATASET_ARGS[@]}" -eq 0 ]]; then
  echo "DATASET_DIRS 至少需要包含一个数据集目录" >&2
  exit 1
fi

if [[ "${CLEAR_YOLO_CACHE}" == "1" ]]; then
  echo "Clearing YOLO label caches before training..."
  for dataset_dir in "${DATASET_DIR_ARRAY[@]}"; do
    find "${dataset_dir}" -path "*/labels/.cache" -delete
  done
fi

# 基础训练参数：默认是小模型、YOLO 数据集、多目录联合训练。
MODEL_NAME="${MODEL_NAME:-small}"
DATASET_FILE="${DATASET_FILE:-yolo}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
LR_ENCODER="${LR_ENCODER:-1.5e-4}"
OUTPUT_DIR="${OUTPUT_DIR:-output/20260310_small}"

# 分布式参数交给 tools/train.py / PyTorch Lightning 处理。
# 单卡默认 strategy=auto；多卡默认启用 unused-parameter 检测，适配 RF-DETR
# 部分 batch 下存在未参与 loss 的参数路径；不再外层包 torchrun。
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DEVICES="${DEVICES:-${NPROC_PER_NODE}}"
NUM_NODES="${NUM_NODES:-1}"
STRATEGY="${STRATEGY:-}"
if [[ -z "${STRATEGY}" ]]; then
  if [[ "${DEVICES}" == "1" ]]; then
    STRATEGY="auto"
  else
    STRATEGY="ddp_find_unused_parameters_true"
  fi
fi
DEVICE="${DEVICE:-}"

# resume 表示恢复完整训练状态；pretrain_weights 表示只加载权重开始新训练。
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
PRETRAIN_WEIGHTS="${PRETRAIN_WEIGHTS:-}"

# 这些属于模型构造参数，不再传给 TrainConfig。
NUM_CLASSES="${NUM_CLASSES:-}"
GROUP_DETR="${GROUP_DETR:-13}"
NUM_SELECT="${NUM_SELECT:-300}"
IA_BCE_LOSS="${IA_BCE_LOSS:-1}"

# batch-size=auto 时使用的探测参数；普通整数 batch size 时不会触发自动探测。
AUTO_BATCH_TARGET_EFFECTIVE="${AUTO_BATCH_TARGET_EFFECTIVE:-16}"
AUTO_BATCH_MAX_TARGETS_PER_IMAGE="${AUTO_BATCH_MAX_TARGETS_PER_IMAGE:-100}"
AUTO_BATCH_EMA_HEADROOM="${AUTO_BATCH_EMA_HEADROOM:-0.7}"

# 布尔开关统一用 1/0 控制，下面会转换成 --foo / --no-foo。
USE_EMA="${USE_EMA:-1}"
RUN_EDA="${RUN_EDA:-0}"
RUN_TEST="${RUN_TEST:-0}"
TENSORBOARD="${TENSORBOARD:-1}"
SAVE_VAL_PREDICTIONS="${SAVE_VAL_PREDICTIONS:-0}"
PROGRESS_BAR="${PROGRESS_BAR:-1}"
MULTI_SCALE="${MULTI_SCALE:-1}"
EXPANDED_SCALES="${EXPANDED_SCALES:-1}"
SQUARE_RESIZE_DIV_64="${SQUARE_RESIZE_DIV_64:-1}"
DO_RANDOM_RESIZE_VIA_PADDING="${DO_RANDOM_RESIZE_VIA_PADDING:-0}"
SYNC_BN="${SYNC_BN:-0}"
FP16_EVAL="${FP16_EVAL:-0}"

PROJECT="${PROJECT:-}"
RUN="${RUN:-}"
AUG_CONFIG="${AUG_CONFIG:-}"
CLASS_NAMES="${CLASS_NAMES:-}"

# DRY_RUN=1 只打印最终命令，不启动训练，适合改参数后快速检查。
DRY_RUN="${DRY_RUN:-0}"

if [[ -n "${RESUME_CHECKPOINT}" && -n "${PRETRAIN_WEIGHTS}" ]]; then
  echo "RESUME_CHECKPOINT 和 PRETRAIN_WEIGHTS 不能同时设置" >&2
  exit 1
fi

TRAIN_ARGS=(
  --model "${MODEL_NAME}"
  --dataset-file "${DATASET_FILE}"
  "${DATASET_ARGS[@]}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --grad-accum-steps "${GRAD_ACCUM_STEPS}"
  --num-workers "${NUM_WORKERS}"
  --lr "${LR}"
  --lr-encoder "${LR_ENCODER}"
  --output-dir "${OUTPUT_DIR}"
  --strategy "${STRATEGY}"
  --devices "${DEVICES}"
  --num-nodes "${NUM_NODES}"
  --group-detr "${GROUP_DETR}"
  --num-select "${NUM_SELECT}"
  --auto-batch-target-effective "${AUTO_BATCH_TARGET_EFFECTIVE}"
  --auto-batch-max-targets-per-image "${AUTO_BATCH_MAX_TARGETS_PER_IMAGE}"
  --auto-batch-ema-headroom "${AUTO_BATCH_EMA_HEADROOM}"
)

# 可选参数仅在环境变量非空时追加，避免覆盖 tools/train.py 的默认 None 语义。
if [[ -n "${DEVICE}" ]]; then
  TRAIN_ARGS+=(--device "${DEVICE}")
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  TRAIN_ARGS+=(--resume "${RESUME_CHECKPOINT}")
fi
if [[ -n "${PRETRAIN_WEIGHTS}" ]]; then
  TRAIN_ARGS+=(--pretrain-weights "${PRETRAIN_WEIGHTS}")
fi
if [[ -n "${NUM_CLASSES}" ]]; then
  TRAIN_ARGS+=(--num-classes "${NUM_CLASSES}")
fi
if [[ -n "${PROJECT}" ]]; then
  TRAIN_ARGS+=(--project "${PROJECT}")
fi
if [[ -n "${RUN}" ]]; then
  TRAIN_ARGS+=(--run "${RUN}")
fi
if [[ -n "${AUG_CONFIG}" ]]; then
  TRAIN_ARGS+=(--aug-config "${AUG_CONFIG}")
fi
# CLASS_NAMES 用空格分隔，例如 CLASS_NAMES="car bus truck"。
if [[ -n "${CLASS_NAMES}" ]]; then
  read -r -a CLASS_NAME_ARRAY <<< "${CLASS_NAMES}"
  TRAIN_ARGS+=(--class-names "${CLASS_NAME_ARRAY[@]}")
fi

# 显式传入所有布尔开关，保证 shell 脚本行为不受 tools/train.py 默认值变动影响。
if [[ "${USE_EMA}" == "1" ]]; then TRAIN_ARGS+=(--use-ema); else TRAIN_ARGS+=(--no-use-ema); fi
if [[ "${RUN_EDA}" == "1" ]]; then TRAIN_ARGS+=(--run-eda); else TRAIN_ARGS+=(--no-run-eda); fi
if [[ "${RUN_TEST}" == "1" ]]; then TRAIN_ARGS+=(--run-test); else TRAIN_ARGS+=(--no-run-test); fi
if [[ "${TENSORBOARD}" == "1" ]]; then TRAIN_ARGS+=(--tensorboard); else TRAIN_ARGS+=(--no-tensorboard); fi
if [[ "${SAVE_VAL_PREDICTIONS}" == "1" ]]; then
  TRAIN_ARGS+=(--save-val-predictions)
else
  TRAIN_ARGS+=(--no-save-val-predictions)
fi
if [[ "${PROGRESS_BAR}" == "1" ]]; then TRAIN_ARGS+=(--progress-bar); else TRAIN_ARGS+=(--no-progress-bar); fi
if [[ "${MULTI_SCALE}" == "1" ]]; then TRAIN_ARGS+=(--multi-scale); else TRAIN_ARGS+=(--no-multi-scale); fi
if [[ "${EXPANDED_SCALES}" == "1" ]]; then TRAIN_ARGS+=(--expanded-scales); else TRAIN_ARGS+=(--no-expanded-scales); fi
if [[ "${SQUARE_RESIZE_DIV_64}" == "1" ]]; then
  TRAIN_ARGS+=(--square-resize-div-64)
else
  TRAIN_ARGS+=(--no-square-resize-div-64)
fi
if [[ "${DO_RANDOM_RESIZE_VIA_PADDING}" == "1" ]]; then
  TRAIN_ARGS+=(--do-random-resize-via-padding)
else
  TRAIN_ARGS+=(--no-do-random-resize-via-padding)
fi
if [[ "${IA_BCE_LOSS}" == "1" ]]; then TRAIN_ARGS+=(--ia-bce-loss); else TRAIN_ARGS+=(--no-ia-bce-loss); fi
if [[ "${SYNC_BN}" == "1" ]]; then TRAIN_ARGS+=(--sync-bn); else TRAIN_ARGS+=(--no-sync-bn); fi
if [[ "${FP16_EVAL}" == "1" ]]; then TRAIN_ARGS+=(--fp16-eval); else TRAIN_ARGS+=(--no-fp16-eval); fi

echo "Training command:"
printf ' %q' uv run --no-sync python tools/train.py "${TRAIN_ARGS[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

uv run --no-sync python tools/train.py "${TRAIN_ARGS[@]}"
