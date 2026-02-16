#!/usr/bin/env bash
set -euo pipefail

# Dense checkpoint 기반으로 df1/df2/df3를 순차 실행.
#
# Example:
#   scripts/run_unlearning_connectivity.sh \
#     --dense-ckpt runs/dense/cifar10/seed42/best_model.pth \
#     --dataset cifar10 --arch resnet --layers 20 \
#     --seed-a 43 --seed-b 44 --gpu 0

DENSE_CKPT=""
DATASET="cifar10"
ARCH="resnet"
LAYERS="20"
SEED_A="43"
SEED_B="44"
UNLEARN_EPOCHS="20"
UNLEARN_STEPS="0"
UNLEARN_LR="0.01"
RETRAIN_EPOCHS="0"
RETRAIN_LR=""
RETRAIN_MOMENTUM=""
RETRAIN_WEIGHT_DECAY=""
RETRAIN_NESTEROV="-1"
FORGET_ALPHA="0.05"
FORGET_OBJECTIVE="ce_ascent"
RETAIN_WEIGHT="1.0"
GRAD_CLIP="1.0"
CKPT_SELECT="test_acc"
BATCH_SIZE="128"
WORKERS="4"
DATAPATH="~/Datasets/CIFAR"
LAMBDAS="21"
BN_RECALC=1
BN_BATCHES="200"
SWA_MERGE=1
SWA_SOURCE="both"
SWA_TOPK="5"
SWA_SELECT_METRIC="test_acc"
SWA_T_MIN="0.0"
SWA_T_MAX="1.0"
MASK_METHOD="delta"
MASK_TOPK="0.1"
OUT_DIR="./runs/unlearning_connectivity"
GPU="0"
SKIP_EXISTING=0
STEP1_ONLY=0
USE_DF_PRESETS=1
TRAIN_SCRATCH_RETRAIN_BASELINE=0
SCRATCH_RETRAIN_CKPT=""
SCRATCH_RETRAIN_EPOCHS="200"
SCRATCH_RETRAIN_LR="0.1"
SCRATCH_RETRAIN_MOMENTUM="0.9"
SCRATCH_RETRAIN_WEIGHT_DECAY="0.0005"
SCRATCH_RETRAIN_NESTEROV=0
SCRATCH_RETRAIN_SEED="123"

# Recommended per-DF performance-preserving initial presets.
DF1_UNLEARN_STEPS="${DF1_UNLEARN_STEPS:-80}"
DF1_FORGET_ALPHA="${DF1_FORGET_ALPHA:-0.008}"
DF1_RETRAIN_EPOCHS="${DF1_RETRAIN_EPOCHS:-15}"
DF1_RETRAIN_LR="${DF1_RETRAIN_LR:-0.005}"

DF2_UNLEARN_STEPS="${DF2_UNLEARN_STEPS:-120}"
DF2_FORGET_ALPHA="${DF2_FORGET_ALPHA:-0.010}"
DF2_RETRAIN_EPOCHS="${DF2_RETRAIN_EPOCHS:-20}"
DF2_RETRAIN_LR="${DF2_RETRAIN_LR:-0.004}"

DF3_UNLEARN_STEPS="${DF3_UNLEARN_STEPS:-160}"
DF3_FORGET_ALPHA="${DF3_FORGET_ALPHA:-0.012}"
DF3_RETRAIN_EPOCHS="${DF3_RETRAIN_EPOCHS:-25}"
DF3_RETRAIN_LR="${DF3_RETRAIN_LR:-0.003}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dense-ckpt) DENSE_CKPT="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --arch) ARCH="$2"; shift 2 ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --seed-a) SEED_A="$2"; shift 2 ;;
    --seed-b) SEED_B="$2"; shift 2 ;;
    --unlearn-epochs) UNLEARN_EPOCHS="$2"; shift 2 ;;
    --unlearn-steps) UNLEARN_STEPS="$2"; shift 2 ;;
    --unlearn-lr) UNLEARN_LR="$2"; shift 2 ;;
    --retrain-epochs) RETRAIN_EPOCHS="$2"; shift 2 ;;
    --retrain-lr) RETRAIN_LR="$2"; shift 2 ;;
    --retrain-momentum) RETRAIN_MOMENTUM="$2"; shift 2 ;;
    --retrain-weight-decay) RETRAIN_WEIGHT_DECAY="$2"; shift 2 ;;
    --retrain-nesterov) RETRAIN_NESTEROV="1"; shift 1 ;;
    --no-retrain-nesterov) RETRAIN_NESTEROV="0"; shift 1 ;;
    --forget-alpha) FORGET_ALPHA="$2"; shift 2 ;;
    --forget-objective) FORGET_OBJECTIVE="$2"; shift 2 ;;
    --retain-weight) RETAIN_WEIGHT="$2"; shift 2 ;;
    --grad-clip) GRAD_CLIP="$2"; shift 2 ;;
    --ckpt-select) CKPT_SELECT="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --datapath) DATAPATH="$2"; shift 2 ;;
    --lambdas) LAMBDAS="$2"; shift 2 ;;
    --bn-recalc) BN_RECALC=1; shift 1 ;;
    --no-bn-recalc) BN_RECALC=0; shift 1 ;;
    --bn-batches) BN_BATCHES="$2"; shift 2 ;;
    --swa-merge) SWA_MERGE=1; shift 1 ;;
    --no-swa-merge) SWA_MERGE=0; shift 1 ;;
    --swa-source) SWA_SOURCE="$2"; shift 2 ;;
    --swa-topk) SWA_TOPK="$2"; shift 2 ;;
    --swa-select-metric) SWA_SELECT_METRIC="$2"; shift 2 ;;
    --swa-t-min) SWA_T_MIN="$2"; shift 2 ;;
    --swa-t-max) SWA_T_MAX="$2"; shift 2 ;;
    --mask-method) MASK_METHOD="$2"; shift 2 ;;
    --mask-topk) MASK_TOPK="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --skip-existing) SKIP_EXISTING=1; shift 1 ;;
    --step1-only) STEP1_ONLY=1; shift 1 ;;
    --use-df-presets) USE_DF_PRESETS=1; shift 1 ;;
    --no-df-presets) USE_DF_PRESETS=0; shift 1 ;;
    --train-scratch-retrain-baseline) TRAIN_SCRATCH_RETRAIN_BASELINE=1; shift 1 ;;
    --scratch-retrain-ckpt) SCRATCH_RETRAIN_CKPT="$2"; shift 2 ;;
    --scratch-retrain-epochs) SCRATCH_RETRAIN_EPOCHS="$2"; shift 2 ;;
    --scratch-retrain-lr) SCRATCH_RETRAIN_LR="$2"; shift 2 ;;
    --scratch-retrain-momentum) SCRATCH_RETRAIN_MOMENTUM="$2"; shift 2 ;;
    --scratch-retrain-weight-decay) SCRATCH_RETRAIN_WEIGHT_DECAY="$2"; shift 2 ;;
    --scratch-retrain-nesterov) SCRATCH_RETRAIN_NESTEROV=1; shift 1 ;;
    --scratch-retrain-seed) SCRATCH_RETRAIN_SEED="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "${DENSE_CKPT}" ]]; then
  echo "--dense-ckpt is required"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"

for DF in df1 df2 df3; do
  echo
  echo "============================================================"
  echo "Running ${DF}"
  echo "============================================================"
  CUR_UNLEARN_STEPS="${UNLEARN_STEPS}"
  CUR_FORGET_ALPHA="${FORGET_ALPHA}"
  CUR_RETRAIN_EPOCHS="${RETRAIN_EPOCHS}"
  CUR_RETRAIN_LR="${RETRAIN_LR}"

  if [[ "${USE_DF_PRESETS}" -eq 1 ]]; then
    case "${DF}" in
      df1)
        CUR_UNLEARN_STEPS="${DF1_UNLEARN_STEPS}"
        CUR_FORGET_ALPHA="${DF1_FORGET_ALPHA}"
        CUR_RETRAIN_EPOCHS="${DF1_RETRAIN_EPOCHS}"
        CUR_RETRAIN_LR="${DF1_RETRAIN_LR}"
        ;;
      df2)
        CUR_UNLEARN_STEPS="${DF2_UNLEARN_STEPS}"
        CUR_FORGET_ALPHA="${DF2_FORGET_ALPHA}"
        CUR_RETRAIN_EPOCHS="${DF2_RETRAIN_EPOCHS}"
        CUR_RETRAIN_LR="${DF2_RETRAIN_LR}"
        ;;
      df3)
        CUR_UNLEARN_STEPS="${DF3_UNLEARN_STEPS}"
        CUR_FORGET_ALPHA="${DF3_FORGET_ALPHA}"
        CUR_RETRAIN_EPOCHS="${DF3_RETRAIN_EPOCHS}"
        CUR_RETRAIN_LR="${DF3_RETRAIN_LR}"
        ;;
    esac
  fi

  echo "[hparam] df=${DF} objective=${FORGET_OBJECTIVE} unlearn_steps=${CUR_UNLEARN_STEPS} forget_alpha=${CUR_FORGET_ALPHA} retrain_epochs=${CUR_RETRAIN_EPOCHS} retrain_lr=${CUR_RETRAIN_LR:-auto} ckpt_select=${CKPT_SELECT}"

  ARGS=(
    --dense-ckpt "${DENSE_CKPT}"
    --dataset "${DATASET}"
    --arch "${ARCH}"
    --layers "${LAYERS}"
    --seed-a "${SEED_A}"
    --seed-b "${SEED_B}"
    --df-mode profile
    --df-profile "${DF}"
    --unlearn-epochs "${UNLEARN_EPOCHS}"
    --unlearn-steps "${CUR_UNLEARN_STEPS}"
    --unlearn-lr "${UNLEARN_LR}"
    --retrain-epochs "${CUR_RETRAIN_EPOCHS}"
    --forget-alpha "${CUR_FORGET_ALPHA}"
    --forget-objective "${FORGET_OBJECTIVE}"
    --retain-weight "${RETAIN_WEIGHT}"
    --grad-clip "${GRAD_CLIP}"
    --ckpt-select "${CKPT_SELECT}"
    --batch-size "${BATCH_SIZE}"
    --workers "${WORKERS}"
    --datapath "${DATAPATH}"
    --lambdas "${LAMBDAS}"
    --bn-batches "${BN_BATCHES}"
    --swa-source "${SWA_SOURCE}"
    --swa-topk "${SWA_TOPK}"
    --swa-select-metric "${SWA_SELECT_METRIC}"
    --swa-t-min "${SWA_T_MIN}"
    --swa-t-max "${SWA_T_MAX}"
    --mask-method "${MASK_METHOD}"
    --mask-topk "${MASK_TOPK}"
    --out-dir "${OUT_DIR}"
  )
  if [[ "${BN_RECALC}" -eq 1 ]]; then
    ARGS+=(--bn-recalc)
  else
    ARGS+=(--no-bn-recalc)
  fi
  if [[ "${SWA_MERGE}" -eq 1 ]]; then
    ARGS+=(--swa-merge)
  else
    ARGS+=(--no-swa-merge)
  fi
  if [[ -n "${CUR_RETRAIN_LR}" ]]; then
    ARGS+=(--retrain-lr "${CUR_RETRAIN_LR}")
  fi
  if [[ -n "${RETRAIN_MOMENTUM}" ]]; then
    ARGS+=(--retrain-momentum "${RETRAIN_MOMENTUM}")
  fi
  if [[ -n "${RETRAIN_WEIGHT_DECAY}" ]]; then
    ARGS+=(--retrain-weight-decay "${RETRAIN_WEIGHT_DECAY}")
  fi
  if [[ "${RETRAIN_NESTEROV}" -eq 1 ]]; then
    ARGS+=(--retrain-nesterov)
  elif [[ "${RETRAIN_NESTEROV}" -eq 0 ]]; then
    ARGS+=(--no-retrain-nesterov)
  fi
  if [[ "${SKIP_EXISTING}" -eq 1 ]]; then
    ARGS+=(--skip-existing)
  fi
  if [[ "${STEP1_ONLY}" -eq 1 ]]; then
    ARGS+=(--step1-only)
  fi
  if [[ "${TRAIN_SCRATCH_RETRAIN_BASELINE}" -eq 1 ]]; then
    ARGS+=(
      --train-scratch-retrain-baseline
      --scratch-retrain-epochs "${SCRATCH_RETRAIN_EPOCHS}"
      --scratch-retrain-lr "${SCRATCH_RETRAIN_LR}"
      --scratch-retrain-momentum "${SCRATCH_RETRAIN_MOMENTUM}"
      --scratch-retrain-weight-decay "${SCRATCH_RETRAIN_WEIGHT_DECAY}"
      --scratch-retrain-seed "${SCRATCH_RETRAIN_SEED}"
    )
    if [[ "${SCRATCH_RETRAIN_NESTEROV}" -eq 1 ]]; then
      ARGS+=(--scratch-retrain-nesterov)
    fi
  fi
  if [[ -n "${SCRATCH_RETRAIN_CKPT}" ]]; then
    ARGS+=(--scratch-retrain-ckpt "${SCRATCH_RETRAIN_CKPT}")
  fi
  python train.py "${ARGS[@]}"
done
