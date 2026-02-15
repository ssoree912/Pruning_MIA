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
UNLEARN_LR="0.01"
FORGET_ALPHA="0.05"
RETAIN_WEIGHT="1.0"
GRAD_CLIP="1.0"
BATCH_SIZE="128"
WORKERS="4"
DATAPATH="~/Datasets/CIFAR"
LAMBDAS="21"
BN_RECALC=1
BN_BATCHES="200"
MASK_METHOD="delta"
MASK_TOPK="0.1"
OUT_DIR="./runs/unlearning_connectivity"
GPU="0"
SKIP_EXISTING=0
STEP1_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dense-ckpt) DENSE_CKPT="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --arch) ARCH="$2"; shift 2 ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --seed-a) SEED_A="$2"; shift 2 ;;
    --seed-b) SEED_B="$2"; shift 2 ;;
    --unlearn-epochs) UNLEARN_EPOCHS="$2"; shift 2 ;;
    --unlearn-lr) UNLEARN_LR="$2"; shift 2 ;;
    --forget-alpha) FORGET_ALPHA="$2"; shift 2 ;;
    --retain-weight) RETAIN_WEIGHT="$2"; shift 2 ;;
    --grad-clip) GRAD_CLIP="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --datapath) DATAPATH="$2"; shift 2 ;;
    --lambdas) LAMBDAS="$2"; shift 2 ;;
    --bn-recalc) BN_RECALC=1; shift 1 ;;
    --no-bn-recalc) BN_RECALC=0; shift 1 ;;
    --bn-batches) BN_BATCHES="$2"; shift 2 ;;
    --mask-method) MASK_METHOD="$2"; shift 2 ;;
    --mask-topk) MASK_TOPK="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --skip-existing) SKIP_EXISTING=1; shift 1 ;;
    --step1-only) STEP1_ONLY=1; shift 1 ;;
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
    --unlearn-lr "${UNLEARN_LR}"
    --forget-alpha "${FORGET_ALPHA}"
    --retain-weight "${RETAIN_WEIGHT}"
    --grad-clip "${GRAD_CLIP}"
    --batch-size "${BATCH_SIZE}"
    --workers "${WORKERS}"
    --datapath "${DATAPATH}"
    --lambdas "${LAMBDAS}"
    --bn-batches "${BN_BATCHES}"
    --mask-method "${MASK_METHOD}"
    --mask-topk "${MASK_TOPK}"
    --out-dir "${OUT_DIR}"
  )
  if [[ "${BN_RECALC}" -eq 1 ]]; then
    ARGS+=(--bn-recalc)
  else
    ARGS+=(--no-bn-recalc)
  fi
  if [[ "${SKIP_EXISTING}" -eq 1 ]]; then
    ARGS+=(--skip-existing)
  fi
  if [[ "${STEP1_ONLY}" -eq 1 ]]; then
    ARGS+=(--step1-only)
  fi
  python train.py "${ARGS[@]}"
done
