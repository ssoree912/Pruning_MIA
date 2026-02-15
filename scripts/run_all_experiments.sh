#!/usr/bin/env bash
set -euo pipefail

# Unlearning + connectivity 전용 실행 스크립트.
# df1/df2/df3를 순차 실행한다.
#
# Required:
#   DENSE_CKPT=/path/to/dense/best_model.pth scripts/run_all_experiments.sh

DENSE_CKPT="${DENSE_CKPT:-}"
DATASET="${DATASET:-cifar10}"
ARCH="${ARCH:-resnet}"
LAYERS="${LAYERS:-20}"
SEED_A="${SEED_A:-43}"
SEED_B="${SEED_B:-44}"
UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-20}"
UNLEARN_LR="${UNLEARN_LR:-0.01}"
FORGET_ALPHA="${FORGET_ALPHA:-0.05}"
RETAIN_WEIGHT="${RETAIN_WEIGHT:-1.0}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
WORKERS="${WORKERS:-4}"
DATAPATH="${DATAPATH:-~/Datasets/CIFAR}"
LAMBDAS="${LAMBDAS:-21}"
BN_BATCHES="${BN_BATCHES:-200}"
MASK_METHOD="${MASK_METHOD:-delta}"
MASK_TOPK="${MASK_TOPK:-0.1}"
OUT_DIR="${OUT_DIR:-./runs/unlearning_connectivity}"
GPU="${GPU:-0}"

if [[ -z "${DENSE_CKPT}" ]]; then
  echo "DENSE_CKPT env is required"
  echo "example:"
  echo "  DENSE_CKPT=runs/dense/cifar10/seed42/best_model.pth scripts/run_all_experiments.sh"
  exit 1
fi

scripts/run_unlearning_connectivity.sh \
  --dense-ckpt "${DENSE_CKPT}" \
  --dataset "${DATASET}" \
  --arch "${ARCH}" \
  --layers "${LAYERS}" \
  --seed-a "${SEED_A}" \
  --seed-b "${SEED_B}" \
  --unlearn-epochs "${UNLEARN_EPOCHS}" \
  --unlearn-lr "${UNLEARN_LR}" \
  --forget-alpha "${FORGET_ALPHA}" \
  --retain-weight "${RETAIN_WEIGHT}" \
  --grad-clip "${GRAD_CLIP}" \
  --batch-size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --datapath "${DATAPATH}" \
  --lambdas "${LAMBDAS}" \
  --bn-recalc \
  --bn-batches "${BN_BATCHES}" \
  --mask-method "${MASK_METHOD}" \
  --mask-topk "${MASK_TOPK}" \
  --out-dir "${OUT_DIR}" \
  --gpu "${GPU}"

echo "=== Done. Output dir: ${OUT_DIR} ==="
