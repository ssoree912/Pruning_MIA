#!/usr/bin/env bash
set -euo pipefail

# DF1 endpoint quality grid (8 runs) for unlearn+retrain tradeoff search.
#
# Usage:
#   bash scripts/experiment/run_df1_endpoint_grid.sh
#   DENSE_CKPT=... GPU=0 bash scripts/experiment/run_df1_endpoint_grid.sh
#   DRY_RUN=1 bash scripts/experiment/run_df1_endpoint_grid.sh

DENSE_CKPT="${DENSE_CKPT:-runs/dense/cifar10/seed42/best_model.pth}"
DATASET="${DATASET:-cifar10}"
ARCH="${ARCH:-resnet}"
LAYERS="${LAYERS:-20}"
SEED_A="${SEED_A:-43}"
SEED_B="${SEED_B:-44}"
GPU="${GPU:-0}"
BASE_OUT_DIR="${BASE_OUT_DIR:-./runs/unlearning_df1_grid8}"
UNLEARN_LR="${UNLEARN_LR:-0.01}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

if [[ ! -f "${DENSE_CKPT}" ]]; then
  echo "Dense checkpoint not found: ${DENSE_CKPT}"
  exit 1
fi

mkdir -p "${BASE_OUT_DIR}"

# id alpha unlearn_steps retrain_epochs retrain_lr
EXPERIMENTS=(
  "E1 0.05 200 30 0.05"
  "E2 0.05 200 50 0.05"
  "E3 0.05 500 30 0.05"
  "E4 0.05 500 50 0.05"
  "E5 0.10 200 30 0.05"
  "E6 0.10 200 50 0.05"
  "E7 0.10 500 30 0.05"
  "E8 0.10 500 50 0.05"
)

for row in "${EXPERIMENTS[@]}"; do
  read -r ID ALPHA UNLEARN_STEPS RETRAIN_EPOCHS RETRAIN_LR <<< "${row}"
  OUT_DIR="${BASE_OUT_DIR}/${ID}"
  mkdir -p "${OUT_DIR}"

  CMD=(
    python train.py
    --dense-ckpt "${DENSE_CKPT}"
    --dataset "${DATASET}"
    --arch "${ARCH}"
    --layers "${LAYERS}"
    --df-mode profile
    --df-profile df1
    --seed-a "${SEED_A}"
    --seed-b "${SEED_B}"
    --unlearn-epochs 1
    --unlearn-steps "${UNLEARN_STEPS}"
    --unlearn-lr "${UNLEARN_LR}"
    --forget-alpha "${ALPHA}"
    --retrain-epochs "${RETRAIN_EPOCHS}"
    --retrain-lr "${RETRAIN_LR}"
    --ckpt-select retain_acc
    --step1-only
    --no-swa-merge
    --out-dir "${OUT_DIR}"
    --gpu "${GPU}"
  )

  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    CMD+=(--skip-existing)
  fi

  echo
  echo "============================================================"
  echo "[${ID}] alpha=${ALPHA}, unlearn_steps=${UNLEARN_STEPS}, retrain_epochs=${RETRAIN_EPOCHS}, retrain_lr=${RETRAIN_LR}"
  echo "out_dir=${OUT_DIR}"
  echo "============================================================"

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${CMD[@]}"
    echo
  else
    "${CMD[@]}"
  fi
done

echo
echo "Done. Results root: ${BASE_OUT_DIR}"
