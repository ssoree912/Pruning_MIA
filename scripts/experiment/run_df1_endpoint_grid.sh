#!/usr/bin/env bash
set -euo pipefail

# DF1 endpoint utility-first grid launcher.
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
BASE_OUT_DIR="${BASE_OUT_DIR:-./runs/unlearning_df1_grid_utility}"
UNLEARN_LR="${UNLEARN_LR:-0.01}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

# New objective/baseline knobs.
FORGET_OBJECTIVE="${FORGET_OBJECTIVE:-kl_uniform}"   # ce_ascent | kl_uniform | entropy
RUN_SCRATCH_BASELINE="${RUN_SCRATCH_BASELINE:-1}"    # 1 to train/eval scratch retrain baseline
SCRATCH_EPOCHS="${SCRATCH_EPOCHS:-200}"
SCRATCH_LR="${SCRATCH_LR:-0.1}"
SCRATCH_SEED="${SCRATCH_SEED:-123}"

if [[ ! -f "${DENSE_CKPT}" ]]; then
  echo "Dense checkpoint not found: ${DENSE_CKPT}"
  exit 1
fi

if [[ "${FORGET_OBJECTIVE}" != "ce_ascent" && "${FORGET_OBJECTIVE}" != "kl_uniform" && "${FORGET_OBJECTIVE}" != "entropy" ]]; then
  echo "Invalid FORGET_OBJECTIVE: ${FORGET_OBJECTIVE}"
  echo "Allowed: ce_ascent | kl_uniform | entropy"
  exit 1
fi

mkdir -p "${BASE_OUT_DIR}"

# id alpha unlearn_steps retrain_epochs retrain_lr
EXPERIMENTS=(
  "K1 0.02 100 50 0.10"
  "K2 0.02 200 50 0.10"
  "K3 0.05 100 50 0.10"
  "K4 0.05 200 50 0.10"
  "K5 0.05 300 80 0.10"
  "K6 0.10 100 50 0.10"
  "K7 0.10 200 50 0.10"
  "K8 0.10 300 80 0.10"
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
    --forget-objective "${FORGET_OBJECTIVE}"
    --retrain-epochs "${RETRAIN_EPOCHS}"
    --retrain-lr "${RETRAIN_LR}"
    --ckpt-select retain_acc
    --step1-only
    --no-swa-merge
    --out-dir "${OUT_DIR}"
    --gpu "${GPU}"
  )

  if [[ "${RUN_SCRATCH_BASELINE}" == "1" ]]; then
    CMD+=(
      --train-scratch-retrain-baseline
      --scratch-retrain-epochs "${SCRATCH_EPOCHS}"
      --scratch-retrain-lr "${SCRATCH_LR}"
      --scratch-retrain-seed "${SCRATCH_SEED}"
    )
  fi

  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    CMD+=(--skip-existing)
  fi

  echo
  echo "============================================================"
  echo "[${ID}] obj=${FORGET_OBJECTIVE}, alpha=${ALPHA}, unlearn_steps=${UNLEARN_STEPS}, retrain_epochs=${RETRAIN_EPOCHS}, retrain_lr=${RETRAIN_LR}"
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
