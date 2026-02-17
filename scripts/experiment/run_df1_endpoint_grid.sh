#!/usr/bin/env bash
set -euo pipefail

# DF1 endpoint tradeoff grid launcher.
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
BASE_OUT_DIR="${BASE_OUT_DIR:-./runs/unlearning_df1_grid_tradeoff}"
UNLEARN_LR="${UNLEARN_LR:-0.01}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

# Grid/baseline knobs.
# RUN_SET: partial | complete | both
RUN_SET="${RUN_SET:-both}"
RUN_SCRATCH_BASELINE="${RUN_SCRATCH_BASELINE:-0}"    # default OFF (avoid full scratch retrain overhead)
SCRATCH_EPOCHS="${SCRATCH_EPOCHS:-200}"
SCRATCH_LR="${SCRATCH_LR:-0.1}"
SCRATCH_SEED="${SCRATCH_SEED:-123}"

if [[ ! -f "${DENSE_CKPT}" ]]; then
  echo "Dense checkpoint not found: ${DENSE_CKPT}"
  exit 1
fi

if [[ "${RUN_SET}" != "partial" && "${RUN_SET}" != "complete" && "${RUN_SET}" != "both" ]]; then
  echo "Invalid RUN_SET: ${RUN_SET}"
  echo "Allowed: partial | complete | both"
  exit 1
fi

mkdir -p "${BASE_OUT_DIR}"

# id objective alpha unlearn_steps retrain_epochs retrain_lr
PARTIAL_EXPERIMENTS=(
  "P1 kl_uniform 0.005 50 80 0.10"
  "P2 kl_uniform 0.010 100 80 0.10"
  "P3 entropy 0.010 100 80 0.10"
)
COMPLETE_EXPERIMENTS=(
  "C1 kl_uniform 0.02 100 120 0.10"
  "C2 kl_uniform 0.05 200 120 0.10"
  "C3 entropy 0.05 200 120 0.10"
)

EXPERIMENTS=()
if [[ "${RUN_SET}" == "partial" || "${RUN_SET}" == "both" ]]; then
  EXPERIMENTS+=("${PARTIAL_EXPERIMENTS[@]}")
fi
if [[ "${RUN_SET}" == "complete" || "${RUN_SET}" == "both" ]]; then
  EXPERIMENTS+=("${COMPLETE_EXPERIMENTS[@]}")
fi

for row in "${EXPERIMENTS[@]}"; do
  read -r ID FORGET_OBJECTIVE ALPHA UNLEARN_STEPS RETRAIN_EPOCHS RETRAIN_LR <<< "${row}"
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
