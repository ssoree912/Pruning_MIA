#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Pairwise connectivity pipeline for DF1 unlearning endpoints.
# Phase order:
#   1) raw_linear
#   2) git-re-basin / perm_linear
#   3) learned bezier on aligned endpoints
#   4) tail-SWA for bezier control point
#   5) simplex + simplex soup
#
# Assumes Step1 unlearning endpoints already exist.
# Example:
#   GPU=0 bash scripts/experiment/run_df1_seed42_connectivity.sh

DENSE_CKPT="${DENSE_CKPT:-runs/dense/cifar10/seed42/best_model.pth}"
ENDPOINT_A="${ENDPOINT_A:-runs/unlearning_df1/unlearn/cifar10/df1/seed42_seed43/unlearn_seed42.pth}"
ENDPOINT_B="${ENDPOINT_B:-runs/unlearning_df1/unlearn/cifar10/df1/seed42_seed43/unlearn_seed43.pth}"
SCRATCH_RETRAIN_CKPT="${SCRATCH_RETRAIN_CKPT:-runs/unlearning_df1/retrain/cifar10/df1/seed42_seed43/scratch_retrain_seed42.pth}"

OUT_ROOT="${OUT_ROOT:-runs/unlearning_connectivity_phase15}"
DATASET="${DATASET:-cifar10}"
ARCH="${ARCH:-resnet}"
LAYERS="${LAYERS:-20}"
DATAPATH="${DATAPATH:-~/Datasets/CIFAR}"
DATAPATH="${DATAPATH/#\~/$HOME}"
BATCH_SIZE="${BATCH_SIZE:-128}"
WORKERS="${WORKERS:-4}"
GPU="${GPU:-0}"
SPLIT_SEED="${SPLIT_SEED:-7}"
VAL_RATIO="${VAL_RATIO:-0.1}"

FORGET_OBJECTIVE="${FORGET_OBJECTIVE:-kl_uniform}"
FORGET_ALPHA="${FORGET_ALPHA:-0.05}"
RETAIN_WEIGHT="${RETAIN_WEIGHT:-1.0}"
FORGET_VAL_BUDGET="${FORGET_VAL_BUDGET:-0.01}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

LAMBDAS="${LAMBDAS:-21}"
BN_BATCHES="${BN_BATCHES:-200}"
BN_RECALC="${BN_RECALC:-1}"  # 1: recalibrate BN, 0: keep checkpoint BN stats
PERM_MAX_ITER="${PERM_MAX_ITER:-100}"
SEED="${SEED:-42}"

BEZIER_STEPS="${BEZIER_STEPS:-300}"
BEZIER_LR="${BEZIER_LR:-0.03}"
BEZIER_WEIGHT_DECAY="${BEZIER_WEIGHT_DECAY:-5e-4}"
BEZIER_T_SAMPLES="${BEZIER_T_SAMPLES:-2}"
BEZIER_TAIL_K="${BEZIER_TAIL_K:-20}"

SIMPLEX_STEPS="${SIMPLEX_STEPS:-300}"
SIMPLEX_LR="${SIMPLEX_LR:-0.03}"
SIMPLEX_WEIGHT_DECAY="${SIMPLEX_WEIGHT_DECAY:-5e-4}"
SIMPLEX_DIRICHLET_ALPHA="${SIMPLEX_DIRICHLET_ALPHA:-1.0}"
SIMPLEX_TAIL_K="${SIMPLEX_TAIL_K:-20}"
SIMPLEX_GRID_RESOLUTION="${SIMPLEX_GRID_RESOLUTION:-5}"

for f in "$DENSE_CKPT" "$ENDPOINT_A" "$ENDPOINT_B"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing file: $f"
    exit 1
  fi
done

SCRATCH_ARG=()
if [[ -f "$SCRATCH_RETRAIN_CKPT" ]]; then
  SCRATCH_ARG=(--scratch-retrain-ckpt "$SCRATCH_RETRAIN_CKPT")
else
  echo "Scratch retrain checkpoint not found, proceeding without RT reference: $SCRATCH_RETRAIN_CKPT"
fi

BN_RECALC_ARG=()
if [[ "${BN_RECALC}" == "0" ]]; then
  BN_RECALC_ARG=(--no-bn-recalc)
fi

python -m connectivity.run_connectivity_experiment \
  --dense-ckpt "$DENSE_CKPT" \
  --endpoint-a "$ENDPOINT_A" \
  --endpoint-b "$ENDPOINT_B" \
  "${SCRATCH_ARG[@]}" \
  --out-dir "$OUT_ROOT" \
  --dataset "$DATASET" \
  --arch "$ARCH" \
  --layers "$LAYERS" \
  --datapath "$DATAPATH" \
  --batch-size "$BATCH_SIZE" \
  --workers "$WORKERS" \
  --gpu "$GPU" \
  --split-seed "$SPLIT_SEED" \
  --val-ratio "$VAL_RATIO" \
  --df-mode profile \
  --df-profile df1 \
  --forget-objective "$FORGET_OBJECTIVE" \
  --forget-alpha "$FORGET_ALPHA" \
  --retain-weight "$RETAIN_WEIGHT" \
  --grad-clip "$GRAD_CLIP" \
  --forget-val-budget "$FORGET_VAL_BUDGET" \
  --lambdas "$LAMBDAS" \
  --bn-batches "$BN_BATCHES" \
  "${BN_RECALC_ARG[@]}" \
  --perm-max-iter "$PERM_MAX_ITER" \
  --seed "$SEED" \
  --bezier-steps "$BEZIER_STEPS" \
  --bezier-lr "$BEZIER_LR" \
  --bezier-weight-decay "$BEZIER_WEIGHT_DECAY" \
  --bezier-t-samples "$BEZIER_T_SAMPLES" \
  --bezier-tail-k "$BEZIER_TAIL_K" \
  --simplex-steps "$SIMPLEX_STEPS" \
  --simplex-lr "$SIMPLEX_LR" \
  --simplex-weight-decay "$SIMPLEX_WEIGHT_DECAY" \
  --simplex-dirichlet-alpha "$SIMPLEX_DIRICHLET_ALPHA" \
  --simplex-tail-k "$SIMPLEX_TAIL_K" \
  --simplex-grid-resolution "$SIMPLEX_GRID_RESOLUTION"
