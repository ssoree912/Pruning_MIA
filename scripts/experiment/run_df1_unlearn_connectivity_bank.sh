#!/usr/bin/env bash
set -euo pipefail

# Multi-pair DF1 pipeline:
#   1) build unlearning endpoints (Step1 only)
#   2) run connectivity/merge (phase15 simplex_soup) per pair
#
# Usage:
#   bash scripts/experiment/run_df1_unlearn_connectivity_bank.sh
#   GPU=0 BN_RECALC=0 PAIR_LIST="42:43,44:45,46:47,48:49,50:51" \
#     bash scripts/experiment/run_df1_unlearn_connectivity_bank.sh
#
# Pair format:
#   PAIR_LIST="seedA:seedB,seedC:seedD,..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DENSE_CKPT="${DENSE_CKPT:-runs/dense/cifar10/seed42/best_model.pth}"
PAIR_LIST="${PAIR_LIST:-42:43,44:45,46:47,48:49,50:51}"

OUT_UNLEARN="${OUT_UNLEARN:-runs/unlearning_df1/unlearn}"
OUT_CONNECTIVITY="${OUT_CONNECTIVITY:-runs/unlearning_connectivity_phase15_final}"

DATASET="${DATASET:-cifar10}"
ARCH="${ARCH:-resnet}"
LAYERS="${LAYERS:-20}"
SPLIT_SEED="${SPLIT_SEED:-7}"
GPU="${GPU:-0}"

DATAPATH="${DATAPATH:-~/Datasets/CIFAR}"
DATAPATH="${DATAPATH/#\~/$HOME}"
BATCH_SIZE="${BATCH_SIZE:-128}"
WORKERS="${WORKERS:-4}"

# Step1 (unlearning endpoint) defaults
FORGET_OBJECTIVE="${FORGET_OBJECTIVE:-kl_uniform}"
FORGET_ALPHA="${FORGET_ALPHA:-0.05}"
RETAIN_WEIGHT="${RETAIN_WEIGHT:-1.0}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-1}"
UNLEARN_STEPS="${UNLEARN_STEPS:-100}"
UNLEARN_LR="${UNLEARN_LR:-0.01}"
RETRAIN_EPOCHS="${RETRAIN_EPOCHS:-50}"
RETRAIN_LR="${RETRAIN_LR:-0.1}"
VAL_RATIO="${VAL_RATIO:-0.1}"
FORGET_VAL_BUDGET="${FORGET_VAL_BUDGET:-0.01}"
SAVE_TAIL_K="${SAVE_TAIL_K:-5}"
CKPT_SELECT="${CKPT_SELECT:-composite}"

# Optional scratch baseline in Step1 run_dir (used as RT reference for connectivity)
TRAIN_SCRATCH_RETRAIN_BASELINE="${TRAIN_SCRATCH_RETRAIN_BASELINE:-0}"
SCRATCH_EPOCHS="${SCRATCH_EPOCHS:-200}"
SCRATCH_LR="${SCRATCH_LR:-0.1}"
SCRATCH_MOMENTUM="${SCRATCH_MOMENTUM:-0.9}"
SCRATCH_WEIGHT_DECAY="${SCRATCH_WEIGHT_DECAY:-5e-4}"

# Connectivity/merge defaults
BN_RECALC="${BN_RECALC:-0}"  # 0 recommended for candidate selection
BN_BATCHES="${BN_BATCHES:-200}"
LAMBDAS="${LAMBDAS:-21}"
PERM_MAX_ITER="${PERM_MAX_ITER:-100}"
CONNECTIVITY_SEED="${CONNECTIVITY_SEED:-42}"

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

SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

is_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] '
    printf '%q ' "$@"
    echo
    return 0
  fi
  "$@"
}

parse_pair() {
  local raw="$1"
  local clean="${raw//[[:space:]]/}"
  if [[ -z "${clean}" ]]; then
    return 1
  fi
  if [[ "${clean}" != *":"* ]]; then
    echo "Invalid pair format: ${raw} (expected A:B)" >&2
    return 1
  fi
  local a="${clean%%:*}"
  local b="${clean##*:}"
  if ! is_int "${a}" || ! is_int "${b}"; then
    echo "Invalid pair seed (must be integer): ${raw}" >&2
    return 1
  fi
  if [[ "${a}" == "${b}" ]]; then
    echo "Invalid pair (same seeds): ${raw}" >&2
    return 1
  fi
  echo "${a}:${b}"
}

if [[ ! -f "${DENSE_CKPT}" ]]; then
  echo "Dense ckpt not found: ${DENSE_CKPT}" >&2
  exit 1
fi

IFS=',' read -r -a PAIRS <<< "${PAIR_LIST}"
if [[ "${#PAIRS[@]}" -eq 0 ]]; then
  echo "PAIR_LIST is empty" >&2
  exit 1
fi

echo "================================================================"
echo "DF1 multi-pair unlearning -> connectivity/merge"
echo "repo: ${REPO_ROOT}"
echo "dense: ${DENSE_CKPT}"
echo "pairs: ${PAIR_LIST}"
echo "out_unlearn: ${OUT_UNLEARN}"
echo "out_connectivity: ${OUT_CONNECTIVITY}"
echo "bn_recalc: ${BN_RECALC}"
echo "skip_existing: ${SKIP_EXISTING}"
echo "dry_run: ${DRY_RUN}"
echo "================================================================"

pair_idx=0
for raw_pair in "${PAIRS[@]}"; do
  parsed="$(parse_pair "${raw_pair}")"
  if [[ -z "${parsed}" ]]; then
    exit 1
  fi

  seed_a="${parsed%%:*}"
  seed_b="${parsed##*:}"
  pair_idx=$((pair_idx + 1))

  run_dir_unlearn="${OUT_UNLEARN}/${DATASET}/df1/seed${seed_a}_seed${seed_b}"
  unlearn_a="${run_dir_unlearn}/unlearn_seed${seed_a}.pth"
  unlearn_b="${run_dir_unlearn}/unlearn_seed${seed_b}.pth"
  scratch_a="${run_dir_unlearn}/scratch_retrain_seed${seed_a}.pth"

  conn_run_dir="${OUT_CONNECTIVITY}/${DATASET}/df1/unlearn_seed${seed_a}__unlearn_seed${seed_b}"
  conn_summary="${conn_run_dir}/summary.json"
  conn_soup="${conn_run_dir}/simplex_soup_best.pth"

  echo
  echo "[${pair_idx}/${#PAIRS[@]}] pair seed${seed_a}_seed${seed_b}"

  need_step1=1
  if [[ "${SKIP_EXISTING}" == "1" && -f "${unlearn_a}" && -f "${unlearn_b}" ]]; then
    if [[ "${TRAIN_SCRATCH_RETRAIN_BASELINE}" == "1" && ! -f "${scratch_a}" ]]; then
      need_step1=1
    else
      need_step1=0
    fi
  fi

  if [[ "${need_step1}" == "1" ]]; then
    echo "  - Step1 unlearning endpoints"
    train_cmd=(
      python train.py
      --dense-ckpt "${DENSE_CKPT}"
      --out-dir "${OUT_UNLEARN}"
      --dataset "${DATASET}"
      --arch "${ARCH}"
      --layers "${LAYERS}"
      --seed-a "${seed_a}"
      --seed-b "${seed_b}"
      --split-seed "${SPLIT_SEED}"
      --df-mode profile
      --df-profile df1
      --unlearn-epochs "${UNLEARN_EPOCHS}"
      --unlearn-steps "${UNLEARN_STEPS}"
      --unlearn-lr "${UNLEARN_LR}"
      --forget-alpha "${FORGET_ALPHA}"
      --forget-objective "${FORGET_OBJECTIVE}"
      --retain-weight "${RETAIN_WEIGHT}"
      --grad-clip "${GRAD_CLIP}"
      --retrain-epochs "${RETRAIN_EPOCHS}"
      --retrain-lr "${RETRAIN_LR}"
      --ckpt-select "${CKPT_SELECT}"
      --val-ratio "${VAL_RATIO}"
      --forget-val-budget "${FORGET_VAL_BUDGET}"
      --save-tail-k "${SAVE_TAIL_K}"
      --batch-size "${BATCH_SIZE}"
      --workers "${WORKERS}"
      --datapath "${DATAPATH}"
      --gpu "${GPU}"
      --step1-only
      --no-swa-merge
    )
    if [[ "${SKIP_EXISTING}" == "1" ]]; then
      train_cmd+=(--skip-existing)
    fi
    if [[ "${TRAIN_SCRATCH_RETRAIN_BASELINE}" == "1" ]]; then
      train_cmd+=(
        --train-scratch-retrain-baseline
        --scratch-retrain-epochs "${SCRATCH_EPOCHS}"
        --scratch-retrain-lr "${SCRATCH_LR}"
        --scratch-retrain-momentum "${SCRATCH_MOMENTUM}"
        --scratch-retrain-weight-decay "${SCRATCH_WEIGHT_DECAY}"
        --scratch-retrain-seed "${seed_a}"
      )
    fi
    run_cmd "${train_cmd[@]}"
  else
    echo "  - Step1 skipped (existing endpoints found)"
  fi

  if [[ "${DRY_RUN}" != "1" ]]; then
    if [[ ! -f "${unlearn_a}" || ! -f "${unlearn_b}" ]]; then
      echo "Missing unlearn endpoints for pair seed${seed_a}_seed${seed_b}" >&2
      echo "  expected: ${unlearn_a}" >&2
      echo "  expected: ${unlearn_b}" >&2
      exit 1
    fi

    if [[ "${TRAIN_SCRATCH_RETRAIN_BASELINE}" == "1" && ! -f "${scratch_a}" ]]; then
      echo "TRAIN_SCRATCH_RETRAIN_BASELINE=1 but scratch ckpt missing: ${scratch_a}" >&2
      exit 1
    fi
  fi

  need_conn=1
  if [[ "${SKIP_EXISTING}" == "1" && -f "${conn_summary}" && -f "${conn_soup}" ]]; then
    need_conn=0
  fi

  if [[ "${need_conn}" == "1" ]]; then
    echo "  - Connectivity/merge (simplex_soup)"
    conn_cmd=(
      env
      GPU="${GPU}"
      DENSE_CKPT="${DENSE_CKPT}"
      ENDPOINT_A="${unlearn_a}"
      ENDPOINT_B="${unlearn_b}"
      SCRATCH_RETRAIN_CKPT="${scratch_a}"
      OUT_ROOT="${OUT_CONNECTIVITY}"
      DATASET="${DATASET}"
      ARCH="${ARCH}"
      LAYERS="${LAYERS}"
      DATAPATH="${DATAPATH}"
      BATCH_SIZE="${BATCH_SIZE}"
      WORKERS="${WORKERS}"
      SPLIT_SEED="${SPLIT_SEED}"
      VAL_RATIO="${VAL_RATIO}"
      FORGET_OBJECTIVE="${FORGET_OBJECTIVE}"
      FORGET_ALPHA="${FORGET_ALPHA}"
      RETAIN_WEIGHT="${RETAIN_WEIGHT}"
      FORGET_VAL_BUDGET="${FORGET_VAL_BUDGET}"
      GRAD_CLIP="${GRAD_CLIP}"
      LAMBDAS="${LAMBDAS}"
      BN_RECALC="${BN_RECALC}"
      BN_BATCHES="${BN_BATCHES}"
      PERM_MAX_ITER="${PERM_MAX_ITER}"
      SEED="${CONNECTIVITY_SEED}"
      BEZIER_STEPS="${BEZIER_STEPS}"
      BEZIER_LR="${BEZIER_LR}"
      BEZIER_WEIGHT_DECAY="${BEZIER_WEIGHT_DECAY}"
      BEZIER_T_SAMPLES="${BEZIER_T_SAMPLES}"
      BEZIER_TAIL_K="${BEZIER_TAIL_K}"
      SIMPLEX_STEPS="${SIMPLEX_STEPS}"
      SIMPLEX_LR="${SIMPLEX_LR}"
      SIMPLEX_WEIGHT_DECAY="${SIMPLEX_WEIGHT_DECAY}"
      SIMPLEX_DIRICHLET_ALPHA="${SIMPLEX_DIRICHLET_ALPHA}"
      SIMPLEX_TAIL_K="${SIMPLEX_TAIL_K}"
      SIMPLEX_GRID_RESOLUTION="${SIMPLEX_GRID_RESOLUTION}"
      bash scripts/experiment/run_df1_seed42_connectivity.sh
    )
    run_cmd "${conn_cmd[@]}"
  else
    echo "  - Connectivity skipped (existing simplex_soup found)"
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  - DRY_RUN: skipped output existence checks"
  else
    if [[ ! -f "${conn_soup}" ]]; then
      echo "Connectivity output missing: ${conn_soup}" >&2
      exit 1
    fi
    echo "  - Done: ${conn_soup}"
  fi
done

echo
echo "All done."
echo "  unlearn root      : ${OUT_UNLEARN}"
echo "  connectivity root : ${OUT_CONNECTIVITY}"
