#!/usr/bin/env bash
set -euo pipefail

# Auto-run connectivity with same-init pair.
# Usage:
#   scripts/run_connectivity_auto.sh --method static --sparsity 0.9 --dataset cifar10 --mode static
#   scripts/run_connectivity_auto.sh --method dpf --sparsity 0.9 --dataset cifar10 --mode dynamic_naive
#   scripts/run_connectivity_auto.sh --method dpf --sparsity 0.9 --dataset cifar10 --mode dynamic_common

METHOD=""
SPARSITY=""
DATASET="cifar10"
MODE="static"
FREEZE_TAG=""
LAMS=101
BN_RECALC=1
BN_BATCHES=200
RUNS_DIR="./runs"
OUT_JSON=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) METHOD="$2"; shift 2 ;;
    --sparsity) SPARSITY="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --freeze-tag) FREEZE_TAG="$2"; shift 2 ;;
    --lambdas) LAMS="$2"; shift 2 ;;
    --bn_recalc) BN_RECALC=1; shift 1 ;;
    --no-bn_recalc) BN_RECALC=0; shift 1 ;;
    --bn_batches) BN_BATCHES="$2"; shift 2 ;;
    --runs) RUNS_DIR="$2"; shift 2 ;;
    --out_json) OUT_JSON="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "${METHOD}" ]]; then echo "--method is required"; exit 1; fi
if [[ "${METHOD}" != "dense" && -z "${SPARSITY}" ]]; then echo "--sparsity is required"; exit 1; fi

PAIR=$(python scripts/find_same_init_pair.py \
  --method "${METHOD}" --sparsity "${SPARSITY}" --dataset "${DATASET}" \
  ${FREEZE_TAG:+--freeze-tag "${FREEZE_TAG}"} --runs "${RUNS_DIR}")

SEED0=$(echo "${PAIR}" | awk '{print $1}')
SEED1=$(echo "${PAIR}" | awk '{print $2}')
echo "Using seeds: ${SEED0}, ${SEED1}"

if [[ "${METHOD}" == "dense" ]]; then
  CKPT0="${RUNS_DIR}/dense/${DATASET}/seed${SEED0}/best_model.pth"
  CKPT1="${RUNS_DIR}/dense/${DATASET}/seed${SEED1}/best_model.pth"
elif [[ "${METHOD}" == "static" ]]; then
  CKPT0="${RUNS_DIR}/static/sparsity_${SPARSITY}/${DATASET}/seed${SEED0}/best_model.pth"
  CKPT1="${RUNS_DIR}/static/sparsity_${SPARSITY}/${DATASET}/seed${SEED1}/best_model.pth"
else
  TAG=""
  if [[ -n "${FREEZE_TAG}" ]]; then TAG="_${FREEZE_TAG}"; fi
  CKPT0="${RUNS_DIR}/dpf/sparsity_${SPARSITY}${TAG}/${DATASET}/seed${SEED0}/best_model.pth"
  CKPT1="${RUNS_DIR}/dpf/sparsity_${SPARSITY}${TAG}/${DATASET}/seed${SEED1}/best_model.pth"
fi

MASK_DIR="masks"
mkdir -p "${MASK_DIR}"

ARGS=(--ckpt0 "${CKPT0}" --ckpt1 "${CKPT1}" --mode "${MODE}" --lambdas "${LAMS}")
if [[ "${BN_RECALC}" -eq 1 ]]; then
  ARGS+=(--bn_recalc --bn_recalc_batches "${BN_BATCHES}")
fi

if [[ "${MODE}" == "static" ]]; then
  MASK="${MASK_DIR}/${METHOD}_${SPARSITY}_seed${SEED0}.pt"
  if [[ ! -f "${MASK}" ]]; then
    python scripts/extract_masks.py --ckpt "${CKPT0}" --out "${MASK}"
  fi
  ARGS+=(--mask "${MASK}")
elif [[ "${MODE}" == "dynamic_common" ]]; then
  MASK0="${MASK_DIR}/${METHOD}_${SPARSITY}_seed${SEED0}.pt"
  MASK1="${MASK_DIR}/${METHOD}_${SPARSITY}_seed${SEED1}.pt"
  if [[ ! -f "${MASK0}" ]]; then
    python scripts/extract_masks.py --ckpt "${CKPT0}" --out "${MASK0}"
  fi
  if [[ ! -f "${MASK1}" ]]; then
    python scripts/extract_masks.py --ckpt "${CKPT1}" --out "${MASK1}"
  fi
  ARGS+=(--mask0 "${MASK0}" --mask1 "${MASK1}")
fi

if [[ -z "${OUT_JSON}" ]]; then
  OUT_JSON="results/connectivity_${METHOD}_${SPARSITY}_${MODE}.json"
fi
ARGS+=(--out_json "${OUT_JSON}")

python tools/linear_connectivity.py "${ARGS[@]}"
