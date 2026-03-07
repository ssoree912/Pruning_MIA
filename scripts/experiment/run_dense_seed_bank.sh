#!/usr/bin/env bash
set -euo pipefail

# Train dense checkpoints for a seed bank (default: 42..51).
#
# Usage:
#   bash scripts/experiment/run_dense_seed_bank.sh
#   GPU=0 SEED_LIST="42,43,44,45,46,47,48,49,50,51" \
#     bash scripts/experiment/run_dense_seed_bank.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEED_LIST="${SEED_LIST:-42,43,44,45,46,47,48,49,50,51}"

OUT_ROOT="${OUT_ROOT:-runs/dense}"
DATASET="${DATASET:-cifar10}"
ARCH="${ARCH:-resnet}"
LAYERS="${LAYERS:-20}"

EPOCHS="${EPOCHS:-200}"
LR="${LR:-0.1}"
MOMENTUM="${MOMENTUM:-0.9}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"

GPU="${GPU:-0}"
DATAPATH="${DATAPATH:-~/Datasets/CIFAR}"
DATAPATH="${DATAPATH/#\~/$HOME}"
BATCH_SIZE="${BATCH_SIZE:-128}"
WORKERS="${WORKERS:-4}"

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

IFS=',' read -r -a SEEDS <<< "${SEED_LIST}"
if [[ "${#SEEDS[@]}" -eq 0 ]]; then
  echo "SEED_LIST is empty" >&2
  exit 1
fi

echo "================================================================"
echo "Dense seed-bank training"
echo "repo: ${REPO_ROOT}"
echo "dataset/arch/layers: ${DATASET}/${ARCH}/${LAYERS}"
echo "seeds: ${SEED_LIST}"
echo "out_root: ${OUT_ROOT}"
echo "epochs/lr: ${EPOCHS}/${LR}"
echo "gpu: ${GPU}"
echo "skip_existing: ${SKIP_EXISTING}"
echo "dry_run: ${DRY_RUN}"
echo "================================================================"

idx=0
for raw_seed in "${SEEDS[@]}"; do
  seed="${raw_seed//[[:space:]]/}"
  if [[ -z "${seed}" ]]; then
    continue
  fi
  if ! is_int "${seed}"; then
    echo "Invalid seed in SEED_LIST: ${raw_seed}" >&2
    exit 1
  fi

  idx=$((idx + 1))
  out_dir="${OUT_ROOT}/${DATASET}/seed${seed}"
  best_ckpt="${out_dir}/best_model.pth"
  cfg_json="${out_dir}/config.json"

  echo
  echo "[${idx}/${#SEEDS[@]}] seed ${seed}"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${best_ckpt}" && -f "${cfg_json}" ]]; then
    echo "  - skip: existing outputs found"
    continue
  fi

  cmd=(
    python run_experiment.py
    --name "dense_${DATASET}_seed${seed}"
    --save-dir "${out_dir}"
    --dataset "${DATASET}"
    --arch "${ARCH}"
    --layers "${LAYERS}"
    --epochs "${EPOCHS}"
    --lr "${LR}"
    --momentum "${MOMENTUM}"
    --weight-decay "${WEIGHT_DECAY}"
    --batch-size "${BATCH_SIZE}"
    --workers "${WORKERS}"
    --datapath "${DATAPATH}"
    --gpu "${GPU}"
    --seed "${seed}"
  )
  run_cmd "${cmd[@]}"

  if [[ "${DRY_RUN}" != "1" ]]; then
    if [[ ! -f "${best_ckpt}" || ! -f "${cfg_json}" ]]; then
      echo "Dense outputs missing for seed ${seed}" >&2
      echo "  expected: ${best_ckpt}" >&2
      echo "  expected: ${cfg_json}" >&2
      exit 1
    fi
  fi
done

echo
echo "All done."
echo "Dense root: ${OUT_ROOT}/${DATASET}"
