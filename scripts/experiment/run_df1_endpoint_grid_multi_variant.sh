#!/usr/bin/env bash
set -euo pipefail

# Run DF1 utility-first grid in parallel for multiple variant tags (e.g., ai, nature),
# writing each variant under its own output folder.
#
# Example:
#   bash scripts/experiment/run_df1_endpoint_grid_multi_variant.sh
#
# Custom:
#   VARIANTS="ai nature" GPU_MAP="0 1" \
#   DENSE_CKPT_AI=/path/to/ai.pth DENSE_CKPT_NATURE=/path/to/nature.pth \
#   bash scripts/experiment/run_df1_endpoint_grid_multi_variant.sh

VARIANTS="${VARIANTS:-ai nature}"
GPU_MAP="${GPU_MAP:-0 1}"
BASE_OUT_ROOT="${BASE_OUT_ROOT:-./runs/unlearning_df1_grid_utility_multi}"
COMMON_DENSE_CKPT="${COMMON_DENSE_CKPT:-runs/dense/cifar10/seed42/best_model.pth}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
FORGET_OBJECTIVE="${FORGET_OBJECTIVE:-kl_uniform}"
RUN_SCRATCH_BASELINE="${RUN_SCRATCH_BASELINE:-1}"
SCRATCH_EPOCHS="${SCRATCH_EPOCHS:-200}"
SCRATCH_LR="${SCRATCH_LR:-0.1}"
SCRATCH_SEED="${SCRATCH_SEED:-123}"

mkdir -p "${BASE_OUT_ROOT}"

# shellcheck disable=SC2206
VARIANT_ARR=(${VARIANTS})
# shellcheck disable=SC2206
GPU_ARR=(${GPU_MAP})

if [[ ${#VARIANT_ARR[@]} -eq 0 ]]; then
  echo "No variants provided."
  exit 1
fi

if [[ ${#GPU_ARR[@]} -lt ${#VARIANT_ARR[@]} ]]; then
  echo "GPU_MAP must provide at least as many entries as VARIANTS."
  echo "VARIANTS=${VARIANTS}"
  echo "GPU_MAP=${GPU_MAP}"
  exit 1
fi

pids=()

for idx in "${!VARIANT_ARR[@]}"; do
  variant="${VARIANT_ARR[$idx]}"
  gpu="${GPU_ARR[$idx]}"

  # Per-variant checkpoint override by env var:
  # e.g., DENSE_CKPT_AI=/... , DENSE_CKPT_NATURE=/...
  upper_variant="$(echo "${variant}" | tr '[:lower:]-' '[:upper:]_')"
  ckpt_var="DENSE_CKPT_${upper_variant}"
  dense_ckpt="${!ckpt_var:-${COMMON_DENSE_CKPT}}"

  if [[ ! -f "${dense_ckpt}" ]]; then
    echo "[${variant}] checkpoint not found: ${dense_ckpt}"
    exit 1
  fi

  out_dir="${BASE_OUT_ROOT}/${variant}"
  log_file="${BASE_OUT_ROOT}/${variant}.log"
  mkdir -p "${out_dir}"

  echo
  echo "============================================================"
  echo "[launch:${variant}] gpu=${gpu}"
  echo "ckpt=${dense_ckpt}"
  echo "out_dir=${out_dir}"
  echo "log=${log_file}"
  echo "============================================================"

  cmd=(
    env
    "DENSE_CKPT=${dense_ckpt}"
    "GPU=${gpu}"
    "BASE_OUT_DIR=${out_dir}"
    "DRY_RUN=${DRY_RUN}"
    "SKIP_EXISTING=${SKIP_EXISTING}"
    "FORGET_OBJECTIVE=${FORGET_OBJECTIVE}"
    "RUN_SCRATCH_BASELINE=${RUN_SCRATCH_BASELINE}"
    "SCRATCH_EPOCHS=${SCRATCH_EPOCHS}"
    "SCRATCH_LR=${SCRATCH_LR}"
    "SCRATCH_SEED=${SCRATCH_SEED}"
    bash scripts/experiment/run_df1_endpoint_grid.sh
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"
    echo
    continue
  fi

  (
    "${cmd[@]}"
  ) > "${log_file}" 2>&1 &

  pids+=($!)
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo
  echo "Dry-run complete."
  exit 0
fi

echo
echo "Launched ${#pids[@]} jobs. Waiting..."
for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "All variant jobs completed."
echo "Results root: ${BASE_OUT_ROOT}"
