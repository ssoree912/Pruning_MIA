#!/usr/bin/env bash
set -euo pipefail

# Build split files + plan JSONs and run MIA bank experiments for
# dense / raw_unlearn / scratch_retrain / merge_* on victim 42,43.
#
# Example:
#   bash scripts/experiment/run_df1_seed42_43_mia_bank.sh
#   GPU=0 SPLIT_SEED=7 RESULT_ROOT=runs/mia_merge_bank \
#     bash scripts/experiment/run_df1_seed42_43_mia_bank.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DATASET="${DATASET:-cifar10}"
SPLIT_SEED="${SPLIT_SEED:-7}"
GPU="${GPU:-0}"
ATTACKS="${ATTACKS:-threshold,lira,nn,samia}"
FORWARD_MODE="${FORWARD_MODE:-standard}"
TPR_FPRS="${TPR_FPRS:-0.1,1,5}"
SAVE_SCORES="${SAVE_SCORES:-true}"
BASE_CONFIG_PATH="${BASE_CONFIG_PATH:-runs/dense/cifar10/seed42/config.json}"
RESULT_ROOT="${RESULT_ROOT:-runs/mia_merge_bank}"
PLAN_DIR="${PLAN_DIR:-plans/mia_seed42_43_bank}"

VICTIMS="${VICTIMS:-42 43}"
SHADOW_MODEL_IDS="${SHADOW_MODEL_IDS:-44 46 48 50}"
MERGE_METHODS="${MERGE_METHODS:-raw_linear_best perm_linear_best bezier_best bezier_swa_best simplex_best simplex_soup_best simplex_swa_best}"
DENSE_VICTIMS="${DENSE_VICTIMS:-${VICTIMS}}"
UNLEARN_VICTIMS="${UNLEARN_VICTIMS:-${VICTIMS}}"
RETRAIN_VICTIMS="${RETRAIN_VICTIMS:-${VICTIMS}}"
MERGE_VICTIMS="${MERGE_VICTIMS:-${VICTIMS}}"

SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-0}"
DRY_RUN="${DRY_RUN:-0}"

IFS=' ' read -r -a VICTIM_IDS <<< "${VICTIMS}"
IFS=' ' read -r -a SHADOW_IDS <<< "${SHADOW_MODEL_IDS}"
IFS=' ' read -r -a MERGE_METHOD_IDS <<< "${MERGE_METHODS}"
IFS=' ' read -r -a DENSE_VICTIM_IDS <<< "${DENSE_VICTIMS}"
IFS=' ' read -r -a UNLEARN_VICTIM_IDS <<< "${UNLEARN_VICTIMS}"
IFS=' ' read -r -a RETRAIN_VICTIM_IDS <<< "${RETRAIN_VICTIMS}"
IFS=' ' read -r -a MERGE_VICTIM_IDS <<< "${MERGE_VICTIMS}"

if [[ "${#DENSE_VICTIM_IDS[@]}" -eq 0 && "${#UNLEARN_VICTIM_IDS[@]}" -eq 0 && "${#RETRAIN_VICTIM_IDS[@]}" -eq 0 && "${#MERGE_VICTIM_IDS[@]}" -eq 0 ]]; then
  echo "No victims configured (DENSE/UNLEARN/RETRAIN/MERGE all empty)" >&2
  exit 1
fi
if [[ "${#MERGE_METHOD_IDS[@]}" -eq 0 ]]; then
  echo "MERGE_METHODS is empty" >&2
  exit 1
fi
if [[ "${#SHADOW_IDS[@]}" -lt 4 ]]; then
  echo "Need at least 4 SHADOW_MODEL_IDS, got ${#SHADOW_IDS[@]}" >&2
  exit 1
fi
if [[ ! -f "${BASE_CONFIG_PATH}" ]]; then
  echo "Missing base config: ${BASE_CONFIG_PATH}" >&2
  exit 1
fi

mkdir -p "${PLAN_DIR}"

pair_dir_for_seed() {
  local seed="$1"
  if (( seed % 2 == 0 )); then
    echo "seed${seed}_seed$((seed + 1))"
  else
    echo "seed$((seed - 1))_seed${seed}"
  fi
}

write_plan_dense() {
  local victim="$1"
  local plan="${PLAN_DIR}/mia_dense_bank_plan.victim${victim}.json"
  cat > "${plan}" <<EOF
{
  "dataset_name": "${DATASET}",
  "base_config_path": "${BASE_CONFIG_PATH}",
  "split_seed": ${SPLIT_SEED},
  "device": ${GPU},
  "attacks": "${ATTACKS}",
  "forward_mode": "${FORWARD_MODE}",
  "tpr_fprs": "${TPR_FPRS}",
  "save_scores": ${SAVE_SCORES},
  "result_root": "${RESULT_ROOT}",
  "victim": {
    "name": "dense_${victim}",
    "pipeline": "dense",
    "model_id": ${victim},
    "source_seeds": [${victim}],
    "ckpt_path": "runs/dense/${DATASET}/seed${victim}/best_model.pth"
  },
  "shadows": [
    {
      "name": "dense_44",
      "pipeline": "dense",
      "model_id": 44,
      "source_seeds": [44],
      "ckpt_path": "runs/dense/${DATASET}/seed44/best_model.pth"
    },
    {
      "name": "dense_46",
      "pipeline": "dense",
      "model_id": 46,
      "source_seeds": [46],
      "ckpt_path": "runs/dense/${DATASET}/seed46/best_model.pth"
    },
    {
      "name": "dense_48",
      "pipeline": "dense",
      "model_id": 48,
      "source_seeds": [48],
      "ckpt_path": "runs/dense/${DATASET}/seed48/best_model.pth"
    },
    {
      "name": "dense_50",
      "pipeline": "dense",
      "model_id": 50,
      "source_seeds": [50],
      "ckpt_path": "runs/dense/${DATASET}/seed50/best_model.pth"
    }
  ]
}
EOF
  echo "${plan}"
}

write_plan_raw_unlearn() {
  local victim="$1"
  local pair_dir
  pair_dir="$(pair_dir_for_seed "${victim}")"
  local plan="${PLAN_DIR}/mia_raw_unlearn_bank_plan.victim${victim}.json"
  cat > "${plan}" <<EOF
{
  "dataset_name": "${DATASET}",
  "base_config_path": "${BASE_CONFIG_PATH}",
  "split_seed": ${SPLIT_SEED},
  "device": ${GPU},
  "attacks": "${ATTACKS}",
  "forward_mode": "${FORWARD_MODE}",
  "tpr_fprs": "${TPR_FPRS}",
  "save_scores": ${SAVE_SCORES},
  "result_root": "${RESULT_ROOT}",
  "victim": {
    "name": "raw_unlearn_${victim}",
    "pipeline": "raw_unlearn",
    "model_id": ${victim},
    "source_seeds": [${victim}],
    "ckpt_path": "runs/unlearning_df1/unlearn/${DATASET}/df1/${pair_dir}/unlearn_seed${victim}.pth"
  },
  "shadows": [
    {
      "name": "raw_unlearn_44",
      "pipeline": "raw_unlearn",
      "model_id": 44,
      "source_seeds": [44],
      "ckpt_path": "runs/unlearning_df1/unlearn/${DATASET}/df1/seed44_seed45/unlearn_seed44.pth"
    },
    {
      "name": "raw_unlearn_46",
      "pipeline": "raw_unlearn",
      "model_id": 46,
      "source_seeds": [46],
      "ckpt_path": "runs/unlearning_df1/unlearn/${DATASET}/df1/seed46_seed47/unlearn_seed46.pth"
    },
    {
      "name": "raw_unlearn_48",
      "pipeline": "raw_unlearn",
      "model_id": 48,
      "source_seeds": [48],
      "ckpt_path": "runs/unlearning_df1/unlearn/${DATASET}/df1/seed48_seed49/unlearn_seed48.pth"
    },
    {
      "name": "raw_unlearn_50",
      "pipeline": "raw_unlearn",
      "model_id": 50,
      "source_seeds": [50],
      "ckpt_path": "runs/unlearning_df1/unlearn/${DATASET}/df1/seed50_seed51/unlearn_seed50.pth"
    }
  ]
}
EOF
  echo "${plan}"
}

write_plan_scratch_retrain() {
  local victim="$1"
  local pair_dir
  pair_dir="$(pair_dir_for_seed "${victim}")"
  local plan="${PLAN_DIR}/mia_retrain_bank_plan.victim${victim}.json"
  cat > "${plan}" <<EOF
{
  "dataset_name": "${DATASET}",
  "base_config_path": "${BASE_CONFIG_PATH}",
  "split_seed": ${SPLIT_SEED},
  "device": ${GPU},
  "attacks": "${ATTACKS}",
  "forward_mode": "${FORWARD_MODE}",
  "tpr_fprs": "${TPR_FPRS}",
  "save_scores": ${SAVE_SCORES},
  "result_root": "${RESULT_ROOT}",
  "victim": {
    "name": "scratch_retrain_${victim}",
    "pipeline": "scratch_retrain",
    "model_id": ${victim},
    "source_seeds": [${victim}],
    "ckpt_path": "runs/unlearning_df1/retrain/${DATASET}/df1/${pair_dir}/scratch_retrain_seed${victim}.pth"
  },
  "shadows": [
    {
      "name": "scratch_retrain_44",
      "pipeline": "scratch_retrain",
      "model_id": 44,
      "source_seeds": [44],
      "ckpt_path": "runs/unlearning_df1/retrain/${DATASET}/df1/seed44_seed45/scratch_retrain_seed44.pth"
    },
    {
      "name": "scratch_retrain_46",
      "pipeline": "scratch_retrain",
      "model_id": 46,
      "source_seeds": [46],
      "ckpt_path": "runs/unlearning_df1/retrain/${DATASET}/df1/seed46_seed47/scratch_retrain_seed46.pth"
    },
    {
      "name": "scratch_retrain_48",
      "pipeline": "scratch_retrain",
      "model_id": 48,
      "source_seeds": [48],
      "ckpt_path": "runs/unlearning_df1/retrain/${DATASET}/df1/seed48_seed49/scratch_retrain_seed48.pth"
    },
    {
      "name": "scratch_retrain_50",
      "pipeline": "scratch_retrain",
      "model_id": 50,
      "source_seeds": [50],
      "ckpt_path": "runs/unlearning_df1/retrain/${DATASET}/df1/seed50_seed51/scratch_retrain_seed50.pth"
    }
  ]
}
EOF
  echo "${plan}"
}

_merge_pipeline_from_method() {
  local method="$1"
  case "${method}" in
    raw_linear_best) echo "merge_raw_linear" ;;
    perm_linear_best) echo "merge_perm_linear" ;;
    bezier_best) echo "merge_bezier" ;;
    bezier_swa_best) echo "merge_bezier_swa" ;;
    simplex_best) echo "merge_simplex" ;;
    simplex_soup_best) echo "merge_simplex_soup" ;;
    simplex_swa_best) echo "merge_simplex_swa" ;;
    *)
      echo "Unknown merge method: ${method}" >&2
      return 1
      ;;
  esac
}

write_plan_merge_method() {
  local victim="$1"
  local method="$2"
  local pipeline
  pipeline="$(_merge_pipeline_from_method "${method}")"
  local plan="${PLAN_DIR}/mia_${pipeline}_bank_plan.victim${victim}_from_42_43.json"
  local method_short="${method%_best}"
  cat > "${plan}" <<EOF
{
  "dataset_name": "${DATASET}",
  "base_config_path": "${BASE_CONFIG_PATH}",
  "split_seed": ${SPLIT_SEED},
  "device": ${GPU},
  "attacks": "${ATTACKS}",
  "forward_mode": "${FORWARD_MODE}",
  "tpr_fprs": "${TPR_FPRS}",
  "save_scores": ${SAVE_SCORES},
  "result_root": "${RESULT_ROOT}",
  "victim": {
    "name": "${method_short}_42_43_v${victim}",
    "pipeline": "${pipeline}",
    "model_id": ${victim},
    "source_seeds": [42, 43],
    "ckpt_path": "runs/unlearning_connectivity_phase15_final/${DATASET}/df1/unlearn_seed42__unlearn_seed43/${method}"
  },
  "shadows": [
    {
      "name": "${method_short}_44_45",
      "pipeline": "${pipeline}",
      "model_id": 44,
      "source_seeds": [44, 45],
      "ckpt_path": "runs/unlearning_connectivity_phase15_final/${DATASET}/df1/unlearn_seed44__unlearn_seed45/${method}"
    },
    {
      "name": "${method_short}_46_47",
      "pipeline": "${pipeline}",
      "model_id": 46,
      "source_seeds": [46, 47],
      "ckpt_path": "runs/unlearning_connectivity_phase15_final/${DATASET}/df1/unlearn_seed46__unlearn_seed47/${method}"
    },
    {
      "name": "${method_short}_48_49",
      "pipeline": "${pipeline}",
      "model_id": 48,
      "source_seeds": [48, 49],
      "ckpt_path": "runs/unlearning_connectivity_phase15_final/${DATASET}/df1/unlearn_seed48__unlearn_seed49/${method}"
    },
    {
      "name": "${method_short}_50_51",
      "pipeline": "${pipeline}",
      "model_id": 50,
      "source_seeds": [50, 51],
      "ckpt_path": "runs/unlearning_connectivity_phase15_final/${DATASET}/df1/unlearn_seed50__unlearn_seed51/${method}"
    }
  ]
}
EOF
  echo "${plan}"
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

append_unique_int() {
  local x="$1"
  local y
  for y in "${SPLIT_VICTIMS[@]}"; do
    if [[ "${y}" == "${x}" ]]; then
      return 0
    fi
  done
  SPLIT_VICTIMS+=("${x}")
}

echo "================================================================"
echo "MIA bank run"
echo "dataset: ${DATASET}"
echo "split_seed: ${SPLIT_SEED}"
echo "gpu: ${GPU}"
echo "dense_victims: ${DENSE_VICTIMS}"
echo "unlearn_victims: ${UNLEARN_VICTIMS}"
echo "retrain_victims: ${RETRAIN_VICTIMS}"
echo "merge_victims: ${MERGE_VICTIMS}"
echo "merge_methods: ${MERGE_METHODS}"
echo "plan_dir: ${PLAN_DIR}"
echo "result_root: ${RESULT_ROOT}"
echo "================================================================"

declare -a SPLIT_VICTIMS=()
for victim in "${DENSE_VICTIM_IDS[@]}" "${UNLEARN_VICTIM_IDS[@]}" "${RETRAIN_VICTIM_IDS[@]}" "${MERGE_VICTIM_IDS[@]}"; do
  if [[ -n "${victim}" ]]; then
    append_unique_int "${victim}"
  fi
done

for victim in "${SPLIT_VICTIMS[@]}"; do
  echo "[split] victim=${victim}, shadows=${SHADOW_MODEL_IDS}"
  run_cmd python mia_eval/create_data/create_fixed_data_splits.py \
    --dataset "${DATASET}" \
    --seed "${SPLIT_SEED}" \
    --victim_seed "${victim}" \
    --shadow_seeds "${SHADOW_IDS[@]}" \
    --verify
done

declare -a PLANS=()
for victim in "${DENSE_VICTIM_IDS[@]}"; do
  PLANS+=("$(write_plan_dense "${victim}")")
done
for victim in "${UNLEARN_VICTIM_IDS[@]}"; do
  PLANS+=("$(write_plan_raw_unlearn "${victim}")")
done
for victim in "${RETRAIN_VICTIM_IDS[@]}"; do
  PLANS+=("$(write_plan_scratch_retrain "${victim}")")
done
for victim in "${MERGE_VICTIM_IDS[@]}"; do
  for merge_method in "${MERGE_METHOD_IDS[@]}"; do
    PLANS+=("$(write_plan_merge_method "${victim}" "${merge_method}")")
  done
done

echo "[plans] created:"
for plan in "${PLANS[@]}"; do
  echo "  - ${plan}"
done

for plan in "${PLANS[@]}"; do
  cmd=(
    python scripts/experiment/run_mia_merge_bank.py
    --repo-root "${REPO_ROOT}"
    --plan-json "${plan}"
    --min-shadows 4
  )
  if [[ "${SKIP_IF_EXISTS}" == "1" ]]; then
    cmd+=(--skip-if-exists)
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    cmd+=(--dry-run)
  fi
  run_cmd "${cmd[@]}"
done

echo
echo "Done."
echo "Plans  : ${PLAN_DIR}"
echo "Results: ${RESULT_ROOT}/${DATASET}"
