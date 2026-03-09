#!/usr/bin/env bash
set -euo pipefail

# Build split files + plan JSONs and run MIA bank experiments for
# dense / raw_unlearn / scratch_retrain / merge_simplex_soup on victim 42,43.
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

SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-0}"
DRY_RUN="${DRY_RUN:-0}"

IFS=' ' read -r -a VICTIM_IDS <<< "${VICTIMS}"
IFS=' ' read -r -a SHADOW_IDS <<< "${SHADOW_MODEL_IDS}"

if [[ "${#VICTIM_IDS[@]}" -eq 0 ]]; then
  echo "VICTIMS is empty" >&2
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

write_plan_merge_simplex_soup() {
  local victim="$1"
  local plan="${PLAN_DIR}/mia_merge_bank_plan.victim${victim}_from_42_43_simplex_soup.json"
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
    "name": "simplex_soup_42_43_v${victim}",
    "pipeline": "merge_simplex_soup",
    "model_id": ${victim},
    "source_seeds": [42, 43],
    "ckpt_path": "runs/unlearning_connectivity_phase15_final/${DATASET}/df1/unlearn_seed42__unlearn_seed43/simplex_soup_best.pth"
  },
  "shadows": [
    {
      "name": "simplex_soup_44_45",
      "pipeline": "merge_simplex_soup",
      "model_id": 44,
      "source_seeds": [44, 45],
      "ckpt_path": "runs/unlearning_connectivity_phase15_final/${DATASET}/df1/unlearn_seed44__unlearn_seed45/simplex_soup_best.pth"
    },
    {
      "name": "simplex_soup_46_47",
      "pipeline": "merge_simplex_soup",
      "model_id": 46,
      "source_seeds": [46, 47],
      "ckpt_path": "runs/unlearning_connectivity_phase15_final/${DATASET}/df1/unlearn_seed46__unlearn_seed47/simplex_soup_best.pth"
    },
    {
      "name": "simplex_soup_48_49",
      "pipeline": "merge_simplex_soup",
      "model_id": 48,
      "source_seeds": [48, 49],
      "ckpt_path": "runs/unlearning_connectivity_phase15_final/${DATASET}/df1/unlearn_seed48__unlearn_seed49/simplex_soup_best.pth"
    },
    {
      "name": "simplex_soup_50_51",
      "pipeline": "merge_simplex_soup",
      "model_id": 50,
      "source_seeds": [50, 51],
      "ckpt_path": "runs/unlearning_connectivity_phase15_final/${DATASET}/df1/unlearn_seed50__unlearn_seed51/simplex_soup_best.pth"
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

echo "================================================================"
echo "MIA bank run for victims: ${VICTIMS}"
echo "dataset: ${DATASET}"
echo "split_seed: ${SPLIT_SEED}"
echo "gpu: ${GPU}"
echo "plan_dir: ${PLAN_DIR}"
echo "result_root: ${RESULT_ROOT}"
echo "================================================================"

for victim in "${VICTIM_IDS[@]}"; do
  echo "[split] victim=${victim}, shadows=${SHADOW_MODEL_IDS}"
  run_cmd python mia_eval/create_data/create_fixed_data_splits.py \
    --dataset "${DATASET}" \
    --seed "${SPLIT_SEED}" \
    --victim_seed "${victim}" \
    --shadow_seeds "${SHADOW_IDS[@]}" \
    --verify
done

declare -a PLANS=()
for victim in "${VICTIM_IDS[@]}"; do
  PLANS+=("$(write_plan_dense "${victim}")")
  PLANS+=("$(write_plan_raw_unlearn "${victim}")")
  PLANS+=("$(write_plan_scratch_retrain "${victim}")")
  PLANS+=("$(write_plan_merge_simplex_soup "${victim}")")
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
