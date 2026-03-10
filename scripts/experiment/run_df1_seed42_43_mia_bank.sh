#!/usr/bin/env bash
set -euo pipefail

# Build split files + plan JSONs and run MIA bank experiments for
# dense / raw_unlearn / scratch_retrain / merge_*.
#
# Merge plans are pair-based by default:
#   victim: merge best from 42_43
#   shadows: merge best from 44_45, 46_47, 48_49, 50_51
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
DENSE_VICTIMS="${DENSE_VICTIMS-42}"
UNLEARN_VICTIMS="${UNLEARN_VICTIMS:-${VICTIMS}}"
RETRAIN_VICTIMS="${RETRAIN_VICTIMS:-${VICTIMS}}"

# Legacy merge seed list. Converted to pair tokens via pair_token_for_seed unless MERGE_VICTIM_PAIRS is set.
MERGE_VICTIMS="${MERGE_VICTIMS:-${VICTIMS}}"
# Optional explicit merge victim pairs (space-separated A_B tokens).
MERGE_VICTIM_PAIRS="${MERGE_VICTIM_PAIRS:-}"
# Optional explicit merge shadow pairs (space-separated A_B tokens). If unset, derived from SHADOW_MODEL_IDS.
MERGE_SHADOW_PAIRS="${MERGE_SHADOW_PAIRS:-}"

MIN_SHADOWS="${MIN_SHADOWS:-4}"
SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-0}"
SKIP_MISSING_CKPT="${SKIP_MISSING_CKPT:-1}"
DRY_RUN="${DRY_RUN:-0}"

SHADOW_IDS=()
MERGE_METHOD_IDS=()
DENSE_VICTIM_IDS=()
UNLEARN_VICTIM_IDS=()
RETRAIN_VICTIM_IDS=()
MERGE_VICTIM_SEED_IDS=()
if [[ -n "${SHADOW_MODEL_IDS// }" ]]; then
  IFS=' ' read -r -a SHADOW_IDS <<< "${SHADOW_MODEL_IDS}"
fi
if [[ -n "${MERGE_METHODS// }" ]]; then
  IFS=' ' read -r -a MERGE_METHOD_IDS <<< "${MERGE_METHODS}"
fi
if [[ -n "${DENSE_VICTIMS// }" ]]; then
  IFS=' ' read -r -a DENSE_VICTIM_IDS <<< "${DENSE_VICTIMS}"
fi
if [[ -n "${UNLEARN_VICTIMS// }" ]]; then
  IFS=' ' read -r -a UNLEARN_VICTIM_IDS <<< "${UNLEARN_VICTIMS}"
fi
if [[ -n "${RETRAIN_VICTIMS// }" ]]; then
  IFS=' ' read -r -a RETRAIN_VICTIM_IDS <<< "${RETRAIN_VICTIMS}"
fi
if [[ -n "${MERGE_VICTIMS// }" ]]; then
  IFS=' ' read -r -a MERGE_VICTIM_SEED_IDS <<< "${MERGE_VICTIMS}"
fi

MERGE_VICTIM_PAIR_OVERRIDES=()
MERGE_SHADOW_PAIR_OVERRIDES=()
if [[ -n "${MERGE_VICTIM_PAIRS// }" ]]; then
  IFS=' ' read -r -a MERGE_VICTIM_PAIR_OVERRIDES <<< "${MERGE_VICTIM_PAIRS}"
fi
if [[ -n "${MERGE_SHADOW_PAIRS// }" ]]; then
  IFS=' ' read -r -a MERGE_SHADOW_PAIR_OVERRIDES <<< "${MERGE_SHADOW_PAIRS}"
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

pair_token_for_seed() {
  local seed="$1"
  if (( seed % 2 == 0 )); then
    echo "${seed}_$((seed + 1))"
  else
    echo "$((seed - 1))_${seed}"
  fi
}

pair_split_from_token() {
  local token="$1"
  if [[ ! "${token}" =~ ^([0-9]+)_([0-9]+)$ ]]; then
    echo "Invalid pair token '${token}' (expected A_B)." >&2
    return 1
  fi
  local a="${BASH_REMATCH[1]}"
  local b="${BASH_REMATCH[2]}"
  if (( 10#${a} == 10#${b} )); then
    echo "Invalid pair token '${token}' (A and B must differ)." >&2
    return 1
  fi
  echo "${a} ${b}"
}

normalize_pair_token() {
  local token="$1"
  local a b
  read -r a b <<< "$(pair_split_from_token "${token}")"
  if (( 10#${a} <= 10#${b} )); then
    echo "${a}_${b}"
  else
    echo "${b}_${a}"
  fi
}

pair_dir_for_pair_token() {
  local token="$1"
  local norm
  norm="$(normalize_pair_token "${token}")"
  local a b
  read -r a b <<< "$(pair_split_from_token "${norm}")"
  echo "unlearn_seed${a}__unlearn_seed${b}"
}

pair_model_id_for_pair_token() {
  local token="$1"
  local norm
  norm="$(normalize_pair_token "${token}")"
  local a b
  read -r a b <<< "$(pair_split_from_token "${norm}")"
  # Pair-level ID used only for MIA split/config identity.
  echo "$((10#${a} * 1000 + 10#${b}))"
}

write_single_seed_shadows() {
  local pipeline="$1"
  local active_shadows=()
  local shadow
  for shadow in "${SHADOW_IDS[@]-}"; do
    if [[ -n "${shadow}" ]]; then
      active_shadows+=("${shadow}")
    fi
  done

  local i
  for i in "${!active_shadows[@]}"; do
    local shadow_id pair_dir ckpt_path suffix
    shadow_id="${active_shadows[$i]}"
    pair_dir="$(pair_dir_for_seed "${shadow_id}")"
    case "${pipeline}" in
      dense)
        ckpt_path="runs/dense/${DATASET}/seed${shadow_id}/best_model.pth"
        ;;
      raw_unlearn)
        ckpt_path="runs/unlearning_df1/unlearn/${DATASET}/df1/${pair_dir}/unlearn_seed${shadow_id}.pth"
        ;;
      scratch_retrain)
        ckpt_path="runs/unlearning_df1/retrain/${DATASET}/df1/${pair_dir}/scratch_retrain_seed${shadow_id}.pth"
        ;;
      *)
        echo "Unsupported single-seed pipeline: ${pipeline}" >&2
        return 1
        ;;
    esac
    suffix=","
    if (( i == ${#active_shadows[@]} - 1 )); then
      suffix=""
    fi
    cat <<EOF
    {
      "name": "${pipeline}_${shadow_id}",
      "pipeline": "${pipeline}",
      "model_id": ${shadow_id},
      "source_seeds": [${shadow_id}],
      "ckpt_path": "${ckpt_path}"
    }${suffix}
EOF
  done
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
EOF
  {
    write_single_seed_shadows "dense"
    cat <<EOF
  ]
}
EOF
  } >> "${plan}"
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
EOF
  {
    write_single_seed_shadows "raw_unlearn"
    cat <<EOF
  ]
}
EOF
  } >> "${plan}"
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
EOF
  {
    write_single_seed_shadows "scratch_retrain"
    cat <<EOF
  ]
}
EOF
  } >> "${plan}"
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
  local victim_pair_raw="$1"
  local method="$2"
  local victim_pair
  victim_pair="$(normalize_pair_token "${victim_pair_raw}")"
  local pipeline
  pipeline="$(_merge_pipeline_from_method "${method}")"
  local plan="${PLAN_DIR}/mia_${pipeline}_bank_plan.victim${victim_pair}.json"
  local method_short="${method%_best}"
  local victim_a victim_b
  read -r victim_a victim_b <<< "$(pair_split_from_token "${victim_pair}")"
  local victim_model_id
  victim_model_id="$(pair_model_id_for_pair_token "${victim_pair}")"
  local victim_pair_dir
  victim_pair_dir="$(pair_dir_for_pair_token "${victim_pair}")"

  if [[ "${#MERGE_SHADOW_PAIR_IDS[@]}" -eq 0 ]]; then
    echo "MERGE_SHADOW_PAIR_IDS is empty" >&2
    return 1
  fi

  {
    cat <<EOF
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
    "name": "${method_short}_${victim_pair}",
    "pipeline": "${pipeline}",
    "model_id": ${victim_model_id},
    "source_seeds": [${victim_a}, ${victim_b}],
    "ckpt_path": "runs/unlearning_connectivity_phase15_final/${DATASET}/df1/${victim_pair_dir}/${method}.pth"
  },
  "shadows": [
EOF
    local i
    for i in "${!MERGE_SHADOW_PAIR_IDS[@]}"; do
      local shadow_pair shadow_a shadow_b shadow_model_id shadow_pair_dir suffix
      shadow_pair="$(normalize_pair_token "${MERGE_SHADOW_PAIR_IDS[$i]}")"
      read -r shadow_a shadow_b <<< "$(pair_split_from_token "${shadow_pair}")"
      shadow_model_id="$(pair_model_id_for_pair_token "${shadow_pair}")"
      shadow_pair_dir="$(pair_dir_for_pair_token "${shadow_pair}")"
      suffix=","
      if (( i == ${#MERGE_SHADOW_PAIR_IDS[@]} - 1 )); then
        suffix=""
      fi
      cat <<EOF
    {
      "name": "${method_short}_${shadow_pair}",
      "pipeline": "${pipeline}",
      "model_id": ${shadow_model_id},
      "source_seeds": [${shadow_a}, ${shadow_b}],
      "ckpt_path": "runs/unlearning_connectivity_phase15_final/${DATASET}/df1/${shadow_pair_dir}/${method}.pth"
    }${suffix}
EOF
    done
    cat <<EOF
  ]
}
EOF
  } > "${plan}"
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
  for y in "${SPLIT_VICTIMS[@]-}"; do
    if [[ "${y}" == "${x}" ]]; then
      return 0
    fi
  done
  SPLIT_VICTIMS+=("${x}")
}

append_unique_merge_int() {
  local x="$1"
  local y
  for y in "${MERGE_SPLIT_VICTIMS[@]-}"; do
    if [[ "${y}" == "${x}" ]]; then
      return 0
    fi
  done
  MERGE_SPLIT_VICTIMS+=("${x}")
}

append_unique_pair_victim() {
  local x="$1"
  local y
  for y in "${MERGE_VICTIM_PAIR_IDS[@]-}"; do
    if [[ "${y}" == "${x}" ]]; then
      return 0
    fi
  done
  MERGE_VICTIM_PAIR_IDS+=("${x}")
}

append_unique_pair_shadow() {
  local x="$1"
  local y
  for y in "${MERGE_SHADOW_PAIR_IDS[@]-}"; do
    if [[ "${y}" == "${x}" ]]; then
      return 0
    fi
  done
  MERGE_SHADOW_PAIR_IDS+=("${x}")
}

declare -a MERGE_VICTIM_PAIR_IDS=()
if [[ "${#MERGE_VICTIM_PAIR_OVERRIDES[@]}" -gt 0 ]]; then
  for raw_pair in "${MERGE_VICTIM_PAIR_OVERRIDES[@]-}"; do
    if [[ -z "${raw_pair}" ]]; then
      continue
    fi
    append_unique_pair_victim "$(normalize_pair_token "${raw_pair}")"
  done
else
  for victim_seed in "${MERGE_VICTIM_SEED_IDS[@]-}"; do
    if [[ -z "${victim_seed}" ]]; then
      continue
    fi
    append_unique_pair_victim "$(pair_token_for_seed "${victim_seed}")"
  done
fi

declare -a MERGE_SHADOW_PAIR_IDS=()
if [[ "${#MERGE_SHADOW_PAIR_OVERRIDES[@]}" -gt 0 ]]; then
  for raw_pair in "${MERGE_SHADOW_PAIR_OVERRIDES[@]-}"; do
    if [[ -z "${raw_pair}" ]]; then
      continue
    fi
    append_unique_pair_shadow "$(normalize_pair_token "${raw_pair}")"
  done
else
  for shadow_seed in "${SHADOW_IDS[@]-}"; do
    if [[ -z "${shadow_seed}" ]]; then
      continue
    fi
    append_unique_pair_shadow "$(pair_token_for_seed "${shadow_seed}")"
  done
fi

declare -a MERGE_VICTIM_MODEL_IDS=()
for victim_pair in "${MERGE_VICTIM_PAIR_IDS[@]-}"; do
  MERGE_VICTIM_MODEL_IDS+=("$(pair_model_id_for_pair_token "${victim_pair}")")
done

declare -a MERGE_SHADOW_MODEL_IDS=()
for shadow_pair in "${MERGE_SHADOW_PAIR_IDS[@]-}"; do
  MERGE_SHADOW_MODEL_IDS+=("$(pair_model_id_for_pair_token "${shadow_pair}")")
done

if ! [[ "${MIN_SHADOWS}" =~ ^[0-9]+$ ]] || (( MIN_SHADOWS <= 0 )); then
  echo "MIN_SHADOWS must be a positive integer, got '${MIN_SHADOWS}'" >&2
  exit 1
fi

if [[ "${#DENSE_VICTIM_IDS[@]}" -eq 0 && "${#UNLEARN_VICTIM_IDS[@]}" -eq 0 && "${#RETRAIN_VICTIM_IDS[@]}" -eq 0 && "${#MERGE_VICTIM_PAIR_IDS[@]}" -eq 0 ]]; then
  echo "No victims configured (DENSE/UNLEARN/RETRAIN/MERGE all empty)" >&2
  exit 1
fi
if [[ "${#MERGE_METHOD_IDS[@]}" -eq 0 ]]; then
  echo "MERGE_METHODS is empty" >&2
  exit 1
fi
if [[ "${#MERGE_VICTIM_PAIR_IDS[@]}" -gt 0 && "${#MERGE_SHADOW_PAIR_IDS[@]}" -lt "${MIN_SHADOWS}" ]]; then
  echo "Need at least ${MIN_SHADOWS} MERGE_SHADOW_PAIRS, got ${#MERGE_SHADOW_PAIR_IDS[@]}" >&2
  exit 1
fi
if [[ $(( ${#DENSE_VICTIM_IDS[@]} + ${#UNLEARN_VICTIM_IDS[@]} + ${#RETRAIN_VICTIM_IDS[@]} )) -gt 0 && "${#SHADOW_IDS[@]}" -lt "${MIN_SHADOWS}" ]]; then
  echo "Need at least ${MIN_SHADOWS} SHADOW_MODEL_IDS, got ${#SHADOW_IDS[@]}" >&2
  exit 1
fi

echo "================================================================"
echo "MIA bank run"
echo "dataset: ${DATASET}"
echo "split_seed: ${SPLIT_SEED}"
echo "gpu: ${GPU}"
echo "dense_victims: ${DENSE_VICTIMS}"
echo "unlearn_victims: ${UNLEARN_VICTIMS}"
echo "retrain_victims: ${RETRAIN_VICTIMS}"
echo "merge_victim_pairs: ${MERGE_VICTIM_PAIR_IDS[*]}"
echo "merge_shadow_pairs: ${MERGE_SHADOW_PAIR_IDS[*]}"
echo "merge_methods: ${MERGE_METHODS}"
echo "min_shadows: ${MIN_SHADOWS}"
echo "plan_dir: ${PLAN_DIR}"
echo "result_root: ${RESULT_ROOT}"
echo "================================================================"

declare -a SPLIT_VICTIMS=()
for victim in "${DENSE_VICTIM_IDS[@]-}" "${UNLEARN_VICTIM_IDS[@]-}" "${RETRAIN_VICTIM_IDS[@]-}"; do
  if [[ -n "${victim}" ]]; then
    append_unique_int "${victim}"
  fi
done

for victim in "${SPLIT_VICTIMS[@]-}"; do
  echo "[split] victim=${victim}, shadows=${SHADOW_MODEL_IDS}"
  run_cmd python mia_eval/create_data/create_fixed_data_splits.py \
    --dataset "${DATASET}" \
    --seed "${SPLIT_SEED}" \
    --victim_seed "${victim}" \
    --shadow_seeds "${SHADOW_IDS[@]-}" \
    --verify
done

declare -a MERGE_SPLIT_VICTIMS=()
for victim_model_id in "${MERGE_VICTIM_MODEL_IDS[@]-}"; do
  append_unique_merge_int "${victim_model_id}"
done

for victim_model_id in "${MERGE_SPLIT_VICTIMS[@]-}"; do
  echo "[split:merge] victim_model_id=${victim_model_id}, shadows=${MERGE_SHADOW_MODEL_IDS[*]}"
  run_cmd python mia_eval/create_data/create_fixed_data_splits.py \
    --dataset "${DATASET}" \
    --seed "${SPLIT_SEED}" \
    --victim_seed "${victim_model_id}" \
    --shadow_seeds "${MERGE_SHADOW_MODEL_IDS[@]-}" \
    --verify
done

declare -a PLANS=()
for victim in "${DENSE_VICTIM_IDS[@]-}"; do
  if [[ -z "${victim}" ]]; then
    continue
  fi
  PLANS+=("$(write_plan_dense "${victim}")")
done
for victim in "${UNLEARN_VICTIM_IDS[@]-}"; do
  if [[ -z "${victim}" ]]; then
    continue
  fi
  PLANS+=("$(write_plan_raw_unlearn "${victim}")")
done
for victim in "${RETRAIN_VICTIM_IDS[@]-}"; do
  if [[ -z "${victim}" ]]; then
    continue
  fi
  PLANS+=("$(write_plan_scratch_retrain "${victim}")")
done
for victim_pair in "${MERGE_VICTIM_PAIR_IDS[@]-}"; do
  if [[ -z "${victim_pair}" ]]; then
    continue
  fi
  for merge_method in "${MERGE_METHOD_IDS[@]-}"; do
    if [[ -z "${merge_method}" ]]; then
      continue
    fi
    PLANS+=("$(write_plan_merge_method "${victim_pair}" "${merge_method}")")
  done
done

echo "[plans] created:"
for plan in "${PLANS[@]-}"; do
  echo "  - ${plan}"
done

for plan in "${PLANS[@]-}"; do
  cmd=(
    python scripts/experiment/run_mia_merge_bank.py
    --repo-root "${REPO_ROOT}"
    --plan-json "${plan}"
    --min-shadows "${MIN_SHADOWS}"
  )
  if [[ "${SKIP_MISSING_CKPT}" == "1" ]]; then
    cmd+=(--skip-missing-ckpt)
  fi
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
