#!/usr/bin/env bash
set -euo pipefail

# Wrapper for the pair-based MIA bank runner with defaults centered on 50_51.
#
# Default behavior:
# - dense/raw/retrain victims: 50, 51
# - single-seed shadows: 42, 44, 46, 48
# - merge victim pair: 50_51
# - merge shadow pairs: 42_43, 44_45, 46_47, 48_49

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export BASE_CONFIG_PATH="${BASE_CONFIG_PATH:-runs/dense/cifar10/seed50/config.json}"
export PLAN_DIR="${PLAN_DIR:-plans/mia_seed50_51_bank}"
export VICTIMS="${VICTIMS:-50 51}"
export DENSE_VICTIMS="${DENSE_VICTIMS:-${VICTIMS}}"
export UNLEARN_VICTIMS="${UNLEARN_VICTIMS:-${VICTIMS}}"
export RETRAIN_VICTIMS="${RETRAIN_VICTIMS:-${VICTIMS}}"
export MERGE_VICTIMS="${MERGE_VICTIMS:-${VICTIMS}}"
export SHADOW_MODEL_IDS="${SHADOW_MODEL_IDS:-42 44 46 48}"

exec bash "${SCRIPT_DIR}/run_df1_seed42_43_mia_bank.sh"
