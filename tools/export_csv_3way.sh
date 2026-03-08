#!/usr/bin/env bash
set -euo pipefail

# Export CSVs in three buckets:
# 1) mia_only
# 2) candidates_only
# 3) full_dump
#
# Usage:
#   bash tools/export_csv_3way.sh
#   MIA_ROOT=/home/.../runs/mia_merge_bank \
#   UNLEARN_ROOT=/home/.../runs/unlearning_df1 \
#   CONNECTIVITY_ROOT=/home/.../runs/unlearning_connectivity_phase15_final \
#   OUT_ROOT=/home/.../runs/csv_exports \
#   bash tools/export_csv_3way.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MIA_ROOT="${MIA_ROOT:-runs/mia_merge_bank}"
UNLEARN_ROOT="${UNLEARN_ROOT:-runs/unlearning_df1}"
CONNECTIVITY_ROOT="${CONNECTIVITY_ROOT:-runs/unlearning_connectivity_phase15_final}"
OUT_ROOT="${OUT_ROOT:-runs/csv_exports}"

echo "==============================================================="
echo "CSV 3-way export"
echo "repo: ${REPO_ROOT}"
echo "mia_root: ${MIA_ROOT}"
echo "unlearn_root: ${UNLEARN_ROOT}"
echo "connectivity_root: ${CONNECTIVITY_ROOT}"
echo "out_root: ${OUT_ROOT}"
echo "==============================================================="

echo
echo "[1/3] mia_only"
python tools/export_mia_and_performance_csv.py \
  --mia-results "${MIA_ROOT}" \
  --perf-summaries \
  --out-dir "${OUT_ROOT}/mia_only" \
  --include-mia-performance

echo
echo "[2/3] candidates_only"
python tools/export_mia_and_performance_csv.py \
  --perf-summaries "${UNLEARN_ROOT}" "${CONNECTIVITY_ROOT}" \
  --mia-results \
  --out-dir "${OUT_ROOT}/candidates_only"

echo
echo "[3/3] full_dump"
python tools/export_mia_and_performance_csv.py \
  --perf-summaries "${UNLEARN_ROOT}" "${CONNECTIVITY_ROOT}" \
  --mia-results "${MIA_ROOT}" \
  --out-dir "${OUT_ROOT}/full_dump" \
  --include-mia-performance

echo
echo "Done."
echo "  mia_only      : ${OUT_ROOT}/mia_only"
echo "  candidates_only: ${OUT_ROOT}/candidates_only"
echo "  full_dump     : ${OUT_ROOT}/full_dump"

