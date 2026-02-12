#!/usr/bin/env bash
set -euo pipefail

# --------- Config ---------
DATASET="${DATASET:-cifar10}"
ARCH="${ARCH:-resnet}"
EPOCHS="${EPOCHS:-200}"
GPU="${GPU:-0}"

# Seeds for merging: base seed + 2 extra seeds (3 total)
START_SEED="${START_SEED:-42}"
NUM_SEEDS="${NUM_SEEDS:-3}"

# Same init across particles (paper-style)
INIT_SEED="${INIT_SEED:-1234}"
DATA_SEED_OFFSET="${DATA_SEED_OFFSET:-10000}"

# Sparsity list for pruning methods
SPARSITIES="${SPARSITIES:-0.5 0.6 0.7 0.8 0.9 0.95}"

# Freeze epoch for dpf_freeze
FREEZE_EPOCH="${FREEZE_EPOCH:-180}"

# Output CSV
OUT_CSV="${OUT_CSV:-results/runs_summary.csv}"

echo "=== Running experiments ==="
echo "Dataset=${DATASET} Arch=${ARCH} Epochs=${EPOCHS} GPU=${GPU}"
echo "Seeds: start=${START_SEED} num=${NUM_SEEDS} (init_seed=${INIT_SEED})"
echo "Sparsities: ${SPARSITIES}"
echo

# 1) Dense (multi-seed)
python train.py \
  --methods dense \
  --dataset "${DATASET}" --arch "${ARCH}" \
  --epochs "${EPOCHS}" --gpu "${GPU}" \
  --multi-seed --num-seeds "${NUM_SEEDS}" --start-seed "${START_SEED}" \
  --init-seed "${INIT_SEED}" --data-seed-offset "${DATA_SEED_OFFSET}"

# 2) Static (multi-seed, all sparsities)
python train.py \
  --methods static \
  --sparsities ${SPARSITIES} \
  --dataset "${DATASET}" --arch "${ARCH}" \
  --epochs "${EPOCHS}" --gpu "${GPU}" \
  --multi-seed --num-seeds "${NUM_SEEDS}" --start-seed "${START_SEED}" \
  --init-seed "${INIT_SEED}" --data-seed-offset "${DATA_SEED_OFFSET}"

# 3) DPF (dynamic, no-freeze)
python train.py \
  --methods dpf \
  --sparsities ${SPARSITIES} \
  --dataset "${DATASET}" --arch "${ARCH}" \
  --epochs "${EPOCHS}" --gpu "${GPU}" \
  --freeze-epoch -1 \
  --multi-seed --num-seeds "${NUM_SEEDS}" --start-seed "${START_SEED}" \
  --init-seed "${INIT_SEED}" --data-seed-offset "${DATA_SEED_OFFSET}"

# 4) DPF + freeze (freeze at epoch 180)
python train.py \
  --methods dpf \
  --sparsities ${SPARSITIES} \
  --dataset "${DATASET}" --arch "${ARCH}" \
  --epochs "${EPOCHS}" --gpu "${GPU}" \
  --freeze-epoch "${FREEZE_EPOCH}" \
  --multi-seed --num-seeds "${NUM_SEEDS}" --start-seed "${START_SEED}" \
  --init-seed "${INIT_SEED}" --data-seed-offset "${DATA_SEED_OFFSET}"

# 5) Summarize results to CSV
python scripts/summarize_runs.py --runs ./runs --out "${OUT_CSV}"

# 6) Compact CSV (requested columns)
OUT_CSV_PATH="${OUT_CSV}" python - <<PY
import csv
from pathlib import Path
import os

src = Path(os.environ["OUT_CSV_PATH"])
dst = src.parent / "runs_summary_compact.csv"
cols = ["seed", "method", "sparsity", "best_acc1", "final_acc1", "final_loss"]

with open(src, newline="") as f:
    r = csv.DictReader(f)
    rows = [{k: row.get(k) for k in cols} for row in r]

dst.parent.mkdir(parents=True, exist_ok=True)
with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

print(f"Compact CSV written: {dst}")
PY

echo "=== Done. Summary CSV: ${OUT_CSV} ==="
