#!/usr/bin/env bash
set -euo pipefail

# Seed-42 기준 DF1 비교 실행 스크립트
# - Unlearning endpoint (victim=42, shadow=43)
# - Scratch retrain baseline on Dr only (victim=42, shadow=43)
# - Dense baseline (victim=42, shadow=43)
# - MIA 결과/유틸리티 비교 JSON 생성
#
# Usage:
#   bash scripts/experiment/run_df1_seed42_compare.sh
#   GPU=0 bash scripts/experiment/run_df1_seed42_compare.sh
#   OUT_DIR=runs/unlearning_df1_seed42_compare/K3 bash scripts/experiment/run_df1_seed42_compare.sh

DENSE_CKPT="${DENSE_CKPT:-runs/dense/cifar10/seed42/best_model.pth}"
DENSE_VICTIM_CONFIG="${DENSE_VICTIM_CONFIG:-runs/dense/cifar10/seed42/config.json}"
DENSE_SHADOW_CKPT="${DENSE_SHADOW_CKPT:-runs/dense/cifar10/seed43/best_model.pth}"
DENSE_SHADOW_CONFIG="${DENSE_SHADOW_CONFIG:-runs/dense/cifar10/seed43/config.json}"

OUT_DIR="${OUT_DIR:-runs/unlearning_df1_grid_utility/K3}"
DATASET="${DATASET:-cifar10}"
ARCH="${ARCH:-resnet}"
LAYERS="${LAYERS:-20}"
SEED_A="${SEED_A:-42}"
SEED_B="${SEED_B:-43}"
SPLIT_SEED="${SPLIT_SEED:-7}"
GPU="${GPU:-0}"

DATAPATH="${DATAPATH:-~/Datasets/CIFAR}"
BATCH_SIZE="${BATCH_SIZE:-128}"
WORKERS="${WORKERS:-4}"

# K3 (utility-first) 계열 기본값
FORGET_OBJECTIVE="${FORGET_OBJECTIVE:-kl_uniform}"
FORGET_ALPHA="${FORGET_ALPHA:-0.05}"
UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-1}"
UNLEARN_STEPS="${UNLEARN_STEPS:-100}"
UNLEARN_LR="${UNLEARN_LR:-0.01}"
RETRAIN_EPOCHS="${RETRAIN_EPOCHS:-50}"
RETRAIN_LR="${RETRAIN_LR:-0.1}"

SCRATCH_EPOCHS="${SCRATCH_EPOCHS:-200}"
SCRATCH_LR="${SCRATCH_LR:-0.1}"
SCRATCH_MOMENTUM="${SCRATCH_MOMENTUM:-0.9}"
SCRATCH_WEIGHT_DECAY="${SCRATCH_WEIGHT_DECAY:-5e-4}"

MIA_ATTACKS="${MIA_ATTACKS:-threshold,lira,nn,samia}"
MIA_FORWARD_MODE="${MIA_FORWARD_MODE:-standard}"
MIA_TPR_FPRS="${MIA_TPR_FPRS:-0.1,1,5}"
RUN_DENSE_MIA="${RUN_DENSE_MIA:-1}"   # 1: run dense baseline MIA, 0: skip
FORCE_SPLIT="${FORCE_SPLIT:-0}"       # 1: recreate split even if exists

if [[ ! -f "${DENSE_CKPT}" ]]; then
  echo "Dense ckpt not found: ${DENSE_CKPT}"
  exit 1
fi
if [[ ! -f "${DENSE_VICTIM_CONFIG}" ]]; then
  echo "Dense victim config not found: ${DENSE_VICTIM_CONFIG}"
  exit 1
fi
if [[ ! -f "${DENSE_SHADOW_CKPT}" ]]; then
  echo "Dense shadow ckpt not found: ${DENSE_SHADOW_CKPT}"
  exit 1
fi
if [[ ! -f "${DENSE_SHADOW_CONFIG}" ]]; then
  echo "Dense shadow config not found: ${DENSE_SHADOW_CONFIG}"
  exit 1
fi

RUN_DIR="${OUT_DIR}/${DATASET}/df1/seed${SEED_A}_seed${SEED_B}"
MIA_DIR="${RUN_DIR}/mia_results"
SPLIT_FILE="mia_data_splits/${DATASET}_seed${SPLIT_SEED}_victim${SEED_A}.pkl"

echo "== [1/7] Create fixed MIA split (victim=${SEED_A}, shadow=${SEED_B}) =="
if [[ "${FORCE_SPLIT}" == "1" || ! -f "${SPLIT_FILE}" ]]; then
  python mia_eval/create_data/create_fixed_data_splits.py \
    --dataset "${DATASET}" \
    --seed "${SPLIT_SEED}" \
    --victim_seed "${SEED_A}" \
    --shadow_seeds "${SEED_B}"
else
  echo "Split exists, skip: ${SPLIT_FILE}"
fi

echo "== [2/7] Train unlearning endpoints + scratch retrain victim(seed=${SEED_A}) =="
python train.py \
  --dense-ckpt "${DENSE_CKPT}" \
  --out-dir "${OUT_DIR}" \
  --dataset "${DATASET}" \
  --arch "${ARCH}" \
  --layers "${LAYERS}" \
  --seed-a "${SEED_A}" \
  --seed-b "${SEED_B}" \
  --split-seed "${SPLIT_SEED}" \
  --df-mode profile \
  --df-profile df1 \
  --unlearn-epochs "${UNLEARN_EPOCHS}" \
  --unlearn-steps "${UNLEARN_STEPS}" \
  --unlearn-lr "${UNLEARN_LR}" \
  --forget-alpha "${FORGET_ALPHA}" \
  --forget-objective "${FORGET_OBJECTIVE}" \
  --retrain-epochs "${RETRAIN_EPOCHS}" \
  --retrain-lr "${RETRAIN_LR}" \
  --ckpt-select retain_acc \
  --batch-size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --datapath "${DATAPATH}" \
  --gpu "${GPU}" \
  --no-swa-merge \
  --step1-only \
  --train-scratch-retrain-baseline \
  --scratch-retrain-epochs "${SCRATCH_EPOCHS}" \
  --scratch-retrain-lr "${SCRATCH_LR}" \
  --scratch-retrain-momentum "${SCRATCH_MOMENTUM}" \
  --scratch-retrain-weight-decay "${SCRATCH_WEIGHT_DECAY}" \
  --scratch-retrain-seed "${SEED_A}"

echo "== [3/7] Train scratch retrain shadow(seed=${SEED_B}) reusing endpoints =="
python train.py \
  --dense-ckpt "${DENSE_CKPT}" \
  --out-dir "${OUT_DIR}" \
  --dataset "${DATASET}" \
  --arch "${ARCH}" \
  --layers "${LAYERS}" \
  --seed-a "${SEED_A}" \
  --seed-b "${SEED_B}" \
  --split-seed "${SPLIT_SEED}" \
  --df-mode profile \
  --df-profile df1 \
  --unlearn-epochs "${UNLEARN_EPOCHS}" \
  --unlearn-steps "${UNLEARN_STEPS}" \
  --unlearn-lr "${UNLEARN_LR}" \
  --forget-alpha "${FORGET_ALPHA}" \
  --forget-objective "${FORGET_OBJECTIVE}" \
  --retrain-epochs "${RETRAIN_EPOCHS}" \
  --retrain-lr "${RETRAIN_LR}" \
  --ckpt-select retain_acc \
  --batch-size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --datapath "${DATAPATH}" \
  --gpu "${GPU}" \
  --no-swa-merge \
  --step1-only \
  --skip-existing \
  --train-scratch-retrain-baseline \
  --scratch-retrain-epochs "${SCRATCH_EPOCHS}" \
  --scratch-retrain-lr "${SCRATCH_LR}" \
  --scratch-retrain-momentum "${SCRATCH_MOMENTUM}" \
  --scratch-retrain-weight-decay "${SCRATCH_WEIGHT_DECAY}" \
  --scratch-retrain-seed "${SEED_B}"

SCRATCH_VICTIM_CKPT="${RUN_DIR}/scratch_retrain_seed${SEED_A}.pth"
SCRATCH_SHADOW_CKPT="${RUN_DIR}/scratch_retrain_seed${SEED_B}.pth"
if [[ ! -f "${SCRATCH_VICTIM_CKPT}" ]]; then
  echo "Missing scratch victim checkpoint: ${SCRATCH_VICTIM_CKPT}"
  exit 1
fi
if [[ ! -f "${SCRATCH_SHADOW_CKPT}" ]]; then
  echo "Missing scratch shadow checkpoint: ${SCRATCH_SHADOW_CKPT}"
  exit 1
fi

echo "== [4/7] Run MIA for unlearn + retrain(Dr-only) baseline =="
python train.py \
  --dense-ckpt "${DENSE_CKPT}" \
  --out-dir "${OUT_DIR}" \
  --dataset "${DATASET}" \
  --arch "${ARCH}" \
  --layers "${LAYERS}" \
  --seed-a "${SEED_A}" \
  --seed-b "${SEED_B}" \
  --split-seed "${SPLIT_SEED}" \
  --df-mode profile \
  --df-profile df1 \
  --unlearn-epochs "${UNLEARN_EPOCHS}" \
  --unlearn-steps "${UNLEARN_STEPS}" \
  --unlearn-lr "${UNLEARN_LR}" \
  --forget-alpha "${FORGET_ALPHA}" \
  --forget-objective "${FORGET_OBJECTIVE}" \
  --retrain-epochs "${RETRAIN_EPOCHS}" \
  --retrain-lr "${RETRAIN_LR}" \
  --ckpt-select retain_acc \
  --batch-size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --datapath "${DATAPATH}" \
  --gpu "${GPU}" \
  --no-swa-merge \
  --step1-only \
  --skip-existing \
  --run-mia \
  --mia-stages unlearn,baseline \
  --mia-victim-seed "${SEED_A}" \
  --mia-shadow-seeds "${SEED_B}" \
  --mia-split-seed "${SPLIT_SEED}" \
  --mia-attacks "${MIA_ATTACKS}" \
  --mia-forward-mode "${MIA_FORWARD_MODE}" \
  --mia-tpr-fprs "${MIA_TPR_FPRS}" \
  --mia-save-scores \
  --scratch-retrain-ckpt "${SCRATCH_VICTIM_CKPT}" \
  --mia-baseline-ckpt "${SCRATCH_VICTIM_CKPT}" \
  --mia-baseline-shadow-ckpts "${SCRATCH_SHADOW_CKPT}"

mkdir -p "${MIA_DIR}"
if [[ -f "${MIA_DIR}/baseline.json" ]]; then
  cp "${MIA_DIR}/baseline.json" "${MIA_DIR}/baseline_retrain.json"
fi
if [[ ! -f "${MIA_DIR}/unlearn.json" ]]; then
  echo "Missing MIA result: ${MIA_DIR}/unlearn.json"
  exit 1
fi
if [[ ! -f "${MIA_DIR}/baseline_retrain.json" ]]; then
  echo "Missing MIA result: ${MIA_DIR}/baseline_retrain.json"
  exit 1
fi

echo "== [5/7] Run MIA for dense baseline (42 vs 43) =="
if [[ "${RUN_DENSE_MIA}" == "1" ]]; then
  python mia_eval/core/mia_modi.py \
    --device "${GPU}" \
    --dataset_name "${DATASET}" \
    --seed "${SPLIT_SEED}" \
    --victim_seed "${SEED_A}" \
    --shadow_seeds "${SEED_B}" \
    --victim_ckpt_path "${DENSE_CKPT}" \
    --shadow_ckpt_paths "${DENSE_SHADOW_CKPT}" \
    --victim_config_path "${DENSE_VICTIM_CONFIG}" \
    --shadow_config_paths "${DENSE_SHADOW_CONFIG}" \
    --attacks "${MIA_ATTACKS}" \
    --forward_mode "${MIA_FORWARD_MODE}" \
    --tpr_fprs "${MIA_TPR_FPRS}" \
    --save_scores \
    --result_file "${MIA_DIR}/baseline_dense.json"
  if [[ ! -f "${MIA_DIR}/baseline_dense.json" ]]; then
    echo "Missing MIA result: ${MIA_DIR}/baseline_dense.json"
    exit 1
  fi
else
  echo "RUN_DENSE_MIA=0, skip dense baseline MIA."
fi

echo "== [6/7] Build comparison report JSON =="
python - "${RUN_DIR}" "${SEED_A}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
victim_seed = str(sys.argv[2])
seed_key = f"seed{victim_seed}"

summary_path = run_dir / "summary.json"
unlearn_path = run_dir / "mia_results" / "unlearn.json"
retrain_path = run_dir / "mia_results" / "baseline_retrain.json"
dense_path = run_dir / "mia_results" / "baseline_dense.json"
out_path = run_dir / f"comparison_seed{victim_seed}.json"

def load_json(path: Path):
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)

def as_float(x):
    try:
        return float(x)
    except Exception:
        return None

def mia_metrics(payload: dict):
    results = payload.get("results", {}) if isinstance(payload, dict) else {}
    conf = results.get("confidence_extended", {}) if isinstance(results.get("confidence_extended", {}), dict) else {}
    lira = results.get("lira", {}) if isinstance(results.get("lira", {}), dict) else {}
    nn = results.get("nn", {}) if isinstance(results.get("nn", {}), dict) else {}
    samia = results.get("samia", {}) if isinstance(results.get("samia", {}), dict) else {}
    return {
        "victim_test_acc": as_float(payload.get("victim_test_acc")),
        "threshold_auroc": as_float(conf.get("auroc")),
        "threshold_advantage": as_float(conf.get("advantage")),
        "threshold_tpr_at_1fpr": as_float(conf.get("tpr_at_1fpr")),
        "lira_auc": as_float(lira.get("auc")),
        "lira_advantage": as_float(lira.get("advantage")),
        "lira_tpr_at_1fpr": as_float(lira.get("tpr_at_1fpr")),
        "nn_auc": as_float(nn.get("auc")),
        "nn_advantage": as_float(nn.get("advantage")),
        "nn_tpr_at_1fpr": as_float(nn.get("tpr_at_1fpr")),
        "samia_auc": as_float(samia.get("auc")),
        "samia_advantage": as_float(samia.get("advantage")),
        "samia_tpr_at_1fpr": as_float(samia.get("tpr_at_1fpr")),
    }

def diff(a: dict, b: dict):
    out = {}
    for k, va in a.items():
        vb = b.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            out[f"{k}_delta"] = va - vb
    return out

summary = load_json(summary_path)
unlearn = load_json(unlearn_path)
retrain = load_json(retrain_path)
dense = load_json(dense_path)

endpoint_metrics = (
    summary.get("endpoints", {}).get("metrics", {}).get(seed_key, {})
    if isinstance(summary, dict)
    else {}
)
scratch_metrics = (
    summary.get("scratch_retrain_baseline", {}).get("metrics", {})
    if isinstance(summary, dict)
    else {}
)

unlearn_mia = mia_metrics(unlearn)
retrain_mia = mia_metrics(retrain)
dense_mia = mia_metrics(dense)

report = {
    "paths": {
        "run_dir": str(run_dir),
        "summary_json": str(summary_path),
        "mia_unlearn_json": str(unlearn_path),
        "mia_retrain_json": str(retrain_path),
        "mia_dense_json": str(dense_path),
    },
    "utility": {
        "unlearn_seed42": {
            "test_acc": as_float(endpoint_metrics.get("test_acc")),
            "retain_test_acc": as_float(endpoint_metrics.get("retain_test_acc")),
            "forget_test_acc": as_float(endpoint_metrics.get("forget_test_acc")),
        },
        "retrain_dr_seed42": {
            "test_acc": as_float(scratch_metrics.get("test_acc")),
            "retain_test_acc": as_float(scratch_metrics.get("retain_test_acc")),
            "forget_test_acc": as_float(scratch_metrics.get("forget_test_acc")),
        },
        "delta_unlearn_minus_retrain": {
            "test_acc_delta": as_float(endpoint_metrics.get("test_acc")) - as_float(scratch_metrics.get("test_acc"))
            if as_float(endpoint_metrics.get("test_acc")) is not None and as_float(scratch_metrics.get("test_acc")) is not None else None,
            "retain_test_acc_delta": as_float(endpoint_metrics.get("retain_test_acc")) - as_float(scratch_metrics.get("retain_test_acc"))
            if as_float(endpoint_metrics.get("retain_test_acc")) is not None and as_float(scratch_metrics.get("retain_test_acc")) is not None else None,
            "forget_test_acc_delta": as_float(endpoint_metrics.get("forget_test_acc")) - as_float(scratch_metrics.get("forget_test_acc"))
            if as_float(endpoint_metrics.get("forget_test_acc")) is not None and as_float(scratch_metrics.get("forget_test_acc")) is not None else None,
        },
    },
    "mia": {
        "unlearn_42v43": unlearn_mia,
        "retrain_dr_42v43": retrain_mia,
        "dense_baseline_42v43": dense_mia,
        "delta_unlearn_minus_retrain": diff(unlearn_mia, retrain_mia),
        "delta_unlearn_minus_dense": diff(unlearn_mia, dense_mia),
    },
}

with open(out_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"[report] saved: {out_path}")
print("[report] key metrics:")
print(f"  unlearn threshold_auroc={unlearn_mia.get('threshold_auroc')}")
print(f"  retrain threshold_auroc={retrain_mia.get('threshold_auroc')}")
print(f"  dense threshold_auroc={dense_mia.get('threshold_auroc')}")
PY

echo "== [7/7] Done =="
echo "Run dir         : ${RUN_DIR}"
echo "Comparison JSON : ${RUN_DIR}/comparison_seed${SEED_A}.json"
echo "Unlearn MIA     : ${RUN_DIR}/mia_results/unlearn.json"
echo "Retrain MIA     : ${RUN_DIR}/mia_results/baseline_retrain.json"
if [[ "${RUN_DENSE_MIA}" == "1" ]]; then
  echo "Dense MIA       : ${RUN_DIR}/mia_results/baseline_dense.json"
else
  echo "Dense MIA       : skipped (RUN_DENSE_MIA=0)"
fi
