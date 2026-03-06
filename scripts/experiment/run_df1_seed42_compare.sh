#!/usr/bin/env bash
set -euo pipefail

# Seed-42 기준 DF1 비교 실행 스크립트
# 저장 구조:
#   runs/unlearning_df1/unlearn/...  (언러닝 결과)
#   runs/unlearning_df1/retrain/...  (Dr-only retrain 결과)
#   runs/unlearning_df1/compare/...  (비교 리포트 + dense MIA 결과)
#
# 수행 내용:
# 1) MIA split 준비
# 2) unlearn(42/43) 체크포인트 + unlearn MIA
# 3) retrain(Dr-only, 42/43) 체크포인트 + retrain MIA
# 4) dense baseline MIA(옵션) + 비교 JSON 생성
#
# Usage:
#   bash scripts/experiment/run_df1_seed42_compare.sh
#   RUN_DENSE_MIA=1 FORCE_SPLIT=1 GPU=0 bash scripts/experiment/run_df1_seed42_compare.sh

DENSE_CKPT="${DENSE_CKPT:-runs/dense/cifar10/seed42/best_model.pth}"
DENSE_VICTIM_CONFIG="${DENSE_VICTIM_CONFIG:-runs/dense/cifar10/seed42/config.json}"
DENSE_SHADOW_CKPT="${DENSE_SHADOW_CKPT:-runs/dense/cifar10/seed43/best_model.pth}"
DENSE_SHADOW_CONFIG="${DENSE_SHADOW_CONFIG:-runs/dense/cifar10/seed43/config.json}"

OUT_ROOT="${OUT_ROOT:-runs/unlearning_df1}"
OUT_UNLEARN="${OUT_UNLEARN:-${OUT_ROOT}/unlearn}"
OUT_RETRAIN="${OUT_RETRAIN:-${OUT_ROOT}/retrain}"
OUT_COMPARE="${OUT_COMPARE:-${OUT_ROOT}/compare}"

DATASET="${DATASET:-cifar10}"
ARCH="${ARCH:-resnet}"
LAYERS="${LAYERS:-20}"
SEED_A="${SEED_A:-42}"  # victim
SEED_B="${SEED_B:-43}"  # shadow
SPLIT_SEED="${SPLIT_SEED:-7}"
GPU="${GPU:-0}"

DATAPATH="${DATAPATH:-~/Datasets/CIFAR}"
DATAPATH="${DATAPATH/#\~/$HOME}"
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

RUN_DENSE_MIA="${RUN_DENSE_MIA:-1}"           # 1: run dense MIA
FORCE_SPLIT="${FORCE_SPLIT:-0}"               # 1: recreate split file
SKIP_TRAIN_IF_EXISTS="${SKIP_TRAIN_IF_EXISTS:-1}"  # 1: skip checkpoint training when targets exist
SKIP_MIA_IF_EXISTS="${SKIP_MIA_IF_EXISTS:-1}"      # 1: skip MIA rerun when result json exists

if [[ ! -f "${DENSE_CKPT}" ]]; then
  echo "Dense ckpt not found: ${DENSE_CKPT}"
  exit 1
fi
if [[ "${RUN_DENSE_MIA}" == "1" ]]; then
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
fi

RUN_DIR_UNLEARN="${OUT_UNLEARN}/${DATASET}/df1/seed${SEED_A}_seed${SEED_B}"
RUN_DIR_RETRAIN="${OUT_RETRAIN}/${DATASET}/df1/seed${SEED_A}_seed${SEED_B}"
COMPARE_DIR="${OUT_COMPARE}/${DATASET}/df1/seed${SEED_A}_seed${SEED_B}"

MIA_DIR_UNLEARN="${RUN_DIR_UNLEARN}/mia_results"
MIA_DIR_RETRAIN="${RUN_DIR_RETRAIN}/mia_results"

SPLIT_FILE="mia_data_splits/${DATASET}_seed${SPLIT_SEED}_victim${SEED_A}.pkl"

UNLEARN_A_U="${RUN_DIR_UNLEARN}/unlearn_seed${SEED_A}.pth"
UNLEARN_B_U="${RUN_DIR_UNLEARN}/unlearn_seed${SEED_B}.pth"
UNLEARN_A_R="${RUN_DIR_RETRAIN}/unlearn_seed${SEED_A}.pth"
UNLEARN_B_R="${RUN_DIR_RETRAIN}/unlearn_seed${SEED_B}.pth"
SCRATCH_A_R="${RUN_DIR_RETRAIN}/scratch_retrain_seed${SEED_A}.pth"
SCRATCH_B_R="${RUN_DIR_RETRAIN}/scratch_retrain_seed${SEED_B}.pth"

MIA_UNLEARN_JSON="${MIA_DIR_UNLEARN}/unlearn.json"
MIA_RETRAIN_JSON="${MIA_DIR_RETRAIN}/baseline_retrain.json"
MIA_DENSE_JSON="${COMPARE_DIR}/baseline_dense.json"

COMMON_TRAIN_ARGS=(
  --dense-ckpt "${DENSE_CKPT}"
  --dataset "${DATASET}"
  --arch "${ARCH}"
  --layers "${LAYERS}"
  --seed-a "${SEED_A}"
  --seed-b "${SEED_B}"
  --split-seed "${SPLIT_SEED}"
  --df-mode profile
  --df-profile df1
  --unlearn-epochs "${UNLEARN_EPOCHS}"
  --unlearn-steps "${UNLEARN_STEPS}"
  --unlearn-lr "${UNLEARN_LR}"
  --forget-alpha "${FORGET_ALPHA}"
  --forget-objective "${FORGET_OBJECTIVE}"
  --retrain-epochs "${RETRAIN_EPOCHS}"
  --retrain-lr "${RETRAIN_LR}"
  --ckpt-select retain_acc
  --batch-size "${BATCH_SIZE}"
  --workers "${WORKERS}"
  --datapath "${DATAPATH}"
  --gpu "${GPU}"
  --no-swa-merge
  --step1-only
)

run_train() {
  local out_dir="$1"
  shift
  local cmd=(python train.py "${COMMON_TRAIN_ARGS[@]}" --out-dir "${out_dir}")
  if [[ "${SKIP_TRAIN_IF_EXISTS}" == "1" ]]; then
    cmd+=(--skip-existing)
  fi
  if [[ "$#" -gt 0 ]]; then
    cmd+=("$@")
  fi
  "${cmd[@]}"
}

echo "== [1/9] Create fixed MIA split (victim=${SEED_A}, shadow=${SEED_B}) =="
if [[ "${FORCE_SPLIT}" == "1" || ! -f "${SPLIT_FILE}" ]]; then
  python mia_eval/create_data/create_fixed_data_splits.py \
    --dataset "${DATASET}" \
    --seed "${SPLIT_SEED}" \
    --victim_seed "${SEED_A}" \
    --shadow_seeds "${SEED_B}"
else
  echo "Split exists, skip: ${SPLIT_FILE}"
fi

echo "== [2/9] Train/prepare unlearn checkpoints in ${OUT_UNLEARN} =="
if [[ "${SKIP_TRAIN_IF_EXISTS}" == "1" && -f "${UNLEARN_A_U}" && -f "${UNLEARN_B_U}" ]]; then
  echo "Unlearn checkpoints exist, skip training:"
  echo "  ${UNLEARN_A_U}"
  echo "  ${UNLEARN_B_U}"
else
  run_train "${OUT_UNLEARN}"
fi
if [[ ! -f "${UNLEARN_A_U}" || ! -f "${UNLEARN_B_U}" ]]; then
  echo "Missing unlearn checkpoints in ${RUN_DIR_UNLEARN}"
  exit 1
fi

echo "== [3/9] Run MIA(unlearn) in ${OUT_UNLEARN} =="
if [[ "${SKIP_MIA_IF_EXISTS}" == "1" && -f "${MIA_UNLEARN_JSON}" ]]; then
  echo "MIA(unlearn) exists, skip: ${MIA_UNLEARN_JSON}"
else
  run_train "${OUT_UNLEARN}" \
    --run-mia \
    --mia-stages unlearn \
    --mia-victim-seed "${SEED_A}" \
    --mia-shadow-seeds "${SEED_B}" \
    --mia-split-seed "${SPLIT_SEED}" \
    --mia-attacks "${MIA_ATTACKS}" \
    --mia-forward-mode "${MIA_FORWARD_MODE}" \
    --mia-tpr-fprs "${MIA_TPR_FPRS}" \
    --mia-save-scores
fi
if [[ ! -f "${MIA_UNLEARN_JSON}" ]]; then
  echo "Missing MIA result: ${MIA_UNLEARN_JSON}"
  exit 1
fi

echo "== [4/9] Prepare retrain workspace (copy unlearn endpoints) =="
mkdir -p "${RUN_DIR_RETRAIN}"
if [[ ! -f "${UNLEARN_A_R}" && -f "${UNLEARN_A_U}" ]]; then
  cp "${UNLEARN_A_U}" "${UNLEARN_A_R}"
fi
if [[ ! -f "${UNLEARN_B_R}" && -f "${UNLEARN_B_U}" ]]; then
  cp "${UNLEARN_B_U}" "${UNLEARN_B_R}"
fi

echo "== [5/9] Train/prepare scratch retrain victim(seed=${SEED_A}) in ${OUT_RETRAIN} =="
if [[ "${SKIP_TRAIN_IF_EXISTS}" == "1" && -f "${SCRATCH_A_R}" ]]; then
  echo "Scratch victim exists, skip: ${SCRATCH_A_R}"
else
  run_train "${OUT_RETRAIN}" \
    --train-scratch-retrain-baseline \
    --scratch-retrain-epochs "${SCRATCH_EPOCHS}" \
    --scratch-retrain-lr "${SCRATCH_LR}" \
    --scratch-retrain-momentum "${SCRATCH_MOMENTUM}" \
    --scratch-retrain-weight-decay "${SCRATCH_WEIGHT_DECAY}" \
    --scratch-retrain-seed "${SEED_A}"
fi
if [[ ! -f "${SCRATCH_A_R}" ]]; then
  echo "Missing scratch retrain victim checkpoint: ${SCRATCH_A_R}"
  exit 1
fi
if [[ -f "${RUN_DIR_RETRAIN}/summary.json" ]]; then
  cp "${RUN_DIR_RETRAIN}/summary.json" "${RUN_DIR_RETRAIN}/summary_seed${SEED_A}.json"
fi

echo "== [6/9] Train/prepare scratch retrain shadow(seed=${SEED_B}) in ${OUT_RETRAIN} =="
if [[ "${SKIP_TRAIN_IF_EXISTS}" == "1" && -f "${SCRATCH_B_R}" ]]; then
  echo "Scratch shadow exists, skip: ${SCRATCH_B_R}"
else
  run_train "${OUT_RETRAIN}" \
    --train-scratch-retrain-baseline \
    --scratch-retrain-epochs "${SCRATCH_EPOCHS}" \
    --scratch-retrain-lr "${SCRATCH_LR}" \
    --scratch-retrain-momentum "${SCRATCH_MOMENTUM}" \
    --scratch-retrain-weight-decay "${SCRATCH_WEIGHT_DECAY}" \
    --scratch-retrain-seed "${SEED_B}"
fi
if [[ ! -f "${SCRATCH_B_R}" ]]; then
  echo "Missing scratch retrain shadow checkpoint: ${SCRATCH_B_R}"
  exit 1
fi

echo "== [7/9] Run MIA(retrain baseline) in ${OUT_RETRAIN} =="
if [[ "${SKIP_MIA_IF_EXISTS}" == "1" && -f "${MIA_RETRAIN_JSON}" ]]; then
  echo "MIA(retrain) exists, skip: ${MIA_RETRAIN_JSON}"
else
  run_train "${OUT_RETRAIN}" \
    --run-mia \
    --mia-stages baseline \
    --mia-victim-seed "${SEED_A}" \
    --mia-shadow-seeds "${SEED_B}" \
    --mia-split-seed "${SPLIT_SEED}" \
    --mia-attacks "${MIA_ATTACKS}" \
    --mia-forward-mode "${MIA_FORWARD_MODE}" \
    --mia-tpr-fprs "${MIA_TPR_FPRS}" \
    --mia-save-scores \
    --scratch-retrain-ckpt "${SCRATCH_A_R}" \
    --mia-baseline-ckpt "${SCRATCH_A_R}" \
    --mia-baseline-shadow-ckpts "${SCRATCH_B_R}"
  mkdir -p "${MIA_DIR_RETRAIN}"
  if [[ -f "${MIA_DIR_RETRAIN}/baseline.json" ]]; then
    cp "${MIA_DIR_RETRAIN}/baseline.json" "${MIA_RETRAIN_JSON}"
  fi
fi
if [[ ! -f "${MIA_RETRAIN_JSON}" ]]; then
  echo "Missing MIA result: ${MIA_RETRAIN_JSON}"
  exit 1
fi

echo "== [8/9] Run MIA(dense baseline) =="
mkdir -p "${COMPARE_DIR}"
if [[ "${RUN_DENSE_MIA}" == "1" ]]; then
  if [[ "${SKIP_MIA_IF_EXISTS}" == "1" && -f "${MIA_DENSE_JSON}" ]]; then
    echo "MIA(dense) exists, skip: ${MIA_DENSE_JSON}"
  else
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
      --result_file "${MIA_DENSE_JSON}"
  fi
else
  echo "RUN_DENSE_MIA=0, skip dense baseline MIA."
fi

echo "== [9/9] Build comparison report JSON =="
python - "${RUN_DIR_UNLEARN}" "${RUN_DIR_RETRAIN}" "${COMPARE_DIR}" "${SEED_A}" <<'PY'
import json
import sys
from pathlib import Path

run_unlearn = Path(sys.argv[1])
run_retrain = Path(sys.argv[2])
compare_dir = Path(sys.argv[3])
victim_seed = str(sys.argv[4])
seed_key = f"seed{victim_seed}"

summary_unlearn_path = run_unlearn / "summary.json"
summary_retrain_path = run_retrain / f"summary_seed{victim_seed}.json"
if not summary_retrain_path.exists():
    summary_retrain_path = run_retrain / "summary.json"

unlearn_path = run_unlearn / "mia_results" / "unlearn.json"
retrain_path = run_retrain / "mia_results" / "baseline_retrain.json"
dense_path = compare_dir / "baseline_dense.json"
out_path = compare_dir / f"comparison_seed{victim_seed}.json"

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

summary_unlearn = load_json(summary_unlearn_path)
summary_retrain = load_json(summary_retrain_path)
unlearn = load_json(unlearn_path)
retrain = load_json(retrain_path)
dense = load_json(dense_path)

endpoint_metrics = (
    summary_unlearn.get("endpoints", {}).get("metrics", {}).get(seed_key, {})
    if isinstance(summary_unlearn, dict)
    else {}
)
scratch_metrics = (
    summary_retrain.get("scratch_retrain_baseline", {}).get("metrics", {})
    if isinstance(summary_retrain, dict)
    else {}
)

unlearn_mia = mia_metrics(unlearn)
retrain_mia = mia_metrics(retrain)
dense_mia = mia_metrics(dense)

report = {
    "paths": {
        "run_unlearn": str(run_unlearn),
        "run_retrain": str(run_retrain),
        "summary_unlearn_json": str(summary_unlearn_path),
        "summary_retrain_json": str(summary_retrain_path),
        "mia_unlearn_json": str(unlearn_path),
        "mia_retrain_json": str(retrain_path),
        "mia_dense_json": str(dense_path),
    },
    "utility": {
        f"unlearn_seed{victim_seed}": {
            "test_acc": as_float(endpoint_metrics.get("test_acc")),
            "retain_test_acc": as_float(endpoint_metrics.get("retain_test_acc")),
            "forget_test_acc": as_float(endpoint_metrics.get("forget_test_acc")),
        },
        f"retrain_dr_seed{victim_seed}": {
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
        f"unlearn_{victim_seed}v43": unlearn_mia,
        f"retrain_dr_{victim_seed}v43": retrain_mia,
        f"dense_baseline_{victim_seed}v43": dense_mia,
        "delta_unlearn_minus_retrain": diff(unlearn_mia, retrain_mia),
        "delta_unlearn_minus_dense": diff(unlearn_mia, dense_mia),
    },
}

compare_dir.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"[report] saved: {out_path}")
print("[report] key metrics:")
print(f"  unlearn threshold_auroc={unlearn_mia.get('threshold_auroc')}")
print(f"  retrain threshold_auroc={retrain_mia.get('threshold_auroc')}")
print(f"  dense threshold_auroc={dense_mia.get('threshold_auroc')}")
PY

echo "Done."
echo "Unlearn dir      : ${RUN_DIR_UNLEARN}"
echo "Retrain dir      : ${RUN_DIR_RETRAIN}"
echo "Compare dir      : ${COMPARE_DIR}"
echo "Unlearn MIA      : ${MIA_UNLEARN_JSON}"
echo "Retrain MIA      : ${MIA_RETRAIN_JSON}"
if [[ "${RUN_DENSE_MIA}" == "1" ]]; then
  echo "Dense MIA        : ${MIA_DENSE_JSON}"
else
  echo "Dense MIA        : skipped (RUN_DENSE_MIA=0)"
fi
echo "Comparison JSON  : ${COMPARE_DIR}/comparison_seed${SEED_A}.json"
