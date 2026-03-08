#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _as_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _has_glob_magic(s: str) -> bool:
    return any(ch in s for ch in "*?[]")


def _collect_from_dir(root: Path, kind: str) -> List[Path]:
    if kind == "mia":
        return [p.resolve() for p in root.rglob("result.json") if p.is_file()]
    if kind == "perf":
        # include summary.json + summary_seedXX.json etc.
        return [p.resolve() for p in root.rglob("summary*.json") if p.is_file()]
    raise ValueError(f"Unsupported kind: {kind}")


def _resolve_input_paths(inputs: Sequence[str], kind: str) -> List[Path]:
    out: List[Path] = []
    for raw in inputs:
        expanded = str(Path(raw).expanduser())
        matches: List[Path]

        if _has_glob_magic(expanded):
            matches = [Path(x) for x in glob.glob(expanded, recursive=True)]
        else:
            matches = [Path(expanded)]

        for m in matches:
            if m.is_dir():
                out.extend(_collect_from_dir(m, kind))
            elif m.is_file():
                out.append(m.resolve())

    return sorted(set(out))


def _auto_mia_results() -> List[Path]:
    return sorted(Path("runs").glob("mia_merge_bank/**/result.json"))


def _auto_perf_summaries() -> List[Path]:
    pats = [
        "runs/unlearning_connectivity_phase15*/**/summary.json",
        "runs/unlearning_df1/unlearn/**/summary.json",
        "runs/unlearning_df1/retrain/**/summary*.json",
    ]
    return _resolve_input_paths(pats, kind="perf")


def _attack_row(base: Dict[str, Any], attack_name: str, block: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **base,
        "attack": attack_name,
        "accuracy": _as_float(block.get("accuracy")),
        "auc": _as_float(block.get("auc", block.get("auroc"))),
        "balanced_accuracy": _as_float(block.get("balanced_accuracy")),
        "advantage": _as_float(block.get("advantage")),
        "tpr_at_1fpr": _as_float(block.get("tpr_at_1fpr")),
        "implementation": block.get("implementation"),
        "per_example_reference_models": block.get("per_example_reference_models"),
    }


def extract_mia_attack_rows(result_path: Path, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = payload.get("results", {}) if isinstance(payload, dict) else {}
    config = payload.get("config", {}) if isinstance(payload, dict) else {}

    victim_seed = payload.get("victim_seed", config.get("victim_seed"))
    shadow_seeds = payload.get("shadow_seeds", config.get("shadow_seeds"))
    if isinstance(shadow_seeds, list):
        shadow_seeds_str = ",".join(str(s) for s in shadow_seeds)
    else:
        shadow_seeds_str = str(shadow_seeds) if shadow_seeds is not None else None

    base = {
        "result_file": str(result_path),
        "run_name": result_path.parent.name,
        "dataset": payload.get("dataset_name", config.get("dataset_name")),
        "victim_seed": victim_seed,
        "shadow_seeds": shadow_seeds_str,
        "victim_test_acc": _as_float(payload.get("victim_test_acc")),
    }

    rows: List[Dict[str, Any]] = []

    threshold_attacks = results.get("threshold_attacks", {})
    if isinstance(threshold_attacks, dict):
        for thr_name, thr_block in threshold_attacks.items():
            if isinstance(thr_block, dict):
                rows.append(_attack_row(base, f"threshold_{thr_name}", thr_block))
    elif isinstance(results.get("confidence_extended"), dict):
        rows.append(_attack_row(base, "threshold_confidence", results["confidence_extended"]))

    for atk in ("lira", "nn", "nn_top3", "nn_cls", "samia"):
        block = results.get(atk)
        if isinstance(block, dict):
            rows.append(_attack_row(base, atk, block))

    return rows


def _best_endpoint_test_acc(endpoint_metrics: Dict[str, Any]) -> Optional[float]:
    vals: List[float] = []
    for key in ("endpoint_a", "endpoint_b"):
        block = endpoint_metrics.get(key, {})
        if isinstance(block, dict):
            v = _as_float(block.get("test_acc"))
            if v is not None:
                vals.append(v)
    return max(vals) if vals else None


def _perf_row(
    *,
    summary_path: Path,
    source_type: str,
    method: str,
    run_dir: Any,
    metrics: Dict[str, Any],
    unlearn_test_acc_ref: Optional[float],
    retrain_test_acc_ref: Optional[float],
) -> Dict[str, Any]:
    return {
        "summary_file": str(summary_path),
        "source_type": source_type,
        "run_name": summary_path.parent.name,
        "run_dir": run_dir,
        "method": method,
        "test_acc": _as_float(metrics.get("test_acc")),
        "retain_test_acc": _as_float(metrics.get("retain_test_acc")),
        "forget_test_acc": _as_float(metrics.get("forget_test_acc")),
        "unlearn_test_acc": unlearn_test_acc_ref,
        "retrain_test_acc": retrain_test_acc_ref,
    }


def extract_performance_rows(summary_path: Path, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    run_dir = payload.get("run_dir")

    # connectivity/run_connectivity_experiment.py summary
    endpoint_metrics = payload.get("endpoint_metrics")
    if isinstance(endpoint_metrics, dict):
        unlearn_ref = _best_endpoint_test_acc(endpoint_metrics)
        retrain_ref = _as_float(
            payload.get("scratch_retrain_baseline", {}).get("metrics", {}).get("test_acc")
            if isinstance(payload.get("scratch_retrain_baseline"), dict)
            else None
        )

        method_blocks: List[Tuple[str, Any]] = [
            ("endpoint_a", endpoint_metrics.get("endpoint_a")),
            ("endpoint_b", endpoint_metrics.get("endpoint_b")),
            ("raw_linear_best", payload.get("raw_linear", {}).get("best_by_selector")),
            ("perm_linear_best", payload.get("perm_linear", {}).get("best_by_selector")),
            ("bezier_best", payload.get("bezier", {}).get("best_by_selector")),
            ("bezier_swa_best", payload.get("bezier_swa", {}).get("best_by_selector")),
            ("simplex_best", payload.get("simplex", {}).get("best_sample")),
            ("simplex_soup_best", payload.get("simplex_soup", {}).get("metrics")),
            ("simplex_swa_best", payload.get("simplex_swa", {}).get("best_sample")),
        ]

        for method, block in method_blocks:
            if isinstance(block, dict) and block:
                rows.append(
                    _perf_row(
                        summary_path=summary_path,
                        source_type="connectivity_summary",
                        method=method,
                        run_dir=run_dir,
                        metrics=block,
                        unlearn_test_acc_ref=unlearn_ref,
                        retrain_test_acc_ref=retrain_ref,
                    )
                )
        if rows:
            return rows

    # train.py summary (step1/step2/step3)
    endpoints = payload.get("endpoints")
    if isinstance(endpoints, dict):
        seed_a = endpoints.get("seed_a")
        key_a = f"seed{seed_a}" if seed_a is not None else None
        endpoint_map = endpoints.get("metrics", {})
        endpoint_a_metrics = endpoint_map.get(key_a, {}) if isinstance(endpoint_map, dict) and key_a else {}
        scratch_metrics = payload.get("scratch_retrain_baseline", {}).get("metrics", {})

        unlearn_ref = _as_float(endpoint_a_metrics.get("test_acc")) if isinstance(endpoint_a_metrics, dict) else None
        retrain_ref = _as_float(scratch_metrics.get("test_acc")) if isinstance(scratch_metrics, dict) else None

        if isinstance(endpoint_a_metrics, dict) and endpoint_a_metrics:
            rows.append(
                _perf_row(
                    summary_path=summary_path,
                    source_type="unlearning_summary",
                    method=f"unlearn_seed{seed_a}" if seed_a is not None else "unlearn_victim",
                    run_dir=run_dir,
                    metrics=endpoint_a_metrics,
                    unlearn_test_acc_ref=unlearn_ref,
                    retrain_test_acc_ref=retrain_ref,
                )
            )
        if isinstance(scratch_metrics, dict) and scratch_metrics:
            rows.append(
                _perf_row(
                    summary_path=summary_path,
                    source_type="unlearning_summary",
                    method="scratch_retrain_baseline",
                    run_dir=run_dir,
                    metrics=scratch_metrics,
                    unlearn_test_acc_ref=unlearn_ref,
                    retrain_test_acc_ref=retrain_ref,
                )
            )
        return rows

    return rows


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], preferred_cols: Sequence[str]) -> int:
    rows_list = list(rows)
    if not rows_list:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(preferred_cols)
        return 0

    all_cols = set()
    for r in rows_list:
        all_cols.update(r.keys())

    ordered: List[str] = []
    for c in preferred_cols:
        if c in all_cols and c not in ordered:
            ordered.append(c)
    for c in sorted(all_cols):
        if c not in ordered:
            ordered.append(c)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        for r in rows_list:
            writer.writerow(r)
    return len(rows_list)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export MIA result.json and performance summary.json into two CSV files."
    )
    parser.add_argument(
        "--mia-results",
        nargs="*",
        default=None,
        help=(
            "Glob/file/dir list for MIA result.json. "
            "If a directory is given, result.json is searched recursively. "
            "(default: runs/mia_merge_bank/**/result.json)"
        ),
    )
    parser.add_argument(
        "--perf-summaries",
        nargs="*",
        default=None,
        help=(
            "Glob/file/dir list for performance summary json. "
            "If a directory is given, summary*.json is searched recursively. "
            "(default scans connectivity/unlearning summary locations)"
        ),
    )
    parser.add_argument("--out-dir", type=str, default="runs/csv_exports")
    parser.add_argument("--mia-csv", type=str, default="mia_attacks.csv")
    parser.add_argument("--perf-csv", type=str, default="performance.csv")
    args = parser.parse_args()

    mia_files = _auto_mia_results() if args.mia_results is None else _resolve_input_paths(args.mia_results, kind="mia")
    perf_files = _auto_perf_summaries() if args.perf_summaries is None else _resolve_input_paths(args.perf_summaries, kind="perf")

    mia_rows: List[Dict[str, Any]] = []
    for p in mia_files:
        try:
            payload = _load_json(p)
            mia_rows.extend(extract_mia_attack_rows(p, payload))
        except Exception as e:
            mia_rows.append(
                {
                    "result_file": str(p),
                    "attack": "ERROR",
                    "error": str(e),
                }
            )

    perf_rows: List[Dict[str, Any]] = []
    for p in perf_files:
        try:
            payload = _load_json(p)
            perf_rows.extend(extract_performance_rows(p, payload))
        except Exception as e:
            perf_rows.append(
                {
                    "summary_file": str(p),
                    "source_type": "ERROR",
                    "method": "ERROR",
                    "error": str(e),
                }
            )

    out_dir = Path(args.out_dir).expanduser().resolve()
    mia_csv = out_dir / args.mia_csv
    perf_csv = out_dir / args.perf_csv

    mia_count = _write_csv(
        mia_csv,
        mia_rows,
        preferred_cols=[
            "result_file",
            "run_name",
            "dataset",
            "victim_seed",
            "shadow_seeds",
            "attack",
            "victim_test_acc",
            "accuracy",
            "auc",
            "balanced_accuracy",
            "advantage",
            "tpr_at_1fpr",
            "implementation",
            "per_example_reference_models",
            "error",
        ],
    )
    perf_count = _write_csv(
        perf_csv,
        perf_rows,
        preferred_cols=[
            "summary_file",
            "source_type",
            "run_name",
            "run_dir",
            "method",
            "test_acc",
            "retrain_test_acc",
            "unlearn_test_acc",
            "retain_test_acc",
            "forget_test_acc",
            "error",
        ],
    )

    print(f"[done] mia csv : {mia_csv} (rows={mia_count})")
    print(f"[done] perf csv: {perf_csv} (rows={perf_count})")
    print(f"[info] mia files scanned : {len(mia_files)}")
    print(f"[info] perf files scanned: {len(perf_files)}")


if __name__ == "__main__":
    main()
