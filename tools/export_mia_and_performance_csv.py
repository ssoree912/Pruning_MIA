#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _as_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _to_ratio_acc(x: Any) -> Optional[float]:
    v = _as_float(x)
    if v is None:
        return None
    # Some MIA result files store test accuracy as percentage (e.g., 90.9).
    if v > 1.5:
        return v / 100.0
    return v


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


def _is_aux_summary(path: Path) -> bool:
    # summary_seed42.json 같은 보조 파일은 같은 폴더의 summary.json과 중복일 수 있음.
    name = path.name
    if not name.startswith("summary_") or not name.endswith(".json"):
        return False
    canonical = path.parent / "summary.json"
    return canonical.exists()


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


def _load_plan_expanded(result_path: Path) -> Dict[str, Any]:
    plan_path = result_path.parent / "plan.expanded.json"
    if not plan_path.exists():
        return {}
    try:
        return _load_json(plan_path)
    except Exception:
        return {}


def extract_mia_attack_rows(result_path: Path, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = payload.get("results", {}) if isinstance(payload, dict) else {}
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    plan_expanded = _load_plan_expanded(result_path)
    victim_meta = plan_expanded.get("victim", {}) if isinstance(plan_expanded, dict) else {}
    shadows_meta = plan_expanded.get("shadows", []) if isinstance(plan_expanded, dict) else []

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
        "victim_name": victim_meta.get("name"),
        "pipeline": victim_meta.get("pipeline"),
        "source_seeds": (
            ",".join(str(s) for s in victim_meta.get("source_seeds", []))
            if isinstance(victim_meta.get("source_seeds"), list)
            else None
        ),
        "shadow_count": len(shadows_meta) if isinstance(shadows_meta, list) else None,
        "victim_test_acc": _to_ratio_acc(payload.get("victim_test_acc")),
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


def _classify_performance_row(method: str) -> Tuple[str, str]:
    m = (method or "").lower()
    if m == "scratch_retrain_baseline" or "scratch_retrain" in m:
        return "comparison", "retrain"
    if m.startswith("unlearn_seed") or m == "unlearn_victim" or "raw_unlearn" in m:
        return "comparison", "unlearn"
    if m == "dense" or m.startswith("dense_") or m == "pipeline_dense":
        return "comparison", "dense"
    return "candidate", ""


def _perf_row(
    *,
    summary_path: Path,
    source_type: str,
    method: str,
    run_dir: Any,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    row_role, comparison_group = _classify_performance_row(method)
    return {
        "summary_file": str(summary_path),
        "source_type": source_type,
        "run_name": summary_path.parent.name,
        "run_dir": run_dir,
        "row_role": row_role,
        "comparison_group": comparison_group,
        "method": method,
        "test_acc": _to_ratio_acc(metrics.get("test_acc")),
        "retain_subset_test_acc": _to_ratio_acc(metrics.get("retain_test_acc")),
        "forget_subset_test_acc": _to_ratio_acc(metrics.get("forget_test_acc")),
    }


def _merge_metrics_from_connectivity_summary(
    pipeline: Optional[str],
    ckpt_path: Optional[Path],
) -> Optional[Dict[str, Any]]:
    if not pipeline or ckpt_path is None:
        return None
    summary_path = ckpt_path.parent / "summary.json"
    if not summary_path.exists():
        return None
    try:
        payload = _load_json(summary_path)
    except Exception:
        return None

    if pipeline == "merge_simplex_soup":
        block = payload.get("simplex_soup", {}).get("metrics")
        return block if isinstance(block, dict) else None
    if pipeline == "merge_bezier_swa":
        block = payload.get("bezier_swa", {}).get("best_by_selector")
        return block if isinstance(block, dict) else None
    if pipeline == "merge_simplex":
        block = payload.get("simplex", {}).get("best_sample")
        return block if isinstance(block, dict) else None
    if pipeline == "merge_bezier":
        block = payload.get("bezier", {}).get("best_by_selector")
        return block if isinstance(block, dict) else None
    return None


def _nonmerge_metrics_from_train_summary(
    pipeline: Optional[str],
    ckpt_path: Optional[Path],
    model_id: Optional[int],
) -> Optional[Dict[str, Any]]:
    if pipeline not in {"scratch_retrain", "raw_unlearn"} or ckpt_path is None:
        return None
    run_dir = ckpt_path.parent
    candidate_summaries: List[Path] = []
    if model_id is not None:
        candidate_summaries.append(run_dir / f"summary_seed{int(model_id)}.json")
    candidate_summaries.append(run_dir / "summary.json")

    for sp in candidate_summaries:
        if not sp.exists():
            continue
        try:
            payload = _load_json(sp)
        except Exception:
            continue

        # scratch retrain baseline metrics
        scratch = payload.get("scratch_retrain_baseline", {})
        if isinstance(scratch, dict):
            m = scratch.get("metrics")
            if isinstance(m, dict) and m:
                return m

        # endpoint metrics (raw unlearn victim)
        endpoints = payload.get("endpoints", {})
        if isinstance(endpoints, dict):
            endpoint_map = endpoints.get("metrics", {})
            if isinstance(endpoint_map, dict):
                key = f"seed{int(model_id)}" if model_id is not None else None
                if key and isinstance(endpoint_map.get(key), dict):
                    return endpoint_map[key]
    return None


def extract_perf_row_from_mia_result(result_path: Path, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # run_mia_merge_bank.py always writes plan.expanded.json next to result.json
    plan_path = result_path.parent / "plan.expanded.json"
    plan = _load_json(plan_path) if plan_path.exists() else {}
    victim = plan.get("victim", {}) if isinstance(plan, dict) else {}

    pipeline = victim.get("pipeline")
    method = str(pipeline) if pipeline else "from_mia_result"
    run_dir = result_path.parent
    model_id = victim.get("model_id") if isinstance(victim, dict) else None

    test_acc = _to_ratio_acc(payload.get("victim_test_acc"))
    retain_test_acc = None
    forget_test_acc = None

    ckpt_path = None
    if isinstance(victim, dict) and victim.get("ckpt_path"):
        ckpt_path = Path(str(victim["ckpt_path"]))

    # For merged victims, try to recover richer metrics from connectivity summary.
    merged_metrics = _merge_metrics_from_connectivity_summary(
        pipeline=str(pipeline) if pipeline is not None else None,
        ckpt_path=ckpt_path,
    )
    if isinstance(merged_metrics, dict):
        test_acc = _as_float(merged_metrics.get("test_acc")) or test_acc
        retain_test_acc = _as_float(merged_metrics.get("retain_test_acc"))
        forget_test_acc = _as_float(merged_metrics.get("forget_test_acc"))
    else:
        # For raw_unlearn / scratch_retrain, pull subset metrics from train summary.
        train_metrics = _nonmerge_metrics_from_train_summary(
            pipeline=str(pipeline) if pipeline is not None else None,
            ckpt_path=ckpt_path,
            model_id=int(model_id) if model_id is not None else None,
        )
        if isinstance(train_metrics, dict):
            test_acc = _to_ratio_acc(train_metrics.get("test_acc")) or test_acc
            retain_test_acc = _to_ratio_acc(train_metrics.get("retain_test_acc"))
            forget_test_acc = _to_ratio_acc(train_metrics.get("forget_test_acc"))

    return {
        "summary_file": str(result_path),
        "source_type": "mia_result_perf",
        "run_name": result_path.parent.name,
        "run_dir": str(run_dir),
        "row_role": _classify_performance_row(method)[0],
        "comparison_group": _classify_performance_row(method)[1],
        "method": method,
        "test_acc": test_acc,
        "retain_subset_test_acc": retain_test_acc,
        "forget_subset_test_acc": forget_test_acc,
    }


def extract_performance_rows(summary_path: Path, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    run_dir = payload.get("run_dir")

    # connectivity/run_connectivity_experiment.py summary
    endpoint_metrics = payload.get("endpoint_metrics")
    if isinstance(endpoint_metrics, dict):
        method_blocks: List[Tuple[str, Any]] = [
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
                    )
                )
        if rows:
            return rows

    # train.py summary (step1/step2/step3)
    endpoints = payload.get("endpoints")
    if isinstance(endpoints, dict):
        norm_path = str(summary_path).replace("\\", "/")
        in_unlearn_dir = "/unlearn/" in norm_path
        in_retrain_dir = "/retrain/" in norm_path

        seed_a = endpoints.get("seed_a")
        key_a = f"seed{seed_a}" if seed_a is not None else None
        endpoint_map = endpoints.get("metrics", {})
        endpoint_a_metrics = endpoint_map.get(key_a, {}) if isinstance(endpoint_map, dict) and key_a else {}
        scratch_metrics = payload.get("scratch_retrain_baseline", {}).get("metrics", {})

        # Clean rule:
        # - unlearn 디렉터리에서는 unlearn row만.
        # - retrain 디렉터리에서는 scratch row만.
        # - 기타 위치에서는 둘 다.
        emit_unlearn = (not in_retrain_dir) or in_unlearn_dir
        emit_retrain = (not in_unlearn_dir) or in_retrain_dir

        if emit_unlearn and isinstance(endpoint_a_metrics, dict) and endpoint_a_metrics:
            rows.append(
                _perf_row(
                    summary_path=summary_path,
                    source_type="unlearning_summary",
                    method=f"unlearn_seed{seed_a}" if seed_a is not None else "unlearn_victim",
                    run_dir=run_dir,
                    metrics=endpoint_a_metrics,
                )
            )
        if emit_retrain and isinstance(scratch_metrics, dict) and scratch_metrics:
            rows.append(
                _perf_row(
                    summary_path=summary_path,
                    source_type="unlearning_summary",
                    method="scratch_retrain_baseline",
                    run_dir=run_dir,
                    metrics=scratch_metrics,
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


def _dedup_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[Tuple[Tuple[str, str], ...]] = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        key = tuple(sorted((str(k), str(v)) for k, v in r.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


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
    parser.add_argument(
        "--include-mia-performance",
        action="store_true",
        help="Also append performance rows inferred from mia result.json (disabled by default).",
    )
    parser.add_argument("--no-dedup", action="store_true", help="Disable row-level de-duplication")
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
        if _is_aux_summary(p):
            continue
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

    # Optional: append performance rows inferred from mia result.json.
    if args.include_mia_performance:
        for p in mia_files:
            try:
                payload = _load_json(p)
                row = extract_perf_row_from_mia_result(p, payload)
                if row is not None:
                    perf_rows.append(row)
            except Exception as e:
                perf_rows.append(
                    {
                        "summary_file": str(p),
                        "source_type": "ERROR",
                        "method": "ERROR",
                        "error": str(e),
                    }
                )

    if not args.no_dedup:
        mia_rows = _dedup_rows(mia_rows)
        perf_rows = _dedup_rows(perf_rows)

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
            "victim_name",
            "pipeline",
            "victim_seed",
            "source_seeds",
            "shadow_seeds",
            "shadow_count",
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
            "row_role",
            "comparison_group",
            "method",
            "test_acc",
            "retain_subset_test_acc",
            "forget_subset_test_acc",
            "error",
        ],
    )

    print(f"[done] mia csv : {mia_csv} (rows={mia_count})")
    print(f"[done] perf csv: {perf_csv} (rows={perf_count})")
    print(f"[info] mia files scanned : {len(mia_files)}")
    print(f"[info] perf files scanned: {len(perf_files)}")


if __name__ == "__main__":
    main()
