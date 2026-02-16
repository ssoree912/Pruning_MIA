#!/usr/bin/env python3
"""
Collect endpoint metrics across runs and compute Pareto front.

Pareto criterion:
  - utility: maximize
  - forget metric: minimize
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _pick_utility(metrics: Dict[str, Any], preferred: str) -> Tuple[Optional[str], Optional[float]]:
    if preferred != "auto":
        val = metrics.get(preferred)
        return (preferred, float(val)) if val is not None else (None, None)
    for key in ("normalized_full_test_acc", "normalized_full", "retain_test_acc", "test_acc"):
        val = metrics.get(key)
        if val is not None:
            return key, float(val)
    return None, None


def _pick_forget(metrics: Dict[str, Any], preferred: str) -> Tuple[Optional[str], Optional[float]]:
    if preferred != "auto":
        val = metrics.get(preferred)
        return (preferred, float(val)) if val is not None else (None, None)
    for key in ("forget_test_acc", "forget_acc"):
        val = metrics.get(key)
        if val is not None:
            return key, float(val)
    return None, None


def _safe_mean(vals: List[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _is_dominated(i: int, points: List[Dict[str, Any]]) -> bool:
    p = points[i]
    for j, q in enumerate(points):
        if i == j:
            continue
        better_or_equal = (q["utility"] >= p["utility"]) and (q["forget_metric"] <= p["forget_metric"])
        strictly_better = (q["utility"] > p["utility"]) or (q["forget_metric"] < p["forget_metric"])
        if better_or_equal and strictly_better:
            return True
    return False


def _collect_rows(
    root: Path,
    utility_preferred: str,
    forget_preferred: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metrics_path in sorted(root.rglob("endpoint_metrics.json")):
        run_dir = metrics_path.parent
        summary_path = run_dir / "summary.json"
        summary = _load_json(summary_path) if summary_path.exists() else {}
        endpoint_metrics = _load_json(metrics_path)

        objective = None
        forget_alpha = None
        unlearn_steps = None
        retrain_epochs = None
        retrain_lr = None
        if isinstance(summary, dict):
            uobj = summary.get("unlearning_objective", {})
            ts = summary.get("training_schedule", {})
            objective = uobj.get("forget_objective")
            forget_alpha = uobj.get("forget_alpha")
            unlearn_steps = ts.get("unlearn_steps")
            retrain_epochs = ts.get("retrain_epochs")
            retrain_lr = ts.get("retrain_lr")

        for seed_key, seed_metrics in endpoint_metrics.items():
            if not isinstance(seed_metrics, dict):
                continue
            utility_key, utility_val = _pick_utility(seed_metrics, utility_preferred)
            forget_key, forget_val = _pick_forget(seed_metrics, forget_preferred)
            if utility_val is None or forget_val is None:
                continue

            try:
                run_id = str(run_dir.relative_to(root))
            except ValueError:
                run_id = str(run_dir)

            row = {
                "run_id": run_id,
                "seed": seed_key,
                "utility": utility_val,
                "utility_key": utility_key,
                "forget_metric": forget_val,
                "forget_metric_key": forget_key,
                "test_acc": seed_metrics.get("test_acc"),
                "retain_test_acc": seed_metrics.get("retain_test_acc"),
                "forget_test_acc": seed_metrics.get("forget_test_acc"),
                "normalized_full_test_acc": seed_metrics.get("normalized_full_test_acc"),
                "objective": objective,
                "forget_alpha": forget_alpha,
                "unlearn_steps": unlearn_steps,
                "retrain_epochs": retrain_epochs,
                "retrain_lr": retrain_lr,
                "metrics_path": str(metrics_path),
                "summary_path": str(summary_path) if summary_path.exists() else None,
            }
            rows.append(row)
    return rows


def _aggregate_runs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["run_id"], []).append(row)

    out: List[Dict[str, Any]] = []
    for run_id, items in grouped.items():
        base = items[0]
        agg = {
            "run_id": run_id,
            "n_seeds": len(items),
            "utility": _safe_mean([x["utility"] for x in items]),
            "forget_metric": _safe_mean([x["forget_metric"] for x in items]),
            "test_acc": _safe_mean([x.get("test_acc") for x in items]),
            "retain_test_acc": _safe_mean([x.get("retain_test_acc") for x in items]),
            "forget_test_acc": _safe_mean([x.get("forget_test_acc") for x in items]),
            "normalized_full_test_acc": _safe_mean([x.get("normalized_full_test_acc") for x in items]),
            "utility_key": base.get("utility_key"),
            "forget_metric_key": base.get("forget_metric_key"),
            "objective": base.get("objective"),
            "forget_alpha": base.get("forget_alpha"),
            "unlearn_steps": base.get("unlearn_steps"),
            "retrain_epochs": base.get("retrain_epochs"),
            "retrain_lr": base.get("retrain_lr"),
            "metrics_path": base.get("metrics_path"),
            "summary_path": base.get("summary_path"),
        }
        if agg["utility"] is None or agg["forget_metric"] is None:
            continue
        out.append(agg)
    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize unlearning runs and compute Pareto front.")
    parser.add_argument("--root", type=str, required=True, help="Root directory containing run outputs")
    parser.add_argument(
        "--utility-metric",
        type=str,
        default="auto",
        choices=["auto", "normalized_full_test_acc", "normalized_full", "retain_test_acc", "test_acc"],
        help="Utility metric to maximize",
    )
    parser.add_argument(
        "--forget-metric",
        type=str,
        default="auto",
        choices=["auto", "forget_test_acc", "forget_acc"],
        help="Forget metric to minimize",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="pareto_summary",
        help="Output filename prefix under --root",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"root does not exist: {root}")

    seed_rows = _collect_rows(
        root=root,
        utility_preferred=args.utility_metric,
        forget_preferred=args.forget_metric,
    )
    if not seed_rows:
        raise RuntimeError("No usable rows found. Check --root and metric keys.")

    run_rows = _aggregate_runs(seed_rows)
    if not run_rows:
        raise RuntimeError("No aggregated run rows available.")

    for i in range(len(run_rows)):
        run_rows[i]["pareto"] = not _is_dominated(i, run_rows)

    run_rows_sorted = sorted(
        run_rows,
        key=lambda x: (float(x["pareto"]), float(x["utility"]), -float(x["forget_metric"])),
        reverse=True,
    )
    pareto_rows = [r for r in run_rows_sorted if r["pareto"]]

    payload = {
        "root": str(root),
        "num_seed_rows": len(seed_rows),
        "num_runs": len(run_rows),
        "utility_metric": args.utility_metric,
        "forget_metric": args.forget_metric,
        "pareto_size": len(pareto_rows),
        "pareto_runs": pareto_rows,
        "all_runs": run_rows_sorted,
    }

    out_json = root / f"{args.out_prefix}.json"
    out_csv = root / f"{args.out_prefix}.csv"
    seed_csv = root / f"{args.out_prefix}_seeds.csv"
    out_json.write_text(json.dumps(payload, indent=2))

    run_fields = [
        "run_id",
        "pareto",
        "n_seeds",
        "utility",
        "forget_metric",
        "test_acc",
        "retain_test_acc",
        "forget_test_acc",
        "normalized_full_test_acc",
        "objective",
        "forget_alpha",
        "unlearn_steps",
        "retrain_epochs",
        "retrain_lr",
        "utility_key",
        "forget_metric_key",
        "summary_path",
        "metrics_path",
    ]
    _write_csv(out_csv, run_rows_sorted, run_fields)

    seed_fields = [
        "run_id",
        "seed",
        "utility",
        "forget_metric",
        "test_acc",
        "retain_test_acc",
        "forget_test_acc",
        "normalized_full_test_acc",
        "objective",
        "forget_alpha",
        "unlearn_steps",
        "retrain_epochs",
        "retrain_lr",
        "utility_key",
        "forget_metric_key",
        "summary_path",
        "metrics_path",
    ]
    _write_csv(seed_csv, seed_rows, seed_fields)

    print(f"[DONE] json : {out_json}")
    print(f"[DONE] csv  : {out_csv}")
    print(f"[DONE] seeds: {seed_csv}")
    print(f"[INFO] runs={len(run_rows)} pareto={len(pareto_rows)}")


if __name__ == "__main__":
    main()
