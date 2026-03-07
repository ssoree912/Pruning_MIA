#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _parse_label_path(item: str) -> Tuple[str, Path]:
    if "=" not in item:
        raise ValueError(f"Invalid item '{item}'. Expected format: label=/path/to/summary.json")
    label, path = item.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Empty label in '{item}'")
    return label, Path(path).expanduser().resolve()


def _metric_union(rows: Dict[str, Dict[str, float]]) -> List[str]:
    keys = set()
    for v in rows.values():
        keys.update(v.keys())
    return sorted(keys)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare MIA bank summary metrics across model families."
    )
    parser.add_argument(
        "labeled_summaries",
        nargs="+",
        help="Items like merged=path/to/summary.json raw=... retrain=... dense=...",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Reference label for delta computation (default: first item).",
    )
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    model_rows: Dict[str, Dict[str, float]] = {}
    sources: Dict[str, str] = {}
    ordered_labels: List[str] = []

    for item in args.labeled_summaries:
        label, path = _parse_label_path(item)
        if label in model_rows:
            raise ValueError(f"Duplicate label: {label}")
        payload = _load_json(path)
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            raise ValueError(f"Invalid summary format (missing metrics dict): {path}")
        model_rows[label] = {k: float(v) for k, v in metrics.items()}
        sources[label] = str(path)
        ordered_labels.append(label)

    if not ordered_labels:
        raise ValueError("No inputs provided")

    ref_label = args.reference or ordered_labels[0]
    if ref_label not in model_rows:
        raise ValueError(f"--reference label not found: {ref_label}")

    metric_keys = _metric_union(model_rows)
    deltas: Dict[str, Dict[str, float]] = {}
    for label in ordered_labels:
        if label == ref_label:
            continue
        d: Dict[str, float] = {}
        for k in metric_keys:
            if k in model_rows[ref_label] and k in model_rows[label]:
                d[f"{k}_delta_{ref_label}_minus_{label}"] = (
                    float(model_rows[ref_label][k]) - float(model_rows[label][k])
                )
        deltas[f"{ref_label}_minus_{label}"] = d

    out = {
        "reference": ref_label,
        "sources": sources,
        "models": model_rows,
        "deltas": deltas,
    }

    text = json.dumps(out, indent=2)
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
