#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List


def _load(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multiple MIA bank summary.json files.")
    parser.add_argument("summaries", nargs="+", help="Paths to summary.json files from run_mia_merge_bank.py")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    rows = []
    keys = set()
    for p in args.summaries:
        payload = _load(Path(p).expanduser().resolve())
        row = payload.get("metrics", {})
        row["_source"] = str(p)
        rows.append(row)
        keys.update(k for k in row.keys() if not k.startswith("_"))

    agg: Dict[str, Any] = {"n": len(rows), "rows": rows, "mean": {}, "std": {}}
    for k in sorted(keys):
        vals: List[float] = [float(r[k]) for r in rows if k in r]
        if not vals:
            continue
        agg["mean"][k] = mean(vals)
        agg["std"][k] = pstdev(vals) if len(vals) > 1 else 0.0

    text = json.dumps(agg, indent=2)
    if args.out:
        out = Path(args.out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
