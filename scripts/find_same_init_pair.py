#!/usr/bin/env python3
"""
Find two seeds under runs/ that share identical init_model.pth (max diff == 0).

Usage:
  python scripts/find_same_init_pair.py \
    --method static --sparsity 0.9 --dataset cifar10
"""

import argparse
from pathlib import Path
import torch


def _strip_module_prefix(sd):
    if not any(k.startswith("module.") for k in sd.keys()):
        return sd
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


def load_sd(p: Path):
    ckpt = torch.load(p, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if not isinstance(sd, dict):
        raise ValueError("Checkpoint must be a dict or contain 'state_dict'.")
    return _strip_module_prefix(sd)


def maxdiff(a, b):
    keys = sorted(set(a.keys()) & set(b.keys()))
    keys = [
        k for k in keys
        if (k.endswith("weight") or k.endswith("bias"))
        and not k.endswith(".mask")
        and "running_mean" not in k
        and "running_var" not in k
        and "num_batches_tracked" not in k
    ]
    md = 0.0
    for k in keys:
        t0, t1 = a[k], b[k]
        if t0.shape != t1.shape:
            continue
        d = (t0 - t1).abs().max().item()
        if d > md:
            md = d
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["static", "dpf", "dense"])
    ap.add_argument("--sparsity", type=float, default=None)
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--freeze-tag", default=None)
    ap.add_argument("--runs", default="./runs")
    args = ap.parse_args()

    runs = Path(args.runs)
    if args.method == "dense":
        base = runs / "dense" / args.dataset
    elif args.method == "static":
        base = runs / "static" / f"sparsity_{args.sparsity}" / args.dataset
    else:  # dpf
        tag = f"_{args.freeze_tag}" if args.freeze_tag else ""
        base = runs / "dpf" / f"sparsity_{args.sparsity}{tag}" / args.dataset

    seeds = sorted([p for p in base.glob("seed*") if (p / "init_model.pth").exists()])
    if len(seeds) < 2:
        raise SystemExit("Need at least 2 seeds with init_model.pth")

    # Load first seed and compare
    s0 = seeds[0]
    sd0 = load_sd(s0 / "init_model.pth")
    for s1 in seeds[1:]:
        sd1 = load_sd(s1 / "init_model.pth")
        if maxdiff(sd0, sd1) == 0.0:
            print(s0.name.replace("seed", ""), s1.name.replace("seed", ""))
            return

    raise SystemExit("No identical init pairs found.")


if __name__ == "__main__":
    main()
