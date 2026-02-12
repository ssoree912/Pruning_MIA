#!/usr/bin/env python3
"""
Check if two checkpoints share identical initial weights (excluding masks/BN stats).

Usage:
  python scripts/check_same_init.py \
    --ckpt0 runs/static/sparsity_0.9/cifar10/seed42/best_model.pth \
    --ckpt1 runs/static/sparsity_0.9/cifar10/seed43/best_model.pth
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt0", required=True)
    ap.add_argument("--ckpt1", required=True)
    args = ap.parse_args()

    s0 = load_sd(Path(args.ckpt0))
    s1 = load_sd(Path(args.ckpt1))

    keys = sorted(set(s0.keys()) & set(s1.keys()))
    # Compare only weights/bias; exclude masks and BN running stats
    keys = [
        k for k in keys
        if (k.endswith("weight") or k.endswith("bias"))
        and not k.endswith(".mask")
        and "running_mean" not in k
        and "running_var" not in k
        and "num_batches_tracked" not in k
    ]
    maxdiff = 0.0
    for k in keys:
        a, b = s0[k], s1[k]
        if a.shape != b.shape:
            continue
        d = (a - b).abs().max().item()
        if d > maxdiff:
            maxdiff = d
    print(f"max |diff| over weights/bias = {maxdiff}")


if __name__ == "__main__":
    main()
