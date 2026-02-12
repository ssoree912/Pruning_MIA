#!/usr/bin/env python3
"""
Check whether two checkpoints share identical masks.

Usage:
  python scripts/check_static_masks.py \
    --ckpt0 runs/static/sparsity_0.9/cifar10/seed43/best_model.pth \
    --ckpt1 runs/static/sparsity_0.9/cifar10/seed44/best_model.pth
"""

import argparse
from pathlib import Path
import torch


def _strip_module_prefix(state):
    if not any(k.startswith("module.") for k in state.keys()):
        return state
    out = {}
    for k, v in state.items():
        out[k[7:] if k.startswith("module.") else k] = v
    return out


def load_state_dict(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError("Checkpoint must be a dict or contain 'state_dict'.")
    return _strip_module_prefix(state)


def extract_masks(state):
    return {k: v for k, v in state.items() if k.endswith(".mask")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt0", required=True)
    ap.add_argument("--ckpt1", required=True)
    args = ap.parse_args()

    s0 = load_state_dict(Path(args.ckpt0))
    s1 = load_state_dict(Path(args.ckpt1))
    m0 = extract_masks(s0)
    m1 = extract_masks(s1)

    if not m0 or not m1:
        raise RuntimeError("No mask tensors found in one or both checkpoints.")

    keys0 = set(m0.keys())
    keys1 = set(m1.keys())
    if keys0 != keys1:
        missing0 = sorted(list(keys1 - keys0))[:5]
        missing1 = sorted(list(keys0 - keys1))[:5]
        print(f"[warn] mask key mismatch: only in ckpt0={len(keys0-keys1)}, ckpt1={len(keys1-keys0)}")
        if missing0:
            print(f"  examples only in ckpt1: {missing0}")
        if missing1:
            print(f"  examples only in ckpt0: {missing1}")
        return

    diffs = 0
    total = 0
    max_diff = 0.0
    for k in keys0:
        a = m0[k].detach().cpu()
        b = m1[k].detach().cpu()
        if a.shape != b.shape:
            diffs += 1
            print(f"[diff] shape mismatch: {k} {a.shape} vs {b.shape}")
            continue
        if not torch.equal(a, b):
            diffs += 1
            max_diff = max(max_diff, float((a - b).abs().max().item()))
        total += 1

    if diffs == 0:
        print(f"[ok] masks identical across {total} tensors")
    else:
        print(f"[diff] {diffs}/{total} mask tensors differ (max abs diff={max_diff})")


if __name__ == "__main__":
    main()
