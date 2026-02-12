#!/usr/bin/env python3
"""
Extract mask tensors from a checkpoint and save as a standalone mask dict.

Usage:
  python scripts/extract_masks.py \
    --ckpt runs/static/sparsity_0.9/cifar10/seed43/best_model.pth \
    --out masks/static_seed43.pt
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
    masks = {}
    for k, v in state.items():
        if k.endswith(".mask"):
            masks[k] = v.detach().cpu()
    return masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint (.pth)")
    ap.add_argument("--out", required=True, help="Output mask file (.pt)")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    state = load_state_dict(ckpt_path)
    masks = extract_masks(state)

    if not masks:
        raise RuntimeError("No mask tensors found in checkpoint state_dict.")

    torch.save(masks, out_path)
    print(f"[done] extracted {len(masks)} masks -> {out_path}")


if __name__ == "__main__":
    main()
