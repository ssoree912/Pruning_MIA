#!/usr/bin/env python3
"""
Find a pair of runs that satisfy "paper-like particle" conditions:
  - Same initialization (init_model.pth weights/bias identical)
  - Same subspace for static pruning (init_pruned_model.pth masks identical)
  - Same config (config.json identical up to ignored keys)
  - (Optional) Different SGD noise (data_seed differs)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch


def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in sd.keys()):
        return sd
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


def load_state_dict(p: Path) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(p, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if not isinstance(sd, dict):
        raise ValueError(f"{p} must be a state_dict or contain 'state_dict'.")
    sd = {k: v for k, v in sd.items() if torch.is_tensor(v)}
    return _strip_module_prefix(sd)


def maxdiff_weights_bias(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
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
    return float(md)


def extract_masks(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in sd.items() if k.endswith(".mask")}


def masks_identical(m0: Dict[str, torch.Tensor], m1: Dict[str, torch.Tensor]) -> bool:
    if not m0 or not m1:
        return False
    if set(m0.keys()) != set(m1.keys()):
        return False
    for k in m0.keys():
        a, b = m0[k], m1[k]
        if a.shape != b.shape:
            return False
        if not torch.equal(a, b):
            return False
    return True


def load_config_json(p: Path) -> Dict[str, Any]:
    with open(p, "r") as f:
        return json.load(f)


def _delete_by_dotted_key(d: Dict[str, Any], dotted: str) -> None:
    parts = dotted.split(".")
    cur = d
    for i, part in enumerate(parts):
        if not isinstance(cur, dict):
            return
        if i == len(parts) - 1:
            cur.pop(part, None)
        else:
            cur = cur.get(part, None)


def normalize_config(cfg: Dict[str, Any], ignore_keys: List[str]) -> Dict[str, Any]:
    out = json.loads(json.dumps(cfg))
    for k in ignore_keys:
        _delete_by_dotted_key(out, k)
    return out


def get_seed_and_data_seed(cfg: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    syscfg = cfg.get("system", {}) if isinstance(cfg, dict) else {}
    seed = syscfg.get("seed", None)
    data_seed = syscfg.get("data_seed", None)
    init_seed = syscfg.get("init_seed", None)
    return seed, data_seed, init_seed


def resolve_base(runs: Path, method: str, sparsity: Optional[float], dataset: str, freeze_tag: Optional[str]) -> Path:
    if method == "dense":
        return runs / "dense" / dataset
    if method == "static":
        if sparsity is None:
            raise ValueError("static requires --sparsity")
        return runs / "static" / f"sparsity_{sparsity}" / dataset
    if method == "dpf":
        if sparsity is None:
            raise ValueError("dpf requires --sparsity")
        tag = f"_{freeze_tag}" if freeze_tag else ""
        return runs / "dpf" / f"sparsity_{sparsity}{tag}" / dataset
    raise ValueError(f"unknown method={method}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["static", "dpf", "dense"])
    ap.add_argument("--sparsity", type=float, default=None)
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--freeze-tag", default=None)
    ap.add_argument("--runs", default="./runs")

    ap.add_argument("--init-file", default="init_model.pth")
    ap.add_argument("--pruned-init-file", default="init_pruned_model.pth")
    ap.add_argument("--mask-from", default=None,
                    help="Fallback checkpoint filename to extract masks if pruned-init-file missing.")

    ap.add_argument("--require-data-seed-diff", action="store_true")
    ap.add_argument("--ignore-config-keys", nargs="*",
                    default=["name", "save_dir", "system.gpu", "system.seed", "system.data_seed"],
                    help="Dotted keys to ignore when comparing config.json")

    args = ap.parse_args()

    runs = Path(args.runs)
    base = resolve_base(runs, args.method, args.sparsity, args.dataset, args.freeze_tag)
    if not base.exists():
        raise SystemExit(f"Base path not found: {base}")

    seed_dirs = sorted([p for p in base.glob("seed*") if p.is_dir()])
    if len(seed_dirs) < 2:
        raise SystemExit("Need at least 2 seed directories.")

    candidates = []
    for sd in seed_dirs:
        init_p = sd / args.init_file
        cfg_p = sd / "config.json"
        if init_p.exists() and cfg_p.exists():
            candidates.append(sd)
    if len(candidates) < 2:
        raise SystemExit(f"Need at least 2 seeds with {args.init_file} and config.json under {base}")

    init_sds = {}
    cfgs_raw = {}
    cfgs_norm = {}
    for sd in candidates:
        init_sds[sd] = load_state_dict(sd / args.init_file)
        cfg = load_config_json(sd / "config.json")
        cfgs_raw[sd] = cfg
        cfgs_norm[sd] = normalize_config(cfg, args.ignore_config_keys)

    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            if cfgs_norm[a] != cfgs_norm[b]:
                continue
            if maxdiff_weights_bias(init_sds[a], init_sds[b]) != 0.0:
                continue

            seed_a, data_a, init_a = get_seed_and_data_seed(cfgs_raw[a])
            seed_b, data_b, init_b = get_seed_and_data_seed(cfgs_raw[b])

            if args.require_data_seed_diff:
                if data_a is None or data_b is None or data_a == data_b:
                    continue

            if args.method == "static":
                pruned_a = a / args.pruned_init_file
                pruned_b = b / args.pruned_init_file
                if pruned_a.exists() and pruned_b.exists():
                    sd_a = load_state_dict(pruned_a)
                    sd_b = load_state_dict(pruned_b)
                elif args.mask_from is not None:
                    fa = a / args.mask_from
                    fb = b / args.mask_from
                    if not (fa.exists() and fb.exists()):
                        continue
                    sd_a = load_state_dict(fa)
                    sd_b = load_state_dict(fb)
                else:
                    continue
                if not masks_identical(extract_masks(sd_a), extract_masks(sd_b)):
                    continue

            print(a.name.replace("seed", ""), b.name.replace("seed", ""))
            return

    raise SystemExit("No pair found that satisfies all constraints.")


if __name__ == "__main__":
    main()
