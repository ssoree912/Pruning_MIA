#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear connectivity test (loss barrier) for:
  A) static (fixed mask)
  B1) dynamic_naive (no mask alignment)
  B2) dynamic_common (project endpoints to common mask, then interpolate)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn

import models
import pruning
from data import DataLoader as CIFARLoader


def _find_config_path(ckpt_path: Path) -> Optional[Path]:
    candidates = [
        ckpt_path.parent / "config.json",
        ckpt_path.parent.parent / "config.json",
        ckpt_path.parent.parent.parent / "config.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _strip_module_prefix(state):
    if not any(k.startswith("module.") for k in state.keys()):
        return state
    out = {}
    for k, v in state.items():
        out[k[7:] if k.startswith("module.") else k] = v
    return out


def load_ckpt_state(ckpt_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise ValueError("Checkpoint must be a dict state_dict or contain 'state_dict'.")
    sd = _strip_module_prefix(sd)
    out = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
    return out


def build_model_from_config(cfg: Dict[str, Any]):
    if cfg.get("pruning", {}).get("enabled", False):
        pruner_key = cfg["pruning"]["method"].lower()
        if pruner_key in ("static", "dpf", "dcil"):
            pruner_key = "dcil"
        pruner = pruning.__dict__[pruner_key]
        model, image_size = pruning.models.__dict__[cfg["model"]["arch"]](
            data=cfg["data"]["dataset"],
            num_layers=cfg["model"]["layers"],
            width_mult=cfg["model"].get("width_mult", 1.0),
            depth_mult=cfg["model"].get("depth_mult", 1.0),
            model_mult=cfg["model"].get("model_mult", 1.0),
            mnn=pruner.mnn,
        )
    else:
        model, image_size = models.__dict__[cfg["model"]["arch"]](
            data=cfg["data"]["dataset"],
            num_layers=cfg["model"]["layers"],
            width_mult=cfg["model"].get("width_mult", 1.0),
            depth_mult=cfg["model"].get("depth_mult", 1.0),
            model_mult=cfg["model"].get("model_mult", 1.0),
        )
    return model, image_size


def get_dataloaders_and_eval(cfg: Dict[str, Any], image_size: int):
    dataset = cfg["data"]["dataset"]
    datapath = cfg["data"].get("datapath", "~/Datasets/CIFAR")
    batch_size = cfg["data"].get("batch_size", 128)
    workers = cfg["data"].get("workers", 4)

    train_loader_bn, val_loader = CIFARLoader(
        batch_size, dataset, workers, datapath, image_size, True
    )

    loss_fn = nn.CrossEntropyLoss()

    @torch.no_grad()
    def eval_fn(model: nn.Module, val_loader, device: torch.device, type_value: Optional[int]):
        model.eval()
        total_loss, total, correct = 0.0, 0, 0
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if type_value is None:
                logits = model(x)
            else:
                try:
                    logits = model(x, type_value)
                except TypeError:
                    logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += float(loss.item()) * x.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += x.size(0)
        acc = correct / max(total, 1)
        return total_loss / max(total, 1), acc, "acc"

    return train_loader_bn, val_loader, eval_fn


def load_mask(mask_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    m = torch.load(mask_path, map_location=device)
    if not isinstance(m, dict):
        raise ValueError("Mask file must be a dict: {param_name: mask_tensor}.")
    out = {}
    for k, v in m.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
    return out


def apply_mask_to_state(state: Dict[str, torch.Tensor], mask: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Apply mask to both mask params and corresponding weights.
    Mask keys are expected to end with ".mask".
    """
    out = dict(state)
    for name, m in mask.items():
        if name in out:
            out[name] = m
        if name.endswith(".mask"):
            w_key = name[:-5] + "weight"
            if w_key in out:
                out[w_key] = out[w_key] * m
    return out


def common_mask_intersection(m0: Dict[str, torch.Tensor], m1: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys = set(m0.keys()) & set(m1.keys())
    out = {}
    for k in keys:
        a = (m0[k] != 0)
        b = (m1[k] != 0)
        out[k] = (a & b).to(m0[k].dtype).to(m0[k].device)
    return out


def interpolate_state(sd0: Dict[str, torch.Tensor], sd1: Dict[str, torch.Tensor], lam: float) -> Dict[str, torch.Tensor]:
    out = {}
    keys = set(sd0.keys()) & set(sd1.keys())
    for k in keys:
        out[k] = (1.0 - lam) * sd0[k] + lam * sd1[k]
    return out


@torch.no_grad()
def reset_bn_stats(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if m.running_mean is not None:
                m.running_mean.zero_()
            if m.running_var is not None:
                m.running_var.fill_(1)
            if hasattr(m, "num_batches_tracked") and m.num_batches_tracked is not None:
                m.num_batches_tracked.zero_()


@torch.no_grad()
def bn_recalibrate(model: nn.Module, loader, device: torch.device, type_value: Optional[int], max_batches: int = 200):
    was_training = model.training
    model.train()
    reset_bn_stats(model)
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device, non_blocking=True)
        if type_value is None:
            _ = model(x)
        else:
            try:
                _ = model(x, type_value)
            except TypeError:
                _ = model(x)
    model.train(was_training)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt0", type=str, required=True)
    ap.add_argument("--ckpt1", type=str, required=True)
    ap.add_argument("--mode", type=str, choices=["static", "dynamic_naive", "dynamic_common"], default="static")
    ap.add_argument("--mask", type=str, default=None, help="(static) fixed mask path")
    ap.add_argument("--mask0", type=str, default=None, help="(dynamic_common) final mask for ckpt0")
    ap.add_argument("--mask1", type=str, default=None, help="(dynamic_common) final mask for ckpt1")
    ap.add_argument("--lambdas", type=int, default=101)
    ap.add_argument("--bn_recalc", action="store_true")
    ap.add_argument("--bn_recalc_batches", type=int, default=200)
    ap.add_argument("--type_value", type=int, default=None, help="Override type_value for forward (e.g., 5 static, 6 dpf)")
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    ckpt0 = Path(args.ckpt0)
    ckpt1 = Path(args.ckpt1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_path = _find_config_path(ckpt0)
    if cfg_path is None:
        raise FileNotFoundError(f"config.json not found near {ckpt0}")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    model, image_size = build_model_from_config(cfg)
    model = model.to(device)

    train_loader_bn, val_loader, eval_fn = get_dataloaders_and_eval(cfg, image_size)

    sd0 = load_ckpt_state(ckpt0, device)
    sd1 = load_ckpt_state(ckpt1, device)

    # Decide mask policy
    common_mask: Optional[Dict[str, torch.Tensor]] = None
    if args.mode == "static":
        if args.mask is None:
            raise ValueError("--mask is required for mode=static")
        common_mask = load_mask(Path(args.mask), device)
        sd0 = apply_mask_to_state(sd0, common_mask)
        sd1 = apply_mask_to_state(sd1, common_mask)
    elif args.mode == "dynamic_common":
        if args.mask0 is None or args.mask1 is None:
            raise ValueError("--mask0 and --mask1 are required for mode=dynamic_common")
        m0 = load_mask(Path(args.mask0), device)
        m1 = load_mask(Path(args.mask1), device)
        common_mask = common_mask_intersection(m0, m1)
        sd0 = apply_mask_to_state(sd0, common_mask)
        sd1 = apply_mask_to_state(sd1, common_mask)

    # type_value selection
    type_value = args.type_value
    if type_value is None:
        if cfg.get("pruning", {}).get("enabled", False):
            method = cfg["pruning"].get("method", "").lower()
            if method == "static":
                type_value = 5
            elif method == "dpf":
                type_value = 6
        else:
            type_value = None

    # Lambda grid
    K = args.lambdas
    lambdas = [k / (K - 1) for k in range(K)]

    curve: List[Dict[str, float]] = []
    metric_name: str = "metric"

    for lam in lambdas:
        sd_lam = interpolate_state(sd0, sd1, lam)
        if common_mask is not None:
            sd_lam = apply_mask_to_state(sd_lam, common_mask)
        model.load_state_dict(sd_lam, strict=False)

        if args.bn_recalc:
            bn_recalibrate(model, train_loader_bn, device, type_value, max_batches=args.bn_recalc_batches)

        loss, metric, metric_name = eval_fn(model, val_loader, device, type_value)
        curve.append({"lambda": float(lam), "loss": float(loss), "metric": float(metric)})

    L0 = curve[0]["loss"]
    L1 = curve[-1]["loss"]
    Lmax = max(p["loss"] for p in curve)
    Lend = max(L0, L1)
    barrier = Lmax - Lend

    out = {
        "mode": args.mode,
        "ckpt0": args.ckpt0,
        "ckpt1": args.ckpt1,
        "mask": args.mask,
        "mask0": args.mask0,
        "mask1": args.mask1,
        "bn_recalc": bool(args.bn_recalc),
        "bn_recalc_batches": int(args.bn_recalc_batches if args.bn_recalc else 0),
        "lambdas": int(K),
        "metric_name": metric_name,
        "L0": float(L0),
        "L1": float(L1),
        "Lmax": float(Lmax),
        "Lend": float(Lend),
        "barrier": float(barrier),
        "metric@0": float(curve[0]["metric"]),
        "metric@0.5": float(min(curve, key=lambda p: abs(p["lambda"] - 0.5))["metric"]),
        "metric@1": float(curve[-1]["metric"]),
        "curve": curve,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[done] mode={args.mode} barrier={barrier:.6f}  (Lmax={Lmax:.6f}, Lend={Lend:.6f})")
    print(f"[done] {metric_name}@0={out['metric@0']:.4f}, @0.5={out['metric@0.5']:.4f}, @1={out['metric@1']:.4f}")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
