#!/usr/bin/env python3
"""
Dense checkpoint 기반 unlearning + connectivity (Step 1/2/3) 실행 스크립트.

운영 원칙:
- MIA는 기본적으로 비활성화이며, --run-mia로 선택적으로 수행
- Step 2: 선형 경로 barrier 진단
- Step 3: 고정 mask(subspace) 경로 barrier 진단
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import models  # noqa: E402
from unlearning.utils import (
    build_forget_retain_indices,
    make_subset_loader,
    resolve_df_spec,
    set_seed,
    train_unlearning_endpoint_ascent,
)


@dataclass
class ModelSpec:
    dataset: str
    arch: str
    layers: int
    width_mult: float = 1.0
    depth_mult: float = 1.0
    model_mult: int = 0
    datapath: str = "~/Datasets/CIFAR"
    batch_size: int = 128
    workers: int = 4



def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state):
        return state
    return {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}


def load_checkpoint(ckpt_path: Path) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected checkpoint format: {ckpt_path}")
    if "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    state = _strip_module_prefix(state)
    return ckpt, state


def infer_model_spec(ckpt: Dict[str, Any], args: argparse.Namespace) -> ModelSpec:
    cfg = ckpt.get("config", {})
    cfg_data = cfg.get("data", {})
    cfg_model = cfg.get("model", {})

    dataset = args.dataset or cfg_data.get("dataset", "cifar10")
    arch = args.arch or cfg_model.get("arch", "resnet")
    layers = args.layers if args.layers is not None else int(cfg_model.get("layers", 20))
    width_mult = args.width_mult if args.width_mult is not None else float(cfg_model.get("width_mult", 1.0))
    depth_mult = args.depth_mult if args.depth_mult is not None else float(cfg_model.get("depth_mult", 1.0))
    model_mult = args.model_mult if args.model_mult is not None else int(cfg_model.get("model_mult", 0))
    datapath = args.datapath or cfg_data.get("datapath", "~/Datasets/CIFAR")
    batch_size = args.batch_size if args.batch_size is not None else int(cfg_data.get("batch_size", 128))
    workers = args.workers if args.workers is not None else int(cfg_data.get("workers", 4))

    if dataset not in ("cifar10", "cifar100"):
        raise ValueError(f"Only cifar10/cifar100 are supported currently, got: {dataset}")
    if arch not in ("resnet", "wideresnet"):
        raise ValueError(f"Unsupported arch: {arch}")

    return ModelSpec(
        dataset=dataset,
        arch=arch,
        layers=layers,
        width_mult=width_mult,
        depth_mult=depth_mult,
        model_mult=model_mult,
        datapath=datapath,
        batch_size=batch_size,
        workers=workers,
    )


def build_dense_model(spec: ModelSpec) -> Tuple[nn.Module, int]:
    model, image_size = models.__dict__[spec.arch](
        data=spec.dataset,
        num_layers=spec.layers,
        width_mult=spec.width_mult,
        depth_mult=spec.depth_mult,
        model_mult=spec.model_mult,
    )
    if model is None:
        raise ValueError(
            f"Model construction failed: arch={spec.arch}, layers={spec.layers}, dataset={spec.dataset}"
        )
    return model, image_size


def build_cifar_datasets(spec: ModelSpec) -> Tuple[Any, Any, Any]:
    root = os.path.expanduser(spec.datapath)
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tf = transforms.Compose([transforms.ToTensor(), normalize])

    ds_cls = CIFAR10 if spec.dataset == "cifar10" else CIFAR100
    train_aug = ds_cls(root=root, train=True, download=False, transform=train_tf)
    train_eval = ds_cls(root=root, train=True, download=False, transform=eval_tf)
    test_eval = ds_cls(root=root, train=False, download=False, transform=eval_tf)
    return train_aug, train_eval, test_eval


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        correct += int((logits.argmax(dim=1) == y).sum().item())
        total += x.size(0)
    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
    }


def _get_dataset_targets(dataset: Any) -> List[int]:
    if hasattr(dataset, "targets"):
        return [int(v) for v in dataset.targets]
    if hasattr(dataset, "labels"):
        return [int(v) for v in dataset.labels]
    raise ValueError("Dataset does not expose targets/labels")


def train_scratch_retrain_baseline(
    spec: ModelSpec,
    retain_train_loader: DataLoader,
    retain_eval_loader: DataLoader,
    forget_eval_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    nesterov: bool,
    seed: int,
) -> Dict[str, Any]:
    if epochs <= 0:
        raise ValueError("baseline epochs must be > 0")

    set_seed(seed)
    model, _ = build_dense_model(spec)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_test_acc = float("-inf")
    best_epoch = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_metrics: Dict[str, float] = {}
    history: List[Dict[str, Any]] = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for x, y in retain_train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            bsz = x.size(0)
            seen += bsz
            running_loss += float(loss.item()) * bsz
        scheduler.step()

        retain_stats = evaluate(model, retain_eval_loader, device)
        forget_stats = evaluate(model, forget_eval_loader, device)
        test_stats = evaluate(model, test_loader, device)
        row = {
            "epoch": int(epoch),
            "train_retain_loss": running_loss / max(seen, 1),
            "retain_loss": float(retain_stats["loss"]),
            "retain_acc": float(retain_stats["acc"]),
            "forget_loss": float(forget_stats["loss"]),
            "forget_acc": float(forget_stats["acc"]),
            "test_loss": float(test_stats["loss"]),
            "test_acc": float(test_stats["acc"]),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"[scratch-retrain] epoch {epoch + 1:03d}/{epochs:03d} "
            f"retain_acc={row['retain_acc']:.4f} forget_acc={row['forget_acc']:.4f} test_acc={row['test_acc']:.4f}"
        )
        if row["test_acc"] > best_test_acc:
            best_test_acc = row["test_acc"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "retain_loss": row["retain_loss"],
                "retain_acc": row["retain_acc"],
                "forget_loss": row["forget_loss"],
                "forget_acc": row["forget_acc"],
                "test_loss": row["test_loss"],
                "test_acc": row["test_acc"],
            }

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        retain_stats = evaluate(model, retain_eval_loader, device)
        forget_stats = evaluate(model, forget_eval_loader, device)
        test_stats = evaluate(model, test_loader, device)
        best_metrics = {
            "retain_loss": float(retain_stats["loss"]),
            "retain_acc": float(retain_stats["acc"]),
            "forget_loss": float(forget_stats["loss"]),
            "forget_acc": float(forget_stats["acc"]),
            "test_loss": float(test_stats["loss"]),
            "test_acc": float(test_stats["acc"]),
        }

    return {
        "state_dict": best_state,
        "best_epoch": int(best_epoch),
        "best_metrics": best_metrics,
        "history": history,
        "training_schedule": {
            "epochs": int(epochs),
            "lr": float(lr),
            "momentum": float(momentum),
            "weight_decay": float(weight_decay),
            "nesterov": bool(nesterov),
            "seed": int(seed),
        },
    }


@torch.no_grad()
def reset_bn_stats(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if m.running_mean is not None:
                m.running_mean.zero_()
            if m.running_var is not None:
                m.running_var.fill_(1)
            if hasattr(m, "num_batches_tracked") and m.num_batches_tracked is not None:
                m.num_batches_tracked.zero_()


@torch.no_grad()
def bn_recalibrate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> None:
    was_training = model.training
    model.train()
    reset_bn_stats(model)
    for i, (x, _) in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        _ = model(x)
    model.train(was_training)


def is_float_param_key(key: str, tensor: torch.Tensor) -> bool:
    if not torch.is_tensor(tensor):
        return False
    if not tensor.dtype.is_floating_point:
        return False
    if key.endswith("running_mean") or key.endswith("running_var") or key.endswith("num_batches_tracked"):
        return False
    return key.endswith("weight") or key.endswith("bias")


def interpolate_state(
    s0: Dict[str, torch.Tensor],
    s1: Dict[str, torch.Tensor],
    t: float,
    subspace_mask: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v0 in s0.items():
        if k not in s1:
            out[k] = v0
            continue
        v1 = s1[k]
        if torch.is_tensor(v0) and torch.is_tensor(v1) and v0.dtype.is_floating_point:
            if subspace_mask is None:
                out[k] = (1.0 - t) * v0 + t * v1
            else:
                m = subspace_mask.get(k)
                if m is None:
                    out[k] = (1.0 - t) * v0 + t * v1
                else:
                    m = m.to(v0.device, dtype=v0.dtype)
                    out[k] = v0 + t * (v1 - v0) * m
        else:
            out[k] = v0 if t < 0.5 else v1
    return out


def average_state_dicts(states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not states:
        raise ValueError("states must not be empty")
    out: Dict[str, torch.Tensor] = {}
    ref = states[0]
    for k, v in ref.items():
        if torch.is_tensor(v):
            if v.dtype.is_floating_point:
                acc = torch.zeros_like(v, dtype=torch.float32)
                for st in states:
                    acc += st[k].to(dtype=torch.float32)
                out[k] = (acc / float(len(states))).to(dtype=v.dtype)
            else:
                out[k] = v.clone()
        else:
            out[k] = v
    return out


def build_swa_state_from_curve(
    s0: Dict[str, torch.Tensor],
    s1: Dict[str, torch.Tensor],
    curve: List[Dict[str, float]],
    topk: int,
    metric: str,
    t_min: float,
    t_max: float,
    subspace_mask: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, float]]]:
    if metric not in {"test_acc", "retain_acc", "retain_loss"}:
        raise ValueError(f"Unsupported SWA metric: {metric}")
    points = [p for p in curve if float(p["t"]) >= t_min and float(p["t"]) <= t_max]
    if not points:
        points = list(curve)
    if not points:
        raise ValueError("curve is empty")

    reverse = metric in {"test_acc", "retain_acc"}
    points = sorted(points, key=lambda p: float(p[metric]), reverse=reverse)
    k = len(points) if topk <= 0 else min(topk, len(points))
    selected = points[:k]
    selected_states = [
        interpolate_state(s0, s1, float(p["t"]), subspace_mask=subspace_mask) for p in selected
    ]
    swa_state = average_state_dicts(selected_states)
    return swa_state, selected


def recalibrate_state_bn(
    spec: ModelSpec,
    state: Dict[str, torch.Tensor],
    bn_loader: DataLoader,
    device: torch.device,
    bn_recalc_on: bool,
    bn_batches: int,
) -> Dict[str, torch.Tensor]:
    model, _ = build_dense_model(spec)
    model = model.to(device)
    model.load_state_dict(state, strict=True)
    if bn_recalc_on:
        bn_recalibrate(model, bn_loader, device=device, max_batches=bn_batches)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def build_delta_mask(
    s0: Dict[str, torch.Tensor],
    s1: Dict[str, torch.Tensor],
    topk: float,
) -> Dict[str, torch.Tensor]:
    score_chunks = []
    keys = []
    for k, v0 in s0.items():
        if k not in s1:
            continue
        v1 = s1[k]
        if not is_float_param_key(k, v0):
            continue
        score = (v1 - v0).abs().flatten().cpu()
        if score.numel() == 0:
            continue
        score_chunks.append(score)
        keys.append(k)
    if not score_chunks:
        raise RuntimeError("No eligible tensors found for delta mask")
    all_scores = torch.cat(score_chunks)
    keep = int(max(1, round(all_scores.numel() * topk)))
    threshold = torch.topk(all_scores, keep, largest=True).values.min().item()

    mask: Dict[str, torch.Tensor] = {}
    for k in keys:
        delta = (s1[k] - s0[k]).abs()
        mask[k] = (delta >= threshold).to(dtype=s0[k].dtype, device=s0[k].device)
    return mask


def build_saliency_mask(
    spec: ModelSpec,
    s_ref: Dict[str, torch.Tensor],
    retain_loader: DataLoader,
    device: torch.device,
    topk: float,
    max_batches: int,
) -> Dict[str, torch.Tensor]:
    model, _ = build_dense_model(spec)
    model.load_state_dict(s_ref, strict=True)
    model = model.to(device)
    model.train()

    loss_fn = nn.CrossEntropyLoss()
    grads: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        grads[name] = torch.zeros_like(p, device="cpu")

    model.zero_grad(set_to_none=True)
    for bi, (x, y) in enumerate(retain_loader):
        if max_batches > 0 and bi >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                grads[name] += p.grad.detach().abs().cpu()
        model.zero_grad(set_to_none=True)

    flat_scores = []
    keys = []
    for k, g in grads.items():
        if g.numel() == 0:
            continue
        keys.append(k)
        flat_scores.append(g.flatten())
    if not flat_scores:
        raise RuntimeError("No saliency gradients were collected")
    all_scores = torch.cat(flat_scores)
    keep = int(max(1, round(all_scores.numel() * topk)))
    threshold = torch.topk(all_scores, keep, largest=True).values.min().item()

    mask: Dict[str, torch.Tensor] = {}
    s_ref_dev = {k: v.to(device) for k, v in s_ref.items()}
    for k in keys:
        if k in s_ref_dev:
            mask[k] = (grads[k].to(device) >= threshold).to(dtype=s_ref_dev[k].dtype)
    return mask


def run_connectivity(
    spec: ModelSpec,
    s0: Dict[str, torch.Tensor],
    s1: Dict[str, torch.Tensor],
    retain_eval_loader: DataLoader,
    forget_eval_loader: DataLoader,
    test_loader: DataLoader,
    bn_loader: DataLoader,
    device: torch.device,
    lambdas: List[float],
    bn_recalc_on: bool,
    bn_batches: int,
    subspace_mask: Optional[Dict[str, torch.Tensor]] = None,
    retain_test_loader: Optional[DataLoader] = None,
    forget_test_loader: Optional[DataLoader] = None,
    normalized_full_scale: Optional[float] = None,
) -> Dict[str, Any]:
    model, _ = build_dense_model(spec)
    model = model.to(device)

    curve = []
    for i, t in enumerate(lambdas, start=1):
        s_t = interpolate_state(s0, s1, t, subspace_mask=subspace_mask)
        model.load_state_dict(s_t, strict=True)
        if bn_recalc_on:
            bn_recalibrate(model, bn_loader, device=device, max_batches=bn_batches)
        retain_stats = evaluate(model, retain_eval_loader, device)
        forget_stats = evaluate(model, forget_eval_loader, device)
        test_stats = evaluate(model, test_loader, device)
        point = {
            "t": float(t),
            "retain_loss": float(retain_stats["loss"]),
            "retain_acc": float(retain_stats["acc"]),
            "forget_loss": float(forget_stats["loss"]),
            "forget_acc": float(forget_stats["acc"]),
            "test_loss": float(test_stats["loss"]),
            "test_acc": float(test_stats["acc"]),
        }
        if retain_test_loader is not None:
            retain_test_stats = evaluate(model, retain_test_loader, device)
            point["retain_test_loss"] = float(retain_test_stats["loss"])
            point["retain_test_acc"] = float(retain_test_stats["acc"])
        if forget_test_loader is not None:
            forget_test_stats = evaluate(model, forget_test_loader, device)
            point["forget_test_loss"] = float(forget_test_stats["loss"])
            point["forget_test_acc"] = float(forget_test_stats["acc"])
        if normalized_full_scale is not None and normalized_full_scale > 0.0:
            point["normalized_full_scale"] = float(normalized_full_scale)
            point["normalized_full_test_acc"] = float(point["test_acc"] / normalized_full_scale)
            point["normalized_full"] = point["normalized_full_test_acc"]
        curve.append(point)
        log_msg = (
            f"[interp {i:03d}/{len(lambdas):03d}] t={t:.3f} "
            f"retain_loss={point['retain_loss']:.4f} retain_acc={point['retain_acc']:.4f} "
            f"forget_acc={point['forget_acc']:.4f} test_acc={point['test_acc']:.4f}"
        )
        if "retain_test_acc" in point:
            log_msg += f" retain_test_acc={point['retain_test_acc']:.4f}"
        if "forget_test_acc" in point:
            log_msg += f" forget_test_acc={point['forget_test_acc']:.4f}"
        print(log_msg)

    l0 = curve[0]["retain_loss"]
    l1 = curve[-1]["retain_loss"]
    lmax = max(p["retain_loss"] for p in curve)
    barrier = lmax - max(l0, l1)

    acc_ref = max(curve[0]["retain_acc"], curve[-1]["retain_acc"])
    min_acc = min(p["retain_acc"] for p in curve)
    acc_drop_pp = (acc_ref - min_acc) * 100.0

    return {
        "curve": curve,
        "retain_loss_endpoint0": float(l0),
        "retain_loss_endpoint1": float(l1),
        "retain_loss_max": float(lmax),
        "retain_loss_barrier": float(barrier),
        "retain_acc_drop_pp": float(acc_drop_pp),
        "best_by_test_acc": max(curve, key=lambda x: float(x["test_acc"])),
        "best_by_retain_acc": max(curve, key=lambda x: float(x["retain_acc"])),
        "best_by_retain_loss": min(curve, key=lambda x: float(x["retain_loss"])),
    }


def evaluate_endpoint_state(
    spec: ModelSpec,
    state: Dict[str, torch.Tensor],
    retain_eval_loader: DataLoader,
    forget_eval_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    retain_test_loader: Optional[DataLoader] = None,
    forget_test_loader: Optional[DataLoader] = None,
    normalized_full_scale: Optional[float] = None,
) -> Dict[str, float]:
    model, _ = build_dense_model(spec)
    model = model.to(device)
    model.load_state_dict(state, strict=True)

    retain_stats = evaluate(model, retain_eval_loader, device)
    forget_stats = evaluate(model, forget_eval_loader, device)
    test_stats = evaluate(model, test_loader, device)
    out = {
        "retain_loss": float(retain_stats["loss"]),
        "retain_acc": float(retain_stats["acc"]),
        "forget_loss": float(forget_stats["loss"]),
        "forget_acc": float(forget_stats["acc"]),
        "test_loss": float(test_stats["loss"]),
        "test_acc": float(test_stats["acc"]),
    }
    if retain_test_loader is not None:
        retain_test_stats = evaluate(model, retain_test_loader, device)
        out["retain_test_loss"] = float(retain_test_stats["loss"])
        out["retain_test_acc"] = float(retain_test_stats["acc"])
    if forget_test_loader is not None:
        forget_test_stats = evaluate(model, forget_test_loader, device)
        out["forget_test_loss"] = float(forget_test_stats["loss"])
        out["forget_test_acc"] = float(forget_test_stats["acc"])
    if normalized_full_scale is not None and normalized_full_scale > 0.0:
        out["normalized_full_scale"] = float(normalized_full_scale)
        out["normalized_full_test_acc"] = float(out["test_acc"] / normalized_full_scale)
        out["normalized_full"] = out["normalized_full_test_acc"]
    return out


def make_lambdas(num: int) -> List[float]:
    if num < 2:
        return [0.0, 1.0]
    return [i / (num - 1) for i in range(num)]


def _split_csv(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _build_mia_config(spec: ModelSpec, seed: int, sparsity: float) -> Dict[str, Any]:
    return {
        "seed": int(seed),
        "data": {
            "dataset": spec.dataset,
            "datapath": spec.datapath,
            "batch_size": int(spec.batch_size),
            "workers": int(spec.workers),
        },
        "model": {
            "arch": spec.arch,
            "layers": int(spec.layers),
            "width_mult": float(spec.width_mult),
            "depth_mult": float(spec.depth_mult),
            "model_mult": int(spec.model_mult),
        },
        "pruning": {
            "enabled": True,
            "method": "static",
            "sparsity": float(sparsity),
        },
    }


def _prepare_mia_workspace(
    stage_dir: Path,
    spec: ModelSpec,
    victim_ckpt: Path,
    shadow_ckpts: List[Path],
    victim_seed: int,
    shadow_seeds: List[int],
    mia_sparsity: float,
) -> Path:
    if len(shadow_ckpts) != len(shadow_seeds):
        raise ValueError("shadow_ckpts and shadow_seeds length mismatch")
    runs_base = stage_dir / "runs"
    dataset_root = runs_base / "static" / f"sparsity_{mia_sparsity}" / spec.dataset
    dataset_root.mkdir(parents=True, exist_ok=True)

    victim_seed_dir = dataset_root / f"seed{victim_seed}"
    victim_seed_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(victim_ckpt), str(victim_seed_dir / "best_model.pth"))
    with open(victim_seed_dir / "config.json", "w") as f:
        json.dump(_build_mia_config(spec, seed=victim_seed, sparsity=mia_sparsity), f, indent=2)

    for ckpt, seed in zip(shadow_ckpts, shadow_seeds):
        seed_dir = dataset_root / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(ckpt), str(seed_dir / "best_model.pth"))
        with open(seed_dir / "config.json", "w") as f:
            json.dump(_build_mia_config(spec, seed=seed, sparsity=mia_sparsity), f, indent=2)

    return runs_base


def _ensure_mia_split(
    repo_root: Path,
    dataset: str,
    split_seed: int,
    victim_seed: int,
    shadow_seeds: List[int],
) -> None:
    create_script = repo_root / "mia_eval" / "create_data" / "create_fixed_data_splits.py"
    cmd = [
        sys.executable,
        str(create_script),
        "--dataset",
        dataset,
        "--seed",
        str(split_seed),
        "--victim_seed",
        str(victim_seed),
        "--shadow_seeds",
        *[str(s) for s in shadow_seeds],
    ]
    subprocess.run(cmd, check=True, cwd=str(repo_root))


def _run_mia_core(
    repo_root: Path,
    runs_base: Path,
    result_file: Path,
    dataset: str,
    victim_seed: int,
    shadow_seeds: List[int],
    device: int,
    split_seed: int,
    attacks: str,
    forward_mode: str,
    tpr_fprs: str,
    save_scores: bool,
    debug: bool,
    mia_sparsity: float,
) -> Dict[str, Any]:
    mia_script = repo_root / "mia_eval" / "core" / "mia_modi.py"
    cmd = [
        sys.executable,
        str(mia_script),
        "--device",
        str(device),
        "--dataset_name",
        dataset,
        "--sparsity",
        str(mia_sparsity),
        "--victim_seed",
        str(victim_seed),
        "--seed",
        str(split_seed),
        "--shadow_seeds",
        *[str(s) for s in shadow_seeds],
        "--prune_method",
        "static",
        "--forward_mode",
        forward_mode,
        "--attacks",
        attacks,
        "--tpr_fprs",
        tpr_fprs,
        "--base_path",
        str(runs_base),
        "--result_file",
        str(result_file),
    ]
    if save_scores:
        cmd.append("--save_scores")
    if debug:
        cmd.append("--debug")
    subprocess.run(cmd, check=True, cwd=str(repo_root))
    with open(result_file, "r") as f:
        return json.load(f)


def _choose_curve_point(curve: List[Dict[str, float]], metric: str) -> Dict[str, float]:
    if metric in {"test_acc", "retain_acc"}:
        return max(curve, key=lambda x: float(x[metric]))
    if metric in {"retain_loss", "test_loss"}:
        return min(curve, key=lambda x: float(x[metric]))
    raise ValueError(f"Unsupported mia curve metric: {metric}")


def _save_state_dict_checkpoint(state: Dict[str, torch.Tensor], out_path: Path, stage: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "stage": stage}, str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dense checkpoint -> 2 unlearning endpoints + Step2/3 connectivity"
    )
    parser.add_argument("--dense-ckpt", type=str, required=True, help="Path to dense checkpoint")
    parser.add_argument("--out-dir", type=str, default="./runs/unlearning_connectivity")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing endpoint ckpts if present")
    parser.add_argument("--step1-only", action="store_true", help="Run only Step1(unlearning checkpoints), skip Step2/3 connectivity")

    parser.add_argument("--dataset", type=str, default=None, choices=["cifar10", "cifar100"])
    parser.add_argument("--arch", type=str, default=None, choices=["resnet", "wideresnet"])
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--width-mult", type=float, default=None)
    parser.add_argument("--depth-mult", type=float, default=None)
    parser.add_argument("--model-mult", type=int, default=None)
    parser.add_argument("--datapath", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (used when CUDA is available)")

    parser.add_argument("--seed-a", type=int, default=43, help="Unlearning seed A")
    parser.add_argument("--seed-b", type=int, default=44, help="Unlearning seed B")
    parser.add_argument("--split-seed", type=int, default=7, help="Df/Dr split seed")

    parser.add_argument("--df-mode", type=str, default="profile", choices=["profile", "class", "random"])
    parser.add_argument("--df-profile", type=str, default="df1", choices=["df1", "df2", "df3"])
    parser.add_argument("--forget-classes", type=str, default=None, help="Comma-separated class ids for df-mode=class")
    parser.add_argument("--forget-ratio", type=float, default=0.1, help="Forget ratio for df-mode=random")

    parser.add_argument("--unlearn-epochs", type=int, default=20)
    parser.add_argument("--unlearn-steps", type=int, default=0, help="Max unlearning optimizer steps (0 disables)")
    parser.add_argument("--unlearn-lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--forget-alpha", type=float, default=0.05, help="Df ascent strength in L=Dr-alpha*Df")
    parser.add_argument(
        "--forget-objective",
        type=str,
        default="ce_ascent",
        choices=["ce_ascent", "kl_uniform", "entropy"],
        help="Df objective: CE-ascent or KL-to-uniform / entropy-max variants",
    )
    parser.add_argument("--retain-weight", type=float, default=1.0, help="Dr descent weight")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm (<=0 disables)")
    parser.add_argument("--retrain-epochs", type=int, default=0, help="Retain-only retrain epochs after unlearning")
    parser.add_argument("--retrain-lr", type=float, default=None, help="Retain retrain LR (default: unlearn-lr)")
    parser.add_argument(
        "--retrain-momentum",
        type=float,
        default=None,
        help="Retain retrain momentum (default: momentum)",
    )
    parser.add_argument(
        "--retrain-weight-decay",
        type=float,
        default=None,
        help="Retain retrain weight decay (default: weight-decay)",
    )
    retrain_nesterov_group = parser.add_mutually_exclusive_group()
    retrain_nesterov_group.add_argument(
        "--retrain-nesterov",
        dest="retrain_nesterov",
        action="store_true",
        help="Enable Nesterov for retrain stage",
    )
    retrain_nesterov_group.add_argument(
        "--no-retrain-nesterov",
        dest="retrain_nesterov",
        action="store_false",
        help="Disable Nesterov for retrain stage",
    )
    parser.set_defaults(retrain_nesterov=None)
    parser.add_argument(
        "--ckpt-select",
        type=str,
        default="retain_acc",
        choices=["retain_acc", "test_acc"],
        help="Best checkpoint selection metric for endpoint model",
    )
    parser.add_argument(
        "--train-scratch-retrain-baseline",
        action="store_true",
        help="Train scratch retrain baseline on Dr and report its test/retain-test acc",
    )
    parser.add_argument(
        "--scratch-retrain-ckpt",
        type=str,
        default=None,
        help="Optional precomputed scratch retrain checkpoint path for baseline reporting",
    )
    parser.add_argument("--scratch-retrain-epochs", type=int, default=200)
    parser.add_argument("--scratch-retrain-lr", type=float, default=0.1)
    parser.add_argument("--scratch-retrain-momentum", type=float, default=0.9)
    parser.add_argument("--scratch-retrain-weight-decay", type=float, default=5e-4)
    parser.add_argument("--scratch-retrain-nesterov", action="store_true")
    parser.add_argument("--scratch-retrain-seed", type=int, default=123)

    # optional MIA evaluation
    parser.add_argument("--run-mia", action="store_true", help="Run MIA after checkpoint generation")
    parser.add_argument(
        "--mia-stages",
        type=str,
        default="unlearn,step2,step3",
        help="Comma-separated stages: unlearn,step2,step3,swa_step2,swa_step3,baseline",
    )
    parser.add_argument(
        "--mia-select-metric",
        type=str,
        default="test_acc",
        choices=["test_acc", "retain_acc", "retain_loss", "test_loss"],
        help="Curve metric for selecting step2/step3 checkpoint used in MIA",
    )
    parser.add_argument("--mia-device", type=int, default=-1, help="GPU id for MIA (-1 uses --gpu)")
    parser.add_argument("--mia-split-seed", type=int, default=7, help="Split seed for MIA split generation")
    parser.add_argument("--mia-victim-seed", type=int, default=None, help="Victim seed id used by MIA")
    parser.add_argument(
        "--mia-shadow-seeds",
        type=str,
        default="",
        help="Comma-separated shadow seeds for MIA workspace (default: auto from seed-b)",
    )
    parser.add_argument(
        "--mia-attacks",
        type=str,
        default="samia,threshold,nn,nn_top3,nn_cls,lira",
        help="Comma-separated MIA attacks",
    )
    parser.add_argument(
        "--mia-forward-mode",
        type=str,
        default="standard",
        choices=["standard", "scaling", "dpf"],
        help="MIA model forward mode",
    )
    parser.add_argument("--mia-tpr-fprs", type=str, default="0.1,1,5")
    parser.add_argument("--mia-save-scores", action="store_true")
    parser.add_argument("--mia-debug", action="store_true")
    parser.add_argument("--mia-sparsity", type=float, default=0.0, help="Virtual sparsity tag used in MIA workspace")
    parser.add_argument(
        "--mia-baseline-ckpt",
        type=str,
        default=None,
        help="Optional baseline victim checkpoint path for extra MIA stage",
    )
    parser.add_argument(
        "--mia-baseline-shadow-ckpts",
        type=str,
        default="",
        help="Comma-separated shadow checkpoint paths for baseline MIA stage",
    )

    parser.add_argument("--lambdas", type=int, default=21, help="Interpolation points count")
    bn_group = parser.add_mutually_exclusive_group()
    bn_group.add_argument("--bn-recalc", dest="bn_recalc", action="store_true", help="Enable BN stats recalibration at each t")
    bn_group.add_argument("--no-bn-recalc", dest="bn_recalc", action="store_false", help="Disable BN stats recalibration at each t")
    parser.set_defaults(bn_recalc=True)
    parser.add_argument("--bn-batches", type=int, default=200, help="Max Dr batches for BN recalibration")
    swa_group = parser.add_mutually_exclusive_group()
    swa_group.add_argument("--swa-merge", dest="swa_merge", action="store_true", help="Enable SWA merge from connectivity path")
    swa_group.add_argument("--no-swa-merge", dest="swa_merge", action="store_false", help="Disable SWA merge from connectivity path")
    parser.set_defaults(swa_merge=True)
    parser.add_argument("--swa-source", type=str, default="both", choices=["step2", "step3", "both"])
    parser.add_argument("--swa-topk", type=int, default=5, help="Top-k points on path to average (<=0 means all)")
    parser.add_argument(
        "--swa-select-metric",
        type=str,
        default="test_acc",
        choices=["test_acc", "retain_acc", "retain_loss"],
        help="Metric used to select top-k points for SWA",
    )
    parser.add_argument("--swa-t-min", type=float, default=0.0, help="Min interpolation t for SWA candidate points")
    parser.add_argument("--swa-t-max", type=float, default=1.0, help="Max interpolation t for SWA candidate points")

    parser.add_argument("--mask-method", type=str, default="delta", choices=["delta", "saliency"])
    parser.add_argument("--mask-topk", type=float, default=0.1, help="Top-k ratio for step3 mask")
    parser.add_argument("--saliency-batches", type=int, default=50, help="Batches for saliency mask")

    parser.add_argument("--max-acc-drop-pp", type=float, default=2.0, help="Barrier gate: retain acc drop threshold")
    parser.add_argument("--max-loss-barrier", type=float, default=0.1, help="Barrier gate: retain loss barrier threshold")
    args = parser.parse_args()

    if args.seed_a == args.seed_b:
        raise ValueError("seed-a and seed-b must be different")
    if args.mask_topk <= 0.0 or args.mask_topk > 1.0:
        raise ValueError("--mask-topk must be in (0,1]")
    if args.forget_alpha <= 0.0:
        raise ValueError("--forget-alpha must be > 0")
    if args.unlearn_steps < 0:
        raise ValueError("--unlearn-steps must be >= 0")
    if args.retrain_epochs < 0:
        raise ValueError("--retrain-epochs must be >= 0")
    if args.scratch_retrain_epochs <= 0 and args.train_scratch_retrain_baseline:
        raise ValueError("--scratch-retrain-epochs must be > 0 when --train-scratch-retrain-baseline is set")
    if args.swa_t_min > args.swa_t_max:
        raise ValueError("--swa-t-min must be <= --swa-t-max")
    if args.run_mia:
        valid_stages = {"unlearn", "step2", "step3", "swa_step2", "swa_step3", "baseline"}
        bad = [s for s in _split_csv(args.mia_stages) if s not in valid_stages]
        if bad:
            raise ValueError(f"Invalid --mia-stages entries: {bad}")

    dense_ckpt = Path(args.dense_ckpt)
    if not dense_ckpt.exists():
        raise FileNotFoundError(f"dense checkpoint not found: {dense_ckpt}")
    ckpt_meta, base_state = load_checkpoint(dense_ckpt)
    spec = infer_model_spec(ckpt_meta, args)

    num_classes = 10 if spec.dataset == "cifar10" else 100
    df_spec = resolve_df_spec(args, num_classes=num_classes)
    df_tag = df_spec["name"]

    run_dir = (
        Path(args.out_dir)
        / spec.dataset
        / df_tag
        / f"seed{args.seed_a}_seed{args.seed_b}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent
    mia_selected_stages = set(_split_csv(args.mia_stages)) if args.run_mia else set()
    mia_results: Dict[str, Any] = {"enabled": bool(args.run_mia), "stages": {}, "errors": []}
    mia_victim_seed = int(args.mia_victim_seed) if args.mia_victim_seed is not None else int(args.seed_a)
    if args.mia_shadow_seeds.strip():
        mia_shadow_seeds = [int(x) for x in _split_csv(args.mia_shadow_seeds)]
    else:
        default_shadow = int(args.seed_b) if int(args.seed_b) != mia_victim_seed else int(args.seed_a)
        mia_shadow_seeds = [default_shadow]
    mia_device = int(args.gpu) if int(args.mia_device) < 0 else int(args.mia_device)
    baseline_shadow_ckpts = [Path(p).expanduser().resolve() for p in _split_csv(args.mia_baseline_shadow_ckpts)]

    print("=" * 80)
    print("Step 1: Build Df/Dr and train two unlearning endpoints")
    print("=" * 80)
    print(f"Dense ckpt: {dense_ckpt}")
    print(f"Dataset/Arch: {spec.dataset}/{spec.arch}{spec.layers}")
    print(f"Df spec: {df_spec}")

    train_aug, train_eval, test_eval = build_cifar_datasets(spec)
    targets = train_aug.targets if hasattr(train_aug, "targets") else train_aug.labels
    forget_idx, retain_idx = build_forget_retain_indices(targets, df_spec=df_spec, split_seed=args.split_seed)
    if len(forget_idx) == 0 or len(retain_idx) == 0:
        raise RuntimeError(
            f"Invalid Df/Dr split: forget={len(forget_idx)}, retain={len(retain_idx)}. "
            "Adjust df spec."
        )
    print(f"Df size={len(forget_idx)} | Dr size={len(retain_idx)}")

    with open(run_dir / "split_info.json", "w") as f:
        json.dump(
            {
                "df_spec": df_spec,
                "split_seed": args.split_seed,
                "forget_size": len(forget_idx),
                "retain_size": len(retain_idx),
            },
            f,
            indent=2,
        )

    retain_train_loader_a = make_subset_loader(
        train_aug, retain_idx, spec.batch_size, spec.workers, shuffle=True, seed=args.seed_a
    )
    retain_train_loader_b = make_subset_loader(
        train_aug, retain_idx, spec.batch_size, spec.workers, shuffle=True, seed=args.seed_b
    )
    forget_train_loader_a = make_subset_loader(
        train_aug, forget_idx, spec.batch_size, spec.workers, shuffle=True, seed=args.seed_a + 100
    )
    forget_train_loader_b = make_subset_loader(
        train_aug, forget_idx, spec.batch_size, spec.workers, shuffle=True, seed=args.seed_b + 100
    )
    retain_eval_loader = make_subset_loader(
        train_eval, retain_idx, spec.batch_size, spec.workers, shuffle=False, seed=0
    )
    forget_eval_loader = make_subset_loader(
        train_eval, forget_idx, spec.batch_size, spec.workers, shuffle=False, seed=0
    )
    test_loader = DataLoader(
        test_eval,
        batch_size=spec.batch_size,
        shuffle=False,
        num_workers=spec.workers,
        pin_memory=True,
    )
    bn_loader = make_subset_loader(
        train_eval, retain_idx, spec.batch_size, spec.workers, shuffle=True, seed=args.split_seed + 1000
    )

    retain_test_loader: Optional[DataLoader] = None
    forget_test_loader: Optional[DataLoader] = None
    retain_test_idx: List[int] = []
    forget_test_idx: List[int] = []
    normalized_full_scale: Optional[float] = None
    if df_spec.get("type") == "class":
        test_targets = _get_dataset_targets(test_eval)
        forget_test_idx, retain_test_idx = build_forget_retain_indices(
            test_targets, df_spec=df_spec, split_seed=args.split_seed
        )
        if len(test_targets) > 0:
            normalized_full_scale = float(len(retain_test_idx) / len(test_targets))
        if retain_test_idx:
            retain_test_loader = make_subset_loader(
                test_eval, retain_test_idx, spec.batch_size, spec.workers, shuffle=False, seed=0
            )
        if forget_test_idx:
            forget_test_loader = make_subset_loader(
                test_eval, forget_test_idx, spec.batch_size, spec.workers, shuffle=False, seed=0
            )

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    ckpt_a = run_dir / f"unlearn_seed{args.seed_a}.pth"
    ckpt_b = run_dir / f"unlearn_seed{args.seed_b}.pth"
    scratch_baseline_metrics: Optional[Dict[str, Any]] = None
    scratch_baseline_ckpt: Optional[Path] = None

    if args.scratch_retrain_ckpt:
        scratch_baseline_ckpt = Path(args.scratch_retrain_ckpt)
        if not scratch_baseline_ckpt.exists():
            raise FileNotFoundError(f"scratch retrain checkpoint not found: {scratch_baseline_ckpt}")
        _, scratch_state = load_checkpoint(scratch_baseline_ckpt)
        scratch_baseline_metrics = evaluate_endpoint_state(
            spec=spec,
            state=scratch_state,
            retain_eval_loader=retain_eval_loader,
            forget_eval_loader=forget_eval_loader,
            test_loader=test_loader,
            device=device,
            retain_test_loader=retain_test_loader,
            forget_test_loader=forget_test_loader,
            normalized_full_scale=normalized_full_scale,
        )
        print(
            f"[scratch-retrain baseline] test_acc={scratch_baseline_metrics['test_acc']:.4f} "
            f"retain_acc={scratch_baseline_metrics['retain_acc']:.4f} "
            f"forget_acc={scratch_baseline_metrics['forget_acc']:.4f}"
        )
    elif args.train_scratch_retrain_baseline:
        scratch_baseline_ckpt = run_dir / f"scratch_retrain_seed{args.scratch_retrain_seed}.pth"
        if args.skip_existing and scratch_baseline_ckpt.exists():
            _, scratch_state = load_checkpoint(scratch_baseline_ckpt)
            scratch_baseline_metrics = evaluate_endpoint_state(
                spec=spec,
                state=scratch_state,
                retain_eval_loader=retain_eval_loader,
                forget_eval_loader=forget_eval_loader,
                test_loader=test_loader,
                device=device,
                retain_test_loader=retain_test_loader,
                forget_test_loader=forget_test_loader,
                normalized_full_scale=normalized_full_scale,
            )
            print(f"Reusing scratch retrain baseline: {scratch_baseline_ckpt}")
        else:
            scratch_retain_train_loader = make_subset_loader(
                train_aug,
                retain_idx,
                spec.batch_size,
                spec.workers,
                shuffle=True,
                seed=args.scratch_retrain_seed,
            )
            scratch_payload = train_scratch_retrain_baseline(
                spec=spec,
                retain_train_loader=scratch_retain_train_loader,
                retain_eval_loader=retain_eval_loader,
                forget_eval_loader=forget_eval_loader,
                test_loader=test_loader,
                device=device,
                epochs=args.scratch_retrain_epochs,
                lr=args.scratch_retrain_lr,
                momentum=args.scratch_retrain_momentum,
                weight_decay=args.scratch_retrain_weight_decay,
                nesterov=args.scratch_retrain_nesterov,
                seed=args.scratch_retrain_seed,
            )
            torch.save(scratch_payload, str(scratch_baseline_ckpt))
            scratch_baseline_metrics = evaluate_endpoint_state(
                spec=spec,
                state=scratch_payload["state_dict"],
                retain_eval_loader=retain_eval_loader,
                forget_eval_loader=forget_eval_loader,
                test_loader=test_loader,
                device=device,
                retain_test_loader=retain_test_loader,
                forget_test_loader=forget_test_loader,
                normalized_full_scale=normalized_full_scale,
            )
            print(f"Saved scratch retrain baseline: {scratch_baseline_ckpt}")
            print(
                f"[scratch-retrain baseline] best_test_acc={scratch_baseline_metrics['test_acc']:.4f} "
                f"retain_acc={scratch_baseline_metrics['retain_acc']:.4f}"
            )

    if args.skip_existing and ckpt_a.exists():
        ep_a, s_a = load_checkpoint(ckpt_a)
        print(f"Reusing endpoint A: {ckpt_a}")
    else:
        model_a, _ = build_dense_model(spec)
        ep_a = train_unlearning_endpoint_ascent(
            model=model_a,
            base_state=base_state,
            seed=args.seed_a,
            retain_train_loader=retain_train_loader_a,
            forget_train_loader=forget_train_loader_a,
            retain_eval_loader=retain_eval_loader,
            forget_eval_loader=forget_eval_loader,
            test_loader=test_loader,
            evaluate_fn=evaluate,
            out_path=ckpt_a,
            device=device,
            epochs=args.unlearn_epochs,
            lr=args.unlearn_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
            forget_alpha=args.forget_alpha,
            forget_objective=args.forget_objective,
            retain_weight=args.retain_weight,
            grad_clip=args.grad_clip,
            retrain_epochs=args.retrain_epochs,
            retrain_lr=args.retrain_lr,
            retrain_momentum=args.retrain_momentum,
            retrain_weight_decay=args.retrain_weight_decay,
            retrain_nesterov=args.retrain_nesterov,
            ckpt_select=args.ckpt_select,
            unlearn_steps=args.unlearn_steps,
            model_config={
                "dataset": spec.dataset,
                "arch": spec.arch,
                "layers": spec.layers,
                "width_mult": spec.width_mult,
                "depth_mult": spec.depth_mult,
                "model_mult": spec.model_mult,
                "datapath": spec.datapath,
            },
        )
        s_a = _strip_module_prefix(ep_a["state_dict"])
    if args.skip_existing and ckpt_b.exists():
        ep_b, s_b = load_checkpoint(ckpt_b)
        print(f"Reusing endpoint B: {ckpt_b}")
    else:
        model_b, _ = build_dense_model(spec)
        ep_b = train_unlearning_endpoint_ascent(
            model=model_b,
            base_state=base_state,
            seed=args.seed_b,
            retain_train_loader=retain_train_loader_b,
            forget_train_loader=forget_train_loader_b,
            retain_eval_loader=retain_eval_loader,
            forget_eval_loader=forget_eval_loader,
            test_loader=test_loader,
            evaluate_fn=evaluate,
            out_path=ckpt_b,
            device=device,
            epochs=args.unlearn_epochs,
            lr=args.unlearn_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
            forget_alpha=args.forget_alpha,
            forget_objective=args.forget_objective,
            retain_weight=args.retain_weight,
            grad_clip=args.grad_clip,
            retrain_epochs=args.retrain_epochs,
            retrain_lr=args.retrain_lr,
            retrain_momentum=args.retrain_momentum,
            retrain_weight_decay=args.retrain_weight_decay,
            retrain_nesterov=args.retrain_nesterov,
            ckpt_select=args.ckpt_select,
            unlearn_steps=args.unlearn_steps,
            model_config={
                "dataset": spec.dataset,
                "arch": spec.arch,
                "layers": spec.layers,
                "width_mult": spec.width_mult,
                "depth_mult": spec.depth_mult,
                "model_mult": spec.model_mult,
                "datapath": spec.datapath,
            },
        )
        s_b = _strip_module_prefix(ep_b["state_dict"])
    if args.skip_existing and ckpt_a.exists():
        _, s_a = load_checkpoint(ckpt_a)
    if args.skip_existing and ckpt_b.exists():
        _, s_b = load_checkpoint(ckpt_b)

    endpoint_metrics = {
        f"seed{args.seed_a}": evaluate_endpoint_state(
            spec=spec,
            state=s_a,
            retain_eval_loader=retain_eval_loader,
            forget_eval_loader=forget_eval_loader,
            test_loader=test_loader,
            device=device,
            retain_test_loader=retain_test_loader,
            forget_test_loader=forget_test_loader,
            normalized_full_scale=normalized_full_scale,
        ),
        f"seed{args.seed_b}": evaluate_endpoint_state(
            spec=spec,
            state=s_b,
            retain_eval_loader=retain_eval_loader,
            forget_eval_loader=forget_eval_loader,
            test_loader=test_loader,
            device=device,
            retain_test_loader=retain_test_loader,
            forget_test_loader=forget_test_loader,
            normalized_full_scale=normalized_full_scale,
        ),
    }
    if scratch_baseline_metrics is not None:
        for seed_key in list(endpoint_metrics.keys()):
            endpoint_metrics[seed_key]["test_acc_gap_vs_scratch_retrain"] = (
                endpoint_metrics[seed_key]["test_acc"] - scratch_baseline_metrics["test_acc"]
            )
            if "retain_test_acc" in endpoint_metrics[seed_key] and "retain_test_acc" in scratch_baseline_metrics:
                endpoint_metrics[seed_key]["retain_test_acc_gap_vs_scratch_retrain"] = (
                    endpoint_metrics[seed_key]["retain_test_acc"] - scratch_baseline_metrics["retain_test_acc"]
                )
    with open(run_dir / "endpoint_metrics.json", "w") as f:
        json.dump(endpoint_metrics, f, indent=2)
    print(
        f"[endpoints] seed{args.seed_a} retain_acc={endpoint_metrics[f'seed{args.seed_a}']['retain_acc']:.4f}, "
        f"forget_acc={endpoint_metrics[f'seed{args.seed_a}']['forget_acc']:.4f}, "
        f"test_acc={endpoint_metrics[f'seed{args.seed_a}']['test_acc']:.4f}"
    )
    print(
        f"[endpoints] seed{args.seed_b} retain_acc={endpoint_metrics[f'seed{args.seed_b}']['retain_acc']:.4f}, "
        f"forget_acc={endpoint_metrics[f'seed{args.seed_b}']['forget_acc']:.4f}, "
        f"test_acc={endpoint_metrics[f'seed{args.seed_b}']['test_acc']:.4f}"
    )

    def _run_stage_mia(stage_name: str, victim_ckpt_path: Path, shadow_ckpt_paths: List[Path]) -> None:
        if not args.run_mia:
            return
        if stage_name not in mia_selected_stages:
            return
        try:
            if not victim_ckpt_path.exists():
                raise FileNotFoundError(f"victim ckpt not found: {victim_ckpt_path}")
            if not shadow_ckpt_paths:
                raise ValueError(f"{stage_name}: no shadow ckpts provided for MIA")
            for p in shadow_ckpt_paths:
                if not p.exists():
                    raise FileNotFoundError(f"shadow ckpt not found: {p}")

            shadow_seed_list = list(mia_shadow_seeds)
            if len(shadow_seed_list) < len(shadow_ckpt_paths):
                start = max(shadow_seed_list) + 1 if shadow_seed_list else (mia_victim_seed + 1)
                for i in range(len(shadow_ckpt_paths) - len(shadow_seed_list)):
                    shadow_seed_list.append(start + i)
            shadow_seed_list = shadow_seed_list[:len(shadow_ckpt_paths)]

            stage_dir = run_dir / "mia_workspace" / stage_name
            stage_dir.mkdir(parents=True, exist_ok=True)
            runs_base = _prepare_mia_workspace(
                stage_dir=stage_dir,
                spec=spec,
                victim_ckpt=victim_ckpt_path,
                shadow_ckpts=shadow_ckpt_paths,
                victim_seed=mia_victim_seed,
                shadow_seeds=shadow_seed_list,
                mia_sparsity=float(args.mia_sparsity),
            )
            _ensure_mia_split(
                repo_root=repo_root,
                dataset=spec.dataset,
                split_seed=int(args.mia_split_seed),
                victim_seed=mia_victim_seed,
                shadow_seeds=shadow_seed_list,
            )
            result_file = run_dir / "mia_results" / f"{stage_name}.json"
            result_file.parent.mkdir(parents=True, exist_ok=True)
            result_payload = _run_mia_core(
                repo_root=repo_root,
                runs_base=runs_base,
                result_file=result_file,
                dataset=spec.dataset,
                victim_seed=mia_victim_seed,
                shadow_seeds=shadow_seed_list,
                device=mia_device,
                split_seed=int(args.mia_split_seed),
                attacks=str(args.mia_attacks),
                forward_mode=str(args.mia_forward_mode),
                tpr_fprs=str(args.mia_tpr_fprs),
                save_scores=bool(args.mia_save_scores),
                debug=bool(args.mia_debug),
                mia_sparsity=float(args.mia_sparsity),
            )
            mia_results["stages"][stage_name] = {
                "victim_ckpt": str(victim_ckpt_path),
                "shadow_ckpts": [str(p) for p in shadow_ckpt_paths],
                "result_file": str(result_file),
                "result": result_payload,
            }
            print(f"[MIA:{stage_name}] saved -> {result_file}")
        except Exception as e:
            msg = f"{stage_name}: {e}"
            mia_results["errors"].append(msg)
            print(f"[MIA:{stage_name}] ERROR: {e}")

    # Stage: unlearn endpoint MIA (victim=seed_a, shadow=seed_b)
    _run_stage_mia("unlearn", ckpt_a, [ckpt_b])
    # Optional external baseline MIA
    if args.run_mia and "baseline" in mia_selected_stages and args.mia_baseline_ckpt:
        baseline_victim = Path(args.mia_baseline_ckpt).expanduser().resolve()
        baseline_shadows = baseline_shadow_ckpts if baseline_shadow_ckpts else [ckpt_b]
        _run_stage_mia("baseline", baseline_victim, baseline_shadows)

    if args.step1_only:
        if args.run_mia and "baseline" in mia_selected_stages and not args.mia_baseline_ckpt:
            msg = "baseline stage requested in --mia-stages but --mia-baseline-ckpt was not provided"
            mia_results["errors"].append(msg)
            print(f"[MIA:baseline] ERROR: {msg}")
        step1_summary = {
            "dense_ckpt": str(dense_ckpt),
            "run_dir": str(run_dir),
            "device": str(device),
            "model_spec": spec.__dict__,
            "df_spec": df_spec,
            "evaluation_normalization": {
                "normalized_full_scale": normalized_full_scale,
                "normalized_full_formula": (
                    "normalized_full_test_acc = test_acc / normalized_full_scale"
                    if normalized_full_scale is not None and normalized_full_scale > 0.0
                    else None
                ),
            },
            "unlearning_objective": {
                "type": "retain_descent_with_forget",
                "forget_objective": args.forget_objective,
                "retain_weight": args.retain_weight,
                "forget_alpha": args.forget_alpha,
                "grad_clip": args.grad_clip,
                "ckpt_select": args.ckpt_select,
            },
            "training_schedule": {
                "unlearn_epochs": args.unlearn_epochs,
                "unlearn_steps": args.unlearn_steps,
                "unlearn_lr": args.unlearn_lr,
                "momentum": args.momentum,
                "weight_decay": args.weight_decay,
                "nesterov": args.nesterov,
                "retrain_epochs": args.retrain_epochs,
                "retrain_lr": args.retrain_lr if args.retrain_lr is not None else args.unlearn_lr,
                "retrain_momentum": args.retrain_momentum if args.retrain_momentum is not None else args.momentum,
                "retrain_weight_decay": (
                    args.retrain_weight_decay if args.retrain_weight_decay is not None else args.weight_decay
                ),
                "retrain_nesterov": args.retrain_nesterov if args.retrain_nesterov is not None else args.nesterov,
            },
            "endpoints": {
                "seed_a": args.seed_a,
                "seed_b": args.seed_b,
                "ckpt_a": str(ckpt_a),
                "ckpt_b": str(ckpt_b),
                "metrics_file": str(run_dir / "endpoint_metrics.json"),
                "metrics": endpoint_metrics,
            },
            "scratch_retrain_baseline": {
                "enabled": bool(args.train_scratch_retrain_baseline or args.scratch_retrain_ckpt),
                "ckpt": str(scratch_baseline_ckpt) if scratch_baseline_ckpt is not None else None,
                "metrics": scratch_baseline_metrics,
            },
            "mia": mia_results,
            "step1_only": True,
        }
        with open(run_dir / "summary.json", "w") as f:
            json.dump(step1_summary, f, indent=2)
        print("\n" + "=" * 80)
        print("STEP1 DONE (checkpoint only)")
        print("=" * 80)
        print(f"summary: {run_dir / 'summary.json'}")
        print(f"ckpt_a : {ckpt_a}")
        print(f"ckpt_b : {ckpt_b}")
        return

    lambdas = make_lambdas(args.lambdas)

    print("\n" + "=" * 80)
    print("Step 2: Linear interpolation barrier")
    print("=" * 80)
    step2 = run_connectivity(
        spec=spec,
        s0=s_a,
        s1=s_b,
        retain_eval_loader=retain_eval_loader,
        forget_eval_loader=forget_eval_loader,
        test_loader=test_loader,
        bn_loader=bn_loader,
        device=device,
        lambdas=lambdas,
        bn_recalc_on=args.bn_recalc,
        bn_batches=args.bn_batches,
        subspace_mask=None,
        retain_test_loader=retain_test_loader,
        forget_test_loader=forget_test_loader,
        normalized_full_scale=normalized_full_scale,
    )
    with open(run_dir / "step2_linear.json", "w") as f:
        json.dump(step2, f, indent=2)
    print(
        f"[step2] retain_loss_barrier={step2['retain_loss_barrier']:.6f}, "
        f"retain_acc_drop_pp={step2['retain_acc_drop_pp']:.4f}"
    )
    if args.run_mia and "step2" in mia_selected_stages:
        best_p2 = _choose_curve_point(step2["curve"], metric=args.mia_select_metric)
        t2 = float(best_p2["t"])
        t2_shadow = float(max(0.0, min(1.0, 1.0 - t2)))
        state_p2_v = interpolate_state(s_a, s_b, t2, subspace_mask=None)
        state_p2_s = interpolate_state(s_a, s_b, t2_shadow, subspace_mask=None)
        state_p2_v = recalibrate_state_bn(
            spec=spec,
            state=state_p2_v,
            bn_loader=bn_loader,
            device=device,
            bn_recalc_on=args.bn_recalc,
            bn_batches=args.bn_batches,
        )
        state_p2_s = recalibrate_state_bn(
            spec=spec,
            state=state_p2_s,
            bn_loader=bn_loader,
            device=device,
            bn_recalc_on=args.bn_recalc,
            bn_batches=args.bn_batches,
        )
        step2_v_ckpt = run_dir / "mia_ckpts" / f"step2_victim_t{t2:.3f}.pth"
        step2_s_ckpt = run_dir / "mia_ckpts" / f"step2_shadow_t{t2_shadow:.3f}.pth"
        _save_state_dict_checkpoint(state_p2_v, step2_v_ckpt, stage="step2_victim")
        _save_state_dict_checkpoint(state_p2_s, step2_s_ckpt, stage="step2_shadow")
        _run_stage_mia("step2", step2_v_ckpt, [step2_s_ckpt])

    print("\n" + "=" * 80)
    print("Step 3: Mask 고정 subspace interpolation barrier")
    print("=" * 80)
    if args.mask_method == "delta":
        mask = build_delta_mask(s_a, s_b, topk=args.mask_topk)
    else:
        mask = build_saliency_mask(
            spec=spec,
            s_ref=s_a,
            retain_loader=retain_train_loader_a,
            device=device,
            topk=args.mask_topk,
            max_batches=args.saliency_batches,
        )
    torch.save(mask, str(run_dir / "step3_mask.pt"))

    step3 = run_connectivity(
        spec=spec,
        s0=s_a,
        s1=s_b,
        retain_eval_loader=retain_eval_loader,
        forget_eval_loader=forget_eval_loader,
        test_loader=test_loader,
        bn_loader=bn_loader,
        device=device,
        lambdas=lambdas,
        bn_recalc_on=args.bn_recalc,
        bn_batches=args.bn_batches,
        subspace_mask=mask,
        retain_test_loader=retain_test_loader,
        forget_test_loader=forget_test_loader,
        normalized_full_scale=normalized_full_scale,
    )
    with open(run_dir / "step3_masked_linear.json", "w") as f:
        json.dump(step3, f, indent=2)
    print(
        f"[step3] retain_loss_barrier={step3['retain_loss_barrier']:.6f}, "
        f"retain_acc_drop_pp={step3['retain_acc_drop_pp']:.4f}"
    )
    if args.run_mia and "step3" in mia_selected_stages:
        best_p3 = _choose_curve_point(step3["curve"], metric=args.mia_select_metric)
        t3 = float(best_p3["t"])
        t3_shadow = float(max(0.0, min(1.0, 1.0 - t3)))
        state_p3_v = interpolate_state(s_a, s_b, t3, subspace_mask=mask)
        state_p3_s = interpolate_state(s_a, s_b, t3_shadow, subspace_mask=mask)
        state_p3_v = recalibrate_state_bn(
            spec=spec,
            state=state_p3_v,
            bn_loader=bn_loader,
            device=device,
            bn_recalc_on=args.bn_recalc,
            bn_batches=args.bn_batches,
        )
        state_p3_s = recalibrate_state_bn(
            spec=spec,
            state=state_p3_s,
            bn_loader=bn_loader,
            device=device,
            bn_recalc_on=args.bn_recalc,
            bn_batches=args.bn_batches,
        )
        step3_v_ckpt = run_dir / "mia_ckpts" / f"step3_victim_t{t3:.3f}.pth"
        step3_s_ckpt = run_dir / "mia_ckpts" / f"step3_shadow_t{t3_shadow:.3f}.pth"
        _save_state_dict_checkpoint(state_p3_v, step3_v_ckpt, stage="step3_victim")
        _save_state_dict_checkpoint(state_p3_s, step3_s_ckpt, stage="step3_shadow")
        _run_stage_mia("step3", step3_v_ckpt, [step3_s_ckpt])

    swa_results: List[Dict[str, Any]] = []
    if args.swa_merge:
        print("\n" + "=" * 80)
        print("SWA Merge From Connectivity Path")
        print("=" * 80)

        sources: List[Tuple[str, Dict[str, Any], Optional[Dict[str, torch.Tensor]]]] = []
        if args.swa_source in {"step2", "both"}:
            sources.append(("step2", step2, None))
        if args.swa_source in {"step3", "both"}:
            sources.append(("step3", step3, mask))

        for src_name, src_result, src_mask in sources:
            swa_state_raw, selected_points = build_swa_state_from_curve(
                s0=s_a,
                s1=s_b,
                curve=src_result["curve"],
                topk=args.swa_topk,
                metric=args.swa_select_metric,
                t_min=args.swa_t_min,
                t_max=args.swa_t_max,
                subspace_mask=src_mask,
            )
            swa_state = recalibrate_state_bn(
                spec=spec,
                state=swa_state_raw,
                bn_loader=bn_loader,
                device=device,
                bn_recalc_on=args.bn_recalc,
                bn_batches=args.bn_batches,
            )
            swa_metrics = evaluate_endpoint_state(
                spec=spec,
                state=swa_state,
                retain_eval_loader=retain_eval_loader,
                forget_eval_loader=forget_eval_loader,
                test_loader=test_loader,
                device=device,
                retain_test_loader=retain_test_loader,
                forget_test_loader=forget_test_loader,
                normalized_full_scale=normalized_full_scale,
            )
            ckpt_path = run_dir / f"{src_name}_swa_merge.pth"
            torch.save(
                {
                    "state_dict": swa_state,
                    "source": src_name,
                    "selected_points": selected_points,
                    "select_metric": args.swa_select_metric,
                    "swa_topk": args.swa_topk,
                    "t_min": args.swa_t_min,
                    "t_max": args.swa_t_max,
                    "bn_recalc": args.bn_recalc,
                    "bn_batches": args.bn_batches,
                },
                str(ckpt_path),
            )
            result_row = {
                "source": src_name,
                "select_metric": args.swa_select_metric,
                "topk": args.swa_topk,
                "t_min": args.swa_t_min,
                "t_max": args.swa_t_max,
                "selected_points": selected_points,
                "ckpt": str(ckpt_path),
                "metrics": swa_metrics,
            }
            swa_results.append(result_row)
            print(
                f"[swa:{src_name}] test_acc={swa_metrics['test_acc']:.4f} "
                f"retain_acc={swa_metrics['retain_acc']:.4f} forget_acc={swa_metrics['forget_acc']:.4f} "
                f"(points={len(selected_points)})"
            )
            mia_stage_name = f"swa_{src_name}"
            if args.run_mia and mia_stage_name in mia_selected_stages:
                _run_stage_mia(mia_stage_name, ckpt_path, [ckpt_b])

        with open(run_dir / "swa_merge_results.json", "w") as f:
            json.dump({"swa_results": swa_results}, f, indent=2)

    def is_mia_candidate(result: Dict[str, Any]) -> bool:
        return (
            result["retain_acc_drop_pp"] <= args.max_acc_drop_pp
            or result["retain_loss_barrier"] <= args.max_loss_barrier
        )

    if args.run_mia and "baseline" in mia_selected_stages and not args.mia_baseline_ckpt:
        msg = "baseline stage requested in --mia-stages but --mia-baseline-ckpt was not provided"
        mia_results["errors"].append(msg)
        print(f"[MIA:baseline] ERROR: {msg}")

    summary = {
        "dense_ckpt": str(dense_ckpt),
        "run_dir": str(run_dir),
        "device": str(device),
        "model_spec": spec.__dict__,
        "df_spec": df_spec,
        "evaluation_normalization": {
            "normalized_full_scale": normalized_full_scale,
            "normalized_full_formula": (
                "normalized_full_test_acc = test_acc / normalized_full_scale"
                if normalized_full_scale is not None and normalized_full_scale > 0.0
                else None
            ),
        },
        "step2": {
            "retain_loss_barrier": step2["retain_loss_barrier"],
            "retain_acc_drop_pp": step2["retain_acc_drop_pp"],
            "mia_candidate": is_mia_candidate(step2),
        },
        "step3": {
            "retain_loss_barrier": step3["retain_loss_barrier"],
            "retain_acc_drop_pp": step3["retain_acc_drop_pp"],
            "mia_candidate": is_mia_candidate(step3),
        },
        "barrier_gate": {
            "max_acc_drop_pp": args.max_acc_drop_pp,
            "max_loss_barrier": args.max_loss_barrier,
        },
        "unlearning_objective": {
            "type": "retain_descent_with_forget",
            "forget_objective": args.forget_objective,
            "retain_weight": args.retain_weight,
            "forget_alpha": args.forget_alpha,
            "grad_clip": args.grad_clip,
            "ckpt_select": args.ckpt_select,
        },
        "training_schedule": {
            "unlearn_epochs": args.unlearn_epochs,
            "unlearn_steps": args.unlearn_steps,
            "unlearn_lr": args.unlearn_lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "nesterov": args.nesterov,
            "retrain_epochs": args.retrain_epochs,
            "retrain_lr": args.retrain_lr if args.retrain_lr is not None else args.unlearn_lr,
            "retrain_momentum": args.retrain_momentum if args.retrain_momentum is not None else args.momentum,
            "retrain_weight_decay": (
                args.retrain_weight_decay if args.retrain_weight_decay is not None else args.weight_decay
            ),
            "retrain_nesterov": args.retrain_nesterov if args.retrain_nesterov is not None else args.nesterov,
        },
        "endpoints": {
            "seed_a": args.seed_a,
            "seed_b": args.seed_b,
            "ckpt_a": str(ckpt_a),
            "ckpt_b": str(ckpt_b),
            "metrics_file": str(run_dir / "endpoint_metrics.json"),
            "metrics": endpoint_metrics,
        },
        "scratch_retrain_baseline": {
            "enabled": bool(args.train_scratch_retrain_baseline or args.scratch_retrain_ckpt),
            "ckpt": str(scratch_baseline_ckpt) if scratch_baseline_ckpt is not None else None,
            "metrics": scratch_baseline_metrics,
        },
        "swa_merge": {
            "enabled": bool(args.swa_merge),
            "source": args.swa_source,
            "select_metric": args.swa_select_metric,
            "topk": args.swa_topk,
            "t_min": args.swa_t_min,
            "t_max": args.swa_t_max,
            "results_file": str(run_dir / "swa_merge_results.json") if args.swa_merge else None,
            "results": swa_results,
        },
        "mia": mia_results,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"summary: {run_dir / 'summary.json'}")
    print(f"step2   : {run_dir / 'step2_linear.json'}")
    print(f"step3   : {run_dir / 'step3_masked_linear.json'}")
    print(f"mask    : {run_dir / 'step3_mask.pt'}")
    if args.swa_merge:
        print(f"swa     : {run_dir / 'swa_merge_results.json'}")


if __name__ == "__main__":
    main()
