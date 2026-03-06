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
    composite_unlearning_score,
    make_subset_loader,
    resolve_df_spec,
    set_seed,
    split_indices_train_val,
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
    retain_val_loader: DataLoader,
    forget_val_loader: DataLoader,
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
    ckpt_select: str = "retain_val_acc",
    forget_budget_acc: float = 0.01,
) -> Dict[str, Any]:
    if epochs <= 0:
        raise ValueError("baseline epochs must be > 0")
    if ckpt_select not in {"retain_val_acc", "composite"}:
        raise ValueError(f"Unsupported scratch baseline ckpt_select: {ckpt_select}")

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

    best_score = float("-inf")
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

        retain_val_stats = evaluate(model, retain_val_loader, device)
        forget_val_stats = evaluate(model, forget_val_loader, device)
        retain_stats = evaluate(model, retain_eval_loader, device)
        forget_stats = evaluate(model, forget_eval_loader, device)
        test_stats = evaluate(model, test_loader, device)

        if ckpt_select == "retain_val_acc":
            selection_score = float(retain_val_stats["acc"])
        else:
            selection_score = composite_unlearning_score(
                retain_val_acc=float(retain_val_stats["acc"]),
                forget_val_acc=float(forget_val_stats["acc"]),
                full_val_acc=None,
                forget_budget_acc=float(forget_budget_acc),
                rt_ref=None,
            )

        row = {
            "epoch": int(epoch),
            "train_retain_loss": running_loss / max(seen, 1),
            "retain_val_loss": float(retain_val_stats["loss"]),
            "retain_val_acc": float(retain_val_stats["acc"]),
            "forget_val_loss": float(forget_val_stats["loss"]),
            "forget_val_acc": float(forget_val_stats["acc"]),
            "retain_loss": float(retain_stats["loss"]),
            "retain_acc": float(retain_stats["acc"]),
            "forget_loss": float(forget_stats["loss"]),
            "forget_acc": float(forget_stats["acc"]),
            "test_loss": float(test_stats["loss"]),
            "test_acc": float(test_stats["acc"]),
            "selection_score": float(selection_score),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"[scratch-retrain] epoch {epoch + 1:03d}/{epochs:03d} "
            f"retain_val_acc={row['retain_val_acc']:.4f} forget_val_acc={row['forget_val_acc']:.4f} "
            f"retain_acc={row['retain_acc']:.4f} forget_acc={row['forget_acc']:.4f} "
            f"test_acc={row['test_acc']:.4f} score={row['selection_score']:.6f}"
        )
        if row["selection_score"] > best_score:
            best_score = row["selection_score"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "retain_val_loss": row["retain_val_loss"],
                "retain_val_acc": row["retain_val_acc"],
                "forget_val_loss": row["forget_val_loss"],
                "forget_val_acc": row["forget_val_acc"],
                "retain_loss": row["retain_loss"],
                "retain_acc": row["retain_acc"],
                "forget_loss": row["forget_loss"],
                "forget_acc": row["forget_acc"],
                "test_loss": row["test_loss"],
                "test_acc": row["test_acc"],
                "selection_score": row["selection_score"],
            }

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        retain_val_stats = evaluate(model, retain_val_loader, device)
        forget_val_stats = evaluate(model, forget_val_loader, device)
        retain_stats = evaluate(model, retain_eval_loader, device)
        forget_stats = evaluate(model, forget_eval_loader, device)
        test_stats = evaluate(model, test_loader, device)
        if ckpt_select == "retain_val_acc":
            best_score = float(retain_val_stats["acc"])
        else:
            best_score = composite_unlearning_score(
                retain_val_acc=float(retain_val_stats["acc"]),
                forget_val_acc=float(forget_val_stats["acc"]),
                full_val_acc=None,
                forget_budget_acc=float(forget_budget_acc),
                rt_ref=None,
            )
        best_metrics = {
            "retain_val_loss": float(retain_val_stats["loss"]),
            "retain_val_acc": float(retain_val_stats["acc"]),
            "forget_val_loss": float(forget_val_stats["loss"]),
            "forget_val_acc": float(forget_val_stats["acc"]),
            "retain_loss": float(retain_stats["loss"]),
            "retain_acc": float(retain_stats["acc"]),
            "forget_loss": float(forget_stats["loss"]),
            "forget_acc": float(forget_stats["acc"]),
            "test_loss": float(test_stats["loss"]),
            "test_acc": float(test_stats["acc"]),
            "selection_score": float(best_score),
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
            "ckpt_select": ckpt_select,
            "forget_budget_acc": float(forget_budget_acc),
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


def swa_from_checkpoints(
    spec: ModelSpec,
    ckpt_paths: List[Path],
    bn_loader: DataLoader,
    device: torch.device,
    bn_recalc_on: bool,
    bn_batches: int,
) -> Dict[str, torch.Tensor]:
    if not ckpt_paths:
        raise ValueError("ckpt_paths must not be empty")
    states = [load_checkpoint(p)[1] for p in ckpt_paths]
    swa_state = average_state_dicts(states)
    return recalibrate_state_bn(
        spec=spec,
        state=swa_state,
        bn_loader=bn_loader,
        device=device,
        bn_recalc_on=bn_recalc_on,
        bn_batches=bn_batches,
    )


def greedy_soup(
    candidates: List[Dict[str, Any]],
    eval_state_fn,
) -> Tuple[Dict[str, torch.Tensor], List[str], float]:
    if not candidates:
        raise ValueError("candidates must not be empty")
    ordered = sorted(candidates, key=lambda x: float(x["val_score"]), reverse=True)
    chosen = [ordered[0]]
    soup_state = ordered[0]["state"]
    best_score = float(ordered[0]["val_score"])
    for cand in ordered[1:]:
        trial_state = average_state_dicts([c["state"] for c in chosen] + [cand["state"]])
        trial_score = float(eval_state_fn(trial_state))
        if trial_score >= best_score:
            chosen.append(cand)
            soup_state = trial_state
            best_score = trial_score
    return soup_state, [str(c["name"]) for c in chosen], float(best_score)


def make_linear_state_fn(
    s0: Dict[str, torch.Tensor],
    s1: Dict[str, torch.Tensor],
    subspace_mask: Optional[Dict[str, torch.Tensor]] = None,
):
    return lambda t: interpolate_state(s0, s1, float(t), subspace_mask=subspace_mask)


def bezier_state(
    s0: Dict[str, torch.Tensor],
    sc: Dict[str, torch.Tensor],
    s1: Dict[str, torch.Tensor],
    t: float,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v0 in s0.items():
        if k not in s1:
            out[k] = v0
            continue
        v1 = s1[k]
        vc = sc[k] if k in sc else v0
        if torch.is_tensor(v0) and torch.is_tensor(v1) and v0.dtype.is_floating_point:
            out[k] = ((1.0 - t) ** 2) * v0 + 2.0 * t * (1.0 - t) * vc + (t ** 2) * v1
        else:
            out[k] = v0 if t < 0.5 else v1
    return out


def make_bezier_state_fn(
    s0: Dict[str, torch.Tensor],
    sc: Dict[str, torch.Tensor],
    s1: Dict[str, torch.Tensor],
):
    return lambda t: bezier_state(s0, sc, s1, float(t))


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


def evaluate_connector(
    spec: ModelSpec,
    state_fn,
    retain_val_loader: Optional[DataLoader],
    forget_val_loader: Optional[DataLoader],
    retain_eval_loader: DataLoader,
    forget_eval_loader: DataLoader,
    test_loader: DataLoader,
    bn_loader: DataLoader,
    device: torch.device,
    lambdas: List[float],
    bn_recalc_on: bool,
    bn_batches: int,
    retain_test_loader: Optional[DataLoader] = None,
    forget_test_loader: Optional[DataLoader] = None,
    normalized_full_scale: Optional[float] = None,
) -> Dict[str, Any]:
    model, _ = build_dense_model(spec)
    model = model.to(device)

    curve: List[Dict[str, Any]] = []
    for i, t in enumerate(lambdas, start=1):
        s_t = state_fn(float(t))
        model.load_state_dict(s_t, strict=True)
        if bn_recalc_on:
            bn_recalibrate(model, bn_loader, device=device, max_batches=bn_batches)

        point: Dict[str, Any] = {"t": float(t)}
        if retain_val_loader is not None and forget_val_loader is not None:
            retain_val = evaluate(model, retain_val_loader, device)
            forget_val = evaluate(model, forget_val_loader, device)
            point["retain_val_loss"] = float(retain_val["loss"])
            point["retain_val_acc"] = float(retain_val["acc"])
            point["forget_val_loss"] = float(forget_val["loss"])
            point["forget_val_acc"] = float(forget_val["acc"])

        retain_stats = evaluate(model, retain_eval_loader, device)
        forget_stats = evaluate(model, forget_eval_loader, device)
        test_stats = evaluate(model, test_loader, device)
        point.update(
            {
                "retain_loss": float(retain_stats["loss"]),
                "retain_acc": float(retain_stats["acc"]),
                "full_val_loss": float(retain_stats["loss"]),
                "full_val_acc": float(retain_stats["acc"]),
                "forget_loss": float(forget_stats["loss"]),
                "forget_acc": float(forget_stats["acc"]),
                "test_loss": float(test_stats["loss"]),
                "test_acc": float(test_stats["acc"]),
            }
        )
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

        msg = (
            f"[interp {i:03d}/{len(lambdas):03d}] t={float(t):.3f} "
            f"retain_acc={point['retain_acc']:.4f} forget_acc={point['forget_acc']:.4f} test_acc={point['test_acc']:.4f}"
        )
        if "retain_val_acc" in point and "forget_val_acc" in point:
            msg += f" retain_val_acc={point['retain_val_acc']:.4f} forget_val_acc={point['forget_val_acc']:.4f}"
        print(msg)

    if not curve:
        raise RuntimeError("evaluate_connector produced empty curve")

    barrier_key = "retain_val_loss" if "retain_val_loss" in curve[0] else "retain_loss"
    drop_key = "retain_val_acc" if "retain_val_acc" in curve[0] else "retain_acc"

    l0 = float(curve[0][barrier_key])
    l1 = float(curve[-1][barrier_key])
    lmax = max(float(p[barrier_key]) for p in curve)
    barrier = lmax - max(l0, l1)

    acc_ref = max(float(curve[0][drop_key]), float(curve[-1][drop_key]))
    min_acc = min(float(p[drop_key]) for p in curve)
    acc_drop_pp = (acc_ref - min_acc) * 100.0

    return {
        "curve": curve,
        "barrier_metric": barrier_key,
        "drop_metric": drop_key,
        "retain_loss_endpoint0": float(l0),
        "retain_loss_endpoint1": float(l1),
        "retain_loss_max": float(lmax),
        "retain_loss_barrier": float(barrier),
        "retain_acc_drop_pp": float(acc_drop_pp),
        "best_by_test_acc": max(curve, key=lambda x: float(x["test_acc"])),
        "best_by_retain_acc": max(curve, key=lambda x: float(x.get(drop_key, x["retain_acc"]))),
        "best_by_retain_loss": min(curve, key=lambda x: float(x.get(barrier_key, x["retain_loss"]))),
    }


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
    retain_val_loader: Optional[DataLoader] = None,
    forget_val_loader: Optional[DataLoader] = None,
) -> Dict[str, Any]:
    state_fn = make_linear_state_fn(s0, s1, subspace_mask=subspace_mask)
    return evaluate_connector(
        spec=spec,
        state_fn=state_fn,
        retain_val_loader=retain_val_loader,
        forget_val_loader=forget_val_loader,
        retain_eval_loader=retain_eval_loader,
        forget_eval_loader=forget_eval_loader,
        test_loader=test_loader,
        bn_loader=bn_loader,
        device=device,
        lambdas=lambdas,
        bn_recalc_on=bn_recalc_on,
        bn_batches=bn_batches,
        retain_test_loader=retain_test_loader,
        forget_test_loader=forget_test_loader,
        normalized_full_scale=normalized_full_scale,
    )


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
            # Unlearning endpoints in this script are dense-model checkpoints.
            "enabled": False,
            "method": "dense",
            "sparsity": float(0.0),
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
) -> Dict[str, Any]:
    if len(shadow_ckpts) != len(shadow_seeds):
        raise ValueError("shadow_ckpts and shadow_seeds length mismatch")
    dataset_root = stage_dir / "inputs" / spec.dataset
    dataset_root.mkdir(parents=True, exist_ok=True)

    victim_seed_dir = dataset_root / f"seed{victim_seed}"
    victim_seed_dir.mkdir(parents=True, exist_ok=True)
    victim_ckpt_out = victim_seed_dir / "best_model.pth"
    victim_cfg_out = victim_seed_dir / "config.json"
    shutil.copy2(str(victim_ckpt), str(victim_ckpt_out))
    with open(victim_cfg_out, "w") as f:
        json.dump(_build_mia_config(spec, seed=victim_seed, sparsity=mia_sparsity), f, indent=2)

    shadow_ckpt_outs: List[Path] = []
    shadow_cfg_outs: List[Path] = []
    for ckpt, seed in zip(shadow_ckpts, shadow_seeds):
        seed_dir = dataset_root / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        shadow_ckpt_out = seed_dir / "best_model.pth"
        shadow_cfg_out = seed_dir / "config.json"
        shutil.copy2(str(ckpt), str(shadow_ckpt_out))
        with open(shadow_cfg_out, "w") as f:
            json.dump(_build_mia_config(spec, seed=seed, sparsity=mia_sparsity), f, indent=2)
        shadow_ckpt_outs.append(shadow_ckpt_out)
        shadow_cfg_outs.append(shadow_cfg_out)

    return {
        "victim_ckpt": victim_ckpt_out,
        "victim_config": victim_cfg_out,
        "shadow_ckpts": shadow_ckpt_outs,
        "shadow_configs": shadow_cfg_outs,
    }


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
    result_file: Path,
    dataset: str,
    victim_seed: int,
    shadow_seeds: List[int],
    victim_ckpt: Path,
    victim_config: Path,
    shadow_ckpts: List[Path],
    shadow_configs: List[Path],
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
        "--victim_seed",
        str(victim_seed),
        "--seed",
        str(split_seed),
        "--shadow_seeds",
        *[str(s) for s in shadow_seeds],
        "--victim_ckpt_path",
        str(victim_ckpt),
        "--victim_config_path",
        str(victim_config),
        "--shadow_ckpt_paths",
        *[str(p) for p in shadow_ckpts],
        "--shadow_config_paths",
        *[str(p) for p in shadow_configs],
        "--forward_mode",
        forward_mode,
        "--attacks",
        attacks,
        "--tpr_fprs",
        tpr_fprs,
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


def _extract_mia_metrics(payload: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(payload, dict):
        return out
    res = payload.get("results", {})
    if not isinstance(res, dict):
        return out

    thr_block = None
    threshold_attacks = res.get("threshold_attacks", {})
    if isinstance(threshold_attacks, dict):
        thr_block = threshold_attacks.get("confidence")
    conf_ext = res.get("confidence_extended", {})
    if thr_block is None and isinstance(conf_ext, dict):
        thr_block = conf_ext
    if isinstance(thr_block, dict):
        if "auc" in thr_block:
            out["threshold_auroc"] = float(thr_block["auc"])
        elif "auroc" in thr_block:
            out["threshold_auroc"] = float(thr_block["auroc"])
        if "advantage" in thr_block:
            out["threshold_advantage"] = float(thr_block["advantage"])
        if "tpr_at_1fpr" in thr_block and thr_block["tpr_at_1fpr"] is not None:
            out["threshold_tpr_at_1fpr"] = float(thr_block["tpr_at_1fpr"])

    for attack in ("lira", "nn", "nn_top3", "nn_cls", "samia"):
        a = res.get(attack, {})
        if isinstance(a, dict):
            if "auc" in a:
                out[f"{attack}_auc"] = float(a["auc"])
            if "advantage" in a:
                out[f"{attack}_advantage"] = float(a["advantage"])
            if "tpr_at_1fpr" in a and a["tpr_at_1fpr"] is not None:
                out[f"{attack}_tpr_at_1fpr"] = float(a["tpr_at_1fpr"])
    return out


def _build_retrain_alignment_report(
    endpoint_metrics: Dict[str, Dict[str, float]],
    scratch_baseline_metrics: Optional[Dict[str, Any]],
    mia_results: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if scratch_baseline_metrics is None:
        return None

    baseline_test = float(scratch_baseline_metrics.get("test_acc", 0.0))
    baseline_retain = float(scratch_baseline_metrics.get("retain_acc", 0.0))
    endpoint_rows: Dict[str, Any] = {}
    for seed_key, m in endpoint_metrics.items():
        test_acc = float(m.get("test_acc", 0.0))
        retain_acc = float(m.get("retain_acc", 0.0))
        d_test = test_acc - baseline_test
        d_retain = retain_acc - baseline_retain
        endpoint_rows[seed_key] = {
            "test_acc": test_acc,
            "retain_acc": retain_acc,
            "delta_test_acc_vs_retrain": d_test,
            "delta_retain_acc_vs_retrain": d_retain,
            "abs_delta_test_acc": abs(d_test),
            "abs_delta_retain_acc": abs(d_retain),
            "l1_distance_to_retrain": abs(d_test) + abs(d_retain),
            "l2_distance_to_retrain": (d_test * d_test + d_retain * d_retain) ** 0.5,
        }

    best_seed = None
    best_val = float("inf")
    for seed_key, row in endpoint_rows.items():
        cur = float(row["l2_distance_to_retrain"])
        if cur < best_val:
            best_val = cur
            best_seed = seed_key

    mia_stage_map = mia_results.get("stages", {}) if isinstance(mia_results, dict) else {}
    unlearn_stage = mia_stage_map.get("unlearn", {})
    baseline_stage = mia_stage_map.get("baseline", {})
    unlearn_metrics = _extract_mia_metrics(unlearn_stage.get("result", {})) if isinstance(unlearn_stage, dict) else {}
    baseline_metrics = _extract_mia_metrics(baseline_stage.get("result", {})) if isinstance(baseline_stage, dict) else {}

    mia_deltas: Dict[str, float] = {}
    for k, v in unlearn_metrics.items():
        if k in baseline_metrics:
            mia_deltas[f"{k}_delta_unlearn_minus_retrain"] = float(v - baseline_metrics[k])

    return {
        "retrain_baseline": {
            "test_acc": baseline_test,
            "retain_acc": baseline_retain,
        },
        "endpoint_distances": endpoint_rows,
        "closest_seed_by_l2": best_seed,
        "mia_unlearn": unlearn_metrics,
        "mia_retrain_baseline": baseline_metrics,
        "mia_delta_unlearn_minus_retrain": mia_deltas,
    }


def _choose_curve_point(
    curve: List[Dict[str, float]],
    *,
    selector: str = "composite",
    forget_budget_acc: float = 0.01,
    rt_reference: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    if not curve:
        raise ValueError("curve is empty")

    feasible = [p for p in curve if float(p.get("forget_val_acc", 1.0)) <= float(forget_budget_acc)]
    if not feasible:
        curve_sorted = sorted(curve, key=lambda x: float(x.get("forget_val_acc", 1.0)))
        feasible = curve_sorted[: max(1, len(curve_sorted) // 5)]

    def _score(p: Dict[str, float]) -> float:
        if selector == "retain_val_acc":
            return float(p.get("retain_val_acc", p.get("retain_acc", 0.0)))
        if selector == "retain_acc":
            return float(p.get("retain_acc", 0.0))
        if selector == "test_acc":
            return float(p.get("test_acc", 0.0))
        if selector == "retain_loss":
            return -float(p.get("retain_loss", float("inf")))
        if selector == "test_loss":
            return -float(p.get("test_loss", float("inf")))
        if selector == "composite":
            return composite_unlearning_score(
                retain_val_acc=float(p.get("retain_val_acc", p.get("retain_acc", 0.0))),
                forget_val_acc=float(p.get("forget_val_acc", p.get("forget_acc", 1.0))),
                full_val_acc=None,
                forget_budget_acc=float(forget_budget_acc),
                rt_ref=rt_reference,
            )
        raise ValueError(f"Unsupported selector: {selector}")

    return max(feasible, key=_score)


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
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio split from Df/Dr train indices")

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
    parser.add_argument(
        "--forget-val-budget",
        type=float,
        default=0.01,
        help="Maximum allowed forget_val_acc for checkpoint/curve selection",
    )
    parser.add_argument(
        "--save-tail-k",
        type=int,
        default=0,
        help="Keep last K per-epoch checkpoints in run_dir/tail_seed* (0 disables)",
    )
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
        default="composite",
        choices=["composite", "retain_val_acc", "retain_acc", "test_acc"],
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
        default="composite",
        choices=["composite", "retain_val_acc", "retain_acc", "test_acc", "retain_loss", "test_loss"],
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
    parser.add_argument(
        "--mia-sparsity",
        type=float,
        default=0.0,
        help="Compatibility field written into temporary MIA configs (direct-ckpt mode).",
    )
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
        default="composite",
        choices=["composite", "retain_val_acc", "retain_acc", "test_acc", "retain_loss", "test_loss"],
        help="Metric used to rank top-k connectivity candidates before greedy soup",
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
    if args.val_ratio <= 0.0 or args.val_ratio >= 0.5:
        raise ValueError("--val-ratio must be in (0,0.5)")
    if args.forget_alpha <= 0.0:
        raise ValueError("--forget-alpha must be > 0")
    if args.forget_val_budget < 0.0 or args.forget_val_budget > 1.0:
        raise ValueError("--forget-val-budget must be in [0,1]")
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
    forget_train_idx, forget_val_idx = split_indices_train_val(
        forget_idx, args.val_ratio, args.split_seed + 11
    )
    retain_train_idx, retain_val_idx = split_indices_train_val(
        retain_idx, args.val_ratio, args.split_seed + 17
    )
    if len(forget_train_idx) == 0 or len(retain_train_idx) == 0:
        raise RuntimeError(
            f"Invalid train split after val holdout: forget_train={len(forget_train_idx)}, "
            f"retain_train={len(retain_train_idx)}"
        )
    print(
        f"Df size={len(forget_idx)} (train={len(forget_train_idx)}, val={len(forget_val_idx)}) | "
        f"Dr size={len(retain_idx)} (train={len(retain_train_idx)}, val={len(retain_val_idx)})"
    )

    with open(run_dir / "split_info.json", "w") as f:
        json.dump(
            {
                "df_spec": df_spec,
                "split_seed": args.split_seed,
                "forget_size": len(forget_idx),
                "retain_size": len(retain_idx),
                "val_ratio": float(args.val_ratio),
                "forget_train_size": len(forget_train_idx),
                "forget_val_size": len(forget_val_idx),
                "retain_train_size": len(retain_train_idx),
                "retain_val_size": len(retain_val_idx),
            },
            f,
            indent=2,
        )

    retain_train_loader_a = make_subset_loader(
        train_aug, retain_train_idx, spec.batch_size, spec.workers, shuffle=True, seed=args.seed_a
    )
    retain_train_loader_b = make_subset_loader(
        train_aug, retain_train_idx, spec.batch_size, spec.workers, shuffle=True, seed=args.seed_b
    )
    forget_train_loader_a = make_subset_loader(
        train_aug, forget_train_idx, spec.batch_size, spec.workers, shuffle=True, seed=args.seed_a + 100
    )
    forget_train_loader_b = make_subset_loader(
        train_aug, forget_train_idx, spec.batch_size, spec.workers, shuffle=True, seed=args.seed_b + 100
    )
    retain_val_loader = make_subset_loader(
        train_eval, retain_val_idx, spec.batch_size, spec.workers, shuffle=False, seed=0
    )
    forget_val_loader = make_subset_loader(
        train_eval, forget_val_idx, spec.batch_size, spec.workers, shuffle=False, seed=0
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
        train_eval, retain_train_idx, spec.batch_size, spec.workers, shuffle=True, seed=args.split_seed + 1000
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
    scratch_baseline_state: Optional[Dict[str, torch.Tensor]] = None

    if args.scratch_retrain_ckpt:
        scratch_baseline_ckpt = Path(args.scratch_retrain_ckpt)
        if not scratch_baseline_ckpt.exists():
            raise FileNotFoundError(f"scratch retrain checkpoint not found: {scratch_baseline_ckpt}")
        _, scratch_state = load_checkpoint(scratch_baseline_ckpt)
        scratch_baseline_state = scratch_state
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
            scratch_baseline_state = scratch_state
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
                retain_train_idx,
                spec.batch_size,
                spec.workers,
                shuffle=True,
                seed=args.scratch_retrain_seed,
            )
            scratch_payload = train_scratch_retrain_baseline(
                spec=spec,
                retain_train_loader=scratch_retain_train_loader,
                retain_val_loader=retain_val_loader,
                forget_val_loader=forget_val_loader,
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
                ckpt_select="composite",
                forget_budget_acc=args.forget_val_budget,
            )
            torch.save(scratch_payload, str(scratch_baseline_ckpt))
            scratch_baseline_state = _strip_module_prefix(scratch_payload["state_dict"])
            scratch_baseline_metrics = evaluate_endpoint_state(
                spec=spec,
                state=scratch_baseline_state,
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

    rt_reference: Optional[Dict[str, float]] = None
    if scratch_baseline_state is not None:
        rt_model, _ = build_dense_model(spec)
        rt_model = rt_model.to(device)
        rt_model.load_state_dict(scratch_baseline_state, strict=True)
        rt_model.eval()
        rt_retain_val = evaluate(rt_model, retain_val_loader, device)
        rt_reference = {
            "retain_val_acc": float(rt_retain_val["acc"]),
        }

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
            retain_val_loader=retain_val_loader,
            forget_val_loader=forget_val_loader,
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
            forget_budget_acc=args.forget_val_budget,
            rt_reference=rt_reference,
            save_tail_k=args.save_tail_k,
            tail_dir=run_dir / f"tail_seed{args.seed_a}",
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
            retain_val_loader=retain_val_loader,
            forget_val_loader=forget_val_loader,
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
            forget_budget_acc=args.forget_val_budget,
            rt_reference=rt_reference,
            save_tail_k=args.save_tail_k,
            tail_dir=run_dir / f"tail_seed{args.seed_b}",
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
            prepared = _prepare_mia_workspace(
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
                result_file=result_file,
                dataset=spec.dataset,
                victim_seed=mia_victim_seed,
                shadow_seeds=shadow_seed_list,
                victim_ckpt=prepared["victim_ckpt"],
                victim_config=prepared["victim_config"],
                shadow_ckpts=prepared["shadow_ckpts"],
                shadow_configs=prepared["shadow_configs"],
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
    # Baseline MIA: explicit --mia-baseline-ckpt, or auto-fallback to scratch-retrain ckpt.
    baseline_victim: Optional[Path] = None
    if args.mia_baseline_ckpt:
        baseline_victim = Path(args.mia_baseline_ckpt).expanduser().resolve()
    elif scratch_baseline_ckpt is not None:
        baseline_victim = scratch_baseline_ckpt
    if args.run_mia and "baseline" in mia_selected_stages and baseline_victim is not None:
        baseline_shadows = baseline_shadow_ckpts if baseline_shadow_ckpts else [ckpt_b]
        _run_stage_mia("baseline", baseline_victim, baseline_shadows)

    retrain_alignment = _build_retrain_alignment_report(
        endpoint_metrics=endpoint_metrics,
        scratch_baseline_metrics=scratch_baseline_metrics,
        mia_results=mia_results,
    )

    if args.step1_only:
        if args.run_mia and "baseline" in mia_selected_stages and baseline_victim is None:
            msg = "baseline stage requested in --mia-stages but no baseline ckpt available (set --mia-baseline-ckpt or enable scratch baseline)"
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
            "retrain_alignment": retrain_alignment,
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
        retain_val_loader=retain_val_loader,
        forget_val_loader=forget_val_loader,
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
        best_p2 = _choose_curve_point(
            step2["curve"],
            selector=args.mia_select_metric,
            forget_budget_acc=args.forget_val_budget,
            rt_reference=rt_reference,
        )
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
        retain_val_loader=retain_val_loader,
        forget_val_loader=forget_val_loader,
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
        best_p3 = _choose_curve_point(
            step3["curve"],
            selector=args.mia_select_metric,
            forget_budget_acc=args.forget_val_budget,
            rt_reference=rt_reference,
        )
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
        print("SWA/Soup Merge")
        print("=" * 80)

        # Real SWA from endpoint local training trajectories (tail checkpoints).
        for seed_label, endpoint_payload in [(f"seed{args.seed_a}", ep_a), (f"seed{args.seed_b}", ep_b)]:
            tail_paths = []
            if isinstance(endpoint_payload, dict):
                tail_paths = [Path(p) for p in endpoint_payload.get("tail_checkpoints", []) if Path(p).exists()]
            if len(tail_paths) < 2:
                continue
            try:
                tail_swa_state = swa_from_checkpoints(
                    spec=spec,
                    ckpt_paths=tail_paths,
                    bn_loader=bn_loader,
                    device=device,
                    bn_recalc_on=args.bn_recalc,
                    bn_batches=args.bn_batches,
                )
                tail_swa_metrics = evaluate_endpoint_state(
                    spec=spec,
                    state=tail_swa_state,
                    retain_eval_loader=retain_eval_loader,
                    forget_eval_loader=forget_eval_loader,
                    test_loader=test_loader,
                    device=device,
                    retain_test_loader=retain_test_loader,
                    forget_test_loader=forget_test_loader,
                    normalized_full_scale=normalized_full_scale,
                )
                ckpt_path = run_dir / f"{seed_label}_tail_swa.pth"
                torch.save(
                    {
                        "state_dict": tail_swa_state,
                        "source": "tail_swa",
                        "seed": seed_label,
                        "tail_checkpoints": [str(p) for p in tail_paths],
                    },
                    str(ckpt_path),
                )
                swa_results.append(
                    {
                        "source": "tail_swa",
                        "seed": seed_label,
                        "tail_checkpoints": [str(p) for p in tail_paths],
                        "ckpt": str(ckpt_path),
                        "metrics": tail_swa_metrics,
                    }
                )
                print(
                    f"[tail-swa:{seed_label}] test_acc={tail_swa_metrics['test_acc']:.4f} "
                    f"retain_acc={tail_swa_metrics['retain_acc']:.4f} forget_acc={tail_swa_metrics['forget_acc']:.4f} "
                    f"(k={len(tail_paths)})"
                )
            except Exception as e:
                print(f"[tail-swa:{seed_label}] skipped: {e}")

        # Soup over connectivity points (validated by val composite score).
        sources: List[Tuple[str, Dict[str, Any], Optional[Dict[str, torch.Tensor]]]] = []
        if args.swa_source in {"step2", "both"}:
            sources.append(("step2", step2, None))
        if args.swa_source in {"step3", "both"}:
            sources.append(("step3", step3, mask))

        for src_name, src_result, src_mask in sources:
            points = [p for p in src_result["curve"] if args.swa_t_min <= float(p["t"]) <= args.swa_t_max]
            if not points:
                points = list(src_result["curve"])
            if not points:
                continue

            def _rank_key(p: Dict[str, Any]) -> float:
                if args.swa_select_metric in {"retain_loss", "test_loss"}:
                    return -float(p.get(args.swa_select_metric, float("inf")))
                if args.swa_select_metric == "retain_val_acc":
                    return float(p.get("retain_val_acc", p.get("retain_acc", 0.0)))
                if args.swa_select_metric == "composite":
                    return composite_unlearning_score(
                        retain_val_acc=float(p.get("retain_val_acc", p.get("retain_acc", 0.0))),
                        forget_val_acc=float(p.get("forget_val_acc", p.get("forget_acc", 1.0))),
                        full_val_acc=None,
                        forget_budget_acc=float(args.forget_val_budget),
                        rt_ref=rt_reference,
                    )
                return float(p.get(args.swa_select_metric, p.get("retain_acc", 0.0)))

            points = sorted(points, key=_rank_key, reverse=True)
            k = len(points) if args.swa_topk <= 0 else min(args.swa_topk, len(points))
            points = points[:k]

            candidates: List[Dict[str, Any]] = []
            for p in points:
                t = float(p["t"])
                state = interpolate_state(s_a, s_b, t, subspace_mask=src_mask)
                candidates.append(
                    {
                        "name": f"t{t:.4f}",
                        "state": state,
                        "val_score": composite_unlearning_score(
                            retain_val_acc=float(p.get("retain_val_acc", p.get("retain_acc", 0.0))),
                            forget_val_acc=float(p.get("forget_val_acc", p.get("forget_acc", 1.0))),
                            full_val_acc=float(p.get("retain_acc", 0.0)),
                            forget_budget_acc=float(args.forget_val_budget),
                            rt_ref=rt_reference,
                        ),
                    }
                )

            soup_eval_model, _ = build_dense_model(spec)
            soup_eval_model = soup_eval_model.to(device)

            def _eval_state_fn(state: Dict[str, torch.Tensor]) -> float:
                soup_eval_model.load_state_dict(state, strict=True)
                if args.bn_recalc:
                    bn_recalibrate(soup_eval_model, bn_loader, device=device, max_batches=args.bn_batches)
                rv_loader = retain_val_loader if retain_val_loader is not None else retain_eval_loader
                fv_loader = forget_val_loader if forget_val_loader is not None else forget_eval_loader
                retain_val_stats = evaluate(soup_eval_model, rv_loader, device)
                forget_val_stats = evaluate(soup_eval_model, fv_loader, device)
                return composite_unlearning_score(
                    retain_val_acc=float(retain_val_stats["acc"]),
                    forget_val_acc=float(forget_val_stats["acc"]),
                    full_val_acc=None,
                    forget_budget_acc=float(args.forget_val_budget),
                    rt_ref=rt_reference,
                )

            soup_state_raw, selected_names, soup_val_score = greedy_soup(candidates, eval_state_fn=_eval_state_fn)
            soup_state = recalibrate_state_bn(
                spec=spec,
                state=soup_state_raw,
                bn_loader=bn_loader,
                device=device,
                bn_recalc_on=args.bn_recalc,
                bn_batches=args.bn_batches,
            )
            soup_metrics = evaluate_endpoint_state(
                spec=spec,
                state=soup_state,
                retain_eval_loader=retain_eval_loader,
                forget_eval_loader=forget_eval_loader,
                test_loader=test_loader,
                device=device,
                retain_test_loader=retain_test_loader,
                forget_test_loader=forget_test_loader,
                normalized_full_scale=normalized_full_scale,
            )
            ckpt_path = run_dir / f"{src_name}_soup_merge.pth"
            torch.save(
                {
                    "state_dict": soup_state,
                    "source": src_name,
                    "selected_candidates": selected_names,
                    "soup_val_score": float(soup_val_score),
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
                "method": "greedy_soup",
                "select_metric": args.swa_select_metric,
                "topk": args.swa_topk,
                "t_min": args.swa_t_min,
                "t_max": args.swa_t_max,
                "selected_candidates": selected_names,
                "soup_val_score": float(soup_val_score),
                "ckpt": str(ckpt_path),
                "metrics": soup_metrics,
            }
            swa_results.append(result_row)
            print(
                f"[soup:{src_name}] test_acc={soup_metrics['test_acc']:.4f} "
                f"retain_acc={soup_metrics['retain_acc']:.4f} forget_acc={soup_metrics['forget_acc']:.4f} "
                f"(cands={len(selected_names)})"
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

    if args.run_mia and "baseline" in mia_selected_stages and baseline_victim is None:
        msg = "baseline stage requested in --mia-stages but no baseline ckpt available (set --mia-baseline-ckpt or enable scratch baseline)"
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
            "forget_val_budget": args.forget_val_budget,
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
            "val_ratio": args.val_ratio,
            "save_tail_k": args.save_tail_k,
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
        "retrain_alignment": retrain_alignment,
        "swa_merge": {
            "enabled": bool(args.swa_merge),
            "methods": ["tail_swa", "greedy_soup"],
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
