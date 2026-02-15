#!/usr/bin/env python3
"""
Classification unlearning utilities (Df ascent + Dr descent).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset


DF_PROFILES: Dict[str, Dict[str, Any]] = {
    "df1": {
        "type": "class",
        "forget_classes": [0],
        "description": "forget class 0",
    },
    "df2": {
        "type": "class",
        "forget_classes": [0, 1],
        "description": "forget classes 0,1",
    },
    "df3": {
        "type": "class",
        "forget_classes": [0, 1, 2],
        "description": "forget classes 0,1,2",
    },
}


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_df_spec(args, num_classes: int) -> Dict[str, Any]:
    if args.df_mode == "profile":
        if args.df_profile not in DF_PROFILES:
            raise ValueError(f"Unknown df-profile: {args.df_profile}")
        spec = dict(DF_PROFILES[args.df_profile])
        spec["name"] = args.df_profile
        classes = [c for c in spec["forget_classes"] if 0 <= c < num_classes]
        if not classes:
            raise ValueError(f"df-profile {args.df_profile} does not match dataset classes")
        spec["forget_classes"] = classes
        return spec

    if args.df_mode == "class":
        if not args.forget_classes:
            raise ValueError("--forget-classes is required for --df-mode class")
        classes = [int(x.strip()) for x in args.forget_classes.split(",") if x.strip()]
        classes = [c for c in classes if 0 <= c < num_classes]
        if not classes:
            raise ValueError("No valid forget classes after filtering")
        return {
            "name": "custom_class",
            "type": "class",
            "forget_classes": sorted(set(classes)),
            "description": f"forget classes {sorted(set(classes))}",
        }

    if args.df_mode == "random":
        if args.forget_ratio <= 0.0 or args.forget_ratio >= 1.0:
            raise ValueError("--forget-ratio must be in (0,1)")
        return {
            "name": "custom_random",
            "type": "random",
            "forget_ratio": args.forget_ratio,
            "description": f"random forget ratio {args.forget_ratio:.4f}",
        }

    raise ValueError(f"Unsupported df-mode: {args.df_mode}")


def build_forget_retain_indices(
    targets: List[int], df_spec: Dict[str, Any], split_seed: int
) -> Tuple[List[int], List[int]]:
    y = [int(v) for v in targets]
    n = len(y)

    if df_spec["type"] == "class":
        forget_classes = set(df_spec["forget_classes"])
        forget_idx = [i for i, label in enumerate(y) if label in forget_classes]
        retain_idx = [i for i, label in enumerate(y) if label not in forget_classes]
        return forget_idx, retain_idx

    if df_spec["type"] == "random":
        g = random.Random(split_seed)
        perm = list(range(n))
        g.shuffle(perm)
        f = int(round(n * float(df_spec["forget_ratio"])))
        forget_idx = sorted(perm[:f])
        retain_idx = sorted(perm[f:])
        return forget_idx, retain_idx

    raise ValueError(f"Unknown df type: {df_spec['type']}")


def make_subset_loader(
    dataset: Any,
    indices: List[int],
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        generator=gen,
    )


def _next_batch(it, loader):
    try:
        batch = next(it)
    except StopIteration:
        it = iter(loader)
        batch = next(it)
    return batch, it


def _unpack_xy(batch):
    if not isinstance(batch, (list, tuple)) or len(batch) < 2:
        raise ValueError("Batch must be tuple/list with (inputs, targets, ...)")
    return batch[0], batch[1]


def train_unlearning_endpoint_ascent(
    model: nn.Module,
    base_state: Dict[str, torch.Tensor],
    seed: int,
    retain_train_loader: DataLoader,
    forget_train_loader: DataLoader,
    retain_eval_loader: DataLoader,
    forget_eval_loader: DataLoader,
    test_loader: DataLoader,
    evaluate_fn: Callable[[nn.Module, DataLoader, torch.device], Dict[str, float]],
    out_path: Path,
    device: torch.device,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    nesterov: bool,
    forget_alpha: float,
    retain_weight: float,
    grad_clip: float,
    model_config: Dict[str, Any],
) -> Dict[str, Any]:
    if forget_alpha <= 0:
        raise ValueError("forget_alpha must be > 0 for ascent-based unlearning")

    set_seed(seed)
    model.load_state_dict(base_state, strict=True)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
    )

    best_retain_acc = -1.0
    best_state = None
    best_epoch = -1
    best_metrics: Dict[str, float] = {}
    history: List[Dict[str, Any]] = []

    for epoch in range(epochs):
        model.train()
        retain_it = iter(retain_train_loader)
        forget_it = iter(forget_train_loader)
        steps = max(1, max(len(retain_train_loader), len(forget_train_loader)))

        running_retain = 0.0
        running_forget = 0.0
        seen_r = 0
        seen_f = 0

        for _ in range(steps):
            batch_r, retain_it = _next_batch(retain_it, retain_train_loader)
            batch_f, forget_it = _next_batch(forget_it, forget_train_loader)

            xr, yr = _unpack_xy(batch_r)
            xf, yf = _unpack_xy(batch_f)
            xr = xr.to(device, non_blocking=True)
            yr = yr.to(device, non_blocking=True)
            xf = xf.to(device, non_blocking=True)
            yf = yf.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits_r = model(xr)
            logits_f = model(xf)
            loss_r = loss_fn(logits_r, yr)
            loss_f = loss_fn(logits_f, yf)
            # Ascent on Df is implemented by subtracting Df loss.
            loss = retain_weight * loss_r - forget_alpha * loss_f
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            bsz_r = xr.size(0)
            bsz_f = xf.size(0)
            seen_r += bsz_r
            seen_f += bsz_f
            running_retain += float(loss_r.item()) * bsz_r
            running_forget += float(loss_f.item()) * bsz_f

        scheduler.step()

        retain_stats = evaluate_fn(model, retain_eval_loader, device)
        forget_stats = evaluate_fn(model, forget_eval_loader, device)
        test_stats = evaluate_fn(model, test_loader, device)
        train_retain = running_retain / max(seen_r, 1)
        train_forget = running_forget / max(seen_f, 1)
        train_total = retain_weight * train_retain - forget_alpha * train_forget

        row = {
            "epoch": epoch,
            "train_total_loss": train_total,
            "train_retain_loss": train_retain,
            "train_forget_loss": train_forget,
            "train_retain_samples": int(seen_r),
            "train_forget_samples": int(seen_f),
            "retain_loss": retain_stats["loss"],
            "retain_acc": retain_stats["acc"],
            "forget_loss": forget_stats["loss"],
            "forget_acc": forget_stats["acc"],
            "test_loss": test_stats["loss"],
            "test_acc": test_stats["acc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"[seed={seed}] epoch {epoch + 1:03d}/{epochs:03d} "
            f"train_total={train_total:.4f} retain_loss={retain_stats['loss']:.4f} "
            f"forget_loss={forget_stats['loss']:.4f} retain_acc={retain_stats['acc']:.4f} "
            f"forget_acc={forget_stats['acc']:.4f} test_acc={test_stats['acc']:.4f}"
        )

        if retain_stats["acc"] > best_retain_acc:
            best_retain_acc = retain_stats["acc"]
            best_epoch = epoch
            best_metrics = {
                "retain_loss": float(retain_stats["loss"]),
                "retain_acc": float(retain_stats["acc"]),
                "forget_loss": float(forget_stats["loss"]),
                "forget_acc": float(forget_stats["acc"]),
                "test_loss": float(test_stats["loss"]),
                "test_acc": float(test_stats["acc"]),
            }
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": best_state,
        "seed": seed,
        "train_history": history,
        "best_epoch": int(best_epoch),
        "best_metrics": best_metrics,
        "final_metrics": history[-1] if history else {},
        "unlearning": {
            "objective": "retain_descent_with_forget_ascent",
            "retain_weight": retain_weight,
            "forget_alpha": forget_alpha,
            "grad_clip": grad_clip,
        },
        "config": model_config,
    }
    torch.save(payload, str(out_path))
    return payload
