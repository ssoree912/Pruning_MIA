#!/usr/bin/env python3
"""
Dense checkpoint 기반 unlearning + connectivity (Step 1/2/3) 실행 스크립트.

운영 원칙:
- MIA는 이 스크립트에서 수행하지 않음
- Step 2: 선형 경로 barrier 진단
- Step 3: 고정 mask(subspace) 경로 barrier 진단
"""

import argparse
import json
import os
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
    test_loader: DataLoader,
    bn_loader: DataLoader,
    device: torch.device,
    lambdas: List[float],
    bn_recalc_on: bool,
    bn_batches: int,
    subspace_mask: Optional[Dict[str, torch.Tensor]] = None,
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
        test_stats = evaluate(model, test_loader, device)
        point = {
            "t": float(t),
            "retain_loss": float(retain_stats["loss"]),
            "retain_acc": float(retain_stats["acc"]),
            "test_loss": float(test_stats["loss"]),
            "test_acc": float(test_stats["acc"]),
        }
        curve.append(point)
        print(
            f"[interp {i:03d}/{len(lambdas):03d}] t={t:.3f} "
            f"retain_loss={point['retain_loss']:.4f} retain_acc={point['retain_acc']:.4f} "
            f"test_acc={point['test_acc']:.4f}"
        )

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
    }


def evaluate_endpoint_state(
    spec: ModelSpec,
    state: Dict[str, torch.Tensor],
    retain_eval_loader: DataLoader,
    forget_eval_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model, _ = build_dense_model(spec)
    model = model.to(device)
    model.load_state_dict(state, strict=True)

    retain_stats = evaluate(model, retain_eval_loader, device)
    forget_stats = evaluate(model, forget_eval_loader, device)
    test_stats = evaluate(model, test_loader, device)
    return {
        "retain_loss": float(retain_stats["loss"]),
        "retain_acc": float(retain_stats["acc"]),
        "forget_loss": float(forget_stats["loss"]),
        "forget_acc": float(forget_stats["acc"]),
        "test_loss": float(test_stats["loss"]),
        "test_acc": float(test_stats["acc"]),
    }


def make_lambdas(num: int) -> List[float]:
    if num < 2:
        return [0.0, 1.0]
    return [i / (num - 1) for i in range(num)]


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

    parser.add_argument("--seed-a", type=int, default=43, help="Unlearning seed A")
    parser.add_argument("--seed-b", type=int, default=44, help="Unlearning seed B")
    parser.add_argument("--split-seed", type=int, default=7, help="Df/Dr split seed")

    parser.add_argument("--df-mode", type=str, default="profile", choices=["profile", "class", "random"])
    parser.add_argument("--df-profile", type=str, default="df1", choices=["df1", "df2", "df3"])
    parser.add_argument("--forget-classes", type=str, default=None, help="Comma-separated class ids for df-mode=class")
    parser.add_argument("--forget-ratio", type=float, default=0.1, help="Forget ratio for df-mode=random")

    parser.add_argument("--unlearn-epochs", type=int, default=20)
    parser.add_argument("--unlearn-lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--forget-alpha", type=float, default=0.05, help="Df ascent strength in L=Dr-alpha*Df")
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

    parser.add_argument("--lambdas", type=int, default=21, help="Interpolation points count")
    bn_group = parser.add_mutually_exclusive_group()
    bn_group.add_argument("--bn-recalc", dest="bn_recalc", action="store_true", help="Enable BN stats recalibration at each t")
    bn_group.add_argument("--no-bn-recalc", dest="bn_recalc", action="store_false", help="Disable BN stats recalibration at each t")
    parser.set_defaults(bn_recalc=True)
    parser.add_argument("--bn-batches", type=int, default=200, help="Max Dr batches for BN recalibration")

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
        raise ValueError("--forget-alpha must be > 0 for ascent-based unlearning")
    if args.retrain_epochs < 0:
        raise ValueError("--retrain-epochs must be >= 0")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_a = run_dir / f"unlearn_seed{args.seed_a}.pth"
    ckpt_b = run_dir / f"unlearn_seed{args.seed_b}.pth"

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
            retain_weight=args.retain_weight,
            grad_clip=args.grad_clip,
            retrain_epochs=args.retrain_epochs,
            retrain_lr=args.retrain_lr,
            retrain_momentum=args.retrain_momentum,
            retrain_weight_decay=args.retrain_weight_decay,
            retrain_nesterov=args.retrain_nesterov,
            ckpt_select=args.ckpt_select,
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
            retain_weight=args.retain_weight,
            grad_clip=args.grad_clip,
            retrain_epochs=args.retrain_epochs,
            retrain_lr=args.retrain_lr,
            retrain_momentum=args.retrain_momentum,
            retrain_weight_decay=args.retrain_weight_decay,
            retrain_nesterov=args.retrain_nesterov,
            ckpt_select=args.ckpt_select,
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
        ),
        f"seed{args.seed_b}": evaluate_endpoint_state(
            spec=spec,
            state=s_b,
            retain_eval_loader=retain_eval_loader,
            forget_eval_loader=forget_eval_loader,
            test_loader=test_loader,
            device=device,
        ),
    }
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

    if args.step1_only:
        step1_summary = {
            "dense_ckpt": str(dense_ckpt),
            "run_dir": str(run_dir),
            "device": str(device),
            "model_spec": spec.__dict__,
            "df_spec": df_spec,
            "unlearning_objective": {
                "type": "retain_descent_with_forget_ascent",
                "retain_weight": args.retain_weight,
                "forget_alpha": args.forget_alpha,
                "grad_clip": args.grad_clip,
                "ckpt_select": args.ckpt_select,
            },
            "training_schedule": {
                "unlearn_epochs": args.unlearn_epochs,
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
    print("Step 2: Linear interpolation barrier (MIA 없음)")
    print("=" * 80)
    step2 = run_connectivity(
        spec=spec,
        s0=s_a,
        s1=s_b,
        retain_eval_loader=retain_eval_loader,
        test_loader=test_loader,
        bn_loader=bn_loader,
        device=device,
        lambdas=lambdas,
        bn_recalc_on=args.bn_recalc,
        bn_batches=args.bn_batches,
        subspace_mask=None,
    )
    with open(run_dir / "step2_linear.json", "w") as f:
        json.dump(step2, f, indent=2)
    print(
        f"[step2] retain_loss_barrier={step2['retain_loss_barrier']:.6f}, "
        f"retain_acc_drop_pp={step2['retain_acc_drop_pp']:.4f}"
    )

    print("\n" + "=" * 80)
    print("Step 3: Mask 고정 subspace interpolation barrier (MIA 없음)")
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
        test_loader=test_loader,
        bn_loader=bn_loader,
        device=device,
        lambdas=lambdas,
        bn_recalc_on=args.bn_recalc,
        bn_batches=args.bn_batches,
        subspace_mask=mask,
    )
    with open(run_dir / "step3_masked_linear.json", "w") as f:
        json.dump(step3, f, indent=2)
    print(
        f"[step3] retain_loss_barrier={step3['retain_loss_barrier']:.6f}, "
        f"retain_acc_drop_pp={step3['retain_acc_drop_pp']:.4f}"
    )

    def is_mia_candidate(result: Dict[str, Any]) -> bool:
        return (
            result["retain_acc_drop_pp"] <= args.max_acc_drop_pp
            or result["retain_loss_barrier"] <= args.max_loss_barrier
        )

    summary = {
        "dense_ckpt": str(dense_ckpt),
        "run_dir": str(run_dir),
        "device": str(device),
        "model_spec": spec.__dict__,
        "df_spec": df_spec,
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
            "type": "retain_descent_with_forget_ascent",
            "retain_weight": args.retain_weight,
            "forget_alpha": args.forget_alpha,
            "grad_clip": args.grad_clip,
            "ckpt_select": args.ckpt_select,
        },
        "training_schedule": {
            "unlearn_epochs": args.unlearn_epochs,
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


if __name__ == "__main__":
    main()
