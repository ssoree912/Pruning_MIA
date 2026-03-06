#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

# Reuse the repo's existing training/eval helpers.
from train import (  # type: ignore
    ModelSpec,
    _get_dataset_targets,
    build_cifar_datasets,
    build_dense_model,
    evaluate,
    infer_model_spec,
    load_checkpoint,
    make_lambdas,
    recalibrate_state_bn,
)
from unlearning.utils import (  # type: ignore
    build_forget_retain_indices,
    composite_unlearning_score,
    make_subset_loader,
    resolve_df_spec,
    split_indices_train_val,
)

from connectivity.functional_modes import (
    CurveTrainResult,
    SimplexTrainResult,
    average_state_dicts,
    bezier_state_dict,
    linear_state_dict,
    simplex_state_dict,
    train_bezier_control,
    train_simplex_vertex,
)
from connectivity.git_rebasin_resnet import align_state_dict_to_reference


@torch.no_grad()
def evaluate_state(
    spec: ModelSpec,
    state: Dict[str, torch.Tensor],
    device: torch.device,
    bn_loader,
    bn_recalc_on: bool,
    bn_batches: int,
    *,
    retain_val_loader=None,
    forget_val_loader=None,
    retain_eval_loader=None,
    forget_eval_loader=None,
    test_loader=None,
    retain_test_loader=None,
    forget_test_loader=None,
    normalized_full_scale: Optional[float] = None,
) -> Dict[str, float]:
    state_eval = recalibrate_state_bn(
        spec=spec,
        state=state,
        bn_loader=bn_loader,
        device=device,
        bn_recalc_on=bn_recalc_on,
        bn_batches=bn_batches,
    )

    model, _ = build_dense_model(spec)
    model = model.to(device)
    model.load_state_dict(state_eval, strict=True)

    out: Dict[str, float] = {}
    if retain_val_loader is not None:
        rv = evaluate(model, retain_val_loader, device)
        out["retain_val_loss"] = float(rv["loss"])
        out["retain_val_acc"] = float(rv["acc"])
    if forget_val_loader is not None:
        fv = evaluate(model, forget_val_loader, device)
        out["forget_val_loss"] = float(fv["loss"])
        out["forget_val_acc"] = float(fv["acc"])
    if retain_eval_loader is not None:
        re = evaluate(model, retain_eval_loader, device)
        out["retain_loss"] = float(re["loss"])
        out["retain_acc"] = float(re["acc"])
    if forget_eval_loader is not None:
        fe = evaluate(model, forget_eval_loader, device)
        out["forget_loss"] = float(fe["loss"])
        out["forget_acc"] = float(fe["acc"])
    if test_loader is not None:
        te = evaluate(model, test_loader, device)
        out["test_loss"] = float(te["loss"])
        out["test_acc"] = float(te["acc"])
    if retain_test_loader is not None:
        rtt = evaluate(model, retain_test_loader, device)
        out["retain_test_loss"] = float(rtt["loss"])
        out["retain_test_acc"] = float(rtt["acc"])
    if forget_test_loader is not None:
        ftt = evaluate(model, forget_test_loader, device)
        out["forget_test_loss"] = float(ftt["loss"])
        out["forget_test_acc"] = float(ftt["acc"])
    if normalized_full_scale is not None and normalized_full_scale > 0.0 and "test_acc" in out:
        out["normalized_full_scale"] = float(normalized_full_scale)
        out["normalized_full_test_acc"] = float(out["test_acc"] / normalized_full_scale)
        out["normalized_full"] = out["normalized_full_test_acc"]
    return out


def build_rt_reference(
    spec: ModelSpec,
    scratch_ckpt: Optional[Path],
    device: torch.device,
    bn_loader,
    bn_recalc_on: bool,
    bn_batches: int,
    retain_val_loader,
) -> Optional[Dict[str, float]]:
    if scratch_ckpt is None or not scratch_ckpt.exists():
        return None
    _, scratch_state = load_checkpoint(scratch_ckpt)
    metrics = evaluate_state(
        spec=spec,
        state=scratch_state,
        device=device,
        bn_loader=bn_loader,
        bn_recalc_on=bn_recalc_on,
        bn_batches=bn_batches,
        retain_val_loader=retain_val_loader,
    )
    return {"retain_val_acc": float(metrics.get("retain_val_acc", 0.0))}


def point_composite_score(
    point: Dict[str, float],
    forget_budget_acc: float,
    rt_reference: Optional[Dict[str, float]],
) -> float:
    return composite_unlearning_score(
        retain_val_acc=float(point.get("retain_val_acc", point.get("retain_acc", 0.0))),
        forget_val_acc=float(point.get("forget_val_acc", point.get("forget_acc", 1.0))),
        full_val_acc=None,
        forget_budget_acc=float(forget_budget_acc),
        rt_ref=rt_reference,
    )


@torch.no_grad()
def evaluate_curve(
    *,
    spec: ModelSpec,
    state_fn,
    lambdas: List[float],
    device: torch.device,
    bn_loader,
    bn_recalc_on: bool,
    bn_batches: int,
    retain_val_loader,
    forget_val_loader,
    retain_eval_loader,
    forget_eval_loader,
    test_loader,
    retain_test_loader=None,
    forget_test_loader=None,
    normalized_full_scale: Optional[float] = None,
    forget_budget_acc: float = 0.01,
    rt_reference: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    curve: List[Dict[str, float]] = []
    for idx, t in enumerate(lambdas, start=1):
        state = state_fn(float(t))
        point = {"t": float(t)}
        point.update(
            evaluate_state(
                spec=spec,
                state=state,
                device=device,
                bn_loader=bn_loader,
                bn_recalc_on=bn_recalc_on,
                bn_batches=bn_batches,
                retain_val_loader=retain_val_loader,
                forget_val_loader=forget_val_loader,
                retain_eval_loader=retain_eval_loader,
                forget_eval_loader=forget_eval_loader,
                test_loader=test_loader,
                retain_test_loader=retain_test_loader,
                forget_test_loader=forget_test_loader,
                normalized_full_scale=normalized_full_scale,
            )
        )
        point["composite"] = float(point_composite_score(point, forget_budget_acc, rt_reference))
        curve.append(point)
        print(
            f"[curve {idx:03d}/{len(lambdas):03d}] t={t:.3f} "
            f"retain_val_acc={point.get('retain_val_acc', 0.0):.4f} "
            f"forget_val_acc={point.get('forget_val_acc', 0.0):.4f} "
            f"test_acc={point.get('test_acc', 0.0):.4f} composite={point['composite']:.6f}"
        )

    barrier_key = "retain_val_loss"
    drop_key = "retain_val_acc"
    l0 = float(curve[0][barrier_key])
    l1 = float(curve[-1][barrier_key])
    lmax = max(float(p[barrier_key]) for p in curve)
    barrier = lmax - max(l0, l1)
    acc_ref = max(float(curve[0][drop_key]), float(curve[-1][drop_key]))
    acc_min = min(float(p[drop_key]) for p in curve)
    acc_drop_pp = (acc_ref - acc_min) * 100.0

    feasible = [p for p in curve if float(p.get("forget_val_acc", 1.0)) <= float(forget_budget_acc)]
    if not feasible:
        feasible = sorted(curve, key=lambda p: float(p.get("forget_val_acc", 1.0)))[: max(1, len(curve) // 5)]
    best_by_composite = max(feasible, key=lambda p: float(p["composite"]))

    return {
        "curve": curve,
        "retain_val_loss_endpoint0": l0,
        "retain_val_loss_endpoint1": l1,
        "retain_val_loss_max": lmax,
        "retain_val_loss_barrier": barrier,
        "retain_val_acc_drop_pp": acc_drop_pp,
        "best_by_composite": best_by_composite,
        "best_by_test_acc": max(curve, key=lambda p: float(p.get("test_acc", 0.0))),
    }


@torch.no_grad()
def sample_simplex_grid(
    resolution: int,
) -> List[torch.Tensor]:
    if resolution < 2:
        return [torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])]
    out: List[torch.Tensor] = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j
            out.append(torch.tensor([i, j, k], dtype=torch.float32) / float(resolution))
    return out


def greedy_soup(
    candidates: List[Dict[str, Any]],
    *,
    spec: ModelSpec,
    device: torch.device,
    bn_loader,
    bn_recalc_on: bool,
    bn_batches: int,
    retain_val_loader,
    forget_val_loader,
    retain_eval_loader,
    forget_eval_loader,
    test_loader,
    retain_test_loader=None,
    forget_test_loader=None,
    normalized_full_scale: Optional[float] = None,
    forget_budget_acc: float = 0.01,
    rt_reference: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, torch.Tensor], List[str], float, Dict[str, float]]:
    if not candidates:
        raise ValueError("candidates must not be empty")
    ordered = sorted(candidates, key=lambda c: float(c["val_score"]), reverse=True)
    chosen = [ordered[0]]
    soup_state = ordered[0]["state"]
    soup_metrics = evaluate_state(
        spec=spec,
        state=soup_state,
        device=device,
        bn_loader=bn_loader,
        bn_recalc_on=bn_recalc_on,
        bn_batches=bn_batches,
        retain_val_loader=retain_val_loader,
        forget_val_loader=forget_val_loader,
        retain_eval_loader=retain_eval_loader,
        forget_eval_loader=forget_eval_loader,
        test_loader=test_loader,
        retain_test_loader=retain_test_loader,
        forget_test_loader=forget_test_loader,
        normalized_full_scale=normalized_full_scale,
    )
    best_score = float(point_composite_score(soup_metrics, forget_budget_acc, rt_reference))

    for cand in ordered[1:]:
        trial_state = average_state_dicts([c["state"] for c in chosen] + [cand["state"]])
        trial_metrics = evaluate_state(
            spec=spec,
            state=trial_state,
            device=device,
            bn_loader=bn_loader,
            bn_recalc_on=bn_recalc_on,
            bn_batches=bn_batches,
            retain_val_loader=retain_val_loader,
            forget_val_loader=forget_val_loader,
            retain_eval_loader=retain_eval_loader,
            forget_eval_loader=forget_eval_loader,
            test_loader=test_loader,
            retain_test_loader=retain_test_loader,
            forget_test_loader=forget_test_loader,
            normalized_full_scale=normalized_full_scale,
        )
        trial_score = float(point_composite_score(trial_metrics, forget_budget_acc, rt_reference))
        if trial_score >= best_score:
            chosen.append(cand)
            soup_state = trial_state
            soup_metrics = trial_metrics
            best_score = trial_score
    return soup_state, [str(c["name"]) for c in chosen], best_score, soup_metrics


@torch.no_grad()
def save_checkpoint_with_state(state: Dict[str, torch.Tensor], out_path: Path, meta: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": state}
    payload.update(meta)
    torch.save(payload, str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Git-Re-Basin + Bezier + Simplex connectivity for unlearned endpoints")
    parser.add_argument("--dense-ckpt", type=str, required=True)
    parser.add_argument("--endpoint-a", type=str, required=True)
    parser.add_argument("--endpoint-b", type=str, required=True)
    parser.add_argument("--scratch-retrain-ckpt", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="runs/unlearning_connectivity_phase15")

    parser.add_argument("--dataset", type=str, default=None, choices=["cifar10", "cifar100"])
    parser.add_argument("--arch", type=str, default=None, choices=["resnet", "wideresnet"])
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--width-mult", type=float, default=None)
    parser.add_argument("--depth-mult", type=float, default=None)
    parser.add_argument("--model-mult", type=int, default=None)
    parser.add_argument("--datapath", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--df-mode", type=str, default="profile", choices=["profile", "class", "random"])
    parser.add_argument("--df-profile", type=str, default="df1", choices=["df1", "df2", "df3"])
    parser.add_argument("--forget-classes", type=str, default=None)
    parser.add_argument("--forget-ratio", type=float, default=0.1)
    parser.add_argument("--forget-objective", type=str, default="kl_uniform", choices=["ce_ascent", "kl_uniform", "entropy"])
    parser.add_argument("--retain-weight", type=float, default=1.0)
    parser.add_argument("--forget-alpha", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--forget-val-budget", type=float, default=0.01)

    parser.add_argument("--lambdas", type=int, default=21)
    parser.add_argument("--bn-batches", type=int, default=200)
    parser.add_argument("--no-bn-recalc", dest="bn_recalc", action="store_false")
    parser.set_defaults(bn_recalc=True)

    parser.add_argument("--perm-max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--bezier-steps", type=int, default=300)
    parser.add_argument("--bezier-lr", type=float, default=0.03)
    parser.add_argument("--bezier-weight-decay", type=float, default=5e-4)
    parser.add_argument("--bezier-t-samples", type=int, default=2)
    parser.add_argument("--bezier-tail-k", type=int, default=20)

    parser.add_argument("--simplex-steps", type=int, default=300)
    parser.add_argument("--simplex-lr", type=float, default=0.03)
    parser.add_argument("--simplex-weight-decay", type=float, default=5e-4)
    parser.add_argument("--simplex-dirichlet-alpha", type=float, default=1.0)
    parser.add_argument("--simplex-tail-k", type=int, default=20)
    parser.add_argument("--simplex-grid-resolution", type=int, default=5)
    args = parser.parse_args()

    dense_ckpt = Path(args.dense_ckpt)
    endpoint_a = Path(args.endpoint_a)
    endpoint_b = Path(args.endpoint_b)
    scratch_ckpt = Path(args.scratch_retrain_ckpt) if args.scratch_retrain_ckpt else None
    for path in (dense_ckpt, endpoint_a, endpoint_b):
        if not path.exists():
            raise FileNotFoundError(path)

    ckpt_meta, _ = load_checkpoint(dense_ckpt)
    spec = infer_model_spec(ckpt_meta, args)
    num_classes = 10 if spec.dataset == "cifar10" else 100
    df_spec = resolve_df_spec(args, num_classes=num_classes)
    df_tag = df_spec["name"]

    run_dir = Path(args.out_dir) / spec.dataset / df_tag / f"{endpoint_a.stem}__{endpoint_b.stem}"
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train_aug, train_eval, test_eval = build_cifar_datasets(spec)
    targets = train_aug.targets if hasattr(train_aug, "targets") else train_aug.labels
    forget_idx, retain_idx = build_forget_retain_indices(targets, df_spec=df_spec, split_seed=args.split_seed)
    forget_train_idx, forget_val_idx = split_indices_train_val(forget_idx, args.val_ratio, args.split_seed + 11)
    retain_train_idx, retain_val_idx = split_indices_train_val(retain_idx, args.val_ratio, args.split_seed + 17)

    retain_train_loader = make_subset_loader(train_aug, retain_train_idx, spec.batch_size, spec.workers, True, args.seed)
    forget_train_loader = make_subset_loader(train_aug, forget_train_idx, spec.batch_size, spec.workers, True, args.seed + 100)
    retain_val_loader = make_subset_loader(train_eval, retain_val_idx, spec.batch_size, spec.workers, False, 0)
    forget_val_loader = make_subset_loader(train_eval, forget_val_idx, spec.batch_size, spec.workers, False, 0)
    retain_eval_loader = make_subset_loader(train_eval, retain_idx, spec.batch_size, spec.workers, False, 0)
    forget_eval_loader = make_subset_loader(train_eval, forget_idx, spec.batch_size, spec.workers, False, 0)
    test_loader = DataLoader(test_eval, batch_size=spec.batch_size, shuffle=False, num_workers=spec.workers, pin_memory=True)
    bn_loader = make_subset_loader(train_eval, retain_train_idx, spec.batch_size, spec.workers, True, args.split_seed + 1000)

    retain_test_loader = None
    forget_test_loader = None
    normalized_full_scale: Optional[float] = None
    if df_spec.get("type") == "class":
        test_targets = _get_dataset_targets(test_eval)
        forget_test_idx, retain_test_idx = build_forget_retain_indices(test_targets, df_spec=df_spec, split_seed=args.split_seed)
        if len(test_targets) > 0:
            normalized_full_scale = float(len(retain_test_idx) / len(test_targets))
        if retain_test_idx:
            retain_test_loader = make_subset_loader(test_eval, retain_test_idx, spec.batch_size, spec.workers, False, 0)
        if forget_test_idx:
            forget_test_loader = make_subset_loader(test_eval, forget_test_idx, spec.batch_size, spec.workers, False, 0)

    _, state_a = load_checkpoint(endpoint_a)
    _, state_b = load_checkpoint(endpoint_b)
    model_for_spec, _ = build_dense_model(spec)

    rt_reference = build_rt_reference(
        spec=spec,
        scratch_ckpt=scratch_ckpt,
        device=device,
        bn_loader=bn_loader,
        bn_recalc_on=args.bn_recalc,
        bn_batches=args.bn_batches,
        retain_val_loader=retain_val_loader,
    )

    # Endpoint metrics for reference.
    endpoint_metrics = {
        "endpoint_a": evaluate_state(
            spec=spec,
            state=state_a,
            device=device,
            bn_loader=bn_loader,
            bn_recalc_on=args.bn_recalc,
            bn_batches=args.bn_batches,
            retain_val_loader=retain_val_loader,
            forget_val_loader=forget_val_loader,
            retain_eval_loader=retain_eval_loader,
            forget_eval_loader=forget_eval_loader,
            test_loader=test_loader,
            retain_test_loader=retain_test_loader,
            forget_test_loader=forget_test_loader,
            normalized_full_scale=normalized_full_scale,
        ),
        "endpoint_b": evaluate_state(
            spec=spec,
            state=state_b,
            device=device,
            bn_loader=bn_loader,
            bn_recalc_on=args.bn_recalc,
            bn_batches=args.bn_batches,
            retain_val_loader=retain_val_loader,
            forget_val_loader=forget_val_loader,
            retain_eval_loader=retain_eval_loader,
            forget_eval_loader=forget_eval_loader,
            test_loader=test_loader,
            retain_test_loader=retain_test_loader,
            forget_test_loader=forget_test_loader,
            normalized_full_scale=normalized_full_scale,
        ),
    }

    # 1) raw linear
    lambdas = make_lambdas(args.lambdas)
    raw_linear = evaluate_curve(
        spec=spec,
        state_fn=lambda t: linear_state_dict(state_a, state_b, t),
        lambdas=lambdas,
        device=device,
        bn_loader=bn_loader,
        bn_recalc_on=args.bn_recalc,
        bn_batches=args.bn_batches,
        retain_val_loader=retain_val_loader,
        forget_val_loader=forget_val_loader,
        retain_eval_loader=retain_eval_loader,
        forget_eval_loader=forget_eval_loader,
        test_loader=test_loader,
        retain_test_loader=retain_test_loader,
        forget_test_loader=forget_test_loader,
        normalized_full_scale=normalized_full_scale,
        forget_budget_acc=args.forget_val_budget,
        rt_reference=rt_reference,
    )
    with open(run_dir / "raw_linear.json", "w") as f:
        json.dump(raw_linear, f, indent=2)

    # 2) Git Re-Basin / perm_linear
    sanity_x, _ = next(iter(retain_val_loader if retain_val_loader is not None else retain_eval_loader))
    rebasin = align_state_dict_to_reference(
        model=model_for_spec,
        state_ref=state_a,
        state_to_align=state_b,
        max_iter=args.perm_max_iter,
        seed=args.seed,
        model_builder=lambda: build_dense_model(spec)[0],
        sanity_batch_x=sanity_x[: min(32, sanity_x.size(0))],
        device=device,
    )
    state_b_perm = rebasin.aligned_state
    save_checkpoint_with_state(
        state_b_perm,
        run_dir / "aligned_endpoint_b.pth",
        {
            "stage": "git_rebasin_aligned_endpoint_b",
            "iterations": rebasin.iterations,
            "converged": rebasin.converged,
            "max_abs_logit_diff": rebasin.max_abs_logit_diff,
            "mean_abs_logit_diff": rebasin.mean_abs_logit_diff,
        },
    )
    perm_linear = evaluate_curve(
        spec=spec,
        state_fn=lambda t: linear_state_dict(state_a, state_b_perm, t),
        lambdas=lambdas,
        device=device,
        bn_loader=bn_loader,
        bn_recalc_on=args.bn_recalc,
        bn_batches=args.bn_batches,
        retain_val_loader=retain_val_loader,
        forget_val_loader=forget_val_loader,
        retain_eval_loader=retain_eval_loader,
        forget_eval_loader=forget_eval_loader,
        test_loader=test_loader,
        retain_test_loader=retain_test_loader,
        forget_test_loader=forget_test_loader,
        normalized_full_scale=normalized_full_scale,
        forget_budget_acc=args.forget_val_budget,
        rt_reference=rt_reference,
    )
    with open(run_dir / "perm_linear.json", "w") as f:
        json.dump(
            {
                **perm_linear,
                "rebasin": {
                    "iterations": rebasin.iterations,
                    "converged": rebasin.converged,
                    "max_abs_logit_diff": rebasin.max_abs_logit_diff,
                    "mean_abs_logit_diff": rebasin.mean_abs_logit_diff,
                },
            },
            f,
            indent=2,
        )

    best_perm_linear_t = float(perm_linear["best_by_composite"]["t"])
    best_perm_linear_state = linear_state_dict(state_a, state_b_perm, best_perm_linear_t)
    save_checkpoint_with_state(
        best_perm_linear_state,
        run_dir / "perm_linear_best.pth",
        {
            "stage": "perm_linear_best",
            "best_t": best_perm_linear_t,
            "composite": float(perm_linear["best_by_composite"]["composite"]),
        },
    )

    # 3) Learned Bezier curve on aligned endpoints.
    model_for_curve, _ = build_dense_model(spec)
    bezier_train: CurveTrainResult = train_bezier_control(
        model=model_for_curve,
        state_a=state_a,
        state_b=state_b_perm,
        retain_train_loader=retain_train_loader,
        forget_train_loader=forget_train_loader,
        device=device,
        steps=args.bezier_steps,
        lr=args.bezier_lr,
        weight_decay=args.bezier_weight_decay,
        retain_weight=args.retain_weight,
        forget_alpha=args.forget_alpha,
        forget_objective=args.forget_objective,
        t_samples=args.bezier_t_samples,
        grad_clip=args.grad_clip,
        seed=args.seed,
        tail_k=args.bezier_tail_k,
    )
    save_checkpoint_with_state(
        bezier_train.final_control_state,
        run_dir / "bezier_control_final.pth",
        {"stage": "bezier_control_final", "history_len": len(bezier_train.history)},
    )
    if bezier_train.swa_control_state is not None:
        save_checkpoint_with_state(
            bezier_train.swa_control_state,
            run_dir / "bezier_control_swa.pth",
            {"stage": "bezier_control_swa", "history_len": len(bezier_train.history)},
        )

    bezier_curve = evaluate_curve(
        spec=spec,
        state_fn=lambda t: bezier_state_dict(state_a, bezier_train.final_control_state, state_b_perm, t),
        lambdas=lambdas,
        device=device,
        bn_loader=bn_loader,
        bn_recalc_on=args.bn_recalc,
        bn_batches=args.bn_batches,
        retain_val_loader=retain_val_loader,
        forget_val_loader=forget_val_loader,
        retain_eval_loader=retain_eval_loader,
        forget_eval_loader=forget_eval_loader,
        test_loader=test_loader,
        retain_test_loader=retain_test_loader,
        forget_test_loader=forget_test_loader,
        normalized_full_scale=normalized_full_scale,
        forget_budget_acc=args.forget_val_budget,
        rt_reference=rt_reference,
    )
    with open(run_dir / "bezier_curve.json", "w") as f:
        json.dump({"history": bezier_train.history, **bezier_curve}, f, indent=2)

    best_bezier_t = float(bezier_curve["best_by_composite"]["t"])
    best_bezier_state = bezier_state_dict(state_a, bezier_train.final_control_state, state_b_perm, best_bezier_t)
    save_checkpoint_with_state(
        best_bezier_state,
        run_dir / "bezier_best.pth",
        {"stage": "bezier_best", "best_t": best_bezier_t, "composite": float(bezier_curve["best_by_composite"]["composite"])},
    )

    bezier_swa_curve = None
    if bezier_train.swa_control_state is not None:
        bezier_swa_curve = evaluate_curve(
            spec=spec,
            state_fn=lambda t: bezier_state_dict(state_a, bezier_train.swa_control_state, state_b_perm, t),
            lambdas=lambdas,
            device=device,
            bn_loader=bn_loader,
            bn_recalc_on=args.bn_recalc,
            bn_batches=args.bn_batches,
            retain_val_loader=retain_val_loader,
            forget_val_loader=forget_val_loader,
            retain_eval_loader=retain_eval_loader,
            forget_eval_loader=forget_eval_loader,
            test_loader=test_loader,
            retain_test_loader=retain_test_loader,
            forget_test_loader=forget_test_loader,
            normalized_full_scale=normalized_full_scale,
            forget_budget_acc=args.forget_val_budget,
            rt_reference=rt_reference,
        )
        with open(run_dir / "bezier_swa_curve.json", "w") as f:
            json.dump({"history": bezier_train.history, **bezier_swa_curve}, f, indent=2)
        best_bezier_swa_t = float(bezier_swa_curve["best_by_composite"]["t"])
        best_bezier_swa_state = bezier_state_dict(state_a, bezier_train.swa_control_state, state_b_perm, best_bezier_swa_t)
        save_checkpoint_with_state(
            best_bezier_swa_state,
            run_dir / "bezier_swa_best.pth",
            {"stage": "bezier_swa_best", "best_t": best_bezier_swa_t, "composite": float(bezier_swa_curve["best_by_composite"]["composite"])},
        )

    # 4) Simplex training anchored at A and aligned B, initialized from best Bezier state.
    model_for_simplex, _ = build_dense_model(spec)
    simplex_train: SimplexTrainResult = train_simplex_vertex(
        model=model_for_simplex,
        state_a=state_a,
        state_b=state_b_perm,
        init_vertex_state=best_bezier_state,
        retain_train_loader=retain_train_loader,
        forget_train_loader=forget_train_loader,
        device=device,
        steps=args.simplex_steps,
        lr=args.simplex_lr,
        weight_decay=args.simplex_weight_decay,
        retain_weight=args.retain_weight,
        forget_alpha=args.forget_alpha,
        forget_objective=args.forget_objective,
        dirichlet_alpha=args.simplex_dirichlet_alpha,
        grad_clip=args.grad_clip,
        seed=args.seed,
        tail_k=args.simplex_tail_k,
    )
    save_checkpoint_with_state(
        simplex_train.final_vertex_state,
        run_dir / "simplex_vertex_final.pth",
        {"stage": "simplex_vertex_final", "history_len": len(simplex_train.history)},
    )
    if simplex_train.swa_vertex_state is not None:
        save_checkpoint_with_state(
            simplex_train.swa_vertex_state,
            run_dir / "simplex_vertex_swa.pth",
            {"stage": "simplex_vertex_swa", "history_len": len(simplex_train.history)},
        )

    simplex_grid = sample_simplex_grid(args.simplex_grid_resolution)
    simplex_samples: List[Dict[str, Any]] = []
    simplex_candidates: List[Dict[str, Any]] = []
    vertices_final = [state_a, state_b_perm, simplex_train.final_vertex_state]
    for idx, lamb in enumerate(simplex_grid):
        state = simplex_state_dict(vertices_final, lamb)
        metrics = evaluate_state(
            spec=spec,
            state=state,
            device=device,
            bn_loader=bn_loader,
            bn_recalc_on=args.bn_recalc,
            bn_batches=args.bn_batches,
            retain_val_loader=retain_val_loader,
            forget_val_loader=forget_val_loader,
            retain_eval_loader=retain_eval_loader,
            forget_eval_loader=forget_eval_loader,
            test_loader=test_loader,
            retain_test_loader=retain_test_loader,
            forget_test_loader=forget_test_loader,
            normalized_full_scale=normalized_full_scale,
        )
        score = float(point_composite_score(metrics, args.forget_val_budget, rt_reference))
        row = {
            "name": f"simplex_{idx:03d}",
            "lambdas": [float(x) for x in lamb.tolist()],
            **metrics,
            "composite": score,
        }
        simplex_samples.append(row)
        simplex_candidates.append({"name": row["name"], "state": state, "val_score": score})

    simplex_best = max(simplex_samples, key=lambda x: float(x["composite"]))
    simplex_best_state = simplex_candidates[[c["name"] for c in simplex_candidates].index(simplex_best["name"])] ["state"]
    save_checkpoint_with_state(
        simplex_best_state,
        run_dir / "simplex_best.pth",
        {"stage": "simplex_best", "lambdas": simplex_best["lambdas"], "composite": float(simplex_best["composite"])},
    )

    simplex_soup_state, simplex_soup_names, simplex_soup_score, simplex_soup_metrics = greedy_soup(
        simplex_candidates,
        spec=spec,
        device=device,
        bn_loader=bn_loader,
        bn_recalc_on=args.bn_recalc,
        bn_batches=args.bn_batches,
        retain_val_loader=retain_val_loader,
        forget_val_loader=forget_val_loader,
        retain_eval_loader=retain_eval_loader,
        forget_eval_loader=forget_eval_loader,
        test_loader=test_loader,
        retain_test_loader=retain_test_loader,
        forget_test_loader=forget_test_loader,
        normalized_full_scale=normalized_full_scale,
        forget_budget_acc=args.forget_val_budget,
        rt_reference=rt_reference,
    )
    save_checkpoint_with_state(
        simplex_soup_state,
        run_dir / "simplex_soup_best.pth",
        {"stage": "simplex_soup_best", "selected_candidates": simplex_soup_names, "composite": float(simplex_soup_score)},
    )

    simplex_swa_samples = None
    simplex_swa_best = None
    if simplex_train.swa_vertex_state is not None:
        vertices_swa = [state_a, state_b_perm, simplex_train.swa_vertex_state]
        simplex_swa_samples = []
        for idx, lamb in enumerate(simplex_grid):
            state = simplex_state_dict(vertices_swa, lamb)
            metrics = evaluate_state(
                spec=spec,
                state=state,
                device=device,
                bn_loader=bn_loader,
                bn_recalc_on=args.bn_recalc,
                bn_batches=args.bn_batches,
                retain_val_loader=retain_val_loader,
                forget_val_loader=forget_val_loader,
                retain_eval_loader=retain_eval_loader,
                forget_eval_loader=forget_eval_loader,
                test_loader=test_loader,
                retain_test_loader=retain_test_loader,
                forget_test_loader=forget_test_loader,
                normalized_full_scale=normalized_full_scale,
            )
            score = float(point_composite_score(metrics, args.forget_val_budget, rt_reference))
            simplex_swa_samples.append(
                {
                    "name": f"simplex_swa_{idx:03d}",
                    "lambdas": [float(x) for x in lamb.tolist()],
                    **metrics,
                    "composite": score,
                }
            )
        simplex_swa_best = max(simplex_swa_samples, key=lambda x: float(x["composite"]))
        best_state = simplex_state_dict(vertices_swa, torch.tensor(simplex_swa_best["lambdas"], dtype=torch.float32))
        save_checkpoint_with_state(
            best_state,
            run_dir / "simplex_swa_best.pth",
            {"stage": "simplex_swa_best", "lambdas": simplex_swa_best["lambdas"], "composite": float(simplex_swa_best["composite"])},
        )

    summary = {
        "dense_ckpt": str(dense_ckpt),
        "endpoint_a": str(endpoint_a),
        "endpoint_b": str(endpoint_b),
        "scratch_retrain_ckpt": str(scratch_ckpt) if scratch_ckpt is not None else None,
        "run_dir": str(run_dir),
        "device": str(device),
        "model_spec": spec.__dict__,
        "df_spec": df_spec,
        "forget_objective": args.forget_objective,
        "forget_alpha": args.forget_alpha,
        "retain_weight": args.retain_weight,
        "grad_clip": args.grad_clip,
        "forget_val_budget": args.forget_val_budget,
        "bn_recalc": bool(args.bn_recalc),
        "bn_batches": int(args.bn_batches),
        "endpoint_metrics": endpoint_metrics,
        "rt_reference": rt_reference,
        "raw_linear": {
            "retain_val_loss_barrier": raw_linear["retain_val_loss_barrier"],
            "retain_val_acc_drop_pp": raw_linear["retain_val_acc_drop_pp"],
            "best_by_composite": raw_linear["best_by_composite"],
        },
        "perm_linear": {
            "retain_val_loss_barrier": perm_linear["retain_val_loss_barrier"],
            "retain_val_acc_drop_pp": perm_linear["retain_val_acc_drop_pp"],
            "best_by_composite": perm_linear["best_by_composite"],
            "rebasin": {
                "iterations": rebasin.iterations,
                "converged": rebasin.converged,
                "max_abs_logit_diff": rebasin.max_abs_logit_diff,
                "mean_abs_logit_diff": rebasin.mean_abs_logit_diff,
            },
        },
        "bezier": {
            "history_len": len(bezier_train.history),
            "best_by_composite": bezier_curve["best_by_composite"],
        },
        "bezier_swa": {
            "enabled": bezier_swa_curve is not None,
            "history_len": len(bezier_train.history),
            "best_by_composite": bezier_swa_curve["best_by_composite"] if bezier_swa_curve is not None else None,
        },
        "simplex": {
            "history_len": len(simplex_train.history),
            "best_sample": simplex_best,
            "num_grid_samples": len(simplex_samples),
        },
        "simplex_soup": {
            "selected_candidates": simplex_soup_names,
            "composite": float(simplex_soup_score),
            "metrics": simplex_soup_metrics,
        },
        "simplex_swa": {
            "enabled": simplex_swa_best is not None,
            "best_sample": simplex_swa_best,
        },
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(run_dir / "simplex_samples.json", "w") as f:
        json.dump({"samples": simplex_samples}, f, indent=2)
    if simplex_swa_samples is not None:
        with open(run_dir / "simplex_swa_samples.json", "w") as f:
            json.dump({"samples": simplex_swa_samples}, f, indent=2)

    print("=" * 80)
    print("CONNECTIVITY DONE")
    print("=" * 80)
    print(f"summary: {run_dir / 'summary.json'}")
    print(f"raw_linear best composite t: {raw_linear['best_by_composite']['t']:.4f}")
    print(f"perm_linear best composite t: {perm_linear['best_by_composite']['t']:.4f}")
    print(f"bezier best composite t: {bezier_curve['best_by_composite']['t']:.4f}")
    print(f"simplex best composite: {simplex_best['composite']:.6f}")


if __name__ == "__main__":
    main()
