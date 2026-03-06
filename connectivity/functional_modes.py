from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call


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


def kl_to_uniform(logits: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    n_cls = logits.size(1)
    log_n = math.log(float(n_cls))
    return (p * logp).sum(dim=1).mean() + log_n


def neg_entropy(logits: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    return (p * logp).sum(dim=1).mean()


def build_param_and_buffer_keys(model: nn.Module) -> Tuple[List[str], List[str]]:
    param_keys = [name for name, _ in model.named_parameters()]
    buffer_keys = [name for name, _ in model.named_buffers()]
    return param_keys, buffer_keys


@torch.no_grad()
def split_state_dict(
    state: Dict[str, torch.Tensor],
    param_keys: Iterable[str],
    buffer_keys: Iterable[str],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    params: Dict[str, torch.Tensor] = {}
    buffers: Dict[str, torch.Tensor] = {}
    others: Dict[str, torch.Tensor] = {}
    param_key_set = set(param_keys)
    buffer_key_set = set(buffer_keys)
    for key, tensor in state.items():
        if key in param_key_set:
            params[key] = tensor.detach().to(device)
        elif key in buffer_key_set:
            buffers[key] = tensor.detach().to(device)
        else:
            others[key] = tensor.detach().cpu().clone()
    return params, buffers, others


@torch.no_grad()
def linear_state_dict(
    s0: Dict[str, torch.Tensor],
    s1: Dict[str, torch.Tensor],
    t: float,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v0 in s0.items():
        if k not in s1:
            out[k] = v0.detach().cpu().clone()
            continue
        v1 = s1[k]
        if torch.is_tensor(v0) and torch.is_tensor(v1) and v0.dtype.is_floating_point:
            out[k] = ((1.0 - t) * v0 + t * v1).detach().cpu().clone()
        else:
            out[k] = (v0 if t < 0.5 else v1).detach().cpu().clone()
    return out


@torch.no_grad()
def average_state_dicts(states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not states:
        raise ValueError("states must not be empty")
    ref = states[0]
    out: Dict[str, torch.Tensor] = {}
    for key, value in ref.items():
        if not torch.is_tensor(value):
            out[key] = value
            continue
        if value.dtype.is_floating_point:
            acc = torch.zeros_like(value, dtype=torch.float32)
            for st in states:
                acc += st[key].to(dtype=torch.float32)
            out[key] = (acc / float(len(states))).to(dtype=value.dtype)
        else:
            out[key] = value.clone()
    return out


@torch.no_grad()
def bezier_state_dict(
    s0: Dict[str, torch.Tensor],
    sc: Dict[str, torch.Tensor],
    s1: Dict[str, torch.Tensor],
    t: float,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v0 in s0.items():
        if k not in s1:
            out[k] = v0.detach().cpu().clone()
            continue
        v1 = s1[k]
        vc = sc.get(k, v0)
        if torch.is_tensor(v0) and torch.is_tensor(v1) and v0.dtype.is_floating_point:
            out[k] = (((1.0 - t) ** 2) * v0 + 2.0 * t * (1.0 - t) * vc + (t ** 2) * v1).detach().cpu().clone()
        else:
            out[k] = (v0 if t < 0.5 else v1).detach().cpu().clone()
    return out


@torch.no_grad()
def simplex_state_dict(
    vertices: List[Dict[str, torch.Tensor]],
    lambdas: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if len(vertices) != len(lambdas):
        raise ValueError("vertices and lambdas must have the same length")
    ref = vertices[0]
    out: Dict[str, torch.Tensor] = {}
    for key, value in ref.items():
        if not torch.is_tensor(value):
            out[key] = value
            continue
        if value.dtype.is_floating_point:
            acc = torch.zeros_like(value, dtype=torch.float32)
            for st, lam in zip(vertices, lambdas):
                acc += float(lam) * st[key].to(dtype=torch.float32)
            out[key] = acc.to(dtype=value.dtype).detach().cpu().clone()
        else:
            pick = int(torch.argmax(lambdas).item())
            out[key] = vertices[pick][key].detach().cpu().clone()
    return out


def combine_params_and_buffers(
    params: Dict[str, torch.Tensor],
    buffers: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {**params, **buffers}


def _bezier_params(
    pa: Dict[str, torch.Tensor],
    pc: Dict[str, torch.Tensor],
    pb: Dict[str, torch.Tensor],
    t: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    one_minus_t = 1.0 - t
    for key, va in pa.items():
        vb = pb[key]
        vc = pc[key]
        out[key] = (one_minus_t * one_minus_t) * va + 2.0 * t * one_minus_t * vc + (t * t) * vb
    return out


def _linear_buffers(
    ba: Dict[str, torch.Tensor],
    bb: Dict[str, torch.Tensor],
    t: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, va in ba.items():
        vb = bb.get(key, va)
        if torch.is_tensor(va) and va.dtype.is_floating_point:
            out[key] = (1.0 - t) * va + t * vb
        else:
            out[key] = va if float(t.item()) < 0.5 else vb
    return out


def _simplex_params(
    vertices: List[Dict[str, torch.Tensor]],
    lambdas: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    ref = vertices[0]
    for key, value in ref.items():
        if value.dtype.is_floating_point:
            acc = torch.zeros_like(value)
            for v, lam in zip(vertices, lambdas):
                acc = acc + lam * v[key]
            out[key] = acc
        else:
            pick = int(torch.argmax(lambdas).item())
            out[key] = vertices[pick][key]
    return out


@dataclass
class CurveTrainResult:
    final_control_state: Dict[str, torch.Tensor]
    swa_control_state: Optional[Dict[str, torch.Tensor]]
    history: List[Dict[str, float]]


@dataclass
class SimplexTrainResult:
    final_vertex_state: Dict[str, torch.Tensor]
    swa_vertex_state: Optional[Dict[str, torch.Tensor]]
    history: List[Dict[str, float]]


class _TailAverage:
    def __init__(self, maxlen: int):
        self._buf: Deque[Dict[str, torch.Tensor]] = deque(maxlen=maxlen)

    def append(self, state: Dict[str, torch.Tensor]) -> None:
        self._buf.append({k: v.detach().cpu().clone() for k, v in state.items()})

    def average(self) -> Optional[Dict[str, torch.Tensor]]:
        if not self._buf:
            return None
        return average_state_dicts(list(self._buf))


def train_bezier_control(
    *,
    model: nn.Module,
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
    retain_train_loader,
    forget_train_loader,
    device: torch.device,
    steps: int,
    lr: float,
    weight_decay: float,
    retain_weight: float,
    forget_alpha: float,
    forget_objective: str,
    t_samples: int,
    grad_clip: float,
    seed: int,
    tail_k: int,
) -> CurveTrainResult:
    if steps <= 0:
        raise ValueError("steps must be > 0")
    if t_samples <= 0:
        raise ValueError("t_samples must be > 0")

    model = model.to(device)
    model.eval()
    param_keys, buffer_keys = build_param_and_buffer_keys(model)
    pa, ba, _ = split_state_dict(state_a, param_keys, buffer_keys, device)
    pb, bb, _ = split_state_dict(state_b, param_keys, buffer_keys, device)

    midpoint = {k: 0.5 * (pa[k] + pb[k]) for k in param_keys}
    control_params = {k: nn.Parameter(v.detach().clone()) for k, v in midpoint.items()}
    optimizer = torch.optim.SGD(control_params.values(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    rng = random.Random(seed)
    loss_fn = nn.CrossEntropyLoss()
    history: List[Dict[str, float]] = []
    tail = _TailAverage(maxlen=max(0, tail_k)) if tail_k > 0 else None

    retain_it = iter(retain_train_loader)
    forget_it = iter(forget_train_loader)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        xr_batch, retain_it = _next_batch(retain_it, retain_train_loader)
        xf_batch, forget_it = _next_batch(forget_it, forget_train_loader)
        xr, yr = _unpack_xy(xr_batch)
        xf, yf = _unpack_xy(xf_batch)
        xr = xr.to(device, non_blocking=True)
        yr = yr.to(device, non_blocking=True)
        xf = xf.to(device, non_blocking=True)
        yf = yf.to(device, non_blocking=True)

        total_loss = 0.0
        total_retain = 0.0
        total_forget = 0.0
        for _ in range(t_samples):
            t = torch.tensor(rng.random(), device=device, dtype=torch.float32)
            params_t = _bezier_params(pa, control_params, pb, t)
            buffers_t = _linear_buffers(ba, bb, t)
            logits_r = functional_call(model, combine_params_and_buffers(params_t, buffers_t), (xr,))
            logits_f = functional_call(model, combine_params_and_buffers(params_t, buffers_t), (xf,))
            loss_r = loss_fn(logits_r, yr)
            if forget_objective == "ce_ascent":
                loss_f = loss_fn(logits_f, yf)
                loss = retain_weight * loss_r - forget_alpha * loss_f
            elif forget_objective == "kl_uniform":
                loss_f = kl_to_uniform(logits_f)
                loss = retain_weight * loss_r + forget_alpha * loss_f
            elif forget_objective == "entropy":
                loss_f = neg_entropy(logits_f)
                loss = retain_weight * loss_r + forget_alpha * loss_f
            else:
                raise ValueError(f"Unsupported forget_objective: {forget_objective}")
            total_loss = total_loss + loss
            total_retain = total_retain + loss_r.detach()
            total_forget = total_forget + loss_f.detach()

        total_loss = total_loss / float(t_samples)
        total_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(list(control_params.values()), grad_clip)
        optimizer.step()

        row = {
            "step": float(step),
            "loss": float(total_loss.detach().item()),
            "retain_loss": float(total_retain.mean().item() / float(t_samples) if hasattr(total_retain, 'mean') else total_retain.item()),
            "forget_loss": float(total_forget.mean().item() / float(t_samples) if hasattr(total_forget, 'mean') else total_forget.item()),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if tail is not None:
            tail.append({k: p.detach().cpu().clone() for k, p in control_params.items()})
        if (step + 1) % max(1, steps // 10) == 0 or step == 0:
            print(
                f"[bezier] step {step + 1:04d}/{steps:04d} "
                f"loss={row['loss']:.6f} retain={row['retain_loss']:.6f} forget={row['forget_loss']:.6f}"
            )

    final_control_state = {k: p.detach().cpu().clone() for k, p in control_params.items()}
    swa_control_state = tail.average() if tail is not None else None
    return CurveTrainResult(
        final_control_state=final_control_state,
        swa_control_state=swa_control_state,
        history=history,
    )


def train_simplex_vertex(
    *,
    model: nn.Module,
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
    init_vertex_state: Dict[str, torch.Tensor],
    retain_train_loader,
    forget_train_loader,
    device: torch.device,
    steps: int,
    lr: float,
    weight_decay: float,
    retain_weight: float,
    forget_alpha: float,
    forget_objective: str,
    dirichlet_alpha: float,
    grad_clip: float,
    seed: int,
    tail_k: int,
) -> SimplexTrainResult:
    if steps <= 0:
        raise ValueError("steps must be > 0")
    if dirichlet_alpha <= 0:
        raise ValueError("dirichlet_alpha must be > 0")

    model = model.to(device)
    model.eval()
    param_keys, buffer_keys = build_param_and_buffer_keys(model)
    pa, ba, _ = split_state_dict(state_a, param_keys, buffer_keys, device)
    pb, bb, _ = split_state_dict(state_b, param_keys, buffer_keys, device)
    pv, bv, _ = split_state_dict(init_vertex_state, param_keys, buffer_keys, device)

    vertex_params = {k: nn.Parameter(v.detach().clone()) for k, v in pv.items()}
    optimizer = torch.optim.SGD(vertex_params.values(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    dist = torch.distributions.Dirichlet(torch.full((3,), float(dirichlet_alpha), device=device))
    loss_fn = nn.CrossEntropyLoss()
    history: List[Dict[str, float]] = []
    tail = _TailAverage(maxlen=max(0, tail_k)) if tail_k > 0 else None

    retain_it = iter(retain_train_loader)
    forget_it = iter(forget_train_loader)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        xr_batch, retain_it = _next_batch(retain_it, retain_train_loader)
        xf_batch, forget_it = _next_batch(forget_it, forget_train_loader)
        xr, yr = _unpack_xy(xr_batch)
        xf, yf = _unpack_xy(xf_batch)
        xr = xr.to(device, non_blocking=True)
        yr = yr.to(device, non_blocking=True)
        xf = xf.to(device, non_blocking=True)
        yf = yf.to(device, non_blocking=True)

        lambdas = dist.sample()
        params_t = _simplex_params([pa, pb, vertex_params], lambdas)
        buffers_t = _simplex_params([ba, bb, bv], lambdas)
        logits_r = functional_call(model, combine_params_and_buffers(params_t, buffers_t), (xr,))
        logits_f = functional_call(model, combine_params_and_buffers(params_t, buffers_t), (xf,))
        loss_r = loss_fn(logits_r, yr)
        if forget_objective == "ce_ascent":
            loss_f = loss_fn(logits_f, yf)
            loss = retain_weight * loss_r - forget_alpha * loss_f
        elif forget_objective == "kl_uniform":
            loss_f = kl_to_uniform(logits_f)
            loss = retain_weight * loss_r + forget_alpha * loss_f
        elif forget_objective == "entropy":
            loss_f = neg_entropy(logits_f)
            loss = retain_weight * loss_r + forget_alpha * loss_f
        else:
            raise ValueError(f"Unsupported forget_objective: {forget_objective}")

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(list(vertex_params.values()), grad_clip)
        optimizer.step()

        row = {
            "step": float(step),
            "loss": float(loss.detach().item()),
            "retain_loss": float(loss_r.detach().item()),
            "forget_loss": float(loss_f.detach().item()),
            "lambda0": float(lambdas[0].detach().item()),
            "lambda1": float(lambdas[1].detach().item()),
            "lambda2": float(lambdas[2].detach().item()),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if tail is not None:
            tail.append({k: p.detach().cpu().clone() for k, p in vertex_params.items()})
        if (step + 1) % max(1, steps // 10) == 0 or step == 0:
            print(
                f"[simplex] step {step + 1:04d}/{steps:04d} "
                f"loss={row['loss']:.6f} retain={row['retain_loss']:.6f} forget={row['forget_loss']:.6f}"
            )

    final_vertex_state = {k: p.detach().cpu().clone() for k, p in vertex_params.items()}
    swa_vertex_state = tail.average() if tail is not None else None
    return SimplexTrainResult(
        final_vertex_state=final_vertex_state,
        swa_vertex_state=swa_vertex_state,
        history=history,
    )
