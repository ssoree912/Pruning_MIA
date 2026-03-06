from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .perm_spec_resnet import PermutationSpec


@dataclass
class WeightMatchingResult:
    perm: Dict[str, torch.Tensor]
    iterations: int
    converged: bool


@torch.no_grad()
def get_permuted_tensor(
    ps: PermutationSpec,
    perm: Dict[str, torch.Tensor],
    key: str,
    state: Dict[str, torch.Tensor],
    except_axis: Optional[int] = None,
) -> torch.Tensor:
    tensor = state[key]
    axis_perms = ps.axes_to_perm.get(key)
    if axis_perms is None:
        return tensor

    out = tensor
    for axis, perm_name in enumerate(axis_perms):
        if axis == except_axis:
            continue
        if perm_name is None:
            continue
        idx = perm[perm_name].to(device=out.device, dtype=torch.long)
        out = torch.index_select(out, axis, idx)
    return out


@torch.no_grad()
def apply_permutation(
    ps: PermutationSpec,
    perm: Dict[str, torch.Tensor],
    state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if key not in ps.axes_to_perm:
            out[key] = tensor.clone()
        else:
            out[key] = get_permuted_tensor(ps, perm, key, state).clone()
    return out


@torch.no_grad()
def weight_matching(
    ps: PermutationSpec,
    params_a: Dict[str, torch.Tensor],
    params_b: Dict[str, torch.Tensor],
    max_iter: int = 100,
    seed: int = 0,
    init_perm: Optional[Dict[str, torch.Tensor]] = None,
    silent: bool = False,
) -> WeightMatchingResult:
    perm_sizes = {
        perm_name: int(params_a[axes[0][0]].shape[axes[0][1]])
        for perm_name, axes in ps.perm_to_axes.items()
    }
    if init_perm is None:
        perm = {name: torch.arange(n, dtype=torch.long) for name, n in perm_sizes.items()}
    else:
        perm = {name: p.detach().cpu().clone().to(dtype=torch.long) for name, p in init_perm.items()}

    perm_names = list(perm.keys())
    converged = False
    iterations = 0

    for iteration in range(max_iter):
        iterations = iteration + 1
        progress = False
        order = list(perm_names)
        random.Random(seed + iteration).shuffle(order)

        for perm_name in order:
            n = perm_sizes[perm_name]
            A = torch.zeros((n, n), dtype=torch.float64)
            for key, axis in ps.perm_to_axes[perm_name]:
                w_a = params_a[key].detach().cpu()
                w_b = get_permuted_tensor(ps, perm, key, params_b, except_axis=axis).detach().cpu()
                w_a = torch.movedim(w_a, axis, 0).reshape(n, -1).to(dtype=torch.float64)
                w_b = torch.movedim(w_b, axis, 0).reshape(n, -1).to(dtype=torch.float64)
                A += w_a @ w_b.T

            ri, ci = linear_sum_assignment(A.numpy(), maximize=True)
            if not np.array_equal(ri, np.arange(len(ri))):
                raise RuntimeError("linear_sum_assignment returned unexpected row indices")

            current_perm = perm[perm_name].detach().cpu().numpy()
            oldL = float(A[np.arange(n), current_perm].sum().item())
            newL = float(A[np.arange(n), ci].sum().item())
            if not silent:
                print(f"[weight_matching] iter={iteration:03d} perm={perm_name} gain={newL - oldL:.6f}")
            if newL > oldL + 1e-12:
                progress = True
                perm[perm_name] = torch.tensor(ci, dtype=torch.long)

        if not progress:
            converged = True
            break

    return WeightMatchingResult(perm=perm, iterations=iterations, converged=converged)
