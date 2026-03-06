from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .perm_spec_resnet import build_resnet_permutation_spec
from .weight_matching_torch import apply_permutation, weight_matching


@dataclass
class RebasinResult:
    aligned_state: Dict[str, torch.Tensor]
    perm: Dict[str, torch.Tensor]
    iterations: int
    converged: bool
    max_abs_logit_diff: Optional[float]
    mean_abs_logit_diff: Optional[float]


@torch.no_grad()
def sanity_check_function_preservation(
    model_builder,
    orig_state: Dict[str, torch.Tensor],
    aligned_state: Dict[str, torch.Tensor],
    device: torch.device,
    batch_x: torch.Tensor,
) -> Dict[str, float]:
    model_orig: nn.Module = model_builder().to(device)
    model_aligned: nn.Module = model_builder().to(device)
    model_orig.load_state_dict(orig_state, strict=True)
    model_aligned.load_state_dict(aligned_state, strict=True)
    model_orig.eval()
    model_aligned.eval()

    x = batch_x.to(device)
    y0 = model_orig(x)
    y1 = model_aligned(x)
    diff = (y0 - y1).abs()
    return {
        "max_abs_logit_diff": float(diff.max().item()),
        "mean_abs_logit_diff": float(diff.mean().item()),
    }


@torch.no_grad()
def align_state_dict_to_reference(
    model: nn.Module,
    state_ref: Dict[str, torch.Tensor],
    state_to_align: Dict[str, torch.Tensor],
    *,
    max_iter: int = 100,
    seed: int = 0,
    model_builder=None,
    sanity_batch_x: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> RebasinResult:
    ps = build_resnet_permutation_spec(model=model, state_dict=state_ref)
    match = weight_matching(
        ps=ps,
        params_a=state_ref,
        params_b=state_to_align,
        max_iter=max_iter,
        seed=seed,
        silent=False,
    )
    aligned = apply_permutation(ps, match.perm, state_to_align)

    max_abs = None
    mean_abs = None
    if model_builder is not None and sanity_batch_x is not None and device is not None:
        sanity = sanity_check_function_preservation(
            model_builder=model_builder,
            orig_state=state_to_align,
            aligned_state=aligned,
            device=device,
            batch_x=sanity_batch_x,
        )
        max_abs = sanity["max_abs_logit_diff"]
        mean_abs = sanity["mean_abs_logit_diff"]

    return RebasinResult(
        aligned_state=aligned,
        perm=match.perm,
        iterations=match.iterations,
        converged=match.converged,
        max_abs_logit_diff=max_abs,
        mean_abs_logit_diff=mean_abs,
    )
