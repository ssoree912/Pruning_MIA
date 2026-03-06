from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PermutationSpec:
    perm_to_axes: Dict[str, List[Tuple[str, int]]]
    axes_to_perm: Dict[str, Tuple[Optional[str], ...]]


def permutation_spec_from_axes_to_perm(
    axes_to_perm: Dict[str, Tuple[Optional[str], ...]]
) -> PermutationSpec:
    perm_to_axes: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for key, axis_perms in axes_to_perm.items():
        for axis, perm_name in enumerate(axis_perms):
            if perm_name is not None:
                perm_to_axes[perm_name].append((key, axis))
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=dict(axes_to_perm))


def _bn_entries(prefix: str, perm_name: str, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Tuple[Optional[str], ...]]:
    out: Dict[str, Tuple[Optional[str], ...]] = {}
    for suffix in ("weight", "bias", "running_mean", "running_var"):
        key = f"{prefix}.{suffix}"
        if key in state_dict:
            out[key] = (perm_name,)
    return out


def _conv_entries(prefix: str, perm_in: Optional[str], perm_out: Optional[str], state_dict: Dict[str, torch.Tensor]) -> Dict[str, Tuple[Optional[str], ...]]:
    out: Dict[str, Tuple[Optional[str], ...]] = {}
    key = f"{prefix}.weight"
    if key in state_dict:
        # PyTorch Conv2d weight layout: [out_channels, in_channels, kH, kW]
        out[key] = (perm_out, perm_in, None, None)
    bias_key = f"{prefix}.bias"
    if bias_key in state_dict:
        out[bias_key] = (perm_out,)
    return out


def _linear_entries(prefix: str, perm_in: Optional[str], perm_out: Optional[str], state_dict: Dict[str, torch.Tensor]) -> Dict[str, Tuple[Optional[str], ...]]:
    out: Dict[str, Tuple[Optional[str], ...]] = {}
    key = f"{prefix}.weight"
    if key in state_dict:
        # PyTorch Linear weight layout: [out_features, in_features]
        out[key] = (perm_out, perm_in)
    bias_key = f"{prefix}.bias"
    if bias_key in state_dict:
        out[bias_key] = (perm_out,)
    return out


def build_cifar_resnet_basicblock_permutation_spec(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> PermutationSpec:
    """
    Build a permutation spec for the CIFAR ResNet implementation in this repo.

    Supported family:
    - conv1 / bn1 stem
    - layer1, layer2, layer3
    - BasicBlock with optional downsample = nn.Sequential(conv1x1, bn)

    This covers the user's current setup (cifar10/cifar100 resnet20 basicblock).
    """
    if not all(hasattr(model, name) for name in ("layer1", "layer2", "layer3", "fc")):
        raise ValueError("Unsupported ResNet structure: expected CIFAR-style layer1/layer2/layer3/fc")

    axes_to_perm: Dict[str, Tuple[Optional[str], ...]] = {}

    # Stem
    axes_to_perm.update(_conv_entries("conv1", None, "P_bg0", state_dict))
    axes_to_perm.update(_bn_entries("bn1", "P_bg0", state_dict))

    stage_names = ["layer1", "layer2", "layer3"]
    stage_out_perms = ["P_bg0", "P_bg1", "P_bg2"]
    stage_in_perms = ["P_bg0", "P_bg0", "P_bg1"]

    for stage_name, stage_in_perm, stage_out_perm in zip(stage_names, stage_in_perms, stage_out_perms):
        stage_module = getattr(model, stage_name)
        blocks = list(stage_module)
        if not blocks:
            continue
        current_in_perm = stage_in_perm
        for block_idx, block in enumerate(blocks):
            prefix = f"{stage_name}.{block_idx}"
            inner_perm = f"P_{stage_name}_{block_idx}_inner"
            out_perm = stage_out_perm

            axes_to_perm.update(_conv_entries(f"{prefix}.conv1", current_in_perm, inner_perm, state_dict))
            axes_to_perm.update(_bn_entries(f"{prefix}.bn1", inner_perm, state_dict))
            axes_to_perm.update(_conv_entries(f"{prefix}.conv2", inner_perm, out_perm, state_dict))
            axes_to_perm.update(_bn_entries(f"{prefix}.bn2", out_perm, state_dict))

            if getattr(block, "downsample", None) is not None:
                axes_to_perm.update(_conv_entries(f"{prefix}.downsample.0", current_in_perm, out_perm, state_dict))
                axes_to_perm.update(_bn_entries(f"{prefix}.downsample.1", out_perm, state_dict))

            current_in_perm = out_perm

    axes_to_perm.update(_linear_entries("fc", "P_bg2", None, state_dict))
    return permutation_spec_from_axes_to_perm(axes_to_perm)


def build_resnet_permutation_spec(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> PermutationSpec:
    # For now we support the CIFAR ResNet family used in the current unlearning experiments.
    return build_cifar_resnet_basicblock_permutation_spec(model=model, state_dict=state_dict)
