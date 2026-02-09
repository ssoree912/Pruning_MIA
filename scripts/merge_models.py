#!/usr/bin/env python3
"""
Merge multiple trained checkpoints (static/dpf/dense) via weight averaging.
Optionally recompute BN stats and run short finetuning.

Merged model is saved as a synthetic seed folder under runs/<method>/.../seed<out_seed>
to keep existing evaluation tools (e.g., MIA runners) working.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn


def _find_repo_root(start: Path) -> Path:
    for cand in [start] + list(start.parents):
        if (cand / '.git').exists():
            return cand
        if (cand / 'base_model.py').exists() and (cand / 'mia_eval').exists():
            return cand
    return start.parents[2]


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import DataLoader as CIFARLoader  # noqa: E402
import models  # noqa: E402
import pruning  # noqa: E402


def _resolve_seed_dir(runs_dir: Path, method: str, dataset: str,
                      sparsity: Optional[float], seed: int,
                      freeze_tag: Optional[str] = None) -> Path:
    if method == 'dense':
        return runs_dir / 'dense' / dataset / f'seed{seed}'
    if method == 'static':
        return runs_dir / 'static' / f'sparsity_{sparsity}' / dataset / f'seed{seed}'
    if method == 'dpf':
        tag = f"_{freeze_tag}" if freeze_tag else ""
        return runs_dir / 'dpf' / f'sparsity_{sparsity}{tag}' / dataset / f'seed{seed}'
    raise ValueError(f"Unsupported method: {method}")


def _auto_discover_seeds(base_dir: Path) -> List[int]:
    seeds = []
    if not base_dir.exists():
        return seeds
    for sdir in sorted(base_dir.glob('seed*')):
        if (sdir / 'best_model.pth').exists():
            try:
                seeds.append(int(sdir.name.replace('seed', '')))
            except Exception:
                continue
    return seeds


def _load_config_from_seed_dir(seed_dir: Path) -> Optional[Dict]:
    candidates = [
        seed_dir / 'config.json',
        seed_dir.parent / 'config.json',
        seed_dir.parent.parent / 'config.json',
    ]
    for c in candidates:
        if c.exists():
            try:
                with open(c) as f:
                    return json.load(f)
            except Exception:
                continue
    return None


def _build_model_from_config(cfg: Dict):
    if cfg.get('pruning', {}).get('enabled', False):
        pruner_key = cfg['pruning']['method'].lower()
        if pruner_key in ('static', 'dpf', 'dcil'):
            pruner_key = 'dcil'
        pruner = pruning.__dict__[pruner_key]
        model, image_size = pruning.models.__dict__[cfg['model']['arch']](
            data=cfg['data']['dataset'],
            num_layers=cfg['model']['layers'],
            width_mult=cfg['model'].get('width_mult', 1.0),
            depth_mult=cfg['model'].get('depth_mult', 1.0),
            model_mult=cfg['model'].get('model_mult', 1.0),
            mnn=pruner.mnn,
        )
    else:
        model, image_size = models.__dict__[cfg['model']['arch']](
            data=cfg['data']['dataset'],
            num_layers=cfg['model']['layers'],
            width_mult=cfg['model'].get('width_mult', 1.0),
            depth_mult=cfg['model'].get('depth_mult', 1.0),
            model_mult=cfg['model'].get('model_mult', 1.0),
        )
    return model, image_size


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith('module.') for k in state.keys()):
        return state
    out = {}
    for k, v in state.items():
        out[k[7:] if k.startswith('module.') else k] = v
    return out


def _load_state_dict(path: Path) -> Tuple[Dict[str, torch.Tensor], Optional[Dict]]:
    ckpt = torch.load(path, map_location='cpu')
    config = None
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
        config = ckpt.get('config')
    else:
        state = ckpt
    state = _strip_module_prefix(state)
    return state, config


def _merge_masks(masks: List[torch.Tensor], strategy: str) -> torch.Tensor:
    if strategy == 'first':
        return masks[0].clone()
    stacked = torch.stack(masks, dim=0)
    if strategy == 'vote':
        return (stacked.float().mean(dim=0) >= 0.5).float()
    if strategy == 'union':
        return (stacked.float().sum(dim=0) > 0).float()
    if strategy == 'intersection':
        return (stacked.float().sum(dim=0) == stacked.size(0)).float()
    if strategy == 'mean':
        return stacked.float().mean(dim=0)
    raise ValueError(f"Unknown mask strategy: {strategy}")


def _merge_state_dicts(state_dicts: List[Dict[str, torch.Tensor]], mask_strategy: str) -> Dict[str, torch.Tensor]:
    if not state_dicts:
        raise ValueError("No state_dicts provided")
    keys = state_dicts[0].keys()
    merged = {}
    for k in keys:
        tensors = [sd[k] for sd in state_dicts if k in sd]
        if len(tensors) != len(state_dicts):
            merged[k] = tensors[0]
            continue
        t0 = tensors[0]
        if not torch.is_tensor(t0):
            merged[k] = t0
            continue
        if k.endswith('num_batches_tracked'):
            merged[k] = t0
            continue
        if k.endswith('mask'):
            merged[k] = _merge_masks(tensors, mask_strategy)
            continue
        if torch.is_floating_point(t0):
            merged[k] = torch.stack(tensors, dim=0).mean(dim=0)
        else:
            merged[k] = t0
    return merged


def _reset_bn_stats(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if m.running_mean is not None:
                m.running_mean.zero_()
            if m.running_var is not None:
                m.running_var.fill_(1)
            if hasattr(m, 'num_batches_tracked') and m.num_batches_tracked is not None:
                m.num_batches_tracked.zero_()


def _forward(model, x, type_value: Optional[int]):
    if type_value is None:
        return model(x)
    try:
        return model(x, type_value)
    except TypeError:
        return model(x)


def _recompute_bn(model: nn.Module, loader, device: str, type_value: Optional[int], max_batches: int):
    model.train()
    _reset_bn_stats(model)
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if max_batches > 0 and i >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            _forward(model, x, type_value)


def _finetune(model: nn.Module, loader, device: str, type_value: Optional[int],
              epochs: int, lr: float, weight_decay: float, momentum: float, max_batches: int):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    model.train()
    for ep in range(epochs):
        for i, (x, y) in enumerate(loader):
            if max_batches > 0 and i >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = _forward(model, x, type_value)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()


def _compute_sparsity(model: nn.Module) -> Optional[float]:
    try:
        from pruning.dcil.mnn import MaskConv2d
    except Exception:
        return None
    total = 0
    zeros = 0
    for m in model.modules():
        if isinstance(m, MaskConv2d):
            total += m.mask.numel()
            zeros += (m.mask == 0).sum().item()
    if total == 0:
        return None
    return zeros / total


def main():
    ap = argparse.ArgumentParser(description="Merge multiple checkpoints by weight averaging")
    ap.add_argument('--method', type=str, required=True, choices=['static', 'dpf', 'dense'])
    ap.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    ap.add_argument('--sparsity', type=float, default=None, help='Required for static/dpf')
    ap.add_argument('--freeze-tag', type=str, default=None, help='DPF only: sparsity_<s>_<tag>')
    ap.add_argument('--seeds', type=int, nargs='+', default=None, help='Seeds to merge (default: auto-discover)')
    ap.add_argument('--out-seed', type=int, default=999, help='Seed index for merged output folder')
    ap.add_argument('--runs-dir', type=str, default='./runs')
    ap.add_argument('--device', type=str, default='cuda:0')
    ap.add_argument('--mask-strategy', type=str, default=None,
                    choices=['first', 'vote', 'union', 'intersection', 'mean'],
                    help='How to merge binary masks (default: static->first, dpf->vote)')
    ap.add_argument('--skip-bn', action='store_true', help='Skip BatchNorm stats recomputation')
    ap.add_argument('--bn-batches', type=int, default=200, help='Max batches for BN recalibration (0=all)')
    ap.add_argument('--finetune-epochs', type=int, default=0, help='Optional short finetune epochs')
    ap.add_argument('--finetune-lr', type=float, default=0.01, help='Finetune learning rate')
    ap.add_argument('--finetune-weight-decay', type=float, default=5e-4)
    ap.add_argument('--finetune-momentum', type=float, default=0.9)
    ap.add_argument('--finetune-batches', type=int, default=0, help='Max batches per epoch for finetune (0=all)')
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--datapath', type=str, default='~/Datasets/CIFAR')
    args = ap.parse_args()

    if args.method in ('static', 'dpf') and args.sparsity is None:
        raise ValueError("--sparsity is required for static/dpf")

    runs_dir = Path(args.runs_dir)
    freeze_tag = args.freeze_tag
    base_dir = _resolve_seed_dir(runs_dir, args.method, args.dataset, args.sparsity,
                                 seed=0, freeze_tag=freeze_tag).parent
    if args.method == 'dpf' and not base_dir.exists() and freeze_tag is None:
        # Auto-detect freeze tag if a single candidate exists
        dpf_root = runs_dir / 'dpf'
        candidates = sorted(dpf_root.glob(f"sparsity_{args.sparsity}*/{args.dataset}"))
        if len(candidates) == 1:
            base_dir = candidates[0]
            sp_dir = base_dir.parent.name  # sparsity_<s>_<tag> or sparsity_<s>
            rest = sp_dir.split('sparsity_', 1)[1]
            parts = rest.split('_', 1)
            if len(parts) > 1:
                freeze_tag = parts[1]
        elif len(candidates) > 1:
            raise ValueError("Multiple DPF freeze tags found. Specify --freeze-tag explicitly.")

    seeds = args.seeds if args.seeds else _auto_discover_seeds(base_dir)
    if len(seeds) < 2:
        raise ValueError(f"Need >=2 seeds to merge. Found: {seeds}")

    # Resolve seed dirs and checkpoints
    seed_dirs = []
    ckpts = []
    for s in seeds:
        sdir = _resolve_seed_dir(runs_dir, args.method, args.dataset, args.sparsity, s, freeze_tag)
        ckpt = sdir / 'best_model.pth'
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        seed_dirs.append(sdir)
        ckpts.append(ckpt)

    # Load config/model from first seed
    cfg = _load_config_from_seed_dir(seed_dirs[0])
    if cfg is None:
        raise FileNotFoundError(f"config.json not found near {seed_dirs[0]}")

    # Optional sanity check: warn if init_seed differs across particles
    init_seeds = []
    for sdir in seed_dirs:
        c = _load_config_from_seed_dir(sdir)
        if c and isinstance(c.get('system'), dict):
            init_seeds.append(c['system'].get('init_seed'))
    uniq_inits = sorted({s for s in init_seeds if s is not None})
    if len(uniq_inits) > 1:
        print(f"[WARN] Multiple init_seed values found across seeds: {uniq_inits}")

    model, image_size = _build_model_from_config(cfg)
    model = model.to(args.device)

    # Determine mask strategy
    if args.mask_strategy:
        mask_strategy = args.mask_strategy
    else:
        mask_strategy = 'first' if args.method == 'static' else ('vote' if args.method == 'dpf' else 'first')

    # Merge state dicts
    states = []
    for ckpt in ckpts:
        state, _ = _load_state_dict(ckpt)
        states.append(state)
    if mask_strategy == 'first' and args.method in ('static', 'dpf'):
        mismatched = 0
        for k in states[0].keys():
            if k.endswith('mask'):
                base = states[0][k]
                for sd in states[1:]:
                    if not torch.equal(base, sd[k]):
                        mismatched += 1
                        break
        if mismatched > 0:
            print(f"[WARN] {mismatched} mask tensors differ across seeds; using first-model masks.")
    merged_state = _merge_state_dicts(states, mask_strategy)
    model.load_state_dict(merged_state, strict=False)

    # Optional BN recalibration + short finetune
    needs_type_value = cfg.get('pruning', {}).get('enabled', False)
    type_value = 5 if needs_type_value else None
    train_loader, _ = CIFARLoader(
        args.batch_size, args.dataset, args.workers, args.datapath, image_size, True
    )
    recompute_bn = not args.skip_bn
    if recompute_bn:
        _recompute_bn(model, train_loader, args.device, type_value, args.bn_batches)
    if args.finetune_epochs > 0:
        _finetune(
            model, train_loader, args.device, type_value,
            args.finetune_epochs, args.finetune_lr, args.finetune_weight_decay,
            args.finetune_momentum, args.finetune_batches
        )

    # Save merged model into seed<out_seed>
    out_dir = _resolve_seed_dir(runs_dir, args.method, args.dataset, args.sparsity, args.out_seed, freeze_tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Update config with merge metadata
    merge_meta = {
        'method': args.method,
        'dataset': args.dataset,
        'sparsity': args.sparsity,
        'freeze_tag': freeze_tag,
        'seeds': seeds,
        'mask_strategy': mask_strategy,
        'out_seed': args.out_seed,
        'recompute_bn': recompute_bn,
        'bn_batches': args.bn_batches,
        'finetune_epochs': args.finetune_epochs,
        'finetune_lr': args.finetune_lr,
    }
    cfg = dict(cfg)
    cfg['name'] = f"merged_{args.method}_s{args.sparsity}_{args.dataset}_seed{args.out_seed}"
    cfg['merge'] = merge_meta

    with open(out_dir / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    with open(out_dir / 'merge_meta.json', 'w') as f:
        json.dump(merge_meta, f, indent=2)

    ckpt_out = {
        'epoch': int(args.finetune_epochs),
        'config': cfg,
        'state_dict': model.state_dict(),
        'best_acc1': None,
        'optimizer': None,
        'iteration': None,
    }
    torch.save(ckpt_out, out_dir / 'best_model.pth')

    merged_sparsity = _compute_sparsity(model)
    if merged_sparsity is not None:
        print(f"✅ Merged model saved to {out_dir} (sparsity≈{merged_sparsity:.4f})")
    else:
        print(f"✅ Merged model saved to {out_dir}")


if __name__ == '__main__':
    main()
