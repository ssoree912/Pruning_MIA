#!/usr/bin/env python3
"""
Run MIA for multiple victim seeds by rotating victim among a seed set.

Unlike the older version, this script now passes explicit checkpoint paths to
mia_eval/core/mia_modi.py so it matches the current CLI.
"""

import argparse
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent


def _find_repo_root(start: Path) -> Path:
    for cand in [start] + list(start.parents):
        if (cand / '.git').exists():
            return cand
        if (cand / 'base_model.py').exists() and (cand / 'mia_eval').exists():
            return cand
    return start.parents[2]


REPO_ROOT = _find_repo_root(THIS_DIR)
MIA_CORE = REPO_ROOT / 'mia_eval' / 'core' / 'mia_modi.py'
CREATE_SPLITS = REPO_ROOT / 'mia_eval' / 'create_data' / 'create_fixed_data_splits.py'


def _resolve_ckpt(method: str, dataset: str, seed: int, sparsity, freeze_tag=None) -> Path:
    base = REPO_ROOT / 'runs'
    if method == 'static':
        return base / 'static' / f'sparsity_{sparsity}' / dataset / f'seed{seed}' / 'best_model.pth'
    if method == 'dpf':
        if freeze_tag:
            return base / 'dpf' / f'sparsity_{sparsity}_{freeze_tag}' / dataset / f'seed{seed}' / 'best_model.pth'
        matches = sorted((base / 'dpf').glob(f'sparsity_{sparsity}*/{dataset}/seed{seed}/best_model.pth'))
        if matches:
            return matches[0]
        return base / 'dpf' / f'sparsity_{sparsity}' / dataset / f'seed{seed}' / 'best_model.pth'
    if method == 'dense':
        return base / 'dense' / dataset / f'seed{seed}' / 'best_model.pth'
    raise ValueError(f'Unknown method: {method}')


def _expected_json_path(args, victim: int, sparsity, effective_original: bool) -> Path:
    suffix = '_original' if effective_original and args.prune_method != 'dense' else ''
    if args.prune_method == 'dense':
        return REPO_ROOT / 'mia_results' / 'dense' / f'{args.dataset}_victim{victim}.json'
    if args.prune_method == 'dpf':
        tag = f'_{args.freeze_tag}' if getattr(args, 'freeze_tag', None) else ''
        return REPO_ROOT / 'mia_results' / f'dpf{tag}' / f'{args.dataset}_sparsity_{sparsity}_victim{victim}{suffix}.json'
    return REPO_ROOT / 'mia_results' / 'static' / f'{args.dataset}_sparsity_{sparsity}_victim{victim}{suffix}.json'


def _ensure_split_pkl(victim: int, shadows: list, args) -> bool:
    pkl = REPO_ROOT / 'mia_data_splits' / f'{args.dataset}_seed{args.split_seed}_victim{victim}.pkl'
    if pkl.exists():
        return True
    cmd = [
        sys.executable,
        str(CREATE_SPLITS),
        '--dataset', args.dataset,
        '--seed', str(args.split_seed),
        '--victim_seed', str(victim),
        '--shadow_seeds', *[str(s) for s in shadows],
        '--save_dir', str(REPO_ROOT / 'mia_data_splits'),
    ]
    print('   [+] Creating split pkl: ' + str(pkl))
    print('       $ ' + ' '.join(cmd))
    try:
        res = subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        return res.returncode == 0 and pkl.exists()
    except subprocess.CalledProcessError as e:
        print(f'   ❌ Failed to create split pkl for victim {victim}: {e}')
        return False


def run_one(victim: int, shadows: list, args, sparsity) -> bool:
    effective_original = bool(args.original and args.prune_method != 'dense')
    out_json = _expected_json_path(args, victim, sparsity=sparsity, effective_original=effective_original)
    if out_json.exists() and not getattr(args, 'force', False):
        print(f'⏭️  Skip victim {victim}: result exists -> {out_json}')
        return True

    if not _ensure_split_pkl(victim, shadows, args):
        return False

    victim_path = _resolve_ckpt(args.prune_method, args.dataset, victim, sparsity=sparsity, freeze_tag=args.freeze_tag)
    if not victim_path.exists():
        print(f'❌ Missing victim checkpoint: {victim_path}')
        return False

    shadow_paths = []
    for seed in shadows:
        shadow_path = _resolve_ckpt(args.prune_method, args.dataset, seed, sparsity=sparsity, freeze_tag=args.freeze_tag)
        if not shadow_path.exists():
            print(f'❌ Missing shadow checkpoint: {shadow_path}')
            return False
        shadow_paths.append(shadow_path)

    cmd = [
        sys.executable,
        str(MIA_CORE),
        '--device', str(args.device),
        '--dataset_name', args.dataset,
        '--victim_seed', str(victim),
        '--seed', str(args.split_seed),
        '--forward_mode', args.forward_mode,
        '--attacks', args.attacks,
        '--tpr_fprs', args.tpr_fprs,
        '--batch_size', str(args.batch_size),
        '--result_file', str(out_json),
        '--victim_ckpt_path', str(victim_path.resolve()),
        '--shadow_seeds', *[str(s) for s in shadows],
        '--shadow_ckpt_paths', *[str(p.resolve()) for p in shadow_paths],
    ]
    if args.debug:
        cmd.append('--debug')
    if getattr(args, 'save_scores', False):
        cmd.append('--save_scores')

    if effective_original:
        victim_dense_path = _resolve_ckpt('dense', args.dataset, victim, sparsity=0.0)
        if not victim_dense_path.exists():
            print(f'❌ Missing dense victim checkpoint for --original: {victim_dense_path}')
            return False
        shadow_dense_paths = []
        for seed in shadows:
            dense_shadow = _resolve_ckpt('dense', args.dataset, seed, sparsity=0.0)
            if not dense_shadow.exists():
                print(f'❌ Missing dense shadow checkpoint for --original: {dense_shadow}')
                return False
            shadow_dense_paths.append(dense_shadow)
        cmd += [
            '--original',
            '--victim_dense_ckpt_path', str(victim_dense_path.resolve()),
            '--shadow_dense_ckpt_paths', *[str(p.resolve()) for p in shadow_dense_paths],
        ]

    print(f"\n▶️ Victim {victim} | Shadows {shadows} | sparsity={sparsity} | original={effective_original}")
    print('   $ ' + ' '.join(cmd))
    try:
        res = subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        return res.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f'❌ Failed for victim {victim}: {e}')
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--dataset', type=str, default='cifar10')
    ap.add_argument('--prune_method', type=str, default='static', choices=['static', 'dpf', 'dense'])
    ap.add_argument('--sparsity', type=float, default=None, help='Single sparsity if --sparsities not given')
    ap.add_argument('--sparsities', type=float, nargs='+', default=None, help='Multiple sparsities to iterate')
    ap.add_argument('--seeds', type=int, nargs='+', required=True, help='Seed set to rotate as victim')
    ap.add_argument('--forward_mode', type=str, default='standard', choices=['standard', 'scaling', 'dpf'])
    ap.add_argument('--attacks', type=str, default='samia,threshold,nn,nn_top3,nn_cls,lira')
    ap.add_argument('--tpr_fprs', type=str, default='0.1,1,5')
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--freeze_tag', type=str, default=None, help='DPF only: freeze tag in runs path (e.g., freeze180 or nofreeze)')
    ap.add_argument('--split_seed', type=int, default=7, help='Seed for fixed MIA data splits')
    ap.add_argument('--force', action='store_true', help='Re-run even if result JSON already exists')
    ap.add_argument('--save_scores', action='store_true', help='Save per-sample labels/scores for each attack')
    ap.add_argument('--original', action='store_true', help='Attack dense baselines instead of target checkpoints')
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    seeds = sorted(set(args.seeds))
    sparsity_list = args.sparsities if args.sparsities else ([args.sparsity] if args.sparsity is not None else [])
    if not sparsity_list:
        sparsity_list = [0.0] if args.prune_method == 'dense' else [0.9]

    ok, fail = 0, 0
    for sp in sparsity_list:
        print(f"\n==== Running MIA: method={args.prune_method} sparsity={sp} seeds={seeds} ====")
        for victim in seeds:
            shadows = [s for s in seeds if s != victim]
            if not shadows:
                print(f'⚠️ Skip victim {victim}: need at least 1 shadow seed')
                continue
            if run_one(victim, shadows, args, sparsity=sp):
                ok += 1
            else:
                fail += 1

    print(f'\nDone. Victims OK={ok}, Fail={fail}')


if __name__ == '__main__':
    main()
