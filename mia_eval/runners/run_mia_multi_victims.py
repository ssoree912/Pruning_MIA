#!/usr/bin/env python3
"""
Run MIA for multiple victim seeds by rotating victim among a seed set.

For each victim seed V in --seeds, uses shadows = (seeds - {V}).
Delegates to mia_eval/core/mia_modi.py so results land under mia_results/.

Example:
  python mia_eval/runners/run_mia_multi_victims.py \
    --device 0 --dataset cifar10 \
    --prune_method dwa --prune_type kill_active_plain_dead \
    --sparsity 0.9 --alpha 5.0 --beta 5.0 \
    --seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 \
    --attacks samia,threshold,nn,nn_top3,nn_cls,lira --debug
"""

import argparse
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
MIA_CORE = REPO_ROOT / 'mia_eval' / 'core' / 'mia_modi.py'
CREATE_SPLITS = REPO_ROOT / 'mia_eval' / 'create_data' / 'create_fixed_data_splits.py'


def _ensure_split_pkl(victim: int, shadows: list, args) -> bool:
    """Create fixed-split pkl if missing for this victim seed."""
    pkl = REPO_ROOT / 'mia_data_splits' / f"{args.dataset}_seed{args.split_seed}_victim{victim}.pkl"
    if pkl.exists():
        return True
    cmd = [
        sys.executable, str(CREATE_SPLITS),
        '--dataset', args.dataset,
        '--seed', str(args.split_seed),
        '--victim_seed', str(victim),
        '--shadow_seeds', *[str(s) for s in shadows],
        '--save_dir', str(REPO_ROOT / 'mia_data_splits')
    ]
    print('   [+] Creating split pkl: ' + str(pkl))
    print('       $ ' + ' '.join(cmd))
    try:
        res = subprocess.run(cmd, check=True)
        return res.returncode == 0 and pkl.exists()
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to create split pkl for victim {victim}: {e}")
        return False


def run_one(victim: int, shadows: list, args) -> bool:
    # Ensure fixed split exists for this victim
    if not _ensure_split_pkl(victim, shadows, args):
        return False
    cmd = [
        sys.executable, str(MIA_CORE),
        '--device', str(args.device),
        '--dataset_name', args.dataset,
        '--sparsity', str(args.sparsity),
        '--victim_seed', str(victim),
        '--seed', str(args.split_seed),
        '--alpha', str(args.alpha),
        '--beta', str(args.beta),
        '--prune_method', args.prune_method,
        '--prune_type', args.prune_type,
        '--forward_mode', args.forward_mode,
        '--attacks', args.attacks,
        '--tpr_fprs', args.tpr_fprs,
        '--batch_size', str(args.batch_size),
    ]
    # shadow seeds (space separated)
    cmd.append('--shadow_seeds')
    cmd.extend([str(s) for s in shadows])
    if args.debug:
        cmd.append('--debug')

    print(f"\n▶️ Victim {victim} | Shadows {shadows}")
    print('   $ ' + ' '.join(cmd))
    try:
        res = subprocess.run(cmd, check=True)
        return res.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed for victim {victim}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--dataset', type=str, default='cifar10')
    ap.add_argument('--prune_method', type=str, default='dwa', choices=['dwa','static','dpf','dense'])
    ap.add_argument('--prune_type', type=str, default='kill_active_plain_dead')
    ap.add_argument('--sparsity', type=float, default=0.9)
    ap.add_argument('--alpha', type=float, default=5.0)
    ap.add_argument('--beta', type=float, default=5.0)
    ap.add_argument('--seeds', type=int, nargs='+', required=True, help='Seed set to rotate as victim')
    ap.add_argument('--forward_mode', type=str, default='standard', choices=['standard','dwa_adaptive','scaling','dpf'])
    ap.add_argument('--attacks', type=str, default='samia,threshold,nn,nn_top3,nn_cls,lira')
    ap.add_argument('--tpr_fprs', type=str, default='0.1,1,5')
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--split_seed', type=int, default=7, help='Seed for fixed MIA data splits')
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    seeds = sorted(set(args.seeds))
    ok, fail = 0, 0
    for v in seeds:
        shadows = [s for s in seeds if s != v]
        if not shadows:
            print(f"⚠️ Skip victim {v}: need at least 1 shadow seed")
            continue
        if run_one(v, shadows, args):
            ok += 1
        else:
            fail += 1

    print(f"\nDone. Victims OK={ok}, Fail={fail}")


if __name__ == '__main__':
    main()
