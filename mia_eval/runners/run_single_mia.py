#!/usr/bin/env python3
"""
단일 모델에 대한 MIA 평가 실행 스크립트.

현재 mia_modi.py는 직접 checkpoint 경로를 요구하므로,
이 runner는 runs/ 아래의 target/dense checkpoint를 찾아서
명시적으로 전달한다.
"""

import json
import os
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
CREATE_SPLITS = REPO_ROOT / 'mia_eval' / 'create_data' / 'create_fixed_data_splits.py'
MIA_CORE = REPO_ROOT / 'mia_eval' / 'core' / 'mia_modi.py'


def _resolve_ckpt(method: str, dataset: str, seed: int, sparsity='0.9', freeze_tag=None) -> Path:
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


def _result_json_path(dataset: str, prune_method: str, sparsity: str, victim_seed: int, freeze_tag=None, original=False) -> Path:
    suffix = '_original' if original else ''
    if prune_method == 'dense':
        return REPO_ROOT / 'mia_results' / 'dense' / f'{dataset}_victim{victim_seed}.json'
    if prune_method == 'dpf':
        tag = f'_{freeze_tag}' if freeze_tag else ''
        return REPO_ROOT / 'mia_results' / f'dpf{tag}' / f'{dataset}_sparsity_{sparsity}_victim{victim_seed}{suffix}.json'
    return REPO_ROOT / 'mia_results' / 'static' / f'{dataset}_sparsity_{sparsity}_victim{victim_seed}{suffix}.json'


def run_single_mia(
    dataset='cifar10',
    sparsity='0.9',
    prune_method='static',
    victim_seed=42,
    shadow_seeds=None,
    device=0,
    split_seed=7,
    forward_mode='standard',
    original=False,
    attacks='samia,threshold,nn,nn_top3,nn_cls,lira',
    debug=False,
    freeze_tag=None,
    auto_shadow=False,
    max_shadows=0,
    tpr_fprs='0.1,1,5',
    save_scores=False,
):
    if shadow_seeds is None:
        shadow_seeds = [43, 44, 45, 46, 47, 48, 49, 50]

    effective_original = bool(original and prune_method != 'dense')
    if original and prune_method == 'dense':
        print('[INFO] prune_method=dense 이므로 --original은 별도 의미가 없습니다. dense target 그대로 평가합니다.')

    print(f"🚀 Running MIA evaluation for dataset={dataset}")
    if prune_method in ('static', 'dpf'):
        tag_info = f" / tag={freeze_tag}" if (prune_method == 'dpf' and freeze_tag) else ''
        print(f"   Method: {prune_method.upper()}{tag_info}")
        print(f"   Sparsity: {sparsity}")
    else:
        print('   Method: DENSE')
    print(f"   Victim seed: {victim_seed}")
    print(f"   Shadow seeds (initial): {shadow_seeds}")
    print(f"   Original baseline mode: {effective_original}")

    print('\n🔍 Step 1: Checking model paths...')
    victim_path = _resolve_ckpt(prune_method, dataset, victim_seed, sparsity=sparsity, freeze_tag=freeze_tag)
    if not victim_path.exists():
        print(f"❌ Victim model not found: {victim_path}")
        return False
    print(f"✅ Found victim model: {victim_path}")

    if auto_shadow:
        if prune_method == 'static':
            base = REPO_ROOT / 'runs' / 'static' / f'sparsity_{sparsity}' / dataset
        elif prune_method == 'dpf':
            tag = f'_{freeze_tag}' if freeze_tag else ''
            base = REPO_ROOT / 'runs' / 'dpf' / f'sparsity_{sparsity}{tag}' / dataset
        else:
            base = REPO_ROOT / 'runs' / 'dense' / dataset
        auto_list = []
        if base.exists():
            for sd in sorted(base.glob('seed*')):
                cand = sd / 'best_model.pth'
                if not cand.exists():
                    continue
                try:
                    sid = int(sd.name.replace('seed', ''))
                except Exception:
                    continue
                if sid != victim_seed:
                    auto_list.append(sid)
        if max_shadows and max_shadows > 0:
            auto_list = auto_list[:max_shadows]
        if not auto_list:
            print('❌ Auto shadow discovery found no usable seeds. Provide --shadow_seeds explicitly.')
            return False
        shadow_seeds = auto_list
        print(f"🔎 Auto-discovered {len(auto_list)} shadow seeds: {auto_list}")

    print(f"   Using shadow seeds: {shadow_seeds}")
    shadow_paths = []
    for seed in shadow_seeds:
        shadow_path = _resolve_ckpt(prune_method, dataset, seed, sparsity=sparsity, freeze_tag=freeze_tag)
        if not shadow_path.exists():
            print(f"❌ Shadow model not found: {shadow_path}")
            return False
        shadow_paths.append(shadow_path)
    print(f"✅ Found all {len(shadow_paths)} shadow models")

    victim_dense_path = None
    shadow_dense_paths = []
    if effective_original:
        victim_dense_path = _resolve_ckpt('dense', dataset, victim_seed)
        if not victim_dense_path.exists():
            print(f"❌ Dense victim model not found for --original: {victim_dense_path}")
            return False
        for seed in shadow_seeds:
            dense_shadow = _resolve_ckpt('dense', dataset, seed)
            if not dense_shadow.exists():
                print(f"❌ Dense shadow model not found for --original: {dense_shadow}")
                return False
            shadow_dense_paths.append(dense_shadow)
        print('✅ Found dense baseline checkpoints for victim and shadows')

    print('\n🧩 Step 2: Ensuring fixed MIA data splits...')
    split_path = REPO_ROOT / 'mia_data_splits' / f'{dataset}_seed{split_seed}_victim{victim_seed}.pkl'
    if not split_path.exists():
        print(f"📦 Creating fixed splits: {split_path}")
        mk_cmd = [
            sys.executable,
            str(CREATE_SPLITS),
            '--dataset', dataset,
            '--seed', str(split_seed),
            '--victim_seed', str(victim_seed),
            '--shadow_seeds', *[str(s) for s in shadow_seeds],
        ]
        try:
            subprocess.run(mk_cmd, check=True, cwd=str(REPO_ROOT))
            print('✅ Fixed splits created.')
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create data splits: {e}")
            return False

    print('\n🎯 Step 3: Running MIA evaluation...')
    result_json = _result_json_path(
        dataset=dataset,
        prune_method=prune_method,
        sparsity=sparsity,
        victim_seed=victim_seed,
        freeze_tag=freeze_tag,
        original=effective_original,
    )
    result_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(MIA_CORE),
        '--device', str(device),
        '--dataset_name', dataset,
        '--victim_seed', str(victim_seed),
        '--seed', str(split_seed),
        '--forward_mode', forward_mode,
        '--attacks', attacks,
        '--result_file', str(result_json),
        '--victim_ckpt_path', str(victim_path.resolve()),
        '--shadow_seeds', *[str(s) for s in shadow_seeds],
        '--shadow_ckpt_paths', *[str(p.resolve()) for p in shadow_paths],
    ]
    if tpr_fprs:
        cmd += ['--tpr_fprs', str(tpr_fprs)]
    if save_scores:
        cmd += ['--save_scores']
    if debug:
        cmd += ['--debug']
    if effective_original:
        cmd += [
            '--original',
            '--victim_dense_ckpt_path', str(victim_dense_path.resolve()),
            '--shadow_dense_ckpt_paths', *[str(p.resolve()) for p in shadow_dense_paths],
        ]

    try:
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        print('✅ MIA evaluation successful!')
        if result_json.exists():
            print(f"\n📊 Results saved to: {result_json}")
            with open(result_json, 'r') as f:
                payload = json.load(f)
            print('\n📈 MIA Attack Results:')
            print(json.dumps(payload.get('results', {}), indent=2))
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ MIA evaluation failed: {e}")
        return False
    except FileNotFoundError:
        print('❌ mia_modi.py script not found')
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run single model MIA evaluation')
    parser.add_argument('--dataset', default='cifar10', help='Dataset name (cifar10, cifar100)')
    parser.add_argument('--model', default=None, help='(Ignored) Model name; resolved from config.json')
    parser.add_argument('--sparsity', default='0.9', help='Sparsity level')
    parser.add_argument('--prune_method', default='static', choices=['static', 'dpf', 'dense'], help='Pruning method')
    parser.add_argument('--victim_seed', type=int, default=42, help='Victim model seed')
    parser.add_argument('--shadow_seeds', nargs='+', type=int, default=[43, 44, 45, 46, 47, 48, 49, 50], help='Shadow model seeds')
    parser.add_argument('--device', type=int, default=0, help='GPU ID')
    parser.add_argument('--split_seed', type=int, default=7, help='Seed used for fixed MIA splits (must match pkl)')
    parser.add_argument('--forward_mode', type=str, default='standard', choices=['standard', 'scaling', 'dpf'], help='Model forward mode')
    parser.add_argument('--original', action='store_true', help='Attack dense baselines instead of target checkpoints')
    parser.add_argument('--attacks', default='samia,threshold,nn,nn_top3,nn_cls,lira', help='Comma-separated attacks to run')
    parser.add_argument('--debug', action='store_true', help='Enable debug prints inside mia_modi.py')
    parser.add_argument('--freeze_tag', type=str, default=None, help='DPF only: choose sparsity_<s>_<tag> (e.g., freeze180 or nofreeze)')
    parser.add_argument('--auto_shadow', action='store_true', help='Auto-discover all available shadow seeds under runs/')
    parser.add_argument('--max_shadows', type=int, default=0, help='Cap number of shadows when using auto discovery (>0 to cap, 0=all)')
    parser.add_argument('--tpr_fprs', type=str, default='0.1,1,5', help='Comma-separated FPR percentages for TPR@FPR (e.g., 0.1,1,5)')
    parser.add_argument('--save_scores', action='store_true', help='Save per-sample labels/scores for each attack')
    args = parser.parse_args()

    print('=' * 60)
    print('🎯 Single MIA Evaluation')
    print('=' * 60)

    success = run_single_mia(
        dataset=args.dataset,
        sparsity=args.sparsity,
        prune_method=args.prune_method,
        victim_seed=args.victim_seed,
        shadow_seeds=args.shadow_seeds,
        device=args.device,
        split_seed=args.split_seed,
        forward_mode=args.forward_mode,
        original=args.original,
        attacks=args.attacks,
        debug=args.debug,
        freeze_tag=args.freeze_tag,
        auto_shadow=args.auto_shadow,
        max_shadows=args.max_shadows,
        tpr_fprs=args.tpr_fprs,
        save_scores=args.save_scores,
    )

    if success:
        print('\n✅ Pipeline completed successfully!')
    else:
        print('\n❌ Pipeline failed!')
        sys.exit(1)


if __name__ == '__main__':
    main()
