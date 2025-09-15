#!/usr/bin/env python3
"""
단일 모델에 대한 MIA 평가 실행 스크립트
DWA 훈련 결과를 WeMeM-main 구조로 변환하고 MIA 평가 실행
"""

import os
import sys
import subprocess
from pathlib import Path

# Ensure project root and absolute script paths (do NOT hardcode project name)
THIS_DIR = Path(__file__).resolve().parent

def _find_repo_root(start: Path) -> Path:
    # Walk up to find a folder that looks like the project root
    for cand in [start] + list(start.parents):
        if (cand / '.git').exists():
            return cand
        if (cand / 'base_model.py').exists() and (cand / 'mia_eval').exists():
            return cand
    # Fallback: assume two levels up from runners/
    return start.parents[2]

REPO_ROOT = _find_repo_root(THIS_DIR)
CREATE_SPLITS = REPO_ROOT / 'mia_eval' / 'create_data' / 'create_fixed_data_splits.py'
MIA_CORE = REPO_ROOT / 'mia_eval' / 'core' / 'mia_modi.py'

def run_single_mia(dataset='cifar10', sparsity='0.9', alpha='5.0', beta='5.0', 
                  prune_method='dwa', prune_type='reactivate_only', 
                  victim_seed=42, shadow_seeds=[43,44,45,46,47,48,49,50], device=0,
                  split_seed=7, forward_mode='standard', original=False,
                  attacks='samia,threshold,nn,nn_top3,nn_cls,lira', debug=False,
                  freeze_tag=None, auto_shadow=False, max_shadows=0):
    """같은 sparsity, 다른 seed 모델들에 대한 MIA 평가 실행"""
    
    print(f"🚀 Running MIA evaluation for dataset={dataset} (arch=auto from config)")
    if prune_method == 'dwa':
        print(f"   Method: DWA / mode={prune_type}")
        print(f"   Sparsity: {sparsity}")
        print(f"   Alpha: {alpha}, Beta: {beta}")
    elif prune_method in ('static','dpf'):
        tag_info = f" / tag={freeze_tag}" if (prune_method=='dpf' and freeze_tag) else ""
        print(f"   Method: {prune_method.upper()}{tag_info}")
        print(f"   Sparsity: {sparsity}")
    elif prune_method == 'dense':
        print("   Method: DENSE (no sparsity/alpha/beta)")
    print(f"   Victim seed: {victim_seed}")
    print(f"   Shadow seeds (initial): {shadow_seeds}")
    
    # Step 1: 모델 경로 확인
    print("\n🔍 Step 1: Checking model paths...")
    
    base_path = "runs"
    
    def _resolve_path(method: str, seed: int) -> str:
        if method == 'dwa':
            return f"{base_path}/{prune_method}/{prune_type}/sparsity_{sparsity}/{dataset}/alpha{alpha}_beta{beta}/seed{seed}/best_model.pth"
        if method == 'static':
            return f"{base_path}/static/sparsity_{sparsity}/{dataset}/seed{seed}/best_model.pth"
        if method == 'dpf':
            # optional freeze_tag: sparsity_<s>_<tag>
            if freeze_tag:
                return f"{base_path}/dpf/sparsity_{sparsity}_{freeze_tag}/{dataset}/seed{seed}/best_model.pth"
            # try to find any matching folder starting with sparsity_<s>
            root = (REPO_ROOT / base_path / 'dpf').resolve()
            for cand in root.glob(f"sparsity_{sparsity}*/{dataset}/seed{seed}/best_model.pth"):
                return str(cand)
            return f"{base_path}/dpf/sparsity_{sparsity}/{dataset}/seed{seed}/best_model.pth"
        if method == 'dense':
            return f"{base_path}/dense/{dataset}/seed{seed}/best_model.pth"
        return ''

    # Victim 모델 확인
    victim_path = _resolve_path(prune_method, victim_seed)
    if not os.path.exists(victim_path):
        print(f"❌ Victim model not found: {victim_path}")
        return False
    print(f"✅ Found victim model: {victim_path}")

    # Auto-discover shadow seeds if requested
    if auto_shadow:
        base = None
        if prune_method == 'dwa':
            base = REPO_ROOT / 'runs' / 'dwa' / prune_type / f'sparsity_{sparsity}' / dataset / f'alpha{alpha}_beta{beta}'
        elif prune_method == 'static':
            base = REPO_ROOT / 'runs' / 'static' / f'sparsity_{sparsity}' / dataset
        elif prune_method == 'dpf':
            tag = f'_{freeze_tag}' if freeze_tag else ''
            base = REPO_ROOT / 'runs' / 'dpf' / f'sparsity_{sparsity}{tag}' / dataset
        elif prune_method == 'dense':
            base = REPO_ROOT / 'runs' / 'dense' / dataset
        auto_list = []
        if base and base.exists():
            for sd in sorted(base.glob('seed*')):
                cand = sd / 'best_model.pth'
                if cand.exists():
                    try:
                        sid = int(sd.name.replace('seed',''))
                        if sid != victim_seed:
                            auto_list.append(sid)
                    except Exception:
                        continue
        if max_shadows and max_shadows > 0:
            auto_list = auto_list[:max_shadows]
        if not auto_list:
            print("❌ Auto shadow discovery found no usable seeds. Provide --shadow_seeds explicitly.")
            return False
        print(f"🔎 Auto-discovered {len(auto_list)} shadow seeds: {auto_list}")
        shadow_seeds = auto_list
    # Log the final list that will actually be used downstream
    print(f"   Using shadow seeds: {shadow_seeds}")

    # Shadow 모델들 확인
    for seed in shadow_seeds:
        shadow_path = _resolve_path(prune_method, seed)
        if not os.path.exists(shadow_path):
            print(f"❌ Shadow model not found: {shadow_path}")
            return False
    print(f"✅ Found all {len(shadow_seeds)} shadow models")
    
    # Step 2: 고정 스플릿 확인 및 없으면 생성
    print("\n🧩 Step 2: Ensuring fixed MIA data splits...")
    split_path = REPO_ROOT / 'mia_data_splits' / f"{dataset}_seed{split_seed}_victim{victim_seed}.pkl"
    if not split_path.exists():
        print(f"📦 Creating fixed splits: {split_path}")
        mk_cmd = [
            sys.executable, str(CREATE_SPLITS),
            '--dataset', dataset,
            '--seed', str(split_seed),
            '--victim_seed', str(victim_seed),
            '--shadow_seeds', *[str(s) for s in shadow_seeds]
        ]
        try:
            # Stream output live for visibility (dataset load can take time)
            subprocess.run(mk_cmd, check=True, cwd=str(REPO_ROOT))
            print("✅ Fixed splits created.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create data splits: {e}")
            return False

    # Step 3: MIA 평가 실행 (설정 파일 없이 직접 파라미터 전달)
    print("\n🎯 Step 3: Running MIA evaluation...")
    
    cmd = [
        sys.executable, str(MIA_CORE),
        '--device', str(device),
        '--dataset_name', dataset,
        # model_name resolved from config; no CLI override
        '--sparsity', str(sparsity),
        '--victim_seed', str(victim_seed),
        '--seed', str(split_seed),
        '--shadow_seeds'] + [str(s) for s in shadow_seeds] + [
        '--prune_method', prune_method,
        '--prune_type', prune_type,
        '--forward_mode', forward_mode,
        '--attacks', attacks
    ]
    if prune_method == 'dwa':
        cmd += ['--alpha', str(alpha), '--beta', str(beta)]
    if freeze_tag and prune_method == 'dpf':
        cmd += ['--freeze_tag', str(freeze_tag)]
    if original:
        cmd.append('--original')
    if debug:
        cmd.append('--debug')
    
    try:
        # Stream child process output live (no capture) so progress is visible
        result = subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        print("✅ MIA evaluation successful!")
        
        # 결과 파일 확인
        result_file = f"mia_results/{prune_method}_{prune_type}/sparsity_{sparsity}_alpha{alpha}_beta{beta}_victim{victim_seed}.json"
        if os.path.exists(result_file):
            print(f"\n📊 Results saved to: {result_file}")
            print("\n📈 MIA Attack Results:")
            import json
            with open(result_file, 'r') as f:
                results = json.load(f)
                print(json.dumps(results['results'], indent=2))
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ MIA evaluation failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ mia_modi.py script not found")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run single model MIA evaluation')
    parser.add_argument('--dataset', default='cifar10', help='Dataset name (cifar10, cifar100)')
    # Model is resolved from config; keep option for backward-compat but ignore
    parser.add_argument('--model', default=None, help='(Ignored) Model name; resolved from config.json')
    parser.add_argument('--sparsity', default='0.9', help='Sparsity level')
    parser.add_argument('--alpha', default='5.0', help='Alpha value')
    parser.add_argument('--beta', default='5.0', help='Beta value')
    parser.add_argument('--prune_method', default='dwa', choices=['dwa','static','dpf','dense'], help='Pruning method')
    parser.add_argument('--prune_type', default='reactivate_only', help='Pruning type (DWA mode)')
    parser.add_argument('--victim_seed', type=int, default=42, help='Victim model seed')
    parser.add_argument('--shadow_seeds', nargs='+', type=int, default=[43,44,45,46,47,48,49,50], help='Shadow model seeds')
    parser.add_argument('--device', type=int, default=0, help='GPU ID')
    parser.add_argument('--split_seed', type=int, default=7, help='Seed used for fixed MIA splits (must match pkl)')
    parser.add_argument('--forward_mode', type=str, default='standard', choices=['standard','dwa_adaptive','scaling','dpf'], help='Model forward mode')
    parser.add_argument('--original', action='store_true', help='Attack original (unpruned) models')
    parser.add_argument('--attacks', default='samia,threshold,nn,nn_top3,nn_cls,lira', help='Comma-separated attacks to run')
    parser.add_argument('--debug', action='store_true', help='Enable debug prints inside mia_modi.py')
    parser.add_argument('--freeze_tag', type=str, default=None, help='DPF only: choose sparsity_<s>_<tag> (e.g., freeze180 or nofreeze)')
    parser.add_argument('--auto_shadow', action='store_true', help='Auto-discover all available shadow seeds under runs/')
    parser.add_argument('--max_shadows', type=int, default=0, help='Cap number of shadows when using auto discovery (>0 to cap, 0=all)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 Single MIA Evaluation")
    print("=" * 60)
    
    success = run_single_mia(
        dataset=args.dataset,
        sparsity=args.sparsity,
        alpha=args.alpha,
        beta=args.beta,
        prune_method=args.prune_method,
        prune_type=args.prune_type,
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
        max_shadows=args.max_shadows
    )
    
    if success:
        print("\n✅ Pipeline completed successfully!")
    else:
        print("\n❌ Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
