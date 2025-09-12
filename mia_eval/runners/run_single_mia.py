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
                  split_seed=7, forward_mode='standard', original=False):
    """같은 sparsity, 다른 seed 모델들에 대한 MIA 평가 실행"""
    
    print(f"🚀 Running MIA evaluation for dataset={dataset} (arch=auto from config)")
    print(f"   Sparsity: {sparsity}")
    print(f"   Alpha: {alpha}, Beta: {beta}")
    print(f"   Victim seed: {victim_seed}")
    print(f"   Shadow seeds: {shadow_seeds}")
    
    # Step 1: 모델 경로 확인
    print("\n🔍 Step 1: Checking model paths...")
    
    base_path = "runs"
    
    # Victim 모델 확인
    victim_path = f"{base_path}/{prune_method}/{prune_type}/sparsity_{sparsity}/{dataset}/alpha{alpha}_beta{beta}/seed{victim_seed}/best_model.pth"
    if not os.path.exists(victim_path):
        print(f"❌ Victim model not found: {victim_path}")
        return False
    print(f"✅ Found victim model: {victim_path}")
    
    # Shadow 모델들 확인
    for seed in shadow_seeds:
        shadow_path = f"{base_path}/{prune_method}/{prune_type}/sparsity_{sparsity}/{dataset}/alpha{alpha}_beta{beta}/seed{seed}/best_model.pth"
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
            mk_res = subprocess.run(mk_cmd, check=True, capture_output=True, text=True, cwd=str(REPO_ROOT))
            if mk_res.stdout:
                print(mk_res.stdout)
            print("✅ Fixed splits created.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create data splits: {e}")
            if e.stdout:
                print("STDOUT:", e.stdout)
            if e.stderr:
                print("STDERR:", e.stderr)
            return False

    # Step 3: MIA 평가 실행 (설정 파일 없이 직접 파라미터 전달)
    print("\n🎯 Step 3: Running MIA evaluation...")
    
    cmd = [
        sys.executable, str(MIA_CORE),
        '--device', str(device),
        '--dataset_name', dataset,
        # model_name resolved from config; no CLI override
        '--sparsity', str(sparsity),
        '--alpha', str(alpha),
        '--beta', str(beta),
        '--victim_seed', str(victim_seed),
        '--seed', str(split_seed),
        '--shadow_seeds'] + [str(s) for s in shadow_seeds] + [
        '--prune_method', prune_method,
        '--prune_type', prune_type,
        '--forward_mode', forward_mode,
        '--attacks', args.attacks
    ]
    if original:
        cmd.append('--original')
    
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
    parser.add_argument('--prune_method', default='dwa', help='Pruning method')
    parser.add_argument('--prune_type', default='reactivate_only', help='Pruning type')
    parser.add_argument('--victim_seed', type=int, default=42, help='Victim model seed')
    parser.add_argument('--shadow_seeds', nargs='+', type=int, default=[43,44,45,46,47,48,49,50], help='Shadow model seeds')
    parser.add_argument('--device', type=int, default=0, help='GPU ID')
    parser.add_argument('--split_seed', type=int, default=7, help='Seed used for fixed MIA splits (must match pkl)')
    parser.add_argument('--forward_mode', type=str, default='standard', choices=['standard','dwa_adaptive','scaling','dpf'], help='Model forward mode')
    parser.add_argument('--original', action='store_true', help='Attack original (unpruned) models')
    parser.add_argument('--attacks', default='samia,threshold,nn,nn_top3,nn_cls,lira', help='Comma-separated attacks to run')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 DWA MIA Evaluation Pipeline")
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
        original=args.original
    )
    
    if success:
        print("\n✅ Pipeline completed successfully!")
    else:
        print("\n❌ Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
