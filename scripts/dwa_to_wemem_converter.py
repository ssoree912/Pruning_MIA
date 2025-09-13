#!/usr/bin/env python3
"""
DWA 훈련 결과를 WeMeM-main 스타일 구조로 변환하는 스크립트
runs/dwa/ 구조를 result/{dataset}_{model}/ 구조로 변환
"""

import os
import shutil
import pickle
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so `datasets.py` is importable when run from scripts/
sys.path.append(str(Path(__file__).parent.parent))
import torch
from torch.utils.data import Subset
from datasets import get_dataset
import random
import numpy as np

def convert_dwa_to_wemem_structure(runs_dir='./runs', dataset='cifar10', model='resnet18', seed=7, mode=None, sparsity=None, limit_shadows=5, ckpt_globs="best_model.pth,best.pth,ckpt_best.pth"): 
    """DWA 결과를 WeMeM-main 구조로 변환

    Args:
        runs_dir: runs 루트 경로 (예: ./runs)
        dataset: 데이터셋 이름 (예: cifar10)
        model: 모델 이름 (예: resnet18)
        seed: 시드
        mode: 특정 모드만 선택 (kill_active_plain_dead | kill_and_reactivate | reactivate_only)
        sparsity: 특정 sparsity만 선택 (예: 0.6 또는 '0.6')
        limit_shadows: 그림자 모델 개수 제한
    """
    
    print("🔄 Converting DWA results to WeMeM-main structure...")
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create result directory
    result_dir = f"result/{dataset}_{model}"
    os.makedirs(result_dir, exist_ok=True)
    
    # Find DWA models
    dwa_dir = Path(runs_dir) / 'dwa'
    if not dwa_dir.exists():
        print(f"❌ No DWA results found in {runs_dir}")
        return False
    
    # Find first available DWA model as victim
    victim_model_path = None
    shadow_model_paths = []
    
    # Filter modes
    mode_dirs = []
    for md in dwa_dir.iterdir():
        if md.is_dir():
            if mode is None or md.name == mode:
                mode_dirs.append(md)

    if not mode_dirs:
        print(f"❌ No modes matched under {dwa_dir} with mode={mode}")
        return False

    # Helper to check sparsity dir filter
    def sparsity_ok(sp_dir):
        if sparsity is None:
            return sp_dir.name.startswith('sparsity_')
        s_str = str(sparsity)
        return sp_dir.name == f'sparsity_{s_str}'

    ds_key = dataset.casefold()
    ckpt_patterns = [p.strip() for p in ckpt_globs.split(',') if p.strip()]

    def find_ckpt_in_dir(root_dir: Path):
        # Try preferred names first
        for pat in ckpt_patterns:
            for p in root_dir.rglob(pat):
                if p.is_file():
                    return p
        # Fallback: any .pth under root_dir
        for p in root_dir.rglob('*.pth'):
            if p.is_file():
                return p
        return None

    # Collect all available models first
    all_models = []
    
    for mode_dir in mode_dirs:
        for sparsity_dir in mode_dir.iterdir():
            if sparsity_dir.is_dir() and (sparsity is None or sparsity_dir.name.startswith('sparsity_')):
                # Search for dataset dirs and checkpoints
                for sub in sparsity_dir.rglob('*'):
                    if sub.is_dir() and ds_key in sub.name.casefold():
                        model_path = find_ckpt_in_dir(sub)
                        if model_path is not None:
                            all_models.append((model_path, sub, sparsity_dir.name))
    
    if not all_models:
        print(f"❌ No DWA models found for dataset '{dataset}' under {dwa_dir}")
        return False
    
    # Sort by sparsity to get consistent ordering
    all_models.sort(key=lambda x: x[2])
    
    # If specific sparsity requested, use it as victim, others as shadows
    if sparsity is not None:
        target_sparsity = f'sparsity_{sparsity}'
        victim_models = [m for m in all_models if m[2] == target_sparsity]
        shadow_models = [m for m in all_models if m[2] != target_sparsity]
        
        if victim_models:
            victim_model_path = victim_models[0][0]
            victim_experiment_dir = victim_models[0][1]
            shadow_model_paths = [m[0] for m in shadow_models]
        else:
            print(f"❌ No model found for sparsity {sparsity}")
            return False
    else:
        # Use first as victim, rest as shadows
        victim_model_path = all_models[0][0]
        victim_experiment_dir = all_models[0][1]
        shadow_model_paths = [m[0] for m in all_models[1:]]
    
    print(f"✅ Found victim model: {victim_model_path}")
    print(f"✅ Found {len(shadow_model_paths)} shadow models")
    
    # Step 1: Create data_prepare.pkl
    print("📊 Creating data_prepare.pkl...")
    create_data_prepare_pkl(result_dir, dataset, seed)
    
    # Step 2: Convert victim model
    print("👤 Converting victim model...")
    victim_dir = f"{result_dir}/victim_model"
    os.makedirs(victim_dir, exist_ok=True)
    shutil.copy2(victim_model_path, f"{victim_dir}/best.pth")
    
    # Step 3: Convert shadow models  
    print("👥 Converting shadow models...")
    for i, shadow_path in enumerate(shadow_model_paths[:limit_shadows]):  # Limit number of shadow models
        shadow_dir = f"{result_dir}/shadow_model_{i}"
        os.makedirs(shadow_dir, exist_ok=True)
        shutil.copy2(shadow_path, f"{shadow_dir}/best.pth")
    
    # Determine prune tag
    prune_tag = str(sparsity) if sparsity is not None else '0.6'

    # Step 4: Create pruned model directories (use same models as "pruned" versions)
    print("✂️ Creating pruned model structure...")
    
    # Victim pruned (copy victim model as "pruned" version)
    pruned_victim_dir = f"{result_dir}/l1unstructure_{prune_tag}_model"
    os.makedirs(pruned_victim_dir, exist_ok=True)
    shutil.copy2(victim_model_path, f"{pruned_victim_dir}/best.pth")
    
    # Shadow pruned (copy shadow models as "pruned" versions)
    for i, shadow_path in enumerate(shadow_model_paths[:limit_shadows]):
        pruned_shadow_dir = f"{result_dir}/shadow_l1unstructure_{prune_tag}_model_{i}"
        os.makedirs(pruned_shadow_dir, exist_ok=True)
        shutil.copy2(shadow_path, f"{pruned_shadow_dir}/best.pth")
    
    # Step 5: Create log directory
    log_dir = f"log/{dataset}_{model}"
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"✅ Conversion complete!")
    print(f"📁 Result directory: {result_dir}")
    print(f"📁 Log directory: {log_dir}")
    
    return True

def create_data_prepare_pkl(result_dir, dataset_name, seed, shadow_num=5):
    """Create data_prepare.pkl file"""
    
    # Load datasets
    trainset = get_dataset(dataset_name, train=True)
    testset = get_dataset(dataset_name, train=False)
    
    # Create data splits
    n = len(trainset)
    idx = list(range(n))
    random.shuffle(idx)
    half = n // 2
    victim_idx, shadow_pool_idx = idx[:half], idx[half:]
    
    def split_7_1_2(index_list):
        m = len(index_list)
        a = int(m * 0.7)
        b = int(m * 0.1)
        train_idx = index_list[:a]
        dev_idx = index_list[a:a+b]  
        test_idx = index_list[a+b:]
        return Subset(trainset, train_idx), Subset(trainset, dev_idx), Subset(trainset, test_idx)
    
    victim_train_dataset, victim_dev_dataset, victim_test_dataset = split_7_1_2(victim_idx)
    
    # Shadow models
    per = len(shadow_pool_idx) // shadow_num
    attack_split_list = []
    shadow_train_list = []
    
    for s in range(shadow_num):
        part_idx = shadow_pool_idx[s*per:(s+1)*per]
        tr, dv, te = split_7_1_2(part_idx)
        attack_split_list.append((tr, dv, te))
        shadow_train_list.append(part_idx)
    
    # Save data_prepare.pkl
    data_prepare_path = f"{result_dir}/data_prepare.pkl"
    with open(data_prepare_path, "wb") as f:
        pickle.dump((victim_idx, victim_train_dataset, victim_dev_dataset, victim_test_dataset,
                    attack_split_list, shadow_train_list), f)
    
    print(f"💾 Created: {data_prepare_path}")
    print(f"   Victim train: {len(victim_train_dataset)}, dev: {len(victim_dev_dataset)}, test: {len(victim_test_dataset)}")
    print(f"   Shadow models: {len(attack_split_list)}")

def main():
    """Main conversion function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert DWA results to WeMeM-main structure')
    parser.add_argument('--runs_dir', default='./runs', help='DWA results directory')
    parser.add_argument('--dataset', default='cifar10', help='Dataset name')
    parser.add_argument('--model', default='resnet18', help='Model name') 
    parser.add_argument('--seed', type=int, default=7, help='Random seed')
    parser.add_argument('--mode', default=None, help='Select one mode to convert')
    parser.add_argument('--sparsity', default=None, help='Select one sparsity (e.g., 0.6)')
    parser.add_argument('--limit_shadows', type=int, default=5, help='Limit number of shadow models')
    parser.add_argument('--ckpt_globs', default='best_model.pth,best.pth,ckpt_best.pth', help='Comma-separated checkpoint filename patterns to search for')
    
    args = parser.parse_args()
    
    success = convert_dwa_to_wemem_structure(
        runs_dir=args.runs_dir,
        dataset=args.dataset, 
        model=args.model,
        seed=args.seed,
        mode=args.mode,
        sparsity=args.sparsity,
        limit_shadows=args.limit_shadows,
        ckpt_globs=args.ckpt_globs,
    )
    
    if success:
        print("\n🎉 Conversion successful!")
        print("\nNext steps:")
        print("1. Run MIA evaluation:")
        print(f"   python mia_modi.py 0 configs/{args.dataset}_{args.model}.json --attacks samia,threshold,nn,nn_top3,nn_cls")
        print("2. Check results in:")
        print(f"   log/{args.dataset}_{args.model}/")
    else:
        print("\n❌ Conversion failed!")

if __name__ == "__main__":
    main()
