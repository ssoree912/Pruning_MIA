#!/usr/bin/env python3
"""
고정된 MIA 데이터 분할 생성 스크립트
각 데이터셋에 대해 victim/shadow용 데이터 분할을 생성하고 pkl로 저장
"""

import argparse
import os
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import ConcatDataset, Subset

# datasets.py import
import sys
sys.path.append('WeMeM-main')

try:
    from datasets import get_dataset
except:
    print("Warning: Could not import get_dataset, using dummy implementation")
    def get_dataset(name, train=True):
        import torchvision
        import torchvision.transforms as transforms
        
        if name == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            return torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
        elif name == 'cifar100':
            transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            return torchvision.datasets.CIFAR100(root='./data', train=train, download=True, transform=transform)
        else:
            raise ValueError(f"Unsupported dataset: {name}")

def create_data_splits(dataset_name, seed=7, victim_seed=42, shadow_seeds=[43,44,45,46,47,48,49,50], 
                      save_dir="mia_data_splits"):
    """
    우리 구조에 맞는 고정 데이터 분할 생성
    - 모든 seed 모델이 전체 데이터로 학습했다고 가정
    - MIA 평가를 위해 victim은 특정 부분을, shadow들은 나머지 부분을 "멤버"로 간주
    """
    
    print(f"Creating fixed data splits for {dataset_name}")
    print(f"Victim seed: {victim_seed}")
    print(f"Shadow seeds: {shadow_seeds}")
    
    # 데이터셋 로드
    trainset = get_dataset(dataset_name, train=True)
    testset = get_dataset(dataset_name, train=False)
    
    if testset is None:
        total_dataset = trainset
        print(f"Using trainset only: {len(trainset)} samples")
    else:
        total_dataset = ConcatDataset([trainset, testset])
        print(f"Using train+test: {len(trainset)} + {len(testset)} = {len(total_dataset)} samples")
    
    # 전체 데이터를 victim/shadow 풀로 분할 (50:50)
    total_indices = list(range(len(total_dataset)))
    victim_pool_indices, shadow_pool_indices = train_test_split(
        total_indices, test_size=0.5, random_state=seed
    )
    
    print(f"Victim pool: {len(victim_pool_indices)} samples")
    print(f"Shadow pool: {len(shadow_pool_indices)} samples")
    
    # Victim 데이터 분할 (90% train, 10% test)
    victim_train_indices, victim_test_indices = train_test_split(
        victim_pool_indices, test_size=0.1, random_state=victim_seed
    )
    
    print(f"Victim train (member): {len(victim_train_indices)} samples")
    print(f"Victim test (non-member): {len(victim_test_indices)} samples")
    
    # Shadow 데이터 분할들 (각 shadow마다 90% train, 10% test)
    shadow_splits = {}
    for shadow_seed in shadow_seeds:
        shadow_train_indices, shadow_test_indices = train_test_split(
            shadow_pool_indices, test_size=0.1, random_state=shadow_seed
        )
        
        shadow_splits[shadow_seed] = {
            'train_indices': shadow_train_indices,  # member
            'test_indices': shadow_test_indices     # non-member
        }
        
        print(f"Shadow {shadow_seed} train (member): {len(shadow_train_indices)} samples")
        print(f"Shadow {shadow_seed} test (non-member): {len(shadow_test_indices)} samples")
    
    # 저장할 데이터 구조
    data_splits = {
        'dataset_name': dataset_name,
        'seed': seed,
        'victim_seed': victim_seed,
        'shadow_seeds': shadow_seeds,
        'total_size': len(total_dataset),
        'victim': {
            'train_indices': victim_train_indices,  # victim이 "본" 데이터 (member)
            'test_indices': victim_test_indices     # victim이 "안 본" 데이터 (non-member)
        },
        'shadows': shadow_splits
    }
    
    # 저장
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{dataset_name}_seed{seed}_victim{victim_seed}.pkl"
    
    with open(save_path, 'wb') as f:
        pickle.dump(data_splits, f)
    
    print(f"\n✅ Data splits saved to: {save_path}")
    return save_path

def verify_splits(pkl_path):
    """생성된 분할 검증"""
    with open(pkl_path, 'rb') as f:
        splits = pickle.load(f)
    
    print(f"\n📊 Verification:")
    print(f"Dataset: {splits['dataset_name']}")
    print(f"Total size: {splits['total_size']}")
    print(f"Victim {splits['victim_seed']}:")
    print(f"  - Members: {len(splits['victim']['train_indices'])}")
    print(f"  - Non-members: {len(splits['victim']['test_indices'])}")
    
    for shadow_seed, shadow_data in splits['shadows'].items():
        print(f"Shadow {shadow_seed}:")
        print(f"  - Members: {len(shadow_data['train_indices'])}")
        print(f"  - Non-members: {len(shadow_data['test_indices'])}")
    
    # 중복 검사
    victim_all = set(splits['victim']['train_indices'] + splits['victim']['test_indices'])
    shadow_all = set()
    for shadow_data in splits['shadows'].values():
        shadow_all.update(shadow_data['train_indices'])
        shadow_all.update(shadow_data['test_indices'])
    
    overlap = victim_all.intersection(shadow_all)
    print(f"\n🔍 Overlap check:")
    print(f"Victim total indices: {len(victim_all)}")
    print(f"Shadow total indices: {len(shadow_all)}")
    print(f"Overlap: {len(overlap)} (should be 0 for proper split)")

def main():
    parser = argparse.ArgumentParser(description='Create fixed MIA data splits')
    parser.add_argument('--dataset', default='cifar10', help='Dataset name')
    parser.add_argument('--seed', type=int, default=7, help='Random seed for main split')
    parser.add_argument('--victim_seed', type=int, default=42, help='Victim model seed')
    parser.add_argument('--shadow_seeds', nargs='+', type=int, 
                       default=[43,44,45,46,47,48,49,50], help='Shadow model seeds')
    parser.add_argument('--save_dir', default='mia_data_splits', help='Directory to save splits')
    parser.add_argument('--verify', action='store_true', help='Verify after creation')
    
    args = parser.parse_args()
    
    # 분할 생성
    pkl_path = create_data_splits(
        dataset_name=args.dataset,
        seed=args.seed,
        victim_seed=args.victim_seed,
        shadow_seeds=args.shadow_seeds,
        save_dir=args.save_dir
    )
    
    # 검증
    if args.verify:
        verify_splits(pkl_path)

if __name__ == '__main__':
    main()

