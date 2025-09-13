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
    고정 MIA 데이터 분할 생성 (훈련/테스트 분리 일관성 유지)

    핵심 변경점:
    - "멤버"(member)는 오직 학습 데이터셋(train set)에서만 선택
    - "비멤버"(non-member)는 오직 테스트 데이터셋(test set)에서만 선택

    과거 버전은 train+test를 합쳐 무작위로 절반을 victim/shadow 풀로 나눴기 때문에
    실제 학습 여부와 상관없는 레이블 잡음이 발생하여 MIA 지표가 무작위 수준으로 떨어질 수 있었습니다.
    본 구현은 학습 분포와 평가 분포를 일치시켜 지표 신뢰도를 높입니다.
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
        # Train-only 환경에서는 엄밀한 MIA 정의가 어렵지만, 동일 분포 내에서 분할
        total_indices = list(range(len(total_dataset)))
        victim_train_indices, victim_test_indices = train_test_split(
            total_indices, test_size=0.1, random_state=victim_seed
        )
        shadow_splits = {}
        for shadow_seed in shadow_seeds:
            s_tr, s_te = train_test_split(total_indices, test_size=0.1, random_state=shadow_seed)
            shadow_splits[shadow_seed] = {'train_indices': s_tr, 'test_indices': s_te}
            print(f"Shadow {shadow_seed} train(member): {len(s_tr)}, test(non-member): {len(s_te)}")
    else:
        total_dataset = ConcatDataset([trainset, testset])
        print(f"Using train+test: {len(trainset)} + {len(testset)} = {len(total_dataset)} samples")

        # ConcatDataset 인덱스 공간으로 변환
        train_offset = 0
        test_offset = len(trainset)
        train_indices_all = list(range(train_offset, train_offset + len(trainset)))
        test_indices_all  = list(range(test_offset,  test_offset  + len(testset)))

        # Victim: train에서 멤버, test에서 비멤버
        v_tr, _ = train_test_split(train_indices_all, test_size=0.1, random_state=victim_seed)
        v_te, _ = train_test_split(test_indices_all,  test_size=0.9, random_state=victim_seed)
        victim_train_indices = v_tr
        victim_test_indices  = v_te

        print(f"Victim train (member, from trainset): {len(victim_train_indices)}")
        print(f"Victim test (non-member, from testset): {len(victim_test_indices)}")

        # Shadow: victim이 차지한 풀을 제외한 잔여 풀에서 샘플링(중복 허용)
        remaining_train = [i for i in train_indices_all if i not in set(victim_train_indices)]
        remaining_test  = [i for i in test_indices_all if i not in set(victim_test_indices)]

        shadow_splits = {}
        for shadow_seed in shadow_seeds:
            s_tr, _ = train_test_split(remaining_train, test_size=0.1, random_state=shadow_seed)
            s_te, _ = train_test_split(remaining_test,  test_size=0.9, random_state=shadow_seed)
            shadow_splits[shadow_seed] = {
                'train_indices': s_tr,  # member (trainset only)
                'test_indices': s_te    # non-member (testset only)
            }
            print(f"Shadow {shadow_seed} train(member, trainset): {len(s_tr)}  test(non-member, testset): {len(s_te)}")
    
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
