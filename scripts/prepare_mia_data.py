#!/usr/bin/env python3
"""
MIA 실험용 데이터 분할 준비 스크립트
train_dwa.py 실행 전에 MIA용 데이터를 미리 분할해두는 스크립트

Usage:
python scripts/prepare_mia_data.py --dataset cifar10 --output_dir ./mia_data
"""

import os
import sys
import argparse
import random
import pickle
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Subset

# 부모 디렉토리를 path에 추가
sys.path.append(str(Path(__file__).parent.parent))

from data import cifar10_loader, cifar100_loader

def create_mia_data_splits(dataset_name, victim_ratio=0.4, shadow_ratio=0.4, test_ratio=0.1, seed=42):
    """
    MIA용 데이터 분할 생성
    
    Args:
        dataset_name: 데이터셋 이름 ('cifar10', 'cifar100')
        victim_ratio: victim 모델 훈련용 비율
        shadow_ratio: shadow 모델 훈련용 비율  
        test_ratio: 최종 테스트용 비율
        seed: 랜덤 시드
    
    Returns:
        dict: 분할된 인덱스들과 메타정보
    """
    print(f"🔄 Creating MIA data splits for {dataset_name}")
    
    # 시드 설정
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 데이터셋 로드
    if dataset_name == 'cifar10':
        train_loader, val_loader = cifar10_loader(1, 0, '~/Datasets/CIFAR', cuda=False)
        num_classes = 10
    elif dataset_name == 'cifar100':
        train_loader, val_loader = cifar100_loader(1, 0, '~/Datasets/CIFAR', cuda=False) 
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    
    print(f"📊 Dataset info: Train={len(train_dataset)}, Val={len(val_dataset)}, Classes={num_classes}")
    
    # 훈련 데이터 인덱스 섞기
    total_size = len(train_dataset)
    indices = list(range(total_size))
    random.shuffle(indices)
    
    # 비율에 따라 분할
    victim_size = int(total_size * victim_ratio)
    shadow_size = int(total_size * shadow_ratio)
    test_size = int(total_size * test_ratio)
    nonmember_size = total_size - victim_size - shadow_size - test_size
    
    # 인덱스 분할
    victim_indices = indices[:victim_size]
    shadow_indices = indices[victim_size:victim_size + shadow_size] 
    test_indices = indices[victim_size + shadow_size:victim_size + shadow_size + test_size]
    nonmember_indices = indices[victim_size + shadow_size + test_size:]
    
    # 클래스별 분포 확인
    def get_class_distribution(dataset, indices):
        targets = [dataset.targets[i] for i in indices]
        unique, counts = np.unique(targets, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    
    victim_dist = get_class_distribution(train_dataset, victim_indices)
    shadow_dist = get_class_distribution(train_dataset, shadow_indices)
    
    splits = {
        'dataset_name': dataset_name,
        'num_classes': num_classes,
        'total_size': total_size,
        'seed': seed,
        'ratios': {
            'victim': victim_ratio,
            'shadow': shadow_ratio, 
            'test': test_ratio,
            'nonmember': nonmember_size / total_size
        },
        'splits': {
            'victim_indices': victim_indices,
            'shadow_indices': shadow_indices,
            'test_indices': test_indices, 
            'nonmember_indices': nonmember_indices
        },
        'class_distributions': {
            'victim': victim_dist,
            'shadow': shadow_dist
        },
        'split_sizes': {
            'victim': len(victim_indices),
            'shadow': len(shadow_indices),
            'test': len(test_indices),
            'nonmember': len(nonmember_indices)
        }
    }
    
    print(f"✅ Split sizes: Victim={len(victim_indices)}, Shadow={len(shadow_indices)}, Test={len(test_indices)}, NonMember={len(nonmember_indices)}")
    
    return splits

def save_mia_splits(splits, output_dir):
    """MIA 분할 저장"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    dataset_name = splits['dataset_name']
    
    # 메인 분할 파일 저장
    split_file = output_dir / f'{dataset_name}_mia_splits.pkl'
    with open(split_file, 'wb') as f:
        pickle.dump(splits, f)
    
    # 메타정보 JSON으로도 저장
    import json
    meta_file = output_dir / f'{dataset_name}_mia_splits_info.json'
    
    # pickle 직렬화 불가능한 부분 제외하고 저장
    meta_info = {
        'dataset_name': splits['dataset_name'],
        'num_classes': splits['num_classes'],
        'total_size': splits['total_size'], 
        'seed': splits['seed'],
        'ratios': splits['ratios'],
        'split_sizes': splits['split_sizes'],
        'class_distributions': splits['class_distributions']
    }
    
    with open(meta_file, 'w') as f:
        json.dump(meta_info, f, indent=2)
    
    print(f"💾 Saved MIA splits:")
    print(f"   Data: {split_file}")
    print(f"   Info: {meta_file}")
    
    return split_file

def load_mia_splits(dataset_name, output_dir='./mia_data'):
    """저장된 MIA 분할 로드"""
    output_dir = Path(output_dir) 
    split_file = output_dir / f'{dataset_name}_mia_splits.pkl'
    
    if not split_file.exists():
        raise FileNotFoundError(f"MIA splits not found: {split_file}")
    
    with open(split_file, 'rb') as f:
        splits = pickle.load(f)
    
    print(f"📂 Loaded MIA splits from {split_file}")
    return splits

def verify_splits(splits, dataset_name):
    """분할 검증"""
    print(f"\n🔍 Verifying splits for {dataset_name}...")
    
    # 데이터셋 다시 로드해서 검증
    if dataset_name == 'cifar10':
        train_loader, _ = cifar10_loader(1, 0, '~/Datasets/CIFAR', cuda=False)
    elif dataset_name == 'cifar100':
        train_loader, _ = cifar100_loader(1, 0, '~/Datasets/CIFAR', cuda=False)
    else:
        return False
        
    dataset = train_loader.dataset
    
    # 인덱스 중복 확인
    all_indices = set()
    for split_name in ['victim_indices', 'shadow_indices', 'test_indices', 'nonmember_indices']:
        indices = splits['splits'][split_name]
        if all_indices & set(indices):
            print(f"❌ Overlap found in {split_name}")
            return False
        all_indices.update(indices)
    
    # 전체 크기 확인
    if len(all_indices) != len(dataset):
        print(f"❌ Size mismatch: expected {len(dataset)}, got {len(all_indices)}")
        return False
    
    print("✅ All splits are valid!")
    return True

def main():
    parser = argparse.ArgumentParser(description='Prepare MIA data splits')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'cifar100'], help='Dataset name')
    parser.add_argument('--output_dir', type=str, default='./mia_data', 
                       help='Output directory for MIA splits') 
    parser.add_argument('--victim_ratio', type=float, default=0.4,
                       help='Ratio for victim training data')
    parser.add_argument('--shadow_ratio', type=float, default=0.4, 
                       help='Ratio for shadow training data')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Ratio for test data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verify_only', action='store_true', 
                       help='Only verify existing splits')
    
    args = parser.parse_args()
    
    if args.verify_only:
        # 기존 분할만 검증
        try:
            splits = load_mia_splits(args.dataset, args.output_dir)
            verify_splits(splits, args.dataset)
        except FileNotFoundError:
            print(f"❌ No existing splits found for {args.dataset}")
        return
    
    print("🚀 Preparing MIA data splits...")
    print("=" * 50)
    
    # 비율 검증
    total_ratio = args.victim_ratio + args.shadow_ratio + args.test_ratio
    if total_ratio > 1.0:
        print(f"❌ Total ratio ({total_ratio}) exceeds 1.0")
        return
    
    nonmember_ratio = 1.0 - total_ratio
    print(f"📊 Split ratios: Victim={args.victim_ratio}, Shadow={args.shadow_ratio}, Test={args.test_ratio}, NonMember={nonmember_ratio:.3f}")
    
    # 분할 생성
    splits = create_mia_data_splits(
        args.dataset,
        victim_ratio=args.victim_ratio,
        shadow_ratio=args.shadow_ratio, 
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # 저장
    split_file = save_mia_splits(splits, args.output_dir)
    
    # 검증
    if verify_splits(splits, args.dataset):
        print("\n✅ MIA data splits prepared successfully!")
        print(f"📁 Use these splits in your MIA evaluation with: --mia_splits {split_file}")
    else:
        print("\n❌ Split verification failed!")

if __name__ == '__main__':
    main()