#!/usr/bin/env python3
"""
Dense/Static/DPF 모델에 대한 MIA 평가 스크립트
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import sys
import argparse
import re

# Add current directory to path
sys.path.append('.')

from data import cifar10_loader
from models.resnet import resnet

class MIADataset(Dataset):
    """MIA용 데이터셋"""
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x, y = self.data[idx], self.targets[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

def load_model_from_checkpoint(model_path, config_path):
    """체크포인트에서 모델 로드 (설정 파일 참조)"""
    
    # 설정 파일에서 모델 타입 확인
    with open(config_path) as f:
        config = json.load(f)
    
    # 모델 생성 (훈련시와 동일하게)
    if config.get('pruning', {}).get('enabled', False):
        # Pruned 모델인 경우 - run_experiment.py와 동일한 방식 필요
        import pruning.dcil
        model, _ = pruning.models.__dict__['resnet'](
            data=config['data']['dataset'],
            num_layers=config['model']['layers']
        )
    else:
        # Dense 모델
        model, _ = resnet(data='cifar10', num_layers=20)
    
    # DataParallel 적용 (훈련시와 동일)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # 체크포인트 로드
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    return model

def confidence_based_mia(model, member_loader, non_member_loader, device):
    """Confidence 기반 MIA"""
    model.eval()
    
    member_confidences = []
    non_member_confidences = []
    
    with torch.no_grad():
        # Member 데이터
        for data, target in member_loader:
            data = data.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            max_probs = torch.max(probs, dim=1)[0]
            member_confidences.extend(max_probs.cpu().numpy())
        
        # Non-member 데이터
        for data, target in non_member_loader:
            data = data.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            max_probs = torch.max(probs, dim=1)[0]
            non_member_confidences.extend(max_probs.cpu().numpy())
    
    # AUC 계산 (간단한 threshold 방법)
    thresholds = np.linspace(0, 1, 1000)
    best_acc = 0
    
    for threshold in thresholds:
        member_pred = np.array(member_confidences) > threshold
        non_member_pred = np.array(non_member_confidences) <= threshold
        
        accuracy = (member_pred.sum() + non_member_pred.sum()) / (len(member_confidences) + len(non_member_confidences))
        best_acc = max(best_acc, accuracy)
    
    return {
        'member_conf_mean': float(np.mean(member_confidences)),
        'non_member_conf_mean': float(np.mean(non_member_confidences)),
        'attack_accuracy': float(best_acc),
        'member_conf_std': float(np.std(member_confidences)),
        'non_member_conf_std': float(np.std(non_member_confidences))
    }

def evaluate_model_mia(model_path, model_name, method, sparsity_percent):
    """단일 모델에 대한 MIA 평가"""
    
    print(f"🔍 {model_name} MIA 평가 중...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 로드
    try:
        config_path = model_path.parent / 'config.json'
        model = load_model_from_checkpoint(model_path, config_path)
        model = model.to(device)
        
        # 데이터 준비 (간단한 분할)
        train_loader, test_loader = cifar10_loader(128, 4, '~/Datasets/CIFAR', cuda=torch.cuda.is_available())
        train_dataset = train_loader.dataset
        
        # Member/Non-member 분할 (훈련셋의 절반씩)
        total_size = len(train_dataset)
        member_size = total_size // 2
        
        member_indices = list(range(member_size))
        non_member_indices = list(range(member_size, total_size))
        
        member_dataset = torch.utils.data.Subset(train_dataset, member_indices)
        non_member_dataset = torch.utils.data.Subset(train_dataset, non_member_indices)
        
        member_loader = DataLoader(member_dataset, batch_size=128, shuffle=False)
        non_member_loader = DataLoader(non_member_dataset, batch_size=128, shuffle=False)
        
        # MIA 수행
        mia_result = confidence_based_mia(model, member_loader, non_member_loader, device)
        
        # 결과 반환 (NumPy 타입을 Python 기본 타입으로 변환)
        result = {
            'name': model_name,
            'method': method,
            'sparsity_percent': int(sparsity_percent),
            'attack_accuracy': float(mia_result['attack_accuracy']),
            'member_confidence': float(mia_result['member_conf_mean']),
            'non_member_confidence': float(mia_result['non_member_conf_mean']),
            'confidence_gap': float(mia_result['member_conf_mean'] - mia_result['non_member_conf_mean'])
        }
        
        print(f"✅ {model_name}: Attack Acc={mia_result['attack_accuracy']:.3f}, Gap={result['confidence_gap']:.3f}")
        return result
        
    except Exception as e:
        print(f"❌ {model_name} MIA 평가 실패: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='MIA Evaluation')
    parser.add_argument('--model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--pruning_method', type=str, help='Pruning method (static/dpf)')  
    parser.add_argument('--sparsity', type=float, help='Sparsity level')
    
    args = parser.parse_args()
    
    # If specific model path is provided, evaluate only that model
    if args.model_path:
        model_path = Path(args.model_path)
        if model_path.exists():
            # Extract method and sparsity from arguments or path
            method = args.pruning_method if args.pruning_method else 'dense'
            sparsity = args.sparsity if args.sparsity is not None else 0.0
            sparsity_percent = int(sparsity * 100)
            
            config_path = model_path.parent / 'config.json'
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                model_name = config.get('name', 'unknown_model')
            else:
                model_name = model_path.stem
            
            result = evaluate_model_mia(model_path, model_name, method, sparsity_percent)
            if result:
                # Save individual result
                result_path = model_path.parent / 'mia_results.json'
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"MIA results saved to: {result_path}")
            return
    
    # Otherwise, evaluate all models in runs directory
    runs_dir = Path('./runs')
    mia_results = []
    
    print("🎯 Dense vs Static vs DPF MIA 평가")
    print("=" * 50)
    
    # Dense 모델들
    dense_dir = runs_dir / 'dense'
    if dense_dir.exists():
        for seed_dir in dense_dir.iterdir():
            if seed_dir.is_dir():
                model_path = seed_dir / 'best_model.pth'
                if model_path.exists():
                    with open(seed_dir / 'experiment_summary.json') as f:
                        summary = json.load(f)
                    model_name = summary['hyperparameters']['name']
                    
                    result = evaluate_model_mia(model_path, model_name, 'dense', 0)
                    if result:
                        mia_results.append(result)
    
    # Static/DPF 모델들
    for method in ['static', 'dpf']:
        method_dir = runs_dir / method
        if method_dir.exists():
            for sparsity_dir in method_dir.iterdir():
                if sparsity_dir.is_dir() and sparsity_dir.name.startswith('sparsity'):
                    # Parse sparsity from folder name using regex
                    name = sparsity_dir.name  # 예: 'sparsity_0.5'
                    m = re.match(r'^sparsity[_-]?([0-9]*\.?[0-9]+)$', name)
                    if not m:
                        print(f"Warning: Unrecognized sparsity folder name: {name}, skipping")
                        continue
                    sparsity = float(m.group(1))
                    sparsity_percent = int(sparsity * 100)
                    
                    for seed_dir in sparsity_dir.iterdir():
                        if seed_dir.is_dir():
                            model_path = seed_dir / 'best_model.pth'
                            if model_path.exists():
                                with open(seed_dir / 'experiment_summary.json') as f:
                                    summary = json.load(f)
                                model_name = summary['hyperparameters']['name']
                                
                                result = evaluate_model_mia(model_path, model_name, method, sparsity_percent)
                                if result:
                                    mia_results.append(result)
    
    # MIA 결과 CSV 저장
    if mia_results:
        import csv
        with open('mia_evaluation_results.csv', 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['name', 'method', 'sparsity_percent', 'attack_accuracy', 'member_confidence', 'non_member_confidence', 'confidence_gap']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(mia_results)
        
        print(f"\n📁 MIA 결과 저장: mia_evaluation_results.csv")
        print(f"✅ 총 {len(mia_results)}개 모델 MIA 평가 완료!")
        
        # 요약 출력
        print(f"\n📊 MIA 공격 성공률 요약:")
        print(f"{'Method':<8} {'Sparsity':<8} {'Attack Acc':<12} {'Conf Gap':<10}")
        print("-" * 45)
        
        mia_results.sort(key=lambda x: (x['sparsity_percent'], x['method']))
        for result in mia_results:
            sparsity_text = f"{result['sparsity_percent']}%" if result['sparsity_percent'] > 0 else "0%"
            print(f"{result['method']:<8} {sparsity_text:<8} {result['attack_accuracy']:<12.3f} {result['confidence_gap']:<10.3f}")
    
    else:
        print("❌ MIA 평가 결과가 없습니다.")

if __name__ == '__main__':
    main()