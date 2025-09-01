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

# Add current directory to path
sys.path.append('.')

from data import cifar10_loader
from models.resnet import ResNet

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

def load_model_from_checkpoint(model_path, arch='resnet', layers=20):
    """체크포인트에서 모델 로드"""
    
    model = ResNet(arch=arch, layers=layers, num_classes=10)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
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
        'member_conf_mean': np.mean(member_confidences),
        'non_member_conf_mean': np.mean(non_member_confidences),
        'attack_accuracy': best_acc,
        'member_conf_std': np.std(member_confidences),
        'non_member_conf_std': np.std(non_member_confidences)
    }

def evaluate_model_mia(model_path, model_name, method, sparsity_percent):
    """단일 모델에 대한 MIA 평가"""
    
    print(f"🔍 {model_name} MIA 평가 중...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 로드
    try:
        model = load_model_from_checkpoint(model_path)
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
        
        # 결과 반환
        result = {
            'name': model_name,
            'method': method,
            'sparsity_percent': sparsity_percent,
            'attack_accuracy': mia_result['attack_accuracy'],
            'member_confidence': mia_result['member_conf_mean'],
            'non_member_confidence': mia_result['non_member_conf_mean'],
            'confidence_gap': mia_result['member_conf_mean'] - mia_result['non_member_conf_mean']
        }
        
        print(f"✅ {model_name}: Attack Acc={mia_result['attack_accuracy']:.3f}, Gap={result['confidence_gap']:.3f}")
        return result
        
    except Exception as e:
        print(f"❌ {model_name} MIA 평가 실패: {e}")
        return None

def main():
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
                    sparsity = float(sparsity_dir.name.replace('sparsity', ''))
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