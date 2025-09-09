#!/usr/bin/env python3
"""
MIA Utils for DWA Training Results
Compatible with WeMeM-main codebase structure
"""

import os
import json
import pickle
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from pathlib import Path

# Import existing modules (try both WeMeM-main and original structure)
try:
    from data import cifar10_loader, cifar100_loader
    import models
    import pruning
except ImportError:
    try:
        from datasets import get_dataset
        from utils_wemem import get_model, get_optimizer, weight_init
    except ImportError as e:
        print(f"Warning: Could not import some modules: {e}")

# Additional WeMeM-main style imports
try:
    from datasets import get_dataset
    from utils_wemem import get_model, get_optimizer, weight_init
    WEMEM_AVAILABLE = True
except ImportError:
    WEMEM_AVAILABLE = False

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

def seed_worker(worker_id):
    """DataLoader 재현성을 위한 시드 설정"""
    import random, numpy as np, torch
    np.random.seed(1234 + worker_id)
    random.seed(1234 + worker_id)
    torch.manual_seed(1234 + worker_id)

def weight_init(m):
    """모델 가중치 초기화"""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)

def get_optimizer(name, params, lr, weight_decay):
    """옵티마이저 생성"""
    name = (name or "sgd").lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)

def CrossEntropy_soft(logits, soft_targets, reduction='none'):
    """소프트 타겟을 위한 크로스엔트로피"""
    log_p = F.log_softmax(logits, dim=1)
    loss = -(soft_targets * log_p).sum(dim=1)
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    return loss

def one_hot_embedding(labels, num_classes):
    """원핫 인코딩"""
    return F.one_hot(labels, num_classes=num_classes).float()

class MIAFC(nn.Module):
    """MIA용 완전연결 신경망"""
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        return self.net(x)

def get_model(model_type, num_cls, input_dim):
    """
    모델 생성 함수
    - 분류 모델: num_cls=클래스 수, input_dim=채널 수
    - 공격 모델: num_cls=feature_dim, 출력은 2로 고정
    """
    model_type = model_type.lower()
    if model_type == "resnet18":
        m = resnet18(num_classes=num_cls)
        # CIFAR에 맞게 첫 conv/풀링 조정
        m.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        return m
    elif model_type in ["mia_fc", "transformer"]:
        feature_dim = num_cls  # 관례상 feature_dim을 num_cls로 전달
        return MIAFC(feature_dim)
    else:
        raise ValueError(f"Unknown model_type={model_type}")

def load_dwa_model(model_path, config_path=None, device='cuda'):
    """DWA 훈련된 모델 로드 (config.json 참조)"""
    
    if config_path is None:
        config_path = Path(model_path).parent / 'config.json'
    
    # 설정 로드
    with open(config_path) as f:
        config = json.load(f)
    
    # 모델 생성 (훈련시와 동일하게)
    if config.get('pruning', {}).get('enabled', False):
        # Pruned 모델인 경우
        pruner_key = config['pruning']['method'].lower()
        if pruner_key in ('dpf', 'dwa', 'static'):
            pruner_key = 'dcil'  # 동일 백엔드 사용
        
        pruner = pruning.__dict__[pruner_key]
        model, _ = pruning.models.__dict__[config['model']['arch']](
            data=config['data']['dataset'],
            num_layers=config['model']['layers'],
            width_mult=config['model'].get('width_mult', 1.0),
            depth_mult=config['model'].get('depth_mult', 1.0),
            model_mult=config['model'].get('model_mult', 1.0),
            mnn=pruner.mnn
        )
    else:
        # Dense 모델
        model, _ = models.__dict__[config['model']['arch']](
            data=config['data']['dataset'],
            num_layers=config['model']['layers'],
            width_mult=config['model'].get('width_mult', 1.0),
            depth_mult=config['model'].get('depth_mult', 1.0),
            model_mult=config['model'].get('model_mult', 1.0)
        )
    
    # 체크포인트 로드
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # DataParallel 처리
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # module. prefix 처리
    if any(key.startswith('module.') for key in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # remove 'module.'
            new_state_dict[name] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    
    return model, config

def get_dataset_loaders(dataset_name, batch_size=128, num_workers=4, datapath='~/Datasets', cuda=True):
    """데이터셋 로더 생성"""
    if dataset_name == 'cifar10':
        train_loader, val_loader = cifar10_loader(batch_size, num_workers, datapath, cuda=cuda)
    elif dataset_name == 'cifar100':
        train_loader, val_loader = cifar100_loader(batch_size, num_workers, datapath, cuda=cuda)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return train_loader, val_loader

def create_mia_data_splits(dataset_name, victim_ratio=0.4, shadow_ratio=0.4, seed=42):
    """
    MIA용 데이터 분할
    - victim_ratio: victim 모델 훈련용 비율
    - shadow_ratio: shadow 모델 훈련용 비율
    - 나머지: non-member 데이터
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 전체 훈련 데이터 로드
    train_loader, _ = get_dataset_loaders(dataset_name, batch_size=1, num_workers=0, cuda=False)
    dataset = train_loader.dataset
    
    # 인덱스 섞기
    total_size = len(dataset)
    indices = list(range(total_size))
    random.shuffle(indices)
    
    # 분할
    victim_size = int(total_size * victim_ratio)
    shadow_size = int(total_size * shadow_ratio)
    
    victim_indices = indices[:victim_size]
    shadow_indices = indices[victim_size:victim_size + shadow_size]
    non_member_indices = indices[victim_size + shadow_size:]
    
    return {
        'victim_indices': victim_indices,
        'shadow_indices': shadow_indices,
        'non_member_indices': non_member_indices,
        'dataset': dataset
    }

def extract_model_features(model, data_loader, device='cuda', forward_mode='static'):
    """모델에서 피처 추출 (softmax, confidence, entropy 등)"""
    model.eval()
    
    features = {
        'softmax': [],
        'confidence': [],
        'entropy': [],
        'predictions': [],
        'targets': []
    }
    
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            if hasattr(model, 'module'):
                # DataParallel 모델인 경우
                if forward_mode in ['static', 'DPF']:
                    outputs = model(data, forward_mode)
                else:
                    outputs = model(data)
            else:
                outputs = model(data)
            
            # Softmax 확률
            probs = F.softmax(outputs, dim=1)
            
            # Confidence (max probability)
            confidence = probs.max(dim=1)[0]
            
            # Entropy
            entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)
            
            # Predictions
            predictions = outputs.argmax(dim=1)
            
            features['softmax'].append(probs.cpu().numpy())
            features['confidence'].append(confidence.cpu().numpy())
            features['entropy'].append(entropy.cpu().numpy())
            features['predictions'].append(predictions.cpu().numpy())
            features['targets'].append(targets.cpu().numpy())
    
    # Concatenate all batches
    for key in features:
        features[key] = np.concatenate(features[key])
    
    return features

def save_mia_data_splits(dataset_name, save_dir='./mia_data'):
    """MIA 실험용 데이터 분할 저장"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    splits = create_mia_data_splits(dataset_name)
    
    # 분할 정보 저장
    with open(save_dir / f'{dataset_name}_mia_splits.pkl', 'wb') as f:
        pickle.dump(splits, f)
    
    print(f"MIA data splits saved to {save_dir / f'{dataset_name}_mia_splits.pkl'}")
    print(f"Victim: {len(splits['victim_indices'])}, Shadow: {len(splits['shadow_indices'])}, Non-member: {len(splits['non_member_indices'])}")
    
    return splits

def load_mia_data_splits(dataset_name, save_dir='./mia_data'):
    """저장된 MIA 데이터 분할 로드"""
    save_dir = Path(save_dir)
    split_file = save_dir / f'{dataset_name}_mia_splits.pkl'
    
    if not split_file.exists():
        print(f"MIA splits not found, creating new splits...")
        return save_mia_data_splits(dataset_name, save_dir)
    
    with open(split_file, 'rb') as f:
        splits = pickle.load(f)
    
    return splits

def get_model_sparsity(model):
    """모델의 실제 sparsity 계산"""
    total_params = 0
    zero_params = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
    
    sparsity = zero_params / total_params if total_params > 0 else 0.0
    return sparsity

def find_dwa_models(runs_dir='./runs'):
    """DWA 훈련 결과에서 모델들 찾기"""
    runs_dir = Path(runs_dir)
    models = []
    
    # DWA 결과 탐색
    dwa_dir = runs_dir / 'dwa'
    if dwa_dir.exists():
        for mode_dir in dwa_dir.iterdir():
            if mode_dir.is_dir():
                for sparsity_dir in mode_dir.iterdir():
                    if sparsity_dir.is_dir() and sparsity_dir.name.startswith('sparsity_'):
                        for dataset_dir in sparsity_dir.iterdir():
                            if dataset_dir.is_dir():
                                model_path = dataset_dir / 'best_model.pth'
                                config_path = dataset_dir / 'config.json'
                                
                                if model_path.exists() and config_path.exists():
                                    models.append({
                                        'model_path': model_path,
                                        'config_path': config_path,
                                        'dwa_mode': mode_dir.name,
                                        'sparsity_dir': sparsity_dir.name,
                                        'dataset': dataset_dir.name,
                                        'experiment_dir': dataset_dir
                                    })
    
    return models