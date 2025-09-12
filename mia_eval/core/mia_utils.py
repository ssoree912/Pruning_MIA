#!/usr/bin/env python3
"""
MIA Utils for DWA Training Results (relocated under mia_eval/core)
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
except ImportError as e:
    print(f"Warning: Could not import some training modules: {e}")

# Additional WeMeM-main style imports
from datasets import get_dataset
from utils.utils import get_model, get_optimizer, weight_init
WEMEM_AVAILABLE = True

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
