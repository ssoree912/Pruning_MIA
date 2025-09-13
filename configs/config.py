#!/usr/bin/env python3
"""
Experiment configuration system for reproducible experiments
"""
import os
import json
import yaml
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
import argparse
from pathlib import Path

@dataclass
class DataConfig:
    dataset: str = 'cifar10'
    datapath: str = '~/Datasets/CIFAR'
    batch_size: int = 128
    workers: int = 4
    def __post_init__(self):
        if self.dataset not in ['cifar10', 'cifar100']:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

@dataclass
class ModelConfig:
    arch: str = 'resnet'
    layers: int = 20
    width_mult: float = 1.0
    depth_mult: float = 1.0
    model_mult: int = 0
    def __post_init__(self):
        if self.arch not in ['resnet', 'wideresnet']:
            raise ValueError(f"Unsupported architecture: {self.arch}")

@dataclass
class TrainingConfig:
    epochs: int = 200
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = False
    scheduler: str = 'multistep'
    milestones: List[int] = field(default_factory=lambda: [100, 150])
    gamma: float = 0.1
    step_size: int = 30
    warmup_lr: float = 0.1
    warmup_lr_epoch: int = 0
    warmup_loss_epoch: int = 70
    def __post_init__(self):
        if self.scheduler not in ['step', 'multistep', 'cosine', 'exp']:
            raise ValueError(f"Unsupported scheduler: {self.scheduler}")

@dataclass
class PruningConfig:
    enabled: bool = False
    method: str = 'static'              # 'static', 'dpf', 'dcil'
    sparsity: float = 0.5
    prune_freq: int = 16
    target_epoch: int = 75
    freeze_epoch: int = 180
    prune_type: str = 'unstructured'
    importance_method: str = 'L1'
    # ---- DWA fields (추가) ----
    dwa_mode: str = 'reactivate_only'   # 'reactivate_only' | 'kill_active_plain_dead' | 'kill_and_reactivate'
    dwa_alpha: float = 1.0
    dwa_beta: float = 1.0
    dwa_threshold_percentile: int = 50  # [0,100]
    def __post_init__(self):
        if self.method not in ['static', 'dpf', 'dcil']:
            raise ValueError(f"Unsupported pruning method: {self.method}")
        if not 0 <= self.sparsity <= 1:
            raise ValueError(f"Sparsity must be between 0 and 1: {self.sparsity}")
        if self.dwa_mode not in ['reactivate_only','kill_active_plain_dead','kill_and_reactivate']:
            raise ValueError(f"Unsupported DWA mode: {self.dwa_mode}")
        if not (0 <= self.dwa_threshold_percentile <= 100):
            raise ValueError(f"dwa_threshold_percentile must be in [0,100]: {self.dwa_threshold_percentile}")

@dataclass
class MIAConfig:
    enabled: bool = False
    attack_type: str = 'lira'
    num_shadow_models: int = 64
    shadow_epochs: int = 200
    recalibrate: bool = False
    def __post_init__(self):
        if self.attack_type not in ['lira', 'samia']:
            raise ValueError(f"Unsupported MIA type: {self.attack_type}")

@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = 'dcil-pytorch'
    entity: str = None
    name: str = None
    tags: List[str] = field(default_factory=list)
    notes: str = ''

@dataclass
class SystemConfig:
    gpu: int = 0
    seed: int = 42
    deterministic: bool = True
    benchmark: bool = True
    print_freq: int = 100
    save_freq: int = 10

@dataclass
class ExperimentConfig:
    name: str = 'default_experiment'
    description: str = ''
    save_dir: str = './runs'
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    mia: MIAConfig = field(default_factory=MIAConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    def to_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        pruning_config = PruningConfig(**config_dict.get('pruning', {}))
        mia_config = MIAConfig(**config_dict.get('mia', {}))
        system_config = SystemConfig(**config_dict.get('system', {}))
        wandb_config = WandbConfig(**config_dict.get('wandb', {}))
        main_config = {k: v for k, v in config_dict.items()
                       if k not in ['data','model','training','pruning','mia','system','wandb']}
        return cls(
            data=data_config, model=model_config, training=training_config,
            pruning=pruning_config, mia=mia_config, system=system_config,
            wandb=wandb_config, **main_config
        )
    @classmethod
    def from_json(cls, path: str) -> 'ExperimentConfig':
        with open(path, 'r') as f: return cls.from_dict(json.load(f))
    @classmethod
    def from_yaml(cls, path: str) -> 'ExperimentConfig':
        with open(path, 'r') as f: return cls.from_dict(yaml.safe_load(f))
    def get_model_name(self) -> str:
        parts = [f"{self.model.arch}{self.model.layers}", self.data.dataset]
        if self.pruning.enabled:
            parts.extend([self.pruning.method, f"sparsity{self.pruning.sparsity}"])
        else:
            parts.append("dense")
        parts.append(f"seed{self.system.seed}")
        return "_".join(parts)
    def get_save_path(self) -> str:
        return self.save_dir

class ConfigManager:
    def __init__(self, config_dir: str = './configs'):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    def save_config(self, config: ExperimentConfig, name: str):
        path = self.config_dir / f"{name}.yaml"; config.to_yaml(str(path))
        print(f"Configuration saved: {path}")
    def load_config(self, name: str) -> ExperimentConfig:
        y = self.config_dir / f"{name}.yaml"; j = self.config_dir / f"{name}.json"
        if y.exists(): return ExperimentConfig.from_yaml(str(y))
        if j.exists(): return ExperimentConfig.from_json(str(j))
        raise FileNotFoundError(f"Configuration not found: {name}")
    def list_configs(self) -> List[str]:
        names = []
        for ext in ['*.yaml','*.yml','*.json']:
            names.extend([f.stem for f in self.config_dir.glob(ext)])
        return sorted(list(set(names)))
    def create_preset_configs(self):
        dense = ExperimentConfig(name="dense_baseline", description="Dense ResNet-18 baseline")
        self.save_config(dense, "dense_baseline")
        sparsities = [0.5,0.7,0.8,0.9,0.95]
        for s in sparsities:
            self.save_config(ExperimentConfig(
                name=f"static_sparsity{s}",
                description=f"Static pruning with {s:.0%} sparsity",
                pruning=PruningConfig(enabled=True, method="static", sparsity=s)
            ), f"static_sparsity{s}")
        for s in sparsities:
            self.save_config(ExperimentConfig(
                name=f"dpf_sparsity{s}",
                description=f"Dynamic pruning with {s:.0%} sparsity",
                pruning=PruningConfig(enabled=True, method="dpf", sparsity=s)
            ), f"dpf_sparsity{s}")
        self.save_config(ExperimentConfig(
            name="mia_evaluation",
            description="MIA evaluation with LiRA",
            mia=MIAConfig(enabled=True, attack_type="lira", num_shadow_models=64)
        ), "mia_evaluation")
        print(f"Created {1 + 2*len(sparsities) + 1} preset configurations")

def parse_config_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description='Experiment Configuration')
    # Main
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--name', type=str, default='default_experiment')
    parser.add_argument('--save-dir', type=str, default='./runs')
    # Data
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10','cifar100'])
    parser.add_argument('--datapath', type=str, default='~/Datasets/CIFAR')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=4)
    # Model
    parser.add_argument('--arch', type=str, default='resnet')
    parser.add_argument('--layers', type=int, default=20)
    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # Pruning
    parser.add_argument('--prune', action='store_true')
    parser.add_argument('--prune-method', type=str, default='static', choices=['static','dpf','dcil'])
    parser.add_argument('--sparsity', type=float, default=0.5)
    parser.add_argument('--prune-freq', type=int, default=16)
    parser.add_argument('--target-epoch', type=int, default=75)
    parser.add_argument('--freeze-epoch', type=int, default=180)
    parser.add_argument('--prune-type', type=str, default='unstructured')
    # ---- DWA CLI (추가) ----
    parser.add_argument('--dwa-mode', type=str, default='reactivate_only',
                        choices=['reactivate_only','kill_active_plain_dead','kill_and_reactivate'])
    parser.add_argument('--dwa-alpha', type=float, default=1.0)
    parser.add_argument('--dwa-beta', type=float, default=1.0)
    parser.add_argument('--dwa-threshold-percentile', type=int, default=50)
    # MIA
    parser.add_argument('--mia', action='store_true')
    parser.add_argument('--num-shadows', type=int, default=64)
    # Wandb
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', default='dcil-pytorch')
    parser.add_argument('--wandb_entity')
    parser.add_argument('--wandb_name')
    parser.add_argument('--wandb_tags', help='Wandb tags (comma separated)')
    parser.add_argument('--wandb_notes', default='')
    # System
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print-freq', type=int, default=100)

    args = parser.parse_args()

    if args.config:
        cm = ConfigManager()
        try:
            cfg = cm.load_config(args.config)
        except FileNotFoundError:
            if args.config.endswith(('.yaml','.yml')): cfg = ExperimentConfig.from_yaml(args.config)
            elif args.config.endswith('.json'): cfg = ExperimentConfig.from_json(args.config)
            else: raise ValueError(f"Unsupported config file format: {args.config}")
        return cfg

    # CLI -> dataclass 매핑
    cfg = ExperimentConfig(
        name=args.name,
        save_dir=args.save_dir,
        data=DataConfig(dataset=args.dataset, datapath=args.datapath,
                        batch_size=args.batch_size, workers=args.workers),
        model=ModelConfig(arch=args.arch, layers=args.layers),
        training=TrainingConfig(epochs=args.epochs, lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay),
        pruning=PruningConfig(
            enabled=args.prune,
            method=args.prune_method,
            sparsity=args.sparsity,
            prune_freq=args.prune_freq,
            target_epoch=args.target_epoch,
            freeze_epoch=args.freeze_epoch,
            prune_type=getattr(args, 'prune_type', 'unstructured'),
            # ---- DWA 필드 매핑 ----
            dwa_mode=args.dwa_mode,
            dwa_alpha=args.dwa_alpha,
            dwa_beta=args.dwa_beta,
            dwa_threshold_percentile=args.dwa_threshold_percentile,
        ),
        mia=MIAConfig(enabled=args.mia, num_shadow_models=args.num_shadows),
        system=SystemConfig(gpu=args.gpu, seed=args.seed, print_freq=args.print_freq),
        wandb=WandbConfig(
            enabled=args.wandb,
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            tags=[t.strip() for t in args.wandb_tags.split(',')] if args.wandb_tags else [],
            notes=args.wandb_notes,
        ),
    )
    return cfg

def setup_reproducibility(system: SystemConfig):
    import torch, numpy as np, random
    torch.manual_seed(system.seed)
    torch.cuda.manual_seed_all(system.seed)
    np.random.seed(system.seed)
    random.seed(system.seed)
    if system.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = system.benchmark
    os.environ['PYTHONHASHSEED'] = str(system.seed)
    print(f"Reproducibility setup complete (seed={system.seed})")

if __name__ == '__main__':
    cm = ConfigManager(); cm.create_preset_configs()
    print("Available configurations:")
    for name in cm.list_configs(): print(f"  {name}")