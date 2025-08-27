#!/usr/bin/env python3
"""
Experiment configuration system for reproducible experiments
"""

import os
import json
import yaml
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Union, Any
import argparse
from pathlib import Path

@dataclass
class DataConfig:
    """Data configuration"""
    dataset: str = 'cifar10'
    datapath: str = '../data'
    batch_size: int = 128
    workers: int = 4
    
    def __post_init__(self):
        if self.dataset not in ['cifar10', 'cifar100']:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

@dataclass
class ModelConfig:
    """Model configuration"""
    arch: str = 'resnet'
    layers: int = 18
    width_mult: float = 1.0
    depth_mult: float = 1.0
    model_mult: int = 0
    
    def __post_init__(self):
        if self.arch not in ['resnet', 'wideresnet']:
            raise ValueError(f"Unsupported architecture: {self.arch}")

@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 200
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = False
    
    # Learning rate schedule
    scheduler: str = 'multistep'
    milestones: List[int] = field(default_factory=lambda: [100, 150])
    gamma: float = 0.1
    step_size: int = 30
    
    # Warmup
    warmup_lr: float = 0.1
    warmup_lr_epoch: int = 0
    warmup_loss_epoch: int = 70
    
    def __post_init__(self):
        if self.scheduler not in ['step', 'multistep', 'cosine', 'exp']:
            raise ValueError(f"Unsupported scheduler: {self.scheduler}")

@dataclass
class PruningConfig:
    """Pruning configuration"""
    enabled: bool = False
    method: str = 'static'  # 'static', 'dpf', 'dcil'
    sparsity: float = 0.5
    prune_freq: int = 16
    target_epoch: int = 75
    prune_type: str = 'unstructured'
    importance_method: str = 'L1'
    
    def __post_init__(self):
        if self.method not in ['static', 'dpf', 'dcil']:
            raise ValueError(f"Unsupported pruning method: {self.method}")
        
        if not 0 <= self.sparsity <= 1:
            raise ValueError(f"Sparsity must be between 0 and 1: {self.sparsity}")

@dataclass
class MIAConfig:
    """Membership Inference Attack configuration"""
    enabled: bool = False
    attack_type: str = 'lira'
    num_shadow_models: int = 64
    shadow_epochs: int = 200
    recalibrate: bool = False
    
    def __post_init__(self):
        if self.attack_type not in ['lira', 'samia']:
            raise ValueError(f"Unsupported MIA type: {self.attack_type}")

@dataclass
class SystemConfig:
    """System configuration"""
    gpu: int = 0
    seed: int = 42
    deterministic: bool = True
    benchmark: bool = True
    print_freq: int = 100
    save_freq: int = 10
    
@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    name: str = 'default_experiment'
    description: str = ''
    save_dir: str = './runs'
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    mia: MIAConfig = field(default_factory=MIAConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self, path: str):
        """Save to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def to_yaml(self, path: str):
        """Save to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary"""
        # Create sub-configs
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        pruning_config = PruningConfig(**config_dict.get('pruning', {}))
        mia_config = MIAConfig(**config_dict.get('mia', {}))
        system_config = SystemConfig(**config_dict.get('system', {}))
        
        # Create main config
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ['data', 'model', 'training', 'pruning', 'mia', 'system']}
        
        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            pruning=pruning_config,
            mia=mia_config,
            system=system_config,
            **main_config
        )
    
    @classmethod
    def from_json(cls, path: str) -> 'ExperimentConfig':
        """Load from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'ExperimentConfig':
        """Load from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def get_model_name(self) -> str:
        """Generate model name from configuration"""
        name_parts = [
            f"{self.model.arch}{self.model.layers}",
            self.data.dataset
        ]
        
        if self.pruning.enabled:
            name_parts.extend([
                self.pruning.method,
                f"sparsity{self.pruning.sparsity}"
            ])
        else:
            name_parts.append("dense")
        
        name_parts.append(f"seed{self.system.seed}")
        
        return "_".join(name_parts)
    
    def get_save_path(self) -> str:
        """Get full save path for this experiment"""
        if self.pruning.enabled:
            method_dir = self.pruning.method
            sparsity_dir = f"sparsity{self.pruning.sparsity}"
            return os.path.join(self.save_dir, method_dir, sparsity_dir, f"seed{self.system.seed}")
        else:
            return os.path.join(self.save_dir, "dense", f"seed{self.system.seed}")

class ConfigManager:
    """Manage experiment configurations"""
    
    def __init__(self, config_dir: str = './configs'):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    
    def save_config(self, config: ExperimentConfig, name: str):
        """Save configuration with given name"""
        config_path = self.config_dir / f"{name}.yaml"
        config.to_yaml(str(config_path))
        print(f"Configuration saved: {config_path}")
    
    def load_config(self, name: str) -> ExperimentConfig:
        """Load configuration by name"""
        # Try YAML first, then JSON
        yaml_path = self.config_dir / f"{name}.yaml"
        json_path = self.config_dir / f"{name}.json"
        
        if yaml_path.exists():
            return ExperimentConfig.from_yaml(str(yaml_path))
        elif json_path.exists():
            return ExperimentConfig.from_json(str(json_path))
        else:
            raise FileNotFoundError(f"Configuration not found: {name}")
    
    def list_configs(self) -> List[str]:
        """List available configurations"""
        configs = []
        for ext in ['*.yaml', '*.yml', '*.json']:
            configs.extend([f.stem for f in self.config_dir.glob(ext)])
        return sorted(list(set(configs)))
    
    def create_preset_configs(self):
        """Create preset experiment configurations"""
        
        # Dense baseline
        dense_config = ExperimentConfig(
            name="dense_baseline",
            description="Dense ResNet-18 baseline",
        )
        self.save_config(dense_config, "dense_baseline")
        
        # Static pruning configurations
        sparsity_levels = [0.5, 0.7, 0.8, 0.9, 0.95]
        for sparsity in sparsity_levels:
            static_config = ExperimentConfig(
                name=f"static_sparsity{sparsity}",
                description=f"Static pruning with {sparsity:.0%} sparsity",
                pruning=PruningConfig(
                    enabled=True,
                    method="static",
                    sparsity=sparsity
                )
            )
            self.save_config(static_config, f"static_sparsity{sparsity}")
        
        # DPF configurations
        for sparsity in sparsity_levels:
            dpf_config = ExperimentConfig(
                name=f"dpf_sparsity{sparsity}",
                description=f"Dynamic pruning with {sparsity:.0%} sparsity",
                pruning=PruningConfig(
                    enabled=True,
                    method="dpf",
                    sparsity=sparsity
                )
            )
            self.save_config(dpf_config, f"dpf_sparsity{sparsity}")
        
        # MIA evaluation configuration
        mia_config = ExperimentConfig(
            name="mia_evaluation",
            description="MIA evaluation with LiRA",
            mia=MIAConfig(
                enabled=True,
                attack_type="lira",
                num_shadow_models=64
            )
        )
        self.save_config(mia_config, "mia_evaluation")
        
        print(f"Created {1 + 2*len(sparsity_levels) + 1} preset configurations")

def parse_config_args() -> ExperimentConfig:
    """Parse command line arguments and create configuration"""
    parser = argparse.ArgumentParser(description='Experiment Configuration')
    
    # Main experiment parameters
    parser.add_argument('--config', type=str, default=None,
                       help='Configuration file to load')
    parser.add_argument('--name', type=str, default='default_experiment',
                       help='Experiment name')
    parser.add_argument('--save-dir', type=str, default='./runs',
                       help='Save directory')
    
    # Data parameters
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100'])
    parser.add_argument('--datapath', type=str, default='../data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=4)
    
    # Model parameters
    parser.add_argument('--arch', type=str, default='resnet')
    parser.add_argument('--layers', type=int, default=18)
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    
    # Pruning parameters
    parser.add_argument('--prune', action='store_true',
                       help='Enable pruning')
    parser.add_argument('--prune-method', type=str, default='static',
                       choices=['static', 'dpf', 'dcil'])
    parser.add_argument('--sparsity', type=float, default=0.5)
    parser.add_argument('--prune-freq', type=int, default=16)
    parser.add_argument('--target-epoch', type=int, default=75)
    
    # MIA parameters
    parser.add_argument('--mia', action='store_true',
                       help='Enable MIA evaluation')
    parser.add_argument('--num-shadows', type=int, default=64)
    
    # System parameters
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print-freq', type=int, default=100)
    
    args = parser.parse_args()
    
    # Load from config file if specified
    if args.config:
        config_manager = ConfigManager()
        try:
            config = config_manager.load_config(args.config)
        except FileNotFoundError:
            # Try loading as file path
            if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                config = ExperimentConfig.from_yaml(args.config)
            elif args.config.endswith('.json'):
                config = ExperimentConfig.from_json(args.config)
            else:
                raise ValueError(f"Unsupported config file format: {args.config}")
    else:
        # Create config from command line arguments
        config = ExperimentConfig(
            name=args.name,
            save_dir=args.save_dir,
            data=DataConfig(
                dataset=args.dataset,
                datapath=args.datapath,
                batch_size=args.batch_size,
                workers=args.workers
            ),
            model=ModelConfig(
                arch=args.arch,
                layers=args.layers
            ),
            training=TrainingConfig(
                epochs=args.epochs,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            ),
            pruning=PruningConfig(
                enabled=args.prune,
                method=args.prune_method,
                sparsity=args.sparsity,
                prune_freq=args.prune_freq,
                target_epoch=args.target_epoch
            ),
            mia=MIAConfig(
                enabled=args.mia,
                num_shadow_models=args.num_shadows
            ),
            system=SystemConfig(
                gpu=args.gpu,
                seed=args.seed,
                print_freq=args.print_freq
            )
        )
    
    return config

def setup_reproducibility(config: SystemConfig):
    """Setup reproducibility based on system config"""
    import torch
    import numpy as np
    import random
    
    # Set seeds
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    # Set deterministic behavior
    if config.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = config.benchmark
    
    # Set environment variables for reproducibility
    os.environ['PYTHONHASHSEED'] = str(config.seed)
    
    print(f"Reproducibility setup complete (seed={config.seed})")

if __name__ == '__main__':
    # Example usage
    config_manager = ConfigManager()
    config_manager.create_preset_configs()
    
    # List available configurations
    print("Available configurations:")
    for name in config_manager.list_configs():
        print(f"  {name}")
    
    # Example: load and modify configuration
    try:
        config = config_manager.load_config('dense_baseline')
        config.system.seed = 123
        config_manager.save_config(config, 'dense_baseline_seed123')
        print("Example configuration created")
    except FileNotFoundError:
        print("Please run config_manager.create_preset_configs() first")