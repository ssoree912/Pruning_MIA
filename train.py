#!/usr/bin/env python3
"""
통합 훈련 스크립트 (결과 수집/요약 제거)
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import argparse
import shutil
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_organized_save_path(method, sparsity=None, dataset='cifar10', freeze_epoch=None, total_epochs=None, seed=42):
    """Create organized save path structure"""
    base_path = Path('./runs')
    
    if method == 'dense':
        save_path = base_path / 'dense' / dataset
    elif method == 'static':
        if sparsity is None:
            raise ValueError(f"Sparsity must be specified for {method} method")
        save_path = base_path / method / f'sparsity_{sparsity}' / dataset
    elif method == 'dpf':
        if sparsity is None:
            raise ValueError(f"Sparsity must be specified for {method} method")
        
        # Add freeze info to path
        if freeze_epoch is not None and total_epochs is not None:
            if freeze_epoch >= 0 and freeze_epoch < total_epochs:
                freeze_suffix = f'_freeze{freeze_epoch}'
            elif freeze_epoch < 0:
                freeze_suffix = '_nofreeze'
            else:
                freeze_suffix = ''
        else:
            freeze_suffix = ''
            
        save_path = base_path / method / f'sparsity_{sparsity}{freeze_suffix}' / dataset
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Add seed to path if not default seed
    if seed != 42:
        save_path = save_path / f'seed{seed}'
    
    return save_path

def run_training(config_params, dry_run: bool = False):
    """Run training with specified parameters"""
    cmd = ['python', 'run_experiment.py']
    
    # Add all config parameters to command
    for key, value in config_params.items():
        if value is not None:
            if isinstance(value, bool):
                # For boolean flags, only add the flag if True
                if value:
                    cmd.append(f'--{key}')
            else:
                cmd.extend([f'--{key}', str(value)])
    
    print(f"Running training command: {' '.join(cmd)}")
    if dry_run:
        print("[DRY RUN] Skipping execution.")
        return True, "dry-run"

    try:
        # Show output in real-time instead of capturing
        # result = subprocess.run(cmd, text=True, timeout=7200)  # 2 hour timeout
        # if result.returncode != 0:
        #     print(f"Training failed with return code: {result.returncode}")
        #     return False, f"Return code: {result.returncode}"
        # return True, "Training completed"
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
        if result.returncode != 0:
            print(f"Training failed with error: {result.stderr}")
            return False, result.stderr
        return True, result.stdout


    except subprocess.TimeoutExpired:
        print("Training timed out after 2 hours")
        return False, "Training timeout"
    except Exception as e:
        print(f"Training failed with exception: {str(e)}")
        return False, str(e)

def run_comprehensive_mia_evaluation(runs_dir):
    """Deprecated: MIA 평가 로직 제거됨 (보존용 스텁)."""
    return False, "MIA evaluation removed"

def combine_mia_results(results_dir, advanced_success, wemem_success):
    """Deprecated: MIA 결과 병합 로직 제거됨 (보존용 스텁)."""
    return True

def collect_results(results_dir):
    """Deprecated: 실험 결과 수집은 별도 스크립트에서 수행 (보존용 스텁)."""
    return {}

def create_training_summary_csv(all_results, experiment_prefix='experiments'):
    """Deprecated: 요약 CSV는 별도 스크립트에서 생성 (보존용 스텁)."""
    return None

def log_mia_results_to_wandb(args):
    """Deprecated: MIA 결과 로깅 제거됨 (보존용 스텁)."""
    print("MIA results logging disabled")

def reorganize_existing_models():
    """Reorganize existing model folders into new structure"""
    runs_dir = Path('./runs')
    if not runs_dir.exists():
        return
    
    print("Reorganizing existing model folders...")
    
    # Find folders to reorganize
    folders_to_move = []
    for folder in runs_dir.iterdir():
        if folder.is_dir():
            name = folder.name
            if name.startswith('dpf_sparsity'):
                sparsity = name.replace('dpf_sparsity', '')
                folders_to_move.append((folder, 'dpf', sparsity))
            elif name.startswith('static_sparsity'):
                sparsity = name.replace('static_sparsity', '')
                folders_to_move.append((folder, 'static', sparsity))
    
    # Move folders
    for old_folder, method, sparsity in folders_to_move:
        new_path = runs_dir / method / f'sparsity_{sparsity}'
        new_path.parent.mkdir(parents=True, exist_ok=True)
        
        if new_path.exists():
            print(f"Target path {new_path} already exists, skipping {old_folder}")
            continue
            
        print(f"Moving {old_folder} -> {new_path}")
        shutil.move(str(old_folder), str(new_path))

def main():
    parser = argparse.ArgumentParser(description='Train models and collect results')
    parser.add_argument('--reorganize-only', action='store_true',
                       help='Only reorganize existing folders, do not train')
    parser.add_argument('--methods', nargs='+', default=['dense', 'static', 'dpf'],
                       choices=['dense', 'static', 'dpf'],
                       help='Methods to train')
    parser.add_argument('--sparsities', nargs='+', type=float, 
                       default=[0.5, 0.7, 0.8, 0.9, 0.95],
                       help='Sparsity levels for pruned methods')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'],
                       help='Dataset to use')
    parser.add_argument('--arch', default='resnet', choices=['resnet', 'wideresnet'],
                       help='Architecture to use')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--freeze-epoch', type=int, default=180,
                       help='Epoch to freeze masks (default: 180)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip experiments if results already exist')
    
    # Wandb arguments
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', default='dcil-pytorch',
                       help='Wandb project name')
    parser.add_argument('--wandb-entity', default=None,
                       help='Wandb entity (username or team)')
    parser.add_argument('--wandb-tags', nargs='*', default=[],
                       help='Wandb tags for experiment')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    # Multi-seed controls
    parser.add_argument('--multi-seed', action='store_true', help='Enable multi-seed training')
    parser.add_argument('--num-seeds', type=int, default=8, help='Number of seeds for multi-seed')
    parser.add_argument('--start-seed', type=int, default=43, help='Starting seed for multi-seed')
    parser.add_argument('--dry-run', action='store_true', help='Print planned runs without executing')
    
    args = parser.parse_args()
    
    # Reorganize existing folders first
    reorganize_existing_models()
    
    if args.reorganize_only:
        print("Reorganization complete. Exiting.")
        return
    
    # Plan seeds
    if args.multi_seed:
        seed_list = list(range(args.start_seed, args.start_seed + args.num_seeds))
    else:
        seed_list = [args.seed]

    # Collecting per-run results is disabled; we'll aggregate later from runs/ if needed.
    failed_experiments = []
    
    print(f"Starting training for methods: {args.methods}")
    print(f"Sparsities: {args.sparsities}")
    print(f"Dataset: {args.dataset}, Architecture: {args.arch}")
    
    for method in args.methods:
        if method == 'dense':
            sparsities = [None]  # Dense doesn't need sparsity
        else:
            sparsities = args.sparsities
            
        for sparsity in sparsities:
            exp_name = f"{method}"
            if sparsity is not None:
                exp_name += f"_sparsity_{sparsity}"
            
            # Add freeze indicator to experiment name
            if method == 'dpf':
                if args.freeze_epoch >= 0 and args.freeze_epoch < args.epochs:
                    exp_name += f"_freeze{args.freeze_epoch}"
                elif args.freeze_epoch < 0:
                    exp_name += "_nofreeze"
            
            exp_name += f"_{args.dataset}_{args.arch}"
            
            print(f"\n{'='*50}")
            print(f"Running experiment: {exp_name}")
            print(f"{'='*50}")
            
            for cur_seed in seed_list:
                # Create save path (includes seed when cur_seed != 42)
                save_path = create_organized_save_path(method, sparsity, args.dataset, args.freeze_epoch, args.epochs, cur_seed)

                # Skip if results already exist
                if args.skip_existing and (save_path / 'best_model.pth').exists():
                    print(f"Results already exist for {exp_name} (seed={cur_seed}), skipping...")
                    continue

                # Prepare training config
                config_params = {
                    'name': f"{exp_name}",
                    'save-dir': str(save_path),
                    'dataset': args.dataset,
                    'arch': args.arch,
                    'epochs': args.epochs,
                    'freeze-epoch': args.freeze_epoch,
                    'seed': cur_seed,
                    'gpu': args.gpu,
                }

                # Add wandb config if enabled
                if args.wandb:
                    tags = args.wandb_tags + [method, args.dataset, args.arch, f'seed{cur_seed}'] if args.wandb_tags else [method, args.dataset, args.arch, f'seed{cur_seed}']
                    config_params.update({
                        'wandb': True,
                        'wandb_project': args.wandb_project,
                        'wandb_entity': args.wandb_entity,
                        'wandb_name': f"{exp_name}_seed{cur_seed}",
                        'wandb_tags': ','.join(tags),
                    })

                if method != 'dense':
                    config_params.update({
                        'prune': True,
                        'prune-method': method,
                        'sparsity': sparsity,
                    })

                # Run training (or dry-run)
                print(f"Starting training for {exp_name} (seed={cur_seed})...")
                success, output = run_training(config_params, dry_run=args.dry_run)

                if not success:
                    print(f"Training failed for {exp_name} (seed={cur_seed}): {output}")
                    failed_experiments.append((f"{exp_name}_seed{cur_seed}", 'training', output))
                    continue

                print(f"Training completed for {exp_name} (seed={cur_seed})")
    
    # Training summary CSV generation is disabled in this run.
    # Run MIA evaluation on all trained models
    # print(f"\n{'='*50}")
    # print("Running MIA evaluation...")
    # print(f"{'='*50}")
    
    # runs_dir = Path('./runs')
    # success, output = run_comprehensive_mia_evaluation(runs_dir)
    
    # if success:
    #     print("✅ MIA evaluation completed successfully!")
    #     print("📊 MIA results saved in: results/mia/")
    #     print("📁 Results: results/mia/comprehensive_mia_results.csv")
        
    #     # Log MIA results to wandb if available
    #     if args.wandb:
    #         log_mia_results_to_wandb(args)
    # else:
    #     print(f"❌ MIA evaluation failed: {output}")
    #     failed_experiments.append(('mia_evaluation', 'mia', output))
    
    # Report failed experiments
    if failed_experiments:
        print(f"\n{'='*20} FAILED EXPERIMENTS {'='*20}")
        for exp_name, stage, error in failed_experiments:
            print(f"FAILED: {exp_name} ({stage}) - {error}")
    
    print("\nAll experiments completed!")

if __name__ == '__main__':
    main()
