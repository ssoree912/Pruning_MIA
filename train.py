#!/usr/bin/env python3
"""
통합 훈련 및 결과 수집 스크립트
훈련 완료 후 자동으로 MIA 평가 수행하고 결과 수집
"""

import os
import sys
import json
import subprocess
import pandas as pd
from pathlib import Path
import argparse
import shutil
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_organized_save_path(method, sparsity=None, dataset='cifar10'):
    """Create organized save path structure"""
    base_path = Path('./runs')
    
    if method == 'dense':
        save_path = base_path / 'dense' / dataset
    elif method in ['static', 'dpf']:
        if sparsity is None:
            raise ValueError(f"Sparsity must be specified for {method} method")
        save_path = base_path / method / f'sparsity_{sparsity}' / dataset
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return save_path

def run_training(config_params):
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
    
    try:
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

def run_mia_evaluation(runs_dir):
    """Run MIA evaluation for all trained models"""
    cmd = ['python', 'test_mia.py', '--runs-dir', str(runs_dir), '--output-dir', 'results/mia']
    
    print(f"Running MIA evaluation: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 minute timeout
        if result.returncode != 0:
            print(f"MIA evaluation failed: {result.stderr}")
            return False, result.stderr
        return True, result.stdout
    except subprocess.TimeoutExpired:
        print("MIA evaluation timed out after 30 minutes")
        return False, "MIA timeout"
    except Exception as e:
        print(f"MIA evaluation failed: {str(e)}")
        return False, str(e)

def collect_results(results_dir):
    """Collect training and MIA results from directory"""
    results = {}
    
    # Look for training results
    config_file = results_dir / 'config.json'
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
            results['config'] = config
    
    # Look for training logs
    log_files = list(results_dir.glob('*.log'))
    if log_files:
        results['log_file'] = str(log_files[0])
    
    # Look for experiment summary (training results)
    experiment_summary_file = results_dir / 'experiment_summary.json'
    if experiment_summary_file.exists():
        with open(experiment_summary_file, 'r') as f:
            training_results = json.load(f)
            results['training'] = training_results
    
    # Look for validation history
    val_history_file = results_dir / 'validation_history.json'
    if val_history_file.exists():
        with open(val_history_file, 'r') as f:
            val_history = json.load(f)
            results['validation_history'] = val_history
    
    # Look for MIA results
    mia_results_file = results_dir / 'mia_results.json'
    if mia_results_file.exists():
        with open(mia_results_file, 'r') as f:
            mia_results = json.load(f)
            results['mia'] = mia_results
    
    return results

def create_training_summary_csv(all_results, output_file='training_results.csv'):
    """Create summary CSV with all results"""
    summary_data = []
    
    for exp_name, results in all_results.items():
        row = {'experiment': exp_name}
        
        # Add config info
        if 'config' in results:
            config = results['config']
            row.update({
                'method': config.get('pruning', {}).get('method', 'dense'),
                'sparsity': config.get('pruning', {}).get('sparsity', 0.0),
                'dataset': config.get('data', {}).get('dataset', 'cifar10'),
                'arch': config.get('model', {}).get('arch', 'resnet'),
                'layers': config.get('model', {}).get('layers', 20),
                'epochs': config.get('training', {}).get('epochs', 200),
                'lr': config.get('training', {}).get('lr', 0.1),
            })
        
        # Add training results if available
        if 'training' in results:
            training = results['training']
            row.update({
                'best_acc1': training.get('best_metrics', {}).get('best_acc1', None),
                'best_loss': training.get('best_metrics', {}).get('best_loss', None),
                'final_acc1': training.get('final_metrics', {}).get('acc1', None),
                'final_acc5': training.get('final_metrics', {}).get('acc5', None),
                'final_loss': training.get('final_metrics', {}).get('loss', None),
                'total_training_time_hours': training.get('total_duration', 0) / 3600 if training.get('total_duration') else None,
            })
        
        # Add validation history summary if available
        if 'validation_history' in results:
            val_history = results['validation_history']
            if val_history:
                # Get best and final metrics from validation history
                best_val_acc1 = max(epoch.get('acc1', 0) for epoch in val_history if 'acc1' in epoch) if val_history else None
                final_val_acc1 = val_history[-1].get('acc1', None) if val_history else None
                final_val_loss = val_history[-1].get('loss', None) if val_history else None
                
                row.update({
                    'val_best_acc1': best_val_acc1,
                    'val_final_acc1': final_val_acc1,
                    'val_final_loss': final_val_loss,
                })
        
        # MIA 결과는 별도 파일로 분리
        
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    df.to_csv(output_file, index=False)
    print(f"Summary results saved to {output_file}")
    return df

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
    
    args = parser.parse_args()
    
    # Reorganize existing folders first
    reorganize_existing_models()
    
    if args.reorganize_only:
        print("Reorganization complete. Exiting.")
        return
    
    all_results = {}
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
            exp_name += f"_{args.dataset}_{args.arch}"
            
            print(f"\n{'='*50}")
            print(f"Running experiment: {exp_name}")
            print(f"{'='*50}")
            
            # Create save path
            save_path = create_organized_save_path(method, sparsity, args.dataset)
            
            # Skip if results already exist
            if args.skip_existing and save_path.exists():
                best_model = save_path / 'best_model.pth'
                if best_model.exists():
                    print(f"Results already exist for {exp_name}, skipping...")
                    results = collect_results(save_path)
                    all_results[exp_name] = results
                    continue
            
            # Prepare training config
            config_params = {
                'name': exp_name,
                'save-dir': str(save_path),
                'dataset': args.dataset,
                'arch': args.arch,
                'epochs': args.epochs,
            }
            
            # Add wandb config if enabled
            if args.wandb:
                config_params.update({
                    'wandb': True,
                    'wandb_project': args.wandb_project,
                    'wandb_entity': args.wandb_entity,
                    'wandb_name': exp_name,
                })
                if args.wandb_tags:
                    config_params['wandb_tags'] = ','.join(args.wandb_tags + [method, args.dataset, args.arch])
            
            if method != 'dense':
                config_params.update({
                    'prune': True,
                    'prune-method': method,
                    'sparsity': sparsity,
                })
            
            # Run training
            print(f"Starting training for {exp_name}...")
            success, output = run_training(config_params)
            
            if not success:
                print(f"Training failed for {exp_name}: {output}")
                failed_experiments.append((exp_name, 'training', output))
                continue
            
            print(f"Training completed for {exp_name}")
            
            # Collect results
            results = collect_results(save_path)
            all_results[exp_name] = results
            
            print(f"Experiment {exp_name} completed")
    
    # Create training results summary
    print(f"\n{'='*50}")
    print("Creating training summary...")
    print(f"{'='*50}")
    
    if all_results:
        summary_df = create_training_summary_csv(all_results)
        print(f"Completed {len(all_results)} experiments")
        print(f"Training Summary:\n{summary_df.to_string()}")
    
    # Run MIA evaluation on all trained models
    print(f"\n{'='*50}")
    print("Running MIA evaluation...")
    print(f"{'='*50}")
    
    runs_dir = Path('./runs')
    success, output = run_mia_evaluation(runs_dir)
    
    if success:
        print("✅ MIA evaluation completed successfully!")
        print("📊 MIA results saved in: results/mia/")
        print("📁 Results: results/mia/test_mia_results.csv")
        print("📁 Summary: results/mia/test_summary_stats.json")
    else:
        print(f"❌ MIA evaluation failed: {output}")
        failed_experiments.append(('mia_evaluation', 'mia', output))
    
    # Report failed experiments
    if failed_experiments:
        print(f"\n{'='*20} FAILED EXPERIMENTS {'='*20}")
        for exp_name, stage, error in failed_experiments:
            print(f"FAILED: {exp_name} ({stage}) - {error}")
    
    print("\nAll experiments completed!")

if __name__ == '__main__':
    main()