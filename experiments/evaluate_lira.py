#!/usr/bin/env python3
"""
LiRA Evaluation Pipeline
Evaluates all 11 target models (Dense + Static×5 + DPF×5) using LiRA
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import json
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models
import pruning
from data import DataLoader
from mia import LiRAAttacker, create_shadow_datasets

def parse_args():
    parser = argparse.ArgumentParser(description='LiRA Evaluation Pipeline')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--arch', default='resnet', type=str)
    parser.add_argument('--layers', default=18, type=int)
    
    # Paths
    parser.add_argument('--target-models-dir', required=True, type=str,
                        help='Directory containing target models')
    parser.add_argument('--shadow-models-dir', required=True, type=str,
                        help='Directory containing shadow models')
    parser.add_argument('--results-dir', default='./results/lira', type=str,
                        help='Directory to save results')
    
    # Data parameters
    parser.add_argument('--datapath', default='../data', type=str)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--workers', default=4, type=int)
    
    # LiRA parameters
    parser.add_argument('--num-shadows', default=64, type=int,
                        help='Number of shadow models to use')
    parser.add_argument('--recalibrate', action='store_true',
                        help='Recalibrate shadow statistics (slow)')
    
    # System parameters
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)
    
    return parser.parse_args()

def load_model(model_path, arch, layers, dataset, model_type, sparsity=None):
    """Load a trained model from checkpoint"""
    
    # Create model
    if model_type == 'dense':
        model, image_size = models.__dict__[arch](
            data=dataset, 
            num_layers=layers
        )
    else:  # static or dpf
        pruner = pruning.dcil
        model, image_size = pruning.models.__dict__[arch](
            data=dataset, 
            num_layers=layers,
            mnn=pruner.mnn
        )
    
    model = model.cuda()
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cuda:0')
    
    # Handle DataParallel state dict
    state_dict = checkpoint['state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        # Remove 'module.' prefix
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key[7:]  # Remove 'module.' prefix
            new_state_dict[new_key] = value
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict, strict=False)
    model = nn.DataParallel(model)
    
    return model, image_size

def load_shadow_models(shadow_dir, model_type, sparsity, arch, layers, dataset, num_shadows):
    """Load shadow models for a specific configuration"""
    
    # Construct shadow directory path
    if sparsity is not None:
        shadow_subdir = f"{model_type}_sparsity{sparsity}"
    else:
        shadow_subdir = model_type
    
    shadow_path = os.path.join(shadow_dir, shadow_subdir)
    
    if not os.path.exists(shadow_path):
        print(f"Warning: Shadow directory not found: {shadow_path}")
        return []
    
    shadow_models = []
    
    for shadow_id in range(num_shadows):
        shadow_model_path = os.path.join(shadow_path, f'shadow{shadow_id:03d}', 'final_model.pth')
        
        if not os.path.exists(shadow_model_path):
            print(f"Warning: Shadow model {shadow_id} not found at {shadow_model_path}")
            continue
        
        try:
            model, _ = load_model(shadow_model_path, arch, layers, dataset, model_type, sparsity)
            shadow_models.append(model)
        except Exception as e:
            print(f"Error loading shadow model {shadow_id}: {e}")
            continue
    
    print(f"Loaded {len(shadow_models)} shadow models for {model_type}")
    if sparsity is not None:
        print(f"  Sparsity: {sparsity:.2%}")
    
    return shadow_models

def evaluate_model_lira(target_model, shadow_models, shadow_datasets, 
                       train_loader, test_loader, results_dir, 
                       model_name, recalibrate=False):
    """Evaluate a single model using LiRA"""
    
    print(f"\nEvaluating {model_name} with LiRA...")
    
    # Create LiRA attacker
    attacker = LiRAAttacker(num_shadow_models=len(shadow_models))
    
    # Check if calibration exists
    calibration_path = os.path.join(results_dir, f'{model_name}_calibration.pkl')
    
    if os.path.exists(calibration_path) and not recalibrate:
        print("Loading existing calibration data...")
        attacker.load_calibration(calibration_path)
    else:
        print("Collecting shadow statistics for calibration...")
        
        # Collect shadow statistics
        attacker.collect_shadow_statistics(shadow_models, shadow_datasets)
        
        # Fit distributions
        attacker.fit_distributions()
        
        # Save calibration data
        attacker.save_calibration(calibration_path)
        
        # Plot distributions
        plot_path = os.path.join(results_dir, f'{model_name}_distributions.png')
        attacker.plot_distributions(plot_path)
    
    # Perform attack on target model
    results = attacker.attack(target_model, train_loader, test_loader)
    
    # Add model information to results
    results['model_name'] = model_name
    results['num_shadow_models'] = len(shadow_models)
    
    # Save detailed results
    results_path = os.path.join(results_dir, f'{model_name}_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Save summary results as JSON
    summary_results = {
        'model_name': model_name,
        'auc': float(results['auc']),
        'tpr_at_fpr': {k: float(v) for k, v in results['tpr_at_fpr'].items()},
        'num_members': int(results['num_members']),
        'num_non_members': int(results['num_non_members']),
        'num_shadow_models': len(shadow_models)
    }
    
    summary_path = os.path.join(results_dir, f'{model_name}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=2)
    
    print(f"Results saved for {model_name}")
    print(f"  AUC: {results['auc']:.4f}")
    for key, value in results['tpr_at_fpr'].items():
        print(f"  {key}: {value:.4f}")
    
    return results

def main():
    args = parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.cuda.set_device(0)
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    print(f"LiRA Evaluation Pipeline")
    print(f"Target models directory: {args.target_models_dir}")
    print(f"Shadow models directory: {args.shadow_models_dir}")
    print(f"Results directory: {args.results_dir}")
    
    # Create data loaders for evaluation
    if args.arch == 'resnet':
        model, image_size = models.__dict__[args.arch](
            data=args.dataset, num_layers=args.layers
        )
    else:
        pruner = pruning.dcil
        model, image_size = pruning.models.__dict__[args.arch](
            data=args.dataset, num_layers=args.layers, mnn=pruner.mnn
        )
    
    train_loader, test_loader = DataLoader(
        args.batch_size, args.dataset, args.workers, 
        args.datapath, image_size, True
    )
    
    # Create shadow datasets (same for all models)
    print("Creating shadow datasets...")
    shadow_datasets = create_shadow_datasets(
        train_loader, test_loader, 
        num_shadows=args.num_shadows
    )
    
    # Define all model configurations to evaluate
    model_configs = []
    
    # Dense model
    model_configs.append({
        'name': 'dense',
        'type': 'dense',
        'sparsity': None,
        'path': os.path.join(args.target_models_dir, 'dense')
    })
    
    # Static pruned models
    sparsity_levels = [0.5, 0.7, 0.8, 0.9, 0.95]
    for sparsity in sparsity_levels:
        model_configs.append({
            'name': f'static_sparsity{sparsity}',
            'type': 'static',
            'sparsity': sparsity,
            'path': os.path.join(args.target_models_dir, 'static', f'sparsity{sparsity}')
        })
    
    # DPF models
    for sparsity in sparsity_levels:
        model_configs.append({
            'name': f'dpf_sparsity{sparsity}',
            'type': 'dpf',
            'sparsity': sparsity,
            'path': os.path.join(args.target_models_dir, 'dpf', f'sparsity{sparsity}')
        })
    
    # Evaluate each model configuration
    all_results = []
    
    for config in model_configs:
        print(f"\n{'='*60}")
        print(f"Evaluating {config['name']}")
        print(f"{'='*60}")
        
        try:
            # Find target model checkpoint
            target_path = None
            if os.path.exists(config['path']):
                # Look for best model in seed subdirectories
                for seed_dir in os.listdir(config['path']):
                    seed_path = os.path.join(config['path'], seed_dir)
                    if os.path.isdir(seed_path):
                        best_model_path = os.path.join(seed_path, 'best_model.pth')
                        if os.path.exists(best_model_path):
                            target_path = best_model_path
                            break
            
            if target_path is None:
                print(f"Target model not found for {config['name']}")
                continue
            
            # Load target model
            target_model, _ = load_model(
                target_path, args.arch, args.layers, args.dataset, 
                config['type'], config['sparsity']
            )
            
            # Load corresponding shadow models
            shadow_models = load_shadow_models(
                args.shadow_models_dir, config['type'], config['sparsity'],
                args.arch, args.layers, args.dataset, args.num_shadows
            )
            
            if len(shadow_models) < 10:  # Minimum threshold
                print(f"Insufficient shadow models ({len(shadow_models)}) for {config['name']}")
                continue
            
            # Evaluate with LiRA
            results = evaluate_model_lira(
                target_model, shadow_models, shadow_datasets[:len(shadow_models)],
                train_loader, test_loader, args.results_dir,
                config['name'], args.recalibrate
            )
            
            results['config'] = config
            all_results.append(results)
            
        except Exception as e:
            print(f"Error evaluating {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate comprehensive results
    if all_results:
        print(f"\n{'='*80}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'='*80}")
        
        summary_data = []
        
        for result in all_results:
            model_name = result['model_name']
            config = result['config']
            
            row = {
                'model_name': model_name,
                'model_type': config['type'],
                'sparsity': config['sparsity'],
                'auc': result['auc'],
                'num_shadow_models': result['num_shadow_models']
            }
            
            # Add TPR@FPR values
            for key, value in result['tpr_at_fpr'].items():
                row[key] = value
            
            summary_data.append(row)
            
            print(f"{model_name:20s} | AUC: {result['auc']:.4f} | "
                  f"TPR@0.1%: {result['tpr_at_fpr']['TPR@0.001']:.4f} | "
                  f"TPR@1%: {result['tpr_at_fpr']['TPR@0.01']:.4f}")
        
        # Save comprehensive summary
        summary_path = os.path.join(args.results_dir, 'comprehensive_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # Create comparison plots
        create_comparison_plots(all_results, args.results_dir)
        
        print(f"\nComprehensive results saved to {summary_path}")
    else:
        print("No results to summarize")

def create_comparison_plots(all_results, results_dir):
    """Create comparison plots for all models"""
    
    # Extract data for plotting
    model_names = []
    aucs = []
    model_types = []
    sparsities = []
    
    for result in all_results:
        model_names.append(result['model_name'])
        aucs.append(result['auc'])
        model_types.append(result['config']['type'])
        sparsities.append(result['config']['sparsity'] or 0)
    
    # Plot 1: AUC comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # AUC bar plot
    colors = []
    for mtype in model_types:
        if mtype == 'dense':
            colors.append('blue')
        elif mtype == 'static':
            colors.append('red')
        else:  # dpf
            colors.append('green')
    
    bars = ax1.bar(range(len(model_names)), aucs, color=colors, alpha=0.7)
    ax1.set_xlabel('Models')
    ax1.set_ylabel('AUC')
    ax1.set_title('LiRA Attack AUC Comparison')
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Random')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='Dense'),
        Patch(facecolor='red', alpha=0.7, label='Static'),
        Patch(facecolor='green', alpha=0.7, label='DPF'),
    ]
    ax1.legend(handles=legend_elements)
    
    # Plot 2: AUC vs Sparsity
    dense_aucs = [auc for auc, mtype in zip(aucs, model_types) if mtype == 'dense']
    static_sparsities = [s for s, mtype in zip(sparsities, model_types) if mtype == 'static']
    static_aucs = [auc for auc, mtype in zip(aucs, model_types) if mtype == 'static']
    dpf_sparsities = [s for s, mtype in zip(sparsities, model_types) if mtype == 'dpf']
    dpf_aucs = [auc for auc, mtype in zip(aucs, model_types) if mtype == 'dpf']
    
    if dense_aucs:
        ax2.axhline(y=dense_aucs[0], color='blue', linestyle='-', linewidth=2, label='Dense')
    
    if static_sparsities and static_aucs:
        ax2.plot(static_sparsities, static_aucs, 'o-', color='red', label='Static', markersize=8)
    
    if dpf_sparsities and dpf_aucs:
        ax2.plot(dpf_sparsities, dpf_aucs, 's-', color='green', label='DPF', markersize=8)
    
    ax2.set_xlabel('Sparsity')
    ax2.set_ylabel('AUC')
    ax2.set_title('LiRA Attack AUC vs Sparsity')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(-0.05, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'lira_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison plots saved to {results_dir}")

if __name__ == '__main__':
    main()