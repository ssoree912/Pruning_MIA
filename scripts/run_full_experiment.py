#!/usr/bin/env python3
"""
Full experiment runner for Dense/Static/DPF comparison with MIA evaluation
Runs all 11 target models (Dense + Static×5 + DPF×5) with multiple seeds
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
import multiprocessing as mp
from typing import List, Dict
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.experiment_config import ConfigManager, ExperimentConfig
from utils.logger import ResultsCollector

def parse_args():
    parser = argparse.ArgumentParser(description='Full Experiment Runner')
    
    # Experiment parameters
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--arch', default='resnet', type=str)
    parser.add_argument('--layers', default=20, type=int)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456],
                       help='Random seeds to run')
    
    # Training parameters
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    
    # Pruning parameters
    parser.add_argument('--sparsity-levels', nargs='+', type=float, 
                       default=[0.5, 0.7, 0.8, 0.9, 0.95],
                       help='Sparsity levels to evaluate')
    
    # MIA parameters
    parser.add_argument('--run-mia', action='store_true',
                       help='Run MIA evaluation after training')
    parser.add_argument('--num-shadows', default=64, type=int,
                       help='Number of shadow models for MIA')
    
    # System parameters
    parser.add_argument('--gpus', nargs='+', type=int, default=[0],
                       help='GPU IDs to use')
    parser.add_argument('--max-parallel', type=int, default=1,
                       help='Maximum parallel jobs')
    parser.add_argument('--save-dir', default='./runs', type=str)
    parser.add_argument('--datapath', default='~/Datasets/CIFAR', type=str)
    parser.add_argument('--dry-run', action='store_true',
                       help='Print commands without running')
    
    return parser.parse_args()

def create_experiment_configs(args) -> List[ExperimentConfig]:
    """Create all experiment configurations"""
    
    configs = []
    
    for seed in args.seeds:
        # Dense baseline
        dense_config = ExperimentConfig(
            name=f"dense_seed{seed}",
            description=f"Dense {args.arch}-{args.layers} baseline (seed={seed})",
            save_dir=args.save_dir,
            data=ExperimentConfig().data,
            model=ExperimentConfig().model,
            training=ExperimentConfig().training,
            system=ExperimentConfig().system
        )
        
        # Update with command line args
        dense_config.data.dataset = args.dataset
        dense_config.data.batch_size = args.batch_size
        dense_config.data.datapath = args.datapath
        dense_config.model.arch = args.arch
        dense_config.model.layers = args.layers
        dense_config.training.epochs = args.epochs
        dense_config.training.lr = args.lr
        dense_config.system.seed = seed
        
        configs.append(dense_config)
        
        # Static pruning configurations
        for sparsity in args.sparsity_levels:
            static_config = ExperimentConfig(
                name=f"static_sparsity{sparsity}_seed{seed}",
                description=f"Static pruning {sparsity:.0%} (seed={seed})",
                save_dir=args.save_dir,
                data=dense_config.data,
                model=dense_config.model,
                training=dense_config.training,
                system=dense_config.system
            )
            
            static_config.pruning.enabled = True
            static_config.pruning.method = "static"
            static_config.pruning.sparsity = sparsity
            
            configs.append(static_config)
        
        # DPF configurations
        for sparsity in args.sparsity_levels:
            dpf_config = ExperimentConfig(
                name=f"dpf_sparsity{sparsity}_seed{seed}",
                description=f"DPF pruning {sparsity:.0%} (seed={seed})",
                save_dir=args.save_dir,
                data=dense_config.data,
                model=dense_config.model,
                training=dense_config.training,
                system=dense_config.system
            )
            
            dpf_config.pruning.enabled = True
            dpf_config.pruning.method = "dpf"
            dpf_config.pruning.sparsity = sparsity
            
            configs.append(dpf_config)
    
    return configs

def run_single_experiment(config: ExperimentConfig, gpu_id: int, dry_run: bool = False) -> Dict:
    """Run a single experiment"""
    
    # Update GPU
    config.system.gpu = gpu_id
    
    # Create temporary config file
    config_dir = Path('./temp_configs')
    config_dir.mkdir(exist_ok=True)
    
    config_path = config_dir / f'{config.name}_gpu{gpu_id}.yaml'
    config.to_yaml(str(config_path))
    
    # Build command
    cmd = [
        'python', 'run_experiment.py',
        '--config', str(config_path)
    ]
    
    result = {
        'config': config.to_dict(),
        'command': ' '.join(cmd),
        'success': False,
        'runtime': 0,
        'error': None
    }
    
    if dry_run:
        print(f"[DRY RUN] GPU {gpu_id}: {' '.join(cmd)}")
        result['success'] = True
        return result
    
    print(f"Starting experiment on GPU {gpu_id}: {config.name}")
    
    start_time = time.time()
    
    try:
        # Run experiment
        process = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=24*3600  # 24 hour timeout
        )
        
        runtime = time.time() - start_time
        result['runtime'] = runtime
        
        if process.returncode == 0:
            result['success'] = True
            print(f"Completed experiment on GPU {gpu_id}: {config.name} ({runtime/3600:.2f}h)")
        else:
            result['error'] = f"Exit code {process.returncode}: {process.stderr}"
            print(f"Failed experiment on GPU {gpu_id}: {config.name}")
            print(f"Error: {result['error']}")
        
    except subprocess.TimeoutExpired:
        result['error'] = "Timeout (24h)"
        print(f"Timeout for experiment on GPU {gpu_id}: {config.name}")
    except Exception as e:
        result['error'] = str(e)
        print(f"Exception for experiment on GPU {gpu_id}: {config.name} - {e}")
    
    # Clean up temporary config
    try:
        config_path.unlink()
    except:
        pass
    
    return result

def run_experiments_parallel(configs: List[ExperimentConfig], args) -> List[Dict]:
    """Run experiments in parallel across GPUs"""
    
    print(f"Running {len(configs)} experiments across {len(args.gpus)} GPUs")
    print(f"Maximum parallel jobs: {min(args.max_parallel, len(args.gpus))}")
    
    results = []
    
    # Create job queue
    jobs = []
    for i, config in enumerate(configs):
        gpu_id = args.gpus[i % len(args.gpus)]
        jobs.append((config, gpu_id, args.dry_run))
    
    # Run jobs in parallel
    max_workers = min(args.max_parallel, len(args.gpus))
    
    if args.dry_run or max_workers == 1:
        # Sequential execution
        for config, gpu_id, dry_run in jobs:
            result = run_single_experiment(config, gpu_id, dry_run)
            results.append(result)
    else:
        # Parallel execution
        with mp.Pool(max_workers) as pool:
            results = pool.starmap(run_single_experiment, jobs)
    
    return results

def run_shadow_training(configs: List[ExperimentConfig], args) -> bool:
    """Run shadow model training for MIA"""
    
    print("\nStarting shadow model training for MIA...")
    
    # Get unique model configurations (ignoring seeds)
    unique_configs = {}
    for config in configs:
        if config.pruning.enabled:
            key = f"{config.pruning.method}_sparsity{config.pruning.sparsity}"
        else:
            key = "dense"
        
        if key not in unique_configs:
            unique_configs[key] = config
    
    print(f"Training shadow models for {len(unique_configs)} configurations")
    
    shadow_results = []
    
    for key, base_config in unique_configs.items():
        print(f"\nTraining shadows for {key}...")
        
        # Build shadow training command
        cmd = [
            'python', 'experiments/train_shadows.py',
            '--dataset', args.dataset,
            '--arch', args.arch,
            '--layers', str(args.layers),
            '--model-type', 'dense' if not base_config.pruning.enabled else base_config.pruning.method,
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size),
            '--lr', str(args.lr),
            '--num-shadows', str(args.num_shadows),
            '--gpu', str(args.gpus[0]),
            '--save-dir', os.path.join(args.save_dir, 'shadows'),
            '--datapath', args.datapath
        ]
        
        if base_config.pruning.enabled:
            cmd.extend(['--sparsity', str(base_config.pruning.sparsity)])
            
            if base_config.pruning.method == 'static':
                # Need pretrained dense model for static pruning
                dense_model_path = os.path.join(
                    args.save_dir, 'dense', f'seed{args.seeds[0]}', 'best_model.pth'
                )
                if os.path.exists(dense_model_path):
                    cmd.extend(['--pretrained-target', dense_model_path])
        
        if args.dry_run:
            print(f"[DRY RUN] Shadow training: {' '.join(cmd)}")
            continue
        
        try:
            start_time = time.time()
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=48*3600)
            runtime = time.time() - start_time
            
            success = process.returncode == 0
            if success:
                print(f"Completed shadow training for {key} ({runtime/3600:.2f}h)")
            else:
                print(f"Failed shadow training for {key}: {process.stderr}")
            
            shadow_results.append({
                'config': key,
                'success': success,
                'runtime': runtime,
                'error': None if success else process.stderr
            })
            
        except Exception as e:
            print(f"Error in shadow training for {key}: {e}")
            shadow_results.append({
                'config': key,
                'success': False,
                'runtime': 0,
                'error': str(e)
            })
    
    # Check if all shadow training succeeded
    all_success = all(r['success'] for r in shadow_results)
    
    if all_success:
        print("All shadow model training completed successfully")
    else:
        failed = [r['config'] for r in shadow_results if not r['success']]
        print(f"Shadow training failed for: {failed}")
    
    return all_success

def run_mia_evaluation(args) -> bool:
    """Run MIA evaluation on all models"""
    
    print("\nStarting MIA evaluation...")
    
    cmd = [
        'python', 'experiments/evaluate_lira.py',
        '--dataset', args.dataset,
        '--arch', args.arch,
        '--layers', str(args.layers),
        '--target-models-dir', args.save_dir,
        '--shadow-models-dir', os.path.join(args.save_dir, 'shadows'),
        '--results-dir', os.path.join(args.save_dir, 'mia_results'),
        '--num-shadows', str(args.num_shadows),
        '--gpu', str(args.gpus[0]),
        '--batch-size', str(args.batch_size),
        '--datapath', args.datapath
    ]
    
    if args.dry_run:
        print(f"[DRY RUN] MIA evaluation: {' '.join(cmd)}")
        return True
    
    try:
        start_time = time.time()
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=12*3600)
        runtime = time.time() - start_time
        
        success = process.returncode == 0
        if success:
            print(f"MIA evaluation completed successfully ({runtime/3600:.2f}h)")
        else:
            print(f"MIA evaluation failed: {process.stderr}")
        
        return success
        
    except Exception as e:
        print(f"Error in MIA evaluation: {e}")
        return False

def generate_final_report(results: List[Dict], args):
    """Generate comprehensive final report"""
    
    print("\nGenerating final report...")
    
    # Create results directory
    report_dir = Path(args.save_dir) / 'final_report'
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary statistics
    total_experiments = len(results)
    successful_experiments = sum(1 for r in results if r['success'])
    failed_experiments = total_experiments - successful_experiments
    total_runtime = sum(r['runtime'] for r in results)
    
    # Create summary
    summary = {
        'experiment_info': {
            'dataset': args.dataset,
            'architecture': f"{args.arch}-{args.layers}",
            'seeds': args.seeds,
            'sparsity_levels': args.sparsity_levels,
            'total_experiments': total_experiments,
            'successful_experiments': successful_experiments,
            'failed_experiments': failed_experiments,
            'success_rate': successful_experiments / total_experiments,
            'total_runtime_hours': total_runtime / 3600,
            'average_runtime_hours': total_runtime / total_experiments / 3600,
        },
        'results': results
    }
    
    # Save detailed results
    with open(report_dir / 'detailed_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create summary table
    import pandas as pd
    
    summary_data = []
    for result in results:
        config = result['config']
        
        row = {
            'experiment_name': config['name'],
            'model_type': 'dense' if not config['pruning']['enabled'] else config['pruning']['method'],
            'sparsity': config['pruning']['sparsity'] if config['pruning']['enabled'] else 0.0,
            'seed': config['system']['seed'],
            'success': result['success'],
            'runtime_hours': result['runtime'] / 3600,
            'error': result['error'] if result['error'] else ''
        }
        
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    df.to_csv(report_dir / 'experiment_summary.csv', index=False)
    
    # Print summary
    print(f"\nFINAL EXPERIMENT SUMMARY")
    print(f"{'='*50}")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {successful_experiments}")
    print(f"Failed: {failed_experiments}")
    print(f"Success rate: {successful_experiments/total_experiments:.2%}")
    print(f"Total runtime: {total_runtime/3600:.2f} hours")
    print(f"Average runtime: {total_runtime/total_experiments/3600:.2f} hours")
    
    if failed_experiments > 0:
        print(f"\nFailed experiments:")
        for result in results:
            if not result['success']:
                print(f"  {result['config']['name']}: {result['error']}")
    
    # Use ResultsCollector for additional analysis
    if successful_experiments > 0:
        try:
            collector = ResultsCollector(args.save_dir)
            df_comparison, report = collector.generate_report(str(report_dir))
            print(f"\nDetailed analysis saved to: {report_dir}")
        except Exception as e:
            print(f"Error generating detailed report: {e}")
    
    print(f"\nReport saved to: {report_dir}")

def main():
    args = parse_args()
    
    print(f"Full Experiment Runner")
    print(f"Dataset: {args.dataset}")
    print(f"Architecture: {args.arch}-{args.layers}")
    print(f"Seeds: {args.seeds}")
    print(f"Sparsity levels: {args.sparsity_levels}")
    print(f"GPUs: {args.gpus}")
    print(f"MIA evaluation: {args.run_mia}")
    
    if args.dry_run:
        print("\n*** DRY RUN MODE - No actual experiments will be run ***")
    
    # Create all experiment configurations
    configs = create_experiment_configs(args)
    
    print(f"\nCreated {len(configs)} experiment configurations:")
    for config in configs:
        print(f"  {config.name}")
    
    # Run all experiments
    print(f"\n{'='*60}")
    print("PHASE 1: TRAINING TARGET MODELS")
    print(f"{'='*60}")
    
    experiment_results = run_experiments_parallel(configs, args)
    
    # Run shadow model training if MIA is requested
    shadow_success = True
    if args.run_mia:
        print(f"\n{'='*60}")
        print("PHASE 2: TRAINING SHADOW MODELS")
        print(f"{'='*60}")
        
        shadow_success = run_shadow_training(configs, args)
    
    # Run MIA evaluation
    mia_success = True
    if args.run_mia and shadow_success:
        print(f"\n{'='*60}")
        print("PHASE 3: MIA EVALUATION")
        print(f"{'='*60}")
        
        mia_success = run_mia_evaluation(args)
    
    # Generate final report
    print(f"\n{'='*60}")
    print("PHASE 4: GENERATING REPORT")
    print(f"{'='*60}")
    
    generate_final_report(experiment_results, args)
    
    # Final status
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETED")
    print(f"{'='*60}")
    
    successful_targets = sum(1 for r in experiment_results if r['success'])
    total_targets = len(experiment_results)
    
    print(f"Target model training: {successful_targets}/{total_targets} successful")
    
    if args.run_mia:
        print(f"Shadow model training: {'SUCCESS' if shadow_success else 'FAILED'}")
        print(f"MIA evaluation: {'SUCCESS' if mia_success else 'FAILED'}")
    
    if successful_targets == total_targets and (not args.run_mia or (shadow_success and mia_success)):
        print("ALL PHASES COMPLETED SUCCESSFULLY! 🎉")
    else:
        print("Some phases failed. Check the detailed report for more information.")

if __name__ == '__main__':
    main()