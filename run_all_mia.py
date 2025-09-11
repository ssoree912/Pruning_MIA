#!/usr/bin/env python3
"""
한번에 모든 DWA 실험에 대한 MIA 평가 실행
runs/dwa 하위의 모든 완료된 훈련 결과를 자동 감지하고 MIA 평가를 병렬로 실행
같은 sparsity, 다른 seed 구조에 맞춰 수정됨
"""

import os
import sys
import subprocess
import concurrent.futures
from pathlib import Path
from datetime import datetime

def find_experiments(runs_dir='runs'):
    """seed 구조의 실험 찾기"""
    experiments = []
    runs_path = Path(runs_dir)
    dwa_path = runs_path / 'dwa'
    
    if not dwa_path.exists():
        return experiments
    
    for mode_dir in dwa_path.iterdir():
        if mode_dir.is_dir():
            for sparsity_dir in mode_dir.iterdir():
                if sparsity_dir.is_dir() and sparsity_dir.name.startswith('sparsity_'):
                    sparsity = sparsity_dir.name.split('_')[1]
                    for dataset_dir in sparsity_dir.iterdir():
                        if dataset_dir.is_dir():
                            for alpha_beta_dir in dataset_dir.iterdir():
                                if alpha_beta_dir.is_dir() and 'alpha' in alpha_beta_dir.name:
                                    parts = alpha_beta_dir.name.split('_')
                                    alpha = parts[0].replace('alpha', '')
                                    beta = parts[1].replace('beta', '')
                                    
                                    # seed 폴더들 찾기
                                    seeds = []
                                    for seed_dir in alpha_beta_dir.iterdir():
                                        if seed_dir.is_dir() and seed_dir.name.startswith('seed'):
                                            if (seed_dir / 'best_model.pth').exists():
                                                seed_num = int(seed_dir.name.replace('seed', ''))
                                                seeds.append(seed_num)
                                    
                                    if len(seeds) >= 2:  # 최소 victim + shadow 1개
                                        experiments.append({
                                            'mode': mode_dir.name,
                                            'sparsity': sparsity,
                                            'dataset': dataset_dir.name,
                                            'alpha': alpha,
                                            'beta': beta,
                                            'seeds': sorted(seeds)
                                        })
    
    return experiments

def run_single_experiment_mia(experiment):
    """단일 실험에 대한 MIA 평가 실행"""
    mode = experiment['mode']
    sparsity = experiment['sparsity']
    dataset = experiment['dataset']
    alpha = experiment['alpha']
    beta = experiment['beta']
    seeds = experiment['seeds']
    
    print(f"🚀 Starting MIA for {mode}/sparsity_{sparsity}/{dataset}/alpha{alpha}_beta{beta}")
    print(f"   Seeds: {seeds} (victim: {seeds[0]}, shadows: {seeds[1:]})")
    
    try:
        # victim은 첫 번째 seed, shadow는 나머지
        victim_seed = seeds[0]
        shadow_seeds = seeds[1:]
        
        cmd = [
            'python', 'run_single_mia.py',
            '--dataset', dataset,
            '--model', 'resnet',
            '--sparsity', sparsity,
            '--alpha', alpha,
            '--beta', beta,
            '--prune_method', 'dwa',
            '--prune_type', mode,
            '--victim_seed', str(victim_seed),
            '--shadow_seeds'] + [str(s) for s in shadow_seeds] + [
            '--gpu', '0'
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        return {
            'experiment': experiment,
            'success': True,
            'output': result.stdout,
            'timestamp': datetime.now().isoformat()
        }
        
    except subprocess.CalledProcessError as e:
        return {
            'experiment': experiment,
            'success': False,
            'error': str(e),
            'stdout': e.stdout,
            'stderr': e.stderr,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'experiment': experiment,
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def main():
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Run MIA evaluation for all DWA experiments')
    parser.add_argument('--max_workers', type=int, default=1, help='Maximum number of parallel workers')
    parser.add_argument('--filter_mode', help='Filter by DWA mode')
    parser.add_argument('--filter_sparsity', help='Filter by sparsity level')
    parser.add_argument('--runs_dir', default='runs', help='Runs directory')
    parser.add_argument('--dry_run', action='store_true', help='Only show what would be run')
    
    args = parser.parse_args()
    
    print("🎯 Running MIA Evaluation for ALL DWA Experiments (Same Sparsity, Different Seeds)")
    print("=" * 80)
    
    # 실험 찾기
    print("🔍 Discovering completed experiments...")
    experiments = find_experiments(args.runs_dir)
    
    # 필터링
    if args.filter_mode:
        experiments = [exp for exp in experiments if exp['mode'] == args.filter_mode]
    if args.filter_sparsity:
        experiments = [exp for exp in experiments if exp['sparsity'] == args.filter_sparsity]
    
    if not experiments:
        print("❌ No experiments found matching criteria")
        return
    
    print(f"✅ Found {len(experiments)} experiment groups:")
    for exp in experiments:
        print(f"  - {exp['mode']}/sparsity_{exp['sparsity']}/{exp['dataset']}/alpha{exp['alpha']}_beta{exp['beta']} ({len(exp['seeds'])} seeds)")
    
    if args.dry_run:
        print("\n🏃‍♂️ Dry run - would process these experiments")
        return
    
    # 실제 실행
    print(f"\n🚀 Starting MIA evaluation for {len(experiments)} experiments...")
    
    results = []
    success_count = 0
    
    if args.max_workers == 1:
        # 순차 실행
        for i, exp in enumerate(experiments, 1):
            print(f"\n[{i}/{len(experiments)}] Processing...")
            result = run_single_experiment_mia(exp)
            results.append(result)
            if result['success']:
                success_count += 1
                print("✅ Success")
            else:
                print(f"❌ Failed: {result['error']}")
    else:
        # 병렬 실행
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_exp = {executor.submit(run_single_experiment_mia, exp): exp for exp in experiments}
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_exp), 1):
                result = future.result()
                results.append(result)
                if result['success']:
                    success_count += 1
                    print(f"[{i}/{len(experiments)}] ✅ {result['experiment']['mode']}/sparsity_{result['experiment']['sparsity']}")
                else:
                    print(f"[{i}/{len(experiments)}] ❌ {result['experiment']['mode']}/sparsity_{result['experiment']['sparsity']}: {result['error']}")
    
    # 결과 저장
    result_summary = {
        'total_experiments': len(experiments),
        'successful': success_count,
        'failed': len(experiments) - success_count,
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    summary_file = f"batch_mia_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(result_summary, f, indent=2)
    
    print(f"\n📊 Summary:")
    print(f"   Total: {len(experiments)}")
    print(f"   Success: {success_count}")
    print(f"   Failed: {len(experiments) - success_count}")
    print(f"   Results saved to: {summary_file}")
    print(f"   Individual results in: mia_results/")
    
    if success_count > 0:
        print("\n✅ Batch MIA evaluation completed!")
    else:
        print("\n❌ No experiments completed successfully")
        sys.exit(1)

if __name__ == "__main__":
    main()