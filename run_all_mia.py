#!/usr/bin/env python3
"""
한번에 모든 DWA 실험에 대한 MIA 평가 실행
runs/dwa 하위의 모든 완료된 훈련 결과를 자동 감지하고 MIA 평가를 병렬로 실행
"""

import os
import sys
import subprocess
import concurrent.futures
from pathlib import Path
from datetime import datetime

def run_single_experiment_mia(experiment):
    """단일 실험에 대한 MIA 평가 실행"""
    mode = experiment['mode']
    sparsity = experiment['sparsity']
    dataset = experiment['dataset']
    model = experiment['model']
    
    print(f"🚀 Starting MIA for {mode}/sparsity_{sparsity}/{dataset}")
    
    try:
        cmd = [
            'python', 'run_single_mia.py',
            '--dataset', dataset,
            '--model', model,
            '--mode', mode,
            '--sparsity', sparsity,
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
    
    parser = argparse.ArgumentParser(description='Run MIA evaluation for all DWA experiments')
    parser.add_argument('--max_workers', type=int, default=1, help='Maximum number of parallel workers')
    parser.add_argument('--filter_mode', help='Filter by DWA mode')
    parser.add_argument('--filter_sparsity', help='Filter by sparsity level')
    
    args = parser.parse_args()
    
    print("🎯 Running MIA Evaluation for ALL DWA Experiments")
    print("=" * 80)
    
    # 먼저 실험 발견
    print("🔍 Discovering completed experiments...")
    cmd = ['python', 'run_batch_mia.py', '--dry_run']
    
    if args.filter_mode:
        cmd.extend(['--filter_mode', args.filter_mode])
    if args.filter_sparsity:
        cmd.extend(['--filter_sparsity', args.filter_sparsity])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to discover experiments: {e}")
        print("STDERR:", e.stderr)
        sys.exit(1)
    
    # 실제 실행
    print("\n🚀 Starting batch MIA evaluation...")
    cmd = ['python', 'run_batch_mia.py']
    
    if args.filter_mode:
        cmd.extend(['--filter_mode', args.filter_mode])
    if args.filter_sparsity:
        cmd.extend(['--filter_sparsity', args.filter_sparsity])
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ All MIA evaluations completed successfully!")
        
        # 결과 요약 출력
        print("\n📊 Check the following directories for results:")
        print("   - log/: Individual MIA evaluation logs")
        print("   - batch_mia_results_*.json: Batch processing summary")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Batch processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()