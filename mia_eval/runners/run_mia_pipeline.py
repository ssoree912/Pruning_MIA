#!/usr/bin/env python3
"""
DWA 프루닝된 모델에 대한 전체 MIA 파이프라인 실행 스크립트

Usage:
python run_mia_pipeline.py --dataset cifar10 --device cuda:0

이 스크립트는:
1. MIA용 데이터 분할 준비 (필요시)
2. DWA 훈련 결과에서 모델 찾기
3. 모든 DWA 모델에 대해 MIA 평가 수행
4. 결과 요약 및 비교 리포트 생성
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Resolve project root and absolute script paths
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
RUN_SINGLE = REPO_ROOT / 'mia_eval' / 'runners' / 'run_single_mia.py'
CREATE_SPLITS = REPO_ROOT / 'mia_eval' / 'data' / 'create_fixed_data_splits.py'
from datetime import datetime

def run_command(cmd, cwd=None):
    """명령어 실행"""
    print(f"🔧 Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def check_dwa_results(runs_dir):
    """DWA 훈련 결과 확인 - seed 폴더 구조"""
    runs_path = Path(runs_dir)
    dwa_path = runs_path / 'dwa'
    
    if not dwa_path.exists():
        print(f"❌ No DWA results found in {runs_dir}")
        print("먼저 train_dwa.py를 실행해서 DWA 모델들을 훈련해주세요.")
        return False, 0, []
    
    # seed별 모델 찾기
    experiments = []
    model_count = 0
    
    for mode_dir in dwa_path.iterdir():
        if mode_dir.is_dir():
            for sparsity_dir in mode_dir.iterdir():
                if sparsity_dir.is_dir() and sparsity_dir.name.startswith('sparsity_'):
                    sparsity = sparsity_dir.name.split('_')[1]
                    for dataset_dir in sparsity_dir.iterdir():
                        if dataset_dir.is_dir():
                            for alpha_beta_dir in dataset_dir.iterdir():
                                if alpha_beta_dir.is_dir() and 'alpha' in alpha_beta_dir.name:
                                    # alpha, beta 추출
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
                                                model_count += 1
                                    
                                    if len(seeds) >= 2:  # 최소 victim + shadow 1개
                                        experiments.append({
                                            'mode': mode_dir.name,
                                            'sparsity': sparsity,
                                            'dataset': dataset_dir.name,
                                            'alpha': alpha,
                                            'beta': beta,
                                            'seeds': sorted(seeds)
                                        })
    
    print(f"✅ Found {model_count} trained models in {len(experiments)} experiment groups")
    return True, model_count, experiments

def main():
    parser = argparse.ArgumentParser(description='DWA MIA Evaluation Pipeline')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'cifar100'], help='Dataset name')
    parser.add_argument('--runs_dir', type=str, default='./runs', 
                       help='DWA training results directory')
    parser.add_argument('--output_dir', type=str, default='./mia_results',
                       help='MIA results output directory')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--split_seed', type=int, default=7, help='Seed used for fixed MIA data splits')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--datapath', type=str, default='~/Datasets', help='Dataset path')
    parser.add_argument('--skip_data_prep', action='store_true', 
                       help='Skip MIA data preparation step')
    
    args = parser.parse_args()
    
    print("🚀 DWA MIA Evaluation Pipeline")
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"DWA Results: {args.runs_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print("=" * 50)
    
    # Step 1: DWA 훈련 결과 확인
    print("\n📋 Step 1: Checking DWA training results...")
    has_results, model_count, experiments = check_dwa_results(args.runs_dir)
    if not has_results:
        print("\n💡 DWA 모델을 먼저 훈련하려면:")
        print("   python train_dwa.py --dwa-modes reactivate_only kill_active_plain_dead kill_and_reactivate")
        return
    
    if model_count == 0:
        print("❌ No trained models found. Please run train_dwa.py first.")
        return
    
    print(f"Found {len(experiments)} experiment groups:")
    for exp in experiments:
        print(f"  - {exp['mode']}/sparsity_{exp['sparsity']}/{exp['dataset']}/alpha{exp['alpha']}_beta{exp['beta']} ({len(exp['seeds'])} seeds)")
    
    # Step 2: Ensure fixed data splits exist (auto-create if missing)
    print("\n🧩 Step 2: Ensuring fixed MIA data splits...")
    split_seed = args.split_seed
    
    # Step 3: MIA 평가 실행
    print(f"\n🎯 Step 3: Running MIA evaluation on {len(experiments)} experiment groups...")
    
    success_count = 0
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Processing {exp['mode']}/sparsity_{exp['sparsity']}/{exp['dataset']}...")
        
        if len(exp['seeds']) < 2:
            print(f"⏭️ Skipping - need at least 2 seeds, found {len(exp['seeds'])}")
            continue
            
        # victim은 첫 번째 seed, shadow는 나머지
        victim_seed = exp['seeds'][0]
        shadow_seeds = exp['seeds'][1:]
        
        # Ensure pkl exists per victim seed
        split_path = Path(f"mia_data_splits/{exp['dataset']}_seed{split_seed}_victim{victim_seed}.pkl")
        if not split_path.exists():
            print(f"  📦 Creating fixed splits: {split_path}")
            mk_cmd = [
                sys.executable, str(CREATE_SPLITS),
                '--dataset', exp['dataset'],
                '--seed', str(split_seed),
                '--victim_seed', str(victim_seed),
                '--shadow_seeds', *[str(s) for s in shadow_seeds]
            ]
            if not run_command(mk_cmd, cwd=str(REPO_ROOT)):
                print(f"  ❌ Failed to create data splits for victim_seed={victim_seed}. Skipping.")
                continue

        eval_cmd = [
            sys.executable, str(RUN_SINGLE),
            '--dataset', exp['dataset'],
            '--model', 'resnet18',
            '--sparsity', exp['sparsity'],
            '--alpha', exp['alpha'],
            '--beta', exp['beta'],
            '--prune_method', 'dwa',
            '--prune_type', exp['mode'],
            '--victim_seed', str(victim_seed),
            '--shadow_seeds'] + [str(s) for s in shadow_seeds] + [
            '--device', args.device.replace('cuda:', ''),
            '--split_seed', str(split_seed)
        ]
        
        if run_command(eval_cmd, cwd=str(REPO_ROOT)):
            success_count += 1
        else:
            print(f"❌ Failed to evaluate {exp['mode']}/sparsity_{exp['sparsity']}/{exp['dataset']}")
    
    print(f"\n📊 Completed {success_count}/{len(experiments)} evaluations")
    
    # Step 4: 결과 요약
    print(f"\n📈 Step 4: Results summary...")
    
    result_dir = Path('mia_results')
    if result_dir.exists():
        json_files = list(result_dir.glob('**/*.json'))
        if json_files:
            print(f"\n📁 Found {len(json_files)} result files:")
            
            # 간단한 요약 출력
            import json
            all_results = []
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        config = data['config']
                        results = data['results']
                        
                        summary = {
                            'mode': config.get('prune_type', 'unknown'),
                            'sparsity': config.get('sparsity', 0),
                            'victim_seed': config.get('victim_seed', 0),
                            'victim_acc': data.get('victim_test_acc', 0),
                        }
                        summary.update(results)
                        all_results.append(summary)
                        
                except Exception as e:
                    print(f"⚠️ Error reading {json_file}: {e}")
            
            if all_results:
                print(f"\n📊 MIA Attack Success Summary ({len(all_results)} experiments):")
                print("-" * 80)
                for result in all_results:
                    print(f"{result['mode']:20s} sparsity={result['sparsity']:4.2f} victim_seed={result['victim_seed']:2d} acc={result['victim_acc']:5.3f}")
                    if 'samia' in result:
                        print(f"{'':20s} SAMIA: {result['samia']:5.3f}")
                    if 'confidence' in result:
                        print(f"{'':20s} Conf: {result['confidence']:5.3f}")
                    print()

                # Write CSV summary
                import csv
                summary_file = result_dir / 'summary.csv'
                fieldnames = [
                    'mode', 'sparsity', 'victim_seed', 'victim_acc',
                    'samia', 'confidence', 'entropy', 'modified_entropy', 'top1_conf',
                    'nn', 'nn_top3', 'nn_cls',
                    'confidence_extended_auroc', 'confidence_extended_accuracy', 'confidence_extended_advantage'
                ]
                with open(summary_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for r in all_results:
                        row = {k: r.get(k, '') for k in fieldnames}
                        # Map nested extended metrics if present
                        if 'confidence_extended' in r:
                            ext = r['confidence_extended']
                            row['confidence_extended_auroc'] = ext.get('auroc', '')
                            row['confidence_extended_accuracy'] = ext.get('accuracy', '')
                            row['confidence_extended_advantage'] = ext.get('advantage', '')
                        writer.writerow(row)
                print(f"📄 Wrote CSV summary: {summary_file}")
        else:
            print("📄 No result files found")
    
    if success_count > 0:
        print(f"\n✅ MIA evaluation pipeline completed successfully!")
        print(f"📁 Results saved in: mia_results/")
    else:
        print(f"\n❌ No evaluations completed successfully")

if __name__ == '__main__':
    main()
