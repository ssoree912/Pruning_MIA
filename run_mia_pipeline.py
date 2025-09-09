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
    """DWA 훈련 결과 확인"""
    runs_path = Path(runs_dir)
    dwa_path = runs_path / 'dwa'
    
    if not dwa_path.exists():
        print(f"❌ No DWA results found in {runs_dir}")
        print("먼저 train_dwa.py를 실행해서 DWA 모델들을 훈련해주세요.")
        return False, 0
    
    # DWA 모델 개수 세기
    model_count = 0
    for mode_dir in dwa_path.iterdir():
        if mode_dir.is_dir():
            for sparsity_dir in mode_dir.iterdir():
                if sparsity_dir.is_dir() and sparsity_dir.name.startswith('sparsity_'):
                    for dataset_dir in sparsity_dir.iterdir():
                        if dataset_dir.is_dir():
                            if (dataset_dir / 'best_model.pth').exists():
                                model_count += 1
    
    print(f"✅ Found {model_count} trained DWA models in {runs_dir}")
    return True, model_count

def main():
    parser = argparse.ArgumentParser(description='DWA MIA Evaluation Pipeline')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'cifar100'], help='Dataset name')
    parser.add_argument('--runs_dir', type=str, default='./runs', 
                       help='DWA training results directory')
    parser.add_argument('--output_dir', type=str, default='./mia_results',
                       help='MIA results output directory')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
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
    has_results, model_count = check_dwa_results(args.runs_dir)
    if not has_results:
        print("\n💡 DWA 모델을 먼저 훈련하려면:")
        print("   python train_dwa.py --dwa-modes reactivate_only kill_active_plain_dead kill_and_reactivate")
        return
    
    if model_count == 0:
        print("❌ No trained models found. Please run train_dwa.py first.")
        return
    
    # Step 2: MIA 데이터 준비
    if not args.skip_data_prep:
        print(f"\n📊 Step 2: Preparing MIA data splits for {args.dataset}...")
        prep_cmd = [
            sys.executable, 'scripts/prepare_mia_data.py',
            '--dataset', args.dataset,
            '--output_dir', './mia_data'
        ]
        
        if not run_command(prep_cmd):
            print("❌ MIA data preparation failed")
            return
    else:
        print("\n⏭️  Step 2: Skipping MIA data preparation")
    
    # Step 3: MIA 평가 실행
    print(f"\n🎯 Step 3: Running MIA evaluation on {model_count} models...")
    eval_cmd = [
        sys.executable, 'evaluate_dwa_mia.py',
        '--runs_dir', args.runs_dir,
        '--output_dir', args.output_dir, 
        '--device', args.device,
        '--batch_size', str(args.batch_size),
        '--dataset', args.dataset,
        '--datapath', args.datapath
    ]
    
    if not run_command(eval_cmd):
        print("❌ MIA evaluation failed")
        return
    
    # Step 4: 결과 요약
    print(f"\n📈 Step 4: Generating summary...")
    
    output_path = Path(args.output_dir)
    if output_path.exists():
        # 최신 결과 파일 찾기
        csv_files = list(output_path.glob('dwa_mia_results_*.csv'))
        if csv_files:
            latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
            print(f"📁 Latest results: {latest_csv}")
            
            # 간단한 요약 출력
            try:
                import pandas as pd
                df = pd.read_csv(latest_csv)
                
                print(f"\n📊 MIA Attack Success Summary ({len(df)} models):")
                print("-" * 60)
                
                # 주요 공격 성공률 통계
                attack_cols = ['attack_conf_gt', 'attack_entropy', 'attack_modified_entropy', 'attack_conf_top1']
                df['best_attack'] = df[attack_cols].max(axis=1)
                
                # DWA 모드별 평균
                summary = df.groupby(['dwa_mode', 'sparsity_actual']).agg({
                    'best_attack': ['mean', 'std'],
                    'confidence_gap': ['mean', 'std'],
                    'best_acc1': 'mean'
                }).round(3)
                
                print(summary)
                
                # 최고/최저 공격 성공률
                best_idx = df['best_attack'].idxmax()
                worst_idx = df['best_attack'].idxmin()
                
                print(f"\n🔥 Highest vulnerability:")
                print(f"   {df.loc[best_idx, 'dwa_mode']} (sparsity={df.loc[best_idx, 'sparsity_actual']:.3f}): {df.loc[best_idx, 'best_attack']:.3f}")
                print(f"🛡️  Lowest vulnerability:")
                print(f"   {df.loc[worst_idx, 'dwa_mode']} (sparsity={df.loc[worst_idx, 'sparsity_actual']:.3f}): {df.loc[worst_idx, 'best_attack']:.3f}")
                
            except ImportError:
                print("📄 For detailed analysis, install pandas: pip install pandas")
                print(f"📁 Raw results available at: {latest_csv}")
    
    print(f"\n✅ MIA evaluation pipeline completed successfully!")
    print(f"📁 Results saved in: {args.output_dir}")

if __name__ == '__main__':
    main()