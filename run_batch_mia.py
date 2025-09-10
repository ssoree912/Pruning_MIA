#!/usr/bin/env python3
"""
전체 DWA 훈련 결과에 대한 배치 MIA 평가 스크립트
runs/dwa/ 하위의 모든 훈련 결과를 자동으로 감지하고 MIA 평가 실행
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

def discover_dwa_experiments(runs_dir='./runs'):
    """DWA 실험 결과 자동 발견"""
    experiments = []
    dwa_dir = Path(runs_dir) / 'dwa'
    
    if not dwa_dir.exists():
        print(f"❌ DWA directory not found: {dwa_dir}")
        return experiments
    
    print(f"🔍 Discovering experiments in {dwa_dir}")
    
    for mode_dir in dwa_dir.iterdir():
        if not mode_dir.is_dir():
            continue
            
        for sparsity_dir in mode_dir.iterdir():
            if not sparsity_dir.is_dir() or not sparsity_dir.name.startswith('sparsity_'):
                continue
                
            sparsity = sparsity_dir.name.replace('sparsity_', '')
            
            for dataset_dir in sparsity_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                
                # experiment_summary.json 파일이 있는지 확인 (훈련 완료 확인)
                for exp_dir in dataset_dir.rglob('*'):
                    if exp_dir.is_dir():
                        summary_file = exp_dir / 'experiment_summary.json'
                        if summary_file.exists():
                            try:
                                with open(summary_file) as f:
                                    summary = json.load(f)
                                    
                                experiment = {
                                    'mode': mode_dir.name,
                                    'sparsity': sparsity,
                                    'dataset': summary.get('dataset', dataset_dir.name),
                                    'model': summary.get('architecture', 'resnet18'),
                                    'path': str(exp_dir),
                                    'summary': summary
                                }
                                experiments.append(experiment)
                                print(f"   Found: {mode_dir.name}/sparsity_{sparsity}/{experiment['dataset']}")
                                
                            except (json.JSONDecodeError, KeyError) as e:
                                print(f"   Warning: Invalid summary file {summary_file}: {e}")
    
    print(f"✅ Found {len(experiments)} completed experiments")
    return experiments

def run_batch_mia(runs_dir='./runs', gpu=0, filter_dataset=None, filter_mode=None, filter_sparsity=None, 
                  attacks='samia,threshold,nn,nn_top3,nn_cls', skip_conversion=False):
    """배치 MIA 평가 실행"""
    
    print("🚀 Starting Batch MIA Evaluation")
    print(f"   GPU: {gpu}")
    print(f"   Attacks: {attacks}")
    print(f"   Filters: dataset={filter_dataset}, mode={filter_mode}, sparsity={filter_sparsity}")
    
    # 실험 발견
    experiments = discover_dwa_experiments(runs_dir)
    
    if not experiments:
        print("❌ No experiments found!")
        return []
    
    # 필터 적용
    filtered_experiments = []
    for exp in experiments:
        if filter_dataset and exp['dataset'] != filter_dataset:
            continue
        if filter_mode and exp['mode'] != filter_mode:
            continue  
        if filter_sparsity and exp['sparsity'] != str(filter_sparsity):
            continue
        filtered_experiments.append(exp)
    
    print(f"🎯 Processing {len(filtered_experiments)} experiments (after filtering)")
    
    results = []
    failed_experiments = []
    
    for i, exp in enumerate(filtered_experiments, 1):
        print(f"\n{'='*80}")
        print(f"📊 Processing {i}/{len(filtered_experiments)}: {exp['mode']}/sparsity_{exp['sparsity']}/{exp['dataset']}")
        print(f"{'='*80}")
        
        dataset = exp['dataset']
        model = exp['model']
        mode = exp['mode']
        sparsity = exp['sparsity']
        
        try:
            # Step 1: 변환 (필요시)
            if not skip_conversion:
                print("🔄 Converting DWA to WeMeM structure...")
                
                cmd = [
                    'python', 'scripts/dwa_to_wemem_converter.py',
                    '--dataset', dataset,
                    '--model', model,
                    '--mode', mode,
                    '--sparsity', sparsity,
                    '--runs_dir', runs_dir
                ]
                
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print("✅ Conversion successful")
            
            # Step 2: 설정 파일 생성/업데이트
            config_path = f"configs/{dataset}_{model}.json"
            config = {
                "dataset_name": dataset,
                "model_name": model,
                "num_cls": 10 if dataset == 'cifar10' else 100,
                "input_dim": 3,
                "image_size": 32,
                "hidden_size": 128,
                "seed": 7,
                "early_stop": 5,
                "batch_size": 128,
                "pruner_name": "l1unstructure",
                "prune_sparsity": float(sparsity),
                "shadow_num": 2,
                "attacks": attacks,
                "original": False
            }
            
            os.makedirs("configs", exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Step 3: MIA 평가 실행
            print("🎯 Running MIA evaluation...")
            
            cmd = [
                'python', 'mia_modi.py',
                str(gpu),
                config_path,
                '--dataset_name', dataset,
                '--model_name', model,
                '--attacks', attacks,
                '--prune_sparsity', sparsity
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ MIA evaluation successful")
            
            # 결과 수집
            log_file = f"log/{dataset}_{model}/l1unstructure_{sparsity}_.txt"
            result_data = {
                'experiment': exp,
                'log_file': log_file,
                'success': True,
                'timestamp': datetime.now().isoformat()
            }
            
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    result_data['log_content'] = f.read()
                    print(f"📊 Results saved to: {log_file}")
            
            results.append(result_data)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed: {e}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            
            failed_experiments.append({
                'experiment': exp,
                'error': str(e),
                'stdout': e.stdout,
                'stderr': e.stderr
            })
        
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            failed_experiments.append({
                'experiment': exp,
                'error': str(e)
            })
    
    # 전체 결과 요약
    print(f"\n{'='*80}")
    print("📊 BATCH PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successful: {len(results)}")
    print(f"❌ Failed: {len(failed_experiments)}")
    
    # 성공한 결과들 요약 출력
    if results:
        print("\n🏆 Successful Results:")
        for r in results:
            exp = r['experiment']
            print(f"   {exp['mode']}/sparsity_{exp['sparsity']}/{exp['dataset']} -> {r['log_file']}")
    
    # 실패한 실험들
    if failed_experiments:
        print("\n💥 Failed Experiments:")
        for f in failed_experiments:
            exp = f['experiment']
            print(f"   {exp['mode']}/sparsity_{exp['sparsity']}/{exp['dataset']}: {f['error']}")
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_result_file = f"batch_mia_results_{timestamp}.json"
    
    batch_results = {
        'summary': {
            'total_experiments': len(filtered_experiments),
            'successful': len(results),
            'failed': len(failed_experiments),
            'timestamp': datetime.now().isoformat(),
            'filters': {
                'dataset': filter_dataset,
                'mode': filter_mode, 
                'sparsity': filter_sparsity
            }
        },
        'results': results,
        'failed': failed_experiments
    }
    
    with open(batch_result_file, 'w') as f:
        json.dump(batch_results, f, indent=2)
    
    print(f"\n💾 Batch results saved to: {batch_result_file}")
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run batch MIA evaluation for all DWA experiments')
    parser.add_argument('--runs_dir', default='./runs', help='DWA results directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--filter_dataset', help='Filter by dataset (e.g., cifar10)')
    parser.add_argument('--filter_mode', help='Filter by DWA mode')
    parser.add_argument('--filter_sparsity', help='Filter by sparsity level')
    parser.add_argument('--attacks', default='samia,threshold,nn,nn_top3,nn_cls', help='MIA attacks to run')
    parser.add_argument('--skip_conversion', action='store_true', help='Skip DWA->WeMeM conversion')
    parser.add_argument('--dry_run', action='store_true', help='Only discover experiments, don\'t run MIA')
    
    args = parser.parse_args()
    
    print("🎯 DWA Batch MIA Evaluation Pipeline")
    print("=" * 60)
    
    if args.dry_run:
        print("🔍 DRY RUN: Discovering experiments only...")
        experiments = discover_dwa_experiments(args.runs_dir)
        print(f"\nWould process {len(experiments)} experiments")
        for exp in experiments:
            print(f"  - {exp['mode']}/sparsity_{exp['sparsity']}/{exp['dataset']}")
        return
    
    results = run_batch_mia(
        runs_dir=args.runs_dir,
        gpu=args.gpu,
        filter_dataset=args.filter_dataset,
        filter_mode=args.filter_mode,
        filter_sparsity=args.filter_sparsity,
        attacks=args.attacks,
        skip_conversion=args.skip_conversion
    )
    
    if results:
        print("\n✅ Batch processing completed successfully!")
    else:
        print("\n❌ Batch processing completed with issues!")
        sys.exit(1)

if __name__ == "__main__":
    main()