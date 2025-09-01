#!/usr/bin/env python3
"""
훈련 결과를 스파시티별 CSV 파일로 저장
"""

import os
import json
import csv
from pathlib import Path

def collect_and_save_csv():
    runs_dir = Path('./runs')
    
    # 전체 결과 수집
    all_results = []
    sparsity_results = {}  # 스파시티별로 분리
    
    print("📊 훈련 결과 수집 중...")
    
    for method_dir in runs_dir.iterdir():
        if not method_dir.is_dir() or method_dir.name == 'final_report':
            continue
            
        for sparsity_dir in method_dir.iterdir():
            if not sparsity_dir.is_dir():
                continue
                
            for seed_dir in sparsity_dir.iterdir():
                if not seed_dir.is_dir():
                    continue
                    
                summary_path = seed_dir / 'experiment_summary.json'
                
                if summary_path.exists():
                    try:
                        with open(summary_path) as f:
                            summary = json.load(f)
                        
                        # 정확도 추출
                        best_metrics = summary.get('best_metrics', {})
                        final_metrics = summary.get('final_metrics', {})
                        acc1 = best_metrics.get('best_acc1', final_metrics.get('acc1', 0.0))
                        
                        # 메타정보
                        hyperparameters = summary.get('hyperparameters', {})
                        name = hyperparameters.get('name', seed_dir.name)
                        method = method_dir.name
                        
                        if method == 'dense':
                            sparsity = 0.0
                        else:
                            try:
                                sparsity = float(sparsity_dir.name.replace('sparsity', ''))
                            except:
                                sparsity = hyperparameters.get('pruning', {}).get('sparsity', 0.0)
                        
                        duration = summary.get('total_duration', 0.0)
                        duration_hours = duration / 3600 if duration > 100 else 0.0
                        
                        result = {
                            'name': name,
                            'method': method,
                            'sparsity': sparsity,
                            'sparsity_percent': sparsity * 100,
                            'accuracy': acc1,
                            'time_hours': duration_hours
                        }
                        
                        all_results.append(result)
                        
                        # 스파시티별로 분류
                        sparsity_key = f"{sparsity*100:.0f}%" if sparsity > 0 else "0%"
                        if sparsity_key not in sparsity_results:
                            sparsity_results[sparsity_key] = []
                        sparsity_results[sparsity_key].append(result)
                        
                        print(f"✓ {name}: {acc1:.2f}%")
                        
                    except Exception as e:
                        print(f"✗ Error in {seed_dir}: {e}")
    
    # 결과 디렉토리 생성
    results_dir = Path('./results')
    results_dir.mkdir(exist_ok=True)
    
    # 전체 결과 CSV 저장
    all_csv = results_dir / 'all_results.csv'
    with open(all_csv, 'w', newline='', encoding='utf-8') as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
    
    print(f"\n📁 전체 결과: {all_csv}")
    
    # 스파시티별 CSV 저장
    for sparsity_key, results in sparsity_results.items():
        sparsity_csv = results_dir / f'sparsity_{sparsity_key.replace("%", "percent")}.csv'
        with open(sparsity_csv, 'w', newline='', encoding='utf-8') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        
        print(f"📁 {sparsity_key} 결과: {sparsity_csv}")
    
    # 방법별 CSV 저장
    methods = set(r['method'] for r in all_results)
    for method in methods:
        method_results = [r for r in all_results if r['method'] == method]
        method_csv = results_dir / f'method_{method}.csv'
        with open(method_csv, 'w', newline='', encoding='utf-8') as f:
            if method_results:
                writer = csv.DictWriter(f, fieldnames=method_results[0].keys())
                writer.writeheader()
                writer.writerows(method_results)
        
        print(f"📁 {method.upper()} 결과: {method_csv}")
    
    print(f"\n✅ 총 {len(all_results)}개 모델 결과 저장 완료!")
    return all_results

def main():
    collect_and_save_csv()

if __name__ == '__main__':
    main()