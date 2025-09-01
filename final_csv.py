#!/usr/bin/env python3
"""
Dense/Static/DPF 모든 결과를 하나의 CSV로 저장
"""

import json
import csv
from pathlib import Path

def main():
    runs_dir = Path('./runs')
    results = []
    
    print("📊 모든 훈련 결과 수집 중...")
    
    # Dense 모델 (runs/dense/seed42/, runs/dense/seed123/)
    dense_dir = runs_dir / 'dense'
    if dense_dir.exists():
        for seed_dir in dense_dir.iterdir():
            if seed_dir.is_dir():
                summary_path = seed_dir / 'experiment_summary.json'
                if summary_path.exists():
                    with open(summary_path) as f:
                        summary = json.load(f)
                    
                    best_acc = summary['best_metrics']['best_acc1']
                    final_acc = summary['final_metrics']['acc1']
                    name = summary['hyperparameters']['name']
                    
                    results.append({
                        'name': name,
                        'method': 'dense',
                        'sparsity_percent': 0,
                        'best_acc1': best_acc,
                        'final_acc1': final_acc,
                        'time_hours': round(summary['total_duration'] / 3600, 2)
                    })
                    print(f"✓ {name}: Best={best_acc:.2f}%, Final={final_acc:.2f}%")
    
    # Static/DPF 모델 (runs/method/sparsity/seed/)
    for method in ['static', 'dpf']:
        method_dir = runs_dir / method
        if method_dir.exists():
            for sparsity_dir in method_dir.iterdir():
                if sparsity_dir.is_dir() and sparsity_dir.name.startswith('sparsity'):
                    for seed_dir in sparsity_dir.iterdir():
                        if seed_dir.is_dir():
                            summary_path = seed_dir / 'experiment_summary.json'
                            if summary_path.exists():
                                with open(summary_path) as f:
                                    summary = json.load(f)
                                
                                best_acc = summary['best_metrics']['best_acc1']
                                final_acc = summary['final_metrics']['acc1']
                                name = summary['hyperparameters']['name']
                                sparsity = float(sparsity_dir.name.replace('sparsity', ''))
                                
                                results.append({
                                    'name': name,
                                    'method': method,
                                    'sparsity_percent': int(sparsity * 100),
                                    'best_acc1': best_acc,
                                    'final_acc1': final_acc,
                                    'time_hours': round(summary['total_duration'] / 3600, 2)
                                })
                                print(f"✓ {name}: Best={best_acc:.2f}%, Final={final_acc:.2f}%")
    
    # 스파시티별로 정렬
    results.sort(key=lambda x: (x['sparsity_percent'], x['method']))
    
    # CSV 저장
    with open('complete_training_results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'method', 'sparsity_percent', 'best_acc1', 'final_acc1', 'time_hours'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n📁 CSV 저장: complete_training_results.csv")
    print(f"✅ 총 {len(results)}개 모델 결과!")

if __name__ == '__main__':
    main()