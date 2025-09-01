#!/usr/bin/env python3
"""
완전한 훈련 결과 CSV 생성 (best/final 정확도 포함)
"""

import os
import json
import csv
from pathlib import Path

def collect_complete_results():
    runs_dir = Path('./runs')
    results = []
    
    print("📊 완전한 훈련 결과 수집 중...")
    
    for method_dir in runs_dir.iterdir():
        if not method_dir.is_dir() or method_dir.name == 'final_report':
            continue
        
        # Dense 모델은 직접 seed 폴더 구조
        if method_dir.name == 'dense':
            for seed_dir in method_dir.iterdir():
                if not seed_dir.is_dir():
                    continue
                summary_path = seed_dir / 'experiment_summary.json'
                if summary_path.exists():
                    sparsity_dir = method_dir  # Dense는 sparsity_dir이 없음
                    process_model(summary_path, method_dir, sparsity_dir, seed_dir, results)
        else:
            # Static/DPF는 기존 구조
            for sparsity_dir in method_dir.iterdir():
                if not sparsity_dir.is_dir():
                    continue
                    
                for seed_dir in sparsity_dir.iterdir():
                    if not seed_dir.is_dir():
                        continue
                        
                    summary_path = seed_dir / 'experiment_summary.json'
                    if summary_path.exists():
                        process_model(summary_path, method_dir, sparsity_dir, seed_dir, results)

def process_model(summary_path, method_dir, sparsity_dir, seed_dir, results):
    """모델 결과 처리 함수"""
    try:
        with open(summary_path) as f:
            summary = json.load(f)
        
        # 정확도 추출
        best_metrics = summary.get('best_metrics', {})
        final_metrics = summary.get('final_metrics', {})
        
        best_acc1 = best_metrics.get('best_acc1', 0.0)
        final_acc1 = final_metrics.get('acc1', 0.0)
        best_acc5 = best_metrics.get('best_acc5', final_metrics.get('acc5', 0.0))
        
        # 메타정보
        hyperparameters = summary.get('hyperparameters', {})
        name = hyperparameters.get('name', seed_dir.name)
        method = method_dir.name
        
        if method == 'dense':
            sparsity = 0.0
            sparsity_percent = 0
        else:
            try:
                sparsity = float(sparsity_dir.name.replace('sparsity', ''))
                sparsity_percent = int(sparsity * 100)
            except:
                sparsity = hyperparameters.get('pruning', {}).get('sparsity', 0.0)
                sparsity_percent = int(sparsity * 100)
        
        duration = summary.get('total_duration', 0.0)
        duration_hours = duration / 3600 if duration > 100 else 0.0
        
        result = {
            'name': name,
            'method': method,
            'sparsity': sparsity,
            'sparsity_percent': sparsity_percent,
            'best_acc1': best_acc1,
            'final_acc1': final_acc1,
            'best_acc5': best_acc5,
            'time_hours': round(duration_hours, 2),
            'epochs': hyperparameters.get('training', {}).get('epochs', 0),
            'seed': hyperparameters.get('system', {}).get('seed', 42)
        }
        
        results.append(result)
        print(f"✓ {name}: Best={best_acc1:.2f}%, Final={final_acc1:.2f}%")
        
    except Exception as e:
        print(f"✗ Error in {seed_dir}: {e}")

def collect_complete_results():
    runs_dir = Path('./runs')
    results = []
    
    print("📊 완전한 훈련 결과 수집 중...")
    
    for method_dir in runs_dir.iterdir():
        if not method_dir.is_dir() or method_dir.name == 'final_report':
            continue
        
        # Dense 모델은 직접 seed 폴더 구조
        if method_dir.name == 'dense':
            for seed_dir in method_dir.iterdir():
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
                        
                        best_acc1 = best_metrics.get('best_acc1', 0.0)
                        final_acc1 = final_metrics.get('acc1', 0.0)
                        best_acc5 = best_metrics.get('best_acc5', final_metrics.get('acc5', 0.0))
                        
                        # 메타정보
                        hyperparameters = summary.get('hyperparameters', {})
                        name = hyperparameters.get('name', seed_dir.name)
                        method = method_dir.name
                        
                        if method == 'dense':
                            sparsity = 0.0
                            sparsity_percent = 0
                        else:
                            try:
                                sparsity = float(sparsity_dir.name.replace('sparsity', ''))
                                sparsity_percent = int(sparsity * 100)
                            except:
                                sparsity = hyperparameters.get('pruning', {}).get('sparsity', 0.0)
                                sparsity_percent = int(sparsity * 100)
                        
                        duration = summary.get('total_duration', 0.0)
                        duration_hours = duration / 3600 if duration > 100 else 0.0
                        
                        result = {
                            'name': name,
                            'method': method,
                            'sparsity': sparsity,
                            'sparsity_percent': sparsity_percent,
                            'best_acc1': best_acc1,
                            'final_acc1': final_acc1,
                            'best_acc5': best_acc5,
                            'time_hours': round(duration_hours, 2),
                            'epochs': hyperparameters.get('training', {}).get('epochs', 0),
                            'seed': hyperparameters.get('system', {}).get('seed', 42)
                        }
                        
                        results.append(result)
                        print(f"✓ {name}: Best={best_acc1:.2f}%, Final={final_acc1:.2f}%")
                        
                    except Exception as e:
                        print(f"✗ Error in {seed_dir}: {e}")
    
    # 결과를 sparsity와 method로 정렬
    results.sort(key=lambda x: (x['sparsity_percent'], x['method']))
    
    # CSV 저장
    csv_path = Path('./complete_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\n📁 완전한 결과 CSV: {csv_path}")
    print(f"✅ 총 {len(results)}개 모델 결과 저장!")
    
    # 요약 출력
    print(f"\n📊 결과 요약:")
    print(f"{'Method':<8} {'Sparsity':<8} {'Best Acc':<10} {'Final Acc':<10}")
    print("-" * 40)
    
    for result in results:
        sparsity_text = f"{result['sparsity_percent']}%" if result['sparsity_percent'] > 0 else "0%"
        print(f"{result['method']:<8} {sparsity_text:<8} {result['best_acc1']:<10.2f} {result['final_acc1']:<10.2f}")
    
    return results

def main():
    collect_complete_results()

if __name__ == '__main__':
    main()