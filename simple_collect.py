#!/usr/bin/env python3
"""
간단한 결과 수집 스크립트 (pandas 없이)
"""

import os
import json
from pathlib import Path

def collect_results():
    runs_dir = Path('./runs')
    results = []
    
    print("📊 훈련 결과 수집 중...")
    
    # runs/method/sparsity/seed/ 구조 탐색
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
                        
                        # 직접 정확도 추출 (디버그에서 확인된 구조)
                        best_metrics = summary.get('best_metrics', {})
                        final_metrics = summary.get('final_metrics', {})
                        
                        # best_acc1이 있으면 그것을, 없으면 final acc1 사용
                        acc1 = best_metrics.get('best_acc1', final_metrics.get('acc1', 0.0))
                        
                        # 메타정보 추출
                        hyperparameters = summary.get('hyperparameters', {})
                        name = hyperparameters.get('name', seed_dir.name)
                        
                        # 방법과 스파시티는 디렉토리명에서 직접 추출
                        method = method_dir.name
                        if method == 'dense':
                            sparsity = 0.0
                        else:
                            # sparsity 디렉토리명에서 추출 (예: sparsity0.9 -> 0.9)
                            try:
                                sparsity = float(sparsity_dir.name.replace('sparsity', ''))
                            except:
                                sparsity = hyperparameters.get('pruning', {}).get('sparsity', 0.0)
                        
                        # 시간 계산
                        duration = summary.get('total_duration', 0.0)
                        duration_hours = duration / 3600 if duration > 100 else 0.0
                        
                        print(f"✓ {name}: {acc1:.2f}% ({method} {sparsity*100:.0f}%)")
                        
                        results.append({
                            'name': name,
                            'method': method,
                            'sparsity_percent': sparsity * 100,
                            'accuracy': acc1,
                            'time_hours': duration_hours
                        })
                        
                    except Exception as e:
                        print(f"✗ Error in {seed_dir}: {e}")
    
    return results

def create_simple_report(results):
    """간단한 텍스트 리포트 생성"""
    
    if not results:
        print("⚠️ 결과 없음")
        return
    
    # 정확도 순 정렬
    results_sorted = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    
    print(f"\n🏆 정확도 순위 ({len(results)}개 모델):")
    print("=" * 60)
    
    for i, result in enumerate(results_sorted, 1):
        sparsity_text = f"{result['sparsity_percent']:.0f}%" if result['sparsity_percent'] > 0 else "0%"
        print(f"{i:2d}. {result['name']:<25} {result['accuracy']:6.2f}% ({result['method'].upper()} {sparsity_text})")
    
    # 방법별 통계
    print(f"\n📊 방법별 평균 성능:")
    print("=" * 40)
    
    methods = {}
    for result in results:
        method = result['method']
        if method not in methods:
            methods[method] = []
        methods[method].append(result['accuracy'])
    
    for method, accs in methods.items():
        avg_acc = sum(accs) / len(accs)
        max_acc = max(accs)
        min_acc = min(accs)
        print(f"{method.upper():<8}: {avg_acc:6.2f}% (최고: {max_acc:.2f}%, 최저: {min_acc:.2f}%)")

def main():
    print("🎯 Dense vs Static vs DPF 결과 정리")
    print("=" * 50)
    
    results = collect_results()
    create_simple_report(results)
    
    print(f"\n🏁 완료!")

if __name__ == '__main__':
    main()