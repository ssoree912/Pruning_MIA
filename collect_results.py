#!/usr/bin/env python3
"""
훈련 결과 수집 및 정확도 정리 스크립트
"""

import os
import json
import pandas as pd
from pathlib import Path

def collect_training_results():
    """실제 디렉토리 구조에서 훈련 결과 수집: runs/method/sparsity/seed/"""
    
    runs_dir = Path('./runs')
    results = []
    
    print("📊 훈련 결과 수집 중...")
    print(f"검색 디렉토리: {runs_dir.absolute()}")
    
    # 실제 구조: runs/method/sparsity/seed/
    for method_dir in runs_dir.iterdir():
        if not method_dir.is_dir() or method_dir.name == 'final_report':
            continue
        
        print(f"  방법: {method_dir.name}")
        
        for sparsity_dir in method_dir.iterdir():
            if not sparsity_dir.is_dir():
                continue
                
            for seed_dir in sparsity_dir.iterdir():
                if not seed_dir.is_dir():
                    continue
                    
                config_path = seed_dir / 'config.json'
                summary_path = seed_dir / 'experiment_summary.json'
                
                if config_path.exists() and summary_path.exists():
                    try:
                        with open(config_path) as f:
                            config = json.load(f)
                        
                        with open(summary_path) as f:
                            summary = json.load(f)
                        
                        # 방법 파싱
                        method = method_dir.name
                        
                        # 스파시티 파싱 
                        if method == 'dense':
                            sparsity = 0.0
                        else:
                            sparsity = config.get('pruning', {}).get('sparsity', 0.0)
                        
                        # 정확도 필드 확인 (여러 가능성)
                        best_acc = (summary.get('best_acc1', 0.0) or 
                                   summary.get('acc1', 0.0) or 
                                   summary.get('accuracy', 0.0))
                        
                        final_acc = (summary.get('final_acc1', 0.0) or
                                    summary.get('acc1', 0.0))
                        
                        # 시간 계산 (초 → 시간)
                        duration = summary.get('total_duration', 0.0)
                        if duration > 100:  # 초 단위인 경우
                            duration_hours = duration / 3600
                        else:
                            duration_hours = summary.get('total_duration_hours', 0.0)
                        
                        result = {
                            'name': config['name'],
                            'method': method,
                            'sparsity': sparsity,
                            'sparsity_percent': sparsity * 100,
                            'best_acc1': best_acc,
                            'final_acc1': final_acc,
                            'total_duration_hours': duration_hours,
                            'epochs': config.get('training', {}).get('epochs', 0),
                            'seed': config.get('system', {}).get('seed', 42)
                        }
                        
                        results.append(result)
                        print(f"    ✓ {config['name']}: {best_acc:.2f}%")
                        
                    except Exception as e:
                        print(f"    ✗ Error processing {seed_dir}: {e}")
                else:
                    print(f"    ⚠️ Missing files in {seed_dir}")
    
    return results

def create_summary_report(results):
    """정확도 요약 리포트 생성"""
    
    if not results:
        print("⚠️ 훈련 결과가 없습니다.")
        return
    
    df = pd.DataFrame(results)
    
    # 결과 디렉토리 생성
    os.makedirs('./runs/final_report', exist_ok=True)
    
    # CSV 저장
    df.to_csv('./runs/final_report/experiments_comparison.csv', index=False)
    
    print(f"\n✅ 결과 수집 완료: {len(results)}개 모델")
    print(f"📁 저장 위치: ./runs/final_report/experiments_comparison.csv")
    
    # 정확도 순위 출력
    print("\n🏆 정확도 순위 (Top 10):")
    print("=" * 50)
    
    df_sorted = df.sort_values('best_acc1', ascending=False)
    for i, (_, row) in enumerate(df_sorted.head(10).iterrows(), 1):
        sparsity_text = f"{row.sparsity_percent:.0f}%" if row.sparsity > 0 else "0%"
        method_text = row.method.upper()
        print(f"{i:2d}. {row.name:<25} {row.best_acc1:6.2f}% ({method_text} {sparsity_text})")
    
    # 방법별 통계
    print("\n📊 방법별 성능 통계:")
    print("=" * 50)
    
    methods = ['dense', 'static', 'dpf']
    for method in methods:
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            print(f"\n{method.upper()}:")
            print(f"  모델 수: {len(method_data)}개")
            print(f"  평균 정확도: {method_data['best_acc1'].mean():.2f}% ± {method_data['best_acc1'].std():.2f}%")
            print(f"  최고 정확도: {method_data['best_acc1'].max():.2f}%")
            print(f"  최저 정확도: {method_data['best_acc1'].min():.2f}%")
            print(f"  평균 훈련시간: {method_data['total_duration_hours'].mean():.2f}시간")
    
    # 스파시티별 분석
    print("\n📈 스파시티별 성능 비교:")
    print("=" * 50)
    
    sparsities = sorted(df[df['sparsity'] > 0]['sparsity_percent'].unique())
    
    if len(sparsities) > 0:
        print(f"{'스파시티':<8} {'Static':<8} {'DPF':<8} {'차이':<8}")
        print("-" * 32)
        
        for sparsity in sparsities:
            static_data = df[(df['method'] == 'static') & (df['sparsity_percent'] == sparsity)]
            dpf_data = df[(df['method'] == 'dpf') & (df['sparsity_percent'] == sparsity)]
            
            static_acc = static_data['best_acc1'].mean() if len(static_data) > 0 else 0
            dpf_acc = dpf_data['best_acc1'].mean() if len(dpf_data) > 0 else 0
            diff = dpf_acc - static_acc
            
            print(f"{sparsity:6.0f}%   {static_acc:6.2f}%  {dpf_acc:6.2f}%  {diff:+6.2f}%")
    
    print(f"\n💾 전체 결과 CSV: ./runs/final_report/experiments_comparison.csv")

def main():
    print("🎯 Dense vs Static vs DPF 훈련 결과 정리")
    print("=" * 50)
    
    # 결과 수집
    results = collect_training_results()
    
    # 요약 리포트 생성
    create_summary_report(results)
    
    print("\n🏁 결과 정리 완료!")

if __name__ == '__main__':
    main()