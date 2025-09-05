#!/usr/bin/env python3
"""
테스트용 간단한 MIA 평가 스크립트
훈련 결과가 MIA 평가에 잘 연결되는지 확인
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def extract_model_info_from_runs(runs_dir):
    """runs 디렉토리에서 모델 정보 추출"""
    models_info = []
    runs_path = Path(runs_dir)
    
    print(f"🔍 Scanning: {runs_dir}")
    
    for method_dir in runs_path.iterdir():
        if not method_dir.is_dir():
            continue
            
        method_name = method_dir.name
        print(f"  Found method: {method_name}")
        
        if method_name == 'dense':
            # Dense: runs/dense/dataset/
            for dataset_dir in method_dir.iterdir():
                if dataset_dir.is_dir():
                    model_info = extract_model_details(dataset_dir, method_name, 0.0)
                    if model_info:
                        models_info.append(model_info)
        
        elif method_name in ['static', 'dpf']:
            # Static/DPF: runs/method/sparsity_X/dataset/
            for sparsity_dir in method_dir.iterdir():
                if sparsity_dir.is_dir() and sparsity_dir.name.startswith('sparsity_'):
                    try:
                        sparsity_str = sparsity_dir.name.replace('sparsity_', '')
                        sparsity = float(sparsity_str)
                    except ValueError:
                        print(f"    Warning: Cannot parse sparsity from {sparsity_dir.name}")
                        continue
                    
                    for dataset_dir in sparsity_dir.iterdir():
                        if dataset_dir.is_dir():
                            model_info = extract_model_details(dataset_dir, method_name, sparsity)
                            if model_info:
                                models_info.append(model_info)
    
    print(f"✅ Found {len(models_info)} models")
    return models_info

def extract_model_details(model_dir, method, sparsity):
    """개별 모델 디렉토리에서 세부 정보 추출"""
    
    # 필수 파일들 확인
    best_model_path = model_dir / 'best_model.pth'
    config_path = model_dir / 'config.json'
    experiment_summary_path = model_dir / 'experiment_summary.json'
    
    if not best_model_path.exists():
        print(f"    ❌ No best_model.pth in {model_dir}")
        return None
    
    model_info = {
        'name': model_dir.parent.name + '_' + model_dir.name,
        'method': method,
        'sparsity': sparsity,
        'path': str(model_dir),
        'files_found': {
            'best_model': best_model_path.exists(),
            'config': config_path.exists(), 
            'summary': experiment_summary_path.exists()
        }
    }
    
    # config.json에서 설정 정보 추출
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            
            model_info.update({
                'dataset': config.get('data', {}).get('dataset', 'unknown'),
                'arch': config.get('model', {}).get('arch', 'unknown'),
                'epochs': config.get('training', {}).get('epochs', 0),
                'actual_method': config.get('pruning', {}).get('method', 'dense'),
                'actual_sparsity': config.get('pruning', {}).get('sparsity', 0.0),
                'pruning_enabled': config.get('pruning', {}).get('enabled', False)
            })
        except Exception as e:
            print(f"    Warning: Error reading config.json: {e}")
    
    # experiment_summary.json에서 성능 정보 추출
    if experiment_summary_path.exists():
        try:
            with open(experiment_summary_path) as f:
                summary = json.load(f)
            
            model_info.update({
                'best_acc1': summary.get('best_metrics', {}).get('best_acc1', 0),
                'final_acc1': summary.get('final_metrics', {}).get('acc1', 0),
                'training_time_hours': summary.get('total_duration', 0) / 3600
            })
        except Exception as e:
            print(f"    Warning: Error reading experiment_summary.json: {e}")
    
    print(f"    ✅ {model_info['name']}: {method} (sparsity={sparsity})")
    return model_info

def simulate_simple_mia(models_info):
    """간단한 MIA 시뮬레이션 (테스트용)"""
    
    mia_results = []
    
    for model_info in models_info:
        # 간단한 confidence-based attack 시뮬레이션
        # 실제로는 모델을 로드해서 prediction을 뽑아야 하지만, 
        # 테스트용으로 accuracy 기반 가상 MIA 결과 생성
        
        accuracy = model_info.get('best_acc1', 70) / 100.0
        
        # 높은 정확도일수록 MIA에 더 취약하다고 가정
        base_vulnerability = min(0.8, 0.5 + accuracy * 0.3)
        
        # 방법별로 취약성 조정
        if model_info['method'] == 'dense':
            vulnerability_factor = 1.0
        elif model_info['method'] == 'static':
            # Static pruning은 일반적으로 덜 취약
            vulnerability_factor = 0.9 - model_info['sparsity'] * 0.1
        else:  # dpf
            # Dynamic pruning은 중간 수준
            vulnerability_factor = 0.95 - model_info['sparsity'] * 0.05
        
        mia_accuracy = base_vulnerability * vulnerability_factor
        mia_auc = mia_accuracy + np.random.normal(0, 0.05)  # 약간의 노이즈
        mia_auc = np.clip(mia_auc, 0.5, 1.0)
        
        mia_result = {
            'experiment': model_info['name'],
            'method': model_info['method'],
            'sparsity': model_info['sparsity'],
            'dataset': model_info.get('dataset', 'cifar10'),
            'best_acc1': model_info.get('best_acc1', 0),
            
            # 간단한 MIA 결과 (시뮬레이션)
            'mia_confidence_attack_accuracy': mia_accuracy,
            'mia_confidence_attack_auc': mia_auc,
            'mia_vulnerability_score': mia_accuracy,
            
            # 추가 정보
            'files_available': all(model_info['files_found'].values()),
            'path': model_info['path']
        }
        
        mia_results.append(mia_result)
        
        print(f"  📊 {model_info['name']}: MIA Attack Acc={mia_accuracy:.3f}, AUC={mia_auc:.3f}")
    
    return mia_results

def create_test_summary(mia_results, output_dir):
    """테스트 결과 요약 생성"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # DataFrame 생성
    df = pd.DataFrame(mia_results)
    
    # 정렬 (method와 sparsity가 있는지 확인)
    if 'method' in df.columns and 'sparsity' in df.columns:
        df = df.sort_values(['method', 'sparsity'])
    
    # CSV 저장
    summary_file = os.path.join(output_dir, 'test_mia_results.csv')
    df.to_csv(summary_file, index=False)
    
    # 요약 통계
    summary_stats = {
        'total_models': len(mia_results),
        'methods': df['method'].value_counts().to_dict() if 'method' in df.columns else {},
        'average_vulnerability': df['mia_vulnerability_score'].mean() if 'mia_vulnerability_score' in df.columns else 0,
        'files_available_count': df['files_available'].sum() if 'files_available' in df.columns else 0
    }
    
    stats_file = os.path.join(output_dir, 'test_summary_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"\n📊 테스트 결과:")
    print(f"  - 총 모델 수: {summary_stats['total_models']}")
    print(f"  - 방법별 분포: {summary_stats['methods']}")
    print(f"  - 평균 MIA 취약성: {summary_stats['average_vulnerability']:.3f}")
    print(f"  - 파일 완전한 모델: {summary_stats['files_available_count']}/{summary_stats['total_models']}")
    
    print(f"\n📁 결과 저장:")
    print(f"  - 상세 결과: {summary_file}")
    print(f"  - 요약 통계: {stats_file}")
    
    return df, summary_stats

def main():
    parser = argparse.ArgumentParser(description='Test MIA Evaluation')
    parser.add_argument('--runs-dir', default='./runs', help='Directory with trained models')
    parser.add_argument('--output-dir', default='./test_results', help='Output directory')
    
    args = parser.parse_args()
    
    print("🧪 MIA 평가 테스트")
    print("=" * 50)
    
    # 1. 모델 정보 추출
    models_info = extract_model_info_from_runs(args.runs_dir)
    
    if not models_info:
        print("❌ 훈련된 모델을 찾을 수 없습니다.")
        return
    
    # 2. 간단한 MIA 시뮬레이션
    print(f"\n🎯 MIA 시뮬레이션 ({len(models_info)}개 모델)")
    mia_results = simulate_simple_mia(models_info)
    
    # 3. 결과 요약
    print(f"\n📋 결과 요약 생성")
    df, stats = create_test_summary(mia_results, args.output_dir)
    
    print(f"\n✅ 테스트 완료!")
    
    # 4. 데이터프레임 미리보기
    if not df.empty:
        print(f"\n📋 결과 미리보기:")
        print(df.head().to_string())

if __name__ == '__main__':
    main()