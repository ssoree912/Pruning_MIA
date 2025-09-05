#!/usr/bin/env python3
"""
통합 MIA 평가 스크립트
Advanced MIA (LiRA, Shokri-NN, Top3-NN, ClassLabel-NN, SAMIA)와 
WeMeM (Confidence, Entropy, Modified Entropy, Neural Network) 결과를 통합
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Add current directory to path
sys.path.append('.')
sys.path.append('./mia')

# Import MIA evaluation modules
from mia_advanced import evaluate_advanced_mia
from mia_wemem import evaluate_mia_wemem

def run_unified_mia_evaluation(runs_dir, results_dir):
    """Run both advanced and WeMeM MIA evaluations and combine results"""
    
    os.makedirs(results_dir, exist_ok=True)
    
    print("🚀 통합 MIA 평가 시작")
    print("=" * 50)
    
    # 1. Run Advanced MIA (LiRA, Shokri-NN, etc.)
    print("\n📊 Advanced MIA 평가 (LiRA, Shokri-NN, Top3-NN, ClassLabel-NN, SAMIA)")
    print("-" * 60)
    
    advanced_results_dir = os.path.join(results_dir, 'advanced')
    try:
        advanced_df = evaluate_advanced_mia(runs_dir, advanced_results_dir)
        print("✅ Advanced MIA 평가 완료")
    except Exception as e:
        print(f"❌ Advanced MIA 평가 실패: {e}")
        advanced_df = pd.DataFrame()
    
    # 2. Run WeMeM MIA (Confidence, Entropy, etc.)
    print("\n📊 WeMeM MIA 평가 (Confidence, Entropy, Modified Entropy, Neural Network)")
    print("-" * 60)
    
    wemem_results_dir = os.path.join(results_dir, 'wemem')
    try:
        wemem_df = evaluate_mia_wemem(runs_dir, wemem_results_dir)
        print("✅ WeMeM MIA 평가 완료")
    except Exception as e:
        print(f"❌ WeMeM MIA 평가 실패: {e}")
        wemem_df = pd.DataFrame()
    
    # 3. Combine results
    print("\n🔄 결과 통합 중...")
    
    combined_results = combine_mia_results(advanced_results_dir, wemem_results_dir, results_dir)
    
    # 4. Create final summary
    create_final_mia_summary(combined_results, results_dir)
    
    print(f"\n✅ 통합 MIA 평가 완료!")
    print(f"📁 결과 저장: {results_dir}")
    print(f"📊 통합 요약: {results_dir}/unified_mia_summary.csv")
    
    return combined_results

def combine_mia_results(advanced_dir, wemem_dir, output_dir):
    """Combine Advanced and WeMeM MIA results"""
    
    combined_results = {}
    
    # Load Advanced MIA results
    advanced_results_file = os.path.join(advanced_dir, 'advanced_mia_results.json')
    if os.path.exists(advanced_results_file):
        with open(advanced_results_file) as f:
            advanced_results = json.load(f)
        
        for model_name, model_results in advanced_results.items():
            if model_name not in combined_results:
                combined_results[model_name] = {
                    'model_info': model_results['model_info'],
                    'advanced': {},
                    'wemem': {}
                }
            combined_results[model_name]['advanced'] = model_results.get('mia_results', {})
    
    # Load WeMeM MIA results
    wemem_results_file = os.path.join(wemem_dir, 'wemem_mia_results.json')
    if os.path.exists(wemem_results_file):
        with open(wemem_results_file) as f:
            wemem_results = json.load(f)
        
        for model_name, model_results in wemem_results.items():
            if model_name not in combined_results:
                combined_results[model_name] = {
                    'model_info': model_results['model_info'],
                    'advanced': {},
                    'wemem': {}
                }
            combined_results[model_name]['wemem'] = model_results.get('mia_results', {})
    
    # Save combined results
    combined_file = os.path.join(output_dir, 'combined_mia_results.json')
    with open(combined_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    return combined_results

def create_final_mia_summary(combined_results, output_dir):
    """Create comprehensive MIA summary table"""
    
    if not combined_results:
        print("Warning: No combined results to create summary")
        return pd.DataFrame(), pd.DataFrame()
    
    summary_data = []
    
    for model_name, results in combined_results.items():
        model_info = results['model_info']
        advanced = results.get('advanced', {})
        wemem = results.get('wemem', {})
        
        row = {
            'experiment': model_name,
            'method': model_info.get('method', 'unknown'),
            'sparsity': model_info.get('sparsity', 0.0),
            'type': model_info.get('type', 'unknown')
        }
        
        # Advanced MIA results
        for attack_type in ['lira', 'shokri_nn', 'top3_nn', 'class_label_nn', 'samia']:
            if attack_type in advanced:
                metrics = advanced[attack_type]
                row[f'advanced_{attack_type}_accuracy'] = metrics.get('accuracy', 0.0)
                row[f'advanced_{attack_type}_auc'] = metrics.get('auc', 0.0)
                row[f'advanced_{attack_type}_f1'] = metrics.get('f1', 0.0)
        
        # WeMeM MIA results
        for attack_type in ['confidence', 'entropy', 'modified_entropy', 'neural_network']:
            if attack_type in wemem:
                metrics = wemem[attack_type]
                row[f'wemem_{attack_type}_accuracy'] = metrics.get('accuracy', 0.0)
                if 'auc' in metrics:
                    row[f'wemem_{attack_type}_auc'] = metrics.get('auc', 0.0)
                row[f'wemem_{attack_type}_f1'] = metrics.get('f1', 0.0)
        
        summary_data.append(row)
    
    # Create DataFrame and save
    if not summary_data:
        print("Warning: No summary data to create DataFrame")
        df = pd.DataFrame()
        key_df = pd.DataFrame()
    else:
        df = pd.DataFrame(summary_data)
        
        # Sort by method and sparsity for better readability (if columns exist)
        if 'method' in df.columns and 'sparsity' in df.columns and len(df) > 0:
            df = df.sort_values(['method', 'sparsity'])
        elif len(df) > 0:
            print("Warning: Cannot sort by method/sparsity - columns not found")
            print(f"Available columns: {list(df.columns)}")
    
    summary_file = os.path.join(output_dir, 'unified_mia_summary.csv')
    df.to_csv(summary_file, index=False)
    
    if df.empty:
        print(f"📊 빈 결과: {summary_file}")
        return df, pd.DataFrame()
    
    # Create simplified summary with key metrics
    key_metrics_data = []
    for _, row in df.iterrows():
        key_row = {
            'experiment': row['experiment'],
            'method': row['method'],
            'sparsity': row['sparsity'],
            
            # Best AUC from each category
            'best_advanced_auc': max([
                row.get('advanced_lira_auc', 0),
                row.get('advanced_shokri_nn_auc', 0),
                row.get('advanced_top3_nn_auc', 0),
                row.get('advanced_class_label_nn_auc', 0),
                row.get('advanced_samia_auc', 0)
            ]),
            
            'best_wemem_auc': max([
                row.get('wemem_neural_network_auc', 0),
                # Threshold-based attacks don't have AUC
            ]),
            
            # Best accuracy from each category
            'best_advanced_accuracy': max([
                row.get('advanced_lira_accuracy', 0),
                row.get('advanced_shokri_nn_accuracy', 0),
                row.get('advanced_top3_nn_accuracy', 0),
                row.get('advanced_class_label_nn_accuracy', 0),
                row.get('advanced_samia_accuracy', 0)
            ]),
            
            'best_wemem_accuracy': max([
                row.get('wemem_confidence_accuracy', 0),
                row.get('wemem_entropy_accuracy', 0),
                row.get('wemem_modified_entropy_accuracy', 0),
                row.get('wemem_neural_network_accuracy', 0)
            ])
        }
        key_metrics_data.append(key_row)
    
    key_df = pd.DataFrame(key_metrics_data)
    key_summary_file = os.path.join(output_dir, 'mia_key_metrics.csv')
    key_df.to_csv(key_summary_file, index=False)
    
    print(f"📊 상세 결과: {summary_file}")
    print(f"📊 핵심 지표: {key_summary_file}")
    
    return df, key_df

def main():
    parser = argparse.ArgumentParser(description='Unified MIA Evaluation')
    parser.add_argument('--runs-dir', default='./runs', help='Directory with trained models')
    parser.add_argument('--results-dir', default='./results/unified_mia', help='Output directory')
    
    args = parser.parse_args()
    
    print("🎯 통합 MIA 평가")
    print("=" * 50)
    print("Advanced MIA: LiRA, Shokri-NN, Top3-NN, ClassLabel-NN, SAMIA")
    print("WeMeM MIA: Confidence, Entropy, Modified Entropy, Neural Network")
    print()
    
    combined_results = run_unified_mia_evaluation(args.runs_dir, args.results_dir)
    
    # Display key metrics
    key_file = os.path.join(args.results_dir, 'mia_key_metrics.csv')
    if os.path.exists(key_file):
        key_df = pd.read_csv(key_file)
        print("\n📊 핵심 MIA 공격 성공률:")
        print(key_df.to_string(index=False))

if __name__ == '__main__':
    main()#!/usr/bin/env python3
"""
통합 MIA 평가 스크립트
Advanced MIA (LiRA, Shokri-NN, Top3-NN, ClassLabel-NN, SAMIA)와 
WeMeM (Confidence, Entropy, Modified Entropy, Neural Network) 결과를 통합
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Add current directory to path
sys.path.append('.')
sys.path.append('./mia')

# Import MIA evaluation modules
from mia_advanced import evaluate_advanced_mia
from mia_wemem import evaluate_mia_wemem

def run_unified_mia_evaluation(runs_dir, results_dir):
    """Run both advanced and WeMeM MIA evaluations and combine results"""
    
    os.makedirs(results_dir, exist_ok=True)
    
    print("🚀 통합 MIA 평가 시작")
    print("=" * 50)
    
    # 1. Run Advanced MIA (LiRA, Shokri-NN, etc.)
    print("\n📊 Advanced MIA 평가 (LiRA, Shokri-NN, Top3-NN, ClassLabel-NN, SAMIA)")
    print("-" * 60)
    
    advanced_results_dir = os.path.join(results_dir, 'advanced')
    try:
        advanced_df = evaluate_advanced_mia(runs_dir, advanced_results_dir)
        print("✅ Advanced MIA 평가 완료")
    except Exception as e:
        print(f"❌ Advanced MIA 평가 실패: {e}")
        advanced_df = pd.DataFrame()
    
    # 2. Run WeMeM MIA (Confidence, Entropy, etc.)
    print("\n📊 WeMeM MIA 평가 (Confidence, Entropy, Modified Entropy, Neural Network)")
    print("-" * 60)
    
    wemem_results_dir = os.path.join(results_dir, 'wemem')
    try:
        wemem_df = evaluate_mia_wemem(runs_dir, wemem_results_dir)
        print("✅ WeMeM MIA 평가 완료")
    except Exception as e:
        print(f"❌ WeMeM MIA 평가 실패: {e}")
        wemem_df = pd.DataFrame()
    
    # 3. Combine results
    print("\n🔄 결과 통합 중...")
    
    combined_results = combine_mia_results(advanced_results_dir, wemem_results_dir, results_dir)
    
    # 4. Create final summary
    create_final_mia_summary(combined_results, results_dir)
    
    print(f"\n✅ 통합 MIA 평가 완료!")
    print(f"📁 결과 저장: {results_dir}")
    print(f"📊 통합 요약: {results_dir}/unified_mia_summary.csv")
    
    return combined_results

def combine_mia_results(advanced_dir, wemem_dir, output_dir):
    """Combine Advanced and WeMeM MIA results"""
    
    combined_results = {}
    
    # Load Advanced MIA results
    advanced_results_file = os.path.join(advanced_dir, 'advanced_mia_results.json')
    if os.path.exists(advanced_results_file):
        with open(advanced_results_file) as f:
            advanced_results = json.load(f)
        
        for model_name, model_results in advanced_results.items():
            if model_name not in combined_results:
                combined_results[model_name] = {
                    'model_info': model_results['model_info'],
                    'advanced': {},
                    'wemem': {}
                }
            combined_results[model_name]['advanced'] = model_results.get('mia_results', {})
    
    # Load WeMeM MIA results
    wemem_results_file = os.path.join(wemem_dir, 'wemem_mia_results.json')
    if os.path.exists(wemem_results_file):
        with open(wemem_results_file) as f:
            wemem_results = json.load(f)
        
        for model_name, model_results in wemem_results.items():
            if model_name not in combined_results:
                combined_results[model_name] = {
                    'model_info': model_results['model_info'],
                    'advanced': {},
                    'wemem': {}
                }
            combined_results[model_name]['wemem'] = model_results.get('mia_results', {})
    
    # Save combined results
    combined_file = os.path.join(output_dir, 'combined_mia_results.json')
    with open(combined_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    return combined_results

def create_final_mia_summary(combined_results, output_dir):
    """Create comprehensive MIA summary table"""
    
    if not combined_results:
        print("Warning: No combined results to create summary")
        return pd.DataFrame(), pd.DataFrame()
    
    summary_data = []
    
    for model_name, results in combined_results.items():
        model_info = results['model_info']
        advanced = results.get('advanced', {})
        wemem = results.get('wemem', {})
        
        row = {
            'experiment': model_name,
            'method': model_info.get('method', 'unknown'),
            'sparsity': model_info.get('sparsity', 0.0),
            'type': model_info.get('type', 'unknown')
        }
        
        # Advanced MIA results
        for attack_type in ['lira', 'shokri_nn', 'top3_nn', 'class_label_nn', 'samia']:
            if attack_type in advanced:
                metrics = advanced[attack_type]
                row[f'advanced_{attack_type}_accuracy'] = metrics.get('accuracy', 0.0)
                row[f'advanced_{attack_type}_auc'] = metrics.get('auc', 0.0)
                row[f'advanced_{attack_type}_f1'] = metrics.get('f1', 0.0)
        
        # WeMeM MIA results
        for attack_type in ['confidence', 'entropy', 'modified_entropy', 'neural_network']:
            if attack_type in wemem:
                metrics = wemem[attack_type]
                row[f'wemem_{attack_type}_accuracy'] = metrics.get('accuracy', 0.0)
                if 'auc' in metrics:
                    row[f'wemem_{attack_type}_auc'] = metrics.get('auc', 0.0)
                row[f'wemem_{attack_type}_f1'] = metrics.get('f1', 0.0)
        
        summary_data.append(row)
    
    # Create DataFrame and save
    if not summary_data:
        print("Warning: No summary data to create DataFrame")
        df = pd.DataFrame()
        key_df = pd.DataFrame()
    else:
        df = pd.DataFrame(summary_data)
        
        # Sort by method and sparsity for better readability (if columns exist)
        if 'method' in df.columns and 'sparsity' in df.columns and len(df) > 0:
            df = df.sort_values(['method', 'sparsity'])
        elif len(df) > 0:
            print("Warning: Cannot sort by method/sparsity - columns not found")
            print(f"Available columns: {list(df.columns)}")
    
    summary_file = os.path.join(output_dir, 'unified_mia_summary.csv')
    df.to_csv(summary_file, index=False)
    
    if df.empty:
        print(f"📊 빈 결과: {summary_file}")
        return df, pd.DataFrame()
    
    # Create simplified summary with key metrics
    key_metrics_data = []
    for _, row in df.iterrows():
        key_row = {
            'experiment': row['experiment'],
            'method': row['method'],
            'sparsity': row['sparsity'],
            
            # Best AUC from each category
            'best_advanced_auc': max([
                row.get('advanced_lira_auc', 0),
                row.get('advanced_shokri_nn_auc', 0),
                row.get('advanced_top3_nn_auc', 0),
                row.get('advanced_class_label_nn_auc', 0),
                row.get('advanced_samia_auc', 0)
            ]),
            
            'best_wemem_auc': max([
                row.get('wemem_neural_network_auc', 0),
                # Threshold-based attacks don't have AUC
            ]),
            
            # Best accuracy from each category
            'best_advanced_accuracy': max([
                row.get('advanced_lira_accuracy', 0),
                row.get('advanced_shokri_nn_accuracy', 0),
                row.get('advanced_top3_nn_accuracy', 0),
                row.get('advanced_class_label_nn_accuracy', 0),
                row.get('advanced_samia_accuracy', 0)
            ]),
            
            'best_wemem_accuracy': max([
                row.get('wemem_confidence_accuracy', 0),
                row.get('wemem_entropy_accuracy', 0),
                row.get('wemem_modified_entropy_accuracy', 0),
                row.get('wemem_neural_network_accuracy', 0)
            ])
        }
        key_metrics_data.append(key_row)
    
    key_df = pd.DataFrame(key_metrics_data)
    key_summary_file = os.path.join(output_dir, 'mia_key_metrics.csv')
    key_df.to_csv(key_summary_file, index=False)
    
    print(f"📊 상세 결과: {summary_file}")
    print(f"📊 핵심 지표: {key_summary_file}")
    
    return df, key_df

def main():
    parser = argparse.ArgumentParser(description='Unified MIA Evaluation')
    parser.add_argument('--runs-dir', default='./runs', help='Directory with trained models')
    parser.add_argument('--results-dir', default='./results/unified_mia', help='Output directory')
    
    args = parser.parse_args()
    
    print("🎯 통합 MIA 평가")
    print("=" * 50)
    print("Advanced MIA: LiRA, Shokri-NN, Top3-NN, ClassLabel-NN, SAMIA")
    print("WeMeM MIA: Confidence, Entropy, Modified Entropy, Neural Network")
    print()
    
    combined_results = run_unified_mia_evaluation(args.runs_dir, args.results_dir)
    
    # Display key metrics
    key_file = os.path.join(args.results_dir, 'mia_key_metrics.csv')
    if os.path.exists(key_file):
        key_df = pd.read_csv(key_file)
        print("\n📊 핵심 MIA 공격 성공률:")
        print(key_df.to_string(index=False))

if __name__ == '__main__':
    main()