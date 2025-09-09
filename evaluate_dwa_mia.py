#!/usr/bin/env python3
"""
DWA 프루닝된 모델에 대한 MIA 평가 스크립트 (WeMeM-main 스타일)
train_dwa.py로 생성된 결과물을 기반으로 MIA 공격 수행

사용법:
python evaluate_dwa_mia.py device config_path --dataset_name cifar10 --model_name resnet18 --attacks samia,threshold,nn
"""

import os
import json
import argparse
import pickle
import random
import csv
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset, ConcatDataset

from mia_utils import (
    load_dwa_model, find_dwa_models, get_dataset_loaders,
    create_mia_data_splits, extract_model_features, get_model_sparsity, get_dataset, get_model, get_optimizer, weight_init
)
from attacker_threshold import ThresholdAttacker

def evaluate_single_model(model_info, mia_splits, device='cuda:0', batch_size=128):
    """단일 DWA 모델에 대한 MIA 평가"""
    print(f"📊 Evaluating: {model_info['dwa_mode']}, {model_info['sparsity_dir']}, {model_info['dataset']}")
    
    try:
        # 모델 로드
        model, config = load_dwa_model(
            model_info['model_path'], 
            model_info['config_path'], 
            device=device
        )
        
        # 실제 sparsity 계산
        actual_sparsity = get_model_sparsity(model)
        
        # 데이터셋 정보 가져오기
        dataset_name = config['data']['dataset']
        
        # MIA 데이터 준비 (victim vs shadow + non-member)
        dataset = mia_splits['dataset']
        victim_indices = mia_splits['victim_indices']
        shadow_indices = mia_splits['shadow_indices']
        non_member_indices = mia_splits['non_member_indices']
        
        # Shadow를 절반씩 나누어 shadow member/non-member로 사용
        half_shadow = len(shadow_indices) // 2
        shadow_member_indices = shadow_indices[:half_shadow]
        shadow_nonmember_indices = shadow_indices[half_shadow:]
        
        # 데이터 로더 생성
        victim_member_dataset = Subset(dataset, victim_indices[:len(victim_indices)//2])
        victim_nonmember_dataset = Subset(dataset, non_member_indices[:len(victim_indices)//2])
        
        shadow_member_dataset = Subset(dataset, shadow_member_indices)
        shadow_nonmember_dataset = Subset(dataset, shadow_nonmember_indices)
        
        victim_member_loader = DataLoader(victim_member_dataset, batch_size=batch_size, shuffle=False)
        victim_nonmember_loader = DataLoader(victim_nonmember_dataset, batch_size=batch_size, shuffle=False)
        shadow_member_loader = DataLoader(shadow_member_dataset, batch_size=batch_size, shuffle=False)
        shadow_nonmember_loader = DataLoader(shadow_nonmember_dataset, batch_size=batch_size, shuffle=False)
        
        # 피처 추출 (static 모드로 평가)
        forward_mode = 'static' if config.get('pruning', {}).get('enabled', False) else None
        
        shadow_member_features = extract_model_features(model, shadow_member_loader, device, forward_mode)
        shadow_nonmember_features = extract_model_features(model, shadow_nonmember_loader, device, forward_mode)
        victim_member_features = extract_model_features(model, victim_member_loader, device, forward_mode)
        victim_nonmember_features = extract_model_features(model, victim_nonmember_loader, device, forward_mode)
        
        # Threshold 기반 공격 수행
        num_classes = config['data'].get('num_classes', 10 if dataset_name == 'cifar10' else 100)
        
        attacker = ThresholdAttacker(
            in_pair=(shadow_member_features['softmax'], shadow_member_features['targets']),
            out_pair=(shadow_nonmember_features['softmax'], shadow_nonmember_features['targets']),
            v_in_pair=(victim_member_features['softmax'], victim_member_features['targets']),
            v_out_pair=(victim_nonmember_features['softmax'], victim_nonmember_features['targets']),
            num_classes=num_classes
        )
        
        # 공격 실행
        conf_acc, entr_acc, mentr_acc = attacker._mem_inf_benchmarks()
        hconf_acc, _, _ = attacker._mem_inf_benchmarks_non_cls()
        
        # 결과 정리
        result = {
            'experiment_name': config.get('name', 'unknown'),
            'dwa_mode': model_info['dwa_mode'],
            'sparsity_config': config.get('pruning', {}).get('sparsity', 0.0),
            'sparsity_actual': float(actual_sparsity),
            'dataset': dataset_name,
            'architecture': config['model']['arch'],
            'layers': config['model']['layers'],
            'dwa_alpha': config.get('pruning', {}).get('dwa_alpha', 1.0),
            'dwa_beta': config.get('pruning', {}).get('dwa_beta', 1.0),
            'target_epoch': config.get('pruning', {}).get('target_epoch', -1),
            'prune_freq': config.get('pruning', {}).get('prune_freq', -1),
            'attack_conf_gt': float(conf_acc),
            'attack_entropy': float(entr_acc),
            'attack_modified_entropy': float(mentr_acc),
            'attack_conf_top1': float(hconf_acc),
            'victim_member_confidence_mean': float(np.mean(victim_member_features['confidence'])),
            'victim_nonmember_confidence_mean': float(np.mean(victim_nonmember_features['confidence'])),
            'confidence_gap': float(np.mean(victim_member_features['confidence']) - np.mean(victim_nonmember_features['confidence'])),
            'model_path': str(model_info['model_path']),
            'config_path': str(model_info['config_path'])
        }
        
        # 실험 요약 정보도 포함 (있다면)
        experiment_summary_path = model_info['experiment_dir'] / 'experiment_summary.json'
        if experiment_summary_path.exists():
            with open(experiment_summary_path) as f:
                exp_summary = json.load(f)
            result['best_acc1'] = exp_summary.get('best_metrics', {}).get('best_acc1', 0.0)
            result['final_acc1'] = exp_summary.get('final_metrics', {}).get('acc1', 0.0)
            result['total_epochs'] = exp_summary.get('total_epochs', 0)
            result['training_time_hours'] = exp_summary.get('total_duration', 0) / 3600
        
        print(f"✅ Success! Best attack accuracy: {max(conf_acc, entr_acc, mentr_acc, hconf_acc):.3f}")
        return result
        
    except Exception as e:
        print(f"❌ Failed to evaluate {model_info['model_path']}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='DWA MIA Evaluation')
    parser.add_argument('--runs_dir', type=str, default='./runs', help='DWA training results directory')
    parser.add_argument('--output_dir', type=str, default='./mia_results', help='MIA results output directory')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Dataset used for training')
    parser.add_argument('--datapath', type=str, default='~/Datasets', help='Dataset path')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🔍 DWA 모델에 대한 MIA 평가 시작")
    print("=" * 50)
    
    # DWA 훈련 결과 모델들 찾기
    dwa_models = find_dwa_models(args.runs_dir)
    
    if not dwa_models:
        print(f"❌ No DWA models found in {args.runs_dir}")
        print("먼저 train_dwa.py로 모델을 훈련해주세요.")
        return
    
    print(f"📋 Found {len(dwa_models)} DWA models")
    
    # MIA 데이터 분할 준비 (한 번만 생성)
    print("🔄 Preparing MIA data splits...")
    mia_splits = create_mia_data_splits(args.dataset)
    
    # 모든 모델에 대해 MIA 평가 수행
    all_results = []
    failed_count = 0
    
    for i, model_info in enumerate(dwa_models, 1):
        print(f"\n📊 [{i}/{len(dwa_models)}] Evaluating model...")
        
        result = evaluate_single_model(
            model_info, mia_splits, 
            device=args.device, batch_size=args.batch_size
        )
        
        if result:
            all_results.append(result)
            
            # 개별 결과 저장
            individual_result_path = model_info['experiment_dir'] / 'mia_evaluation.json'
            with open(individual_result_path, 'w') as f:
                json.dump(result, f, indent=2)
        else:
            failed_count += 1
    
    # 전체 결과 저장
    if all_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON 저장
        json_path = output_dir / f'dwa_mia_results_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # CSV 저장
        csv_path = output_dir / f'dwa_mia_results_{timestamp}.csv'
        fieldnames = list(all_results[0].keys())
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\n💾 Results saved:")
        print(f"   JSON: {json_path}")
        print(f"   CSV:  {csv_path}")
        
        # 요약 출력
        print(f"\n📈 MIA Attack Success Summary:")
        print(f"{'DWA Mode':<20} {'Sparsity':<10} {'Best Attack':<12} {'Conf Gap':<12}")
        print("-" * 60)
        
        # 정렬해서 출력
        sorted_results = sorted(all_results, key=lambda x: (x['dwa_mode'], x['sparsity_actual']))
        
        for result in sorted_results:
            best_attack = max(
                result['attack_conf_gt'], 
                result['attack_entropy'], 
                result['attack_modified_entropy'],
                result['attack_conf_top1']
            )
            print(f"{result['dwa_mode']:<20} {result['sparsity_actual']:<10.3f} {best_attack:<12.3f} {result['confidence_gap']:<12.3f}")
        
        print(f"\n✅ Successfully evaluated {len(all_results)} models")
        if failed_count > 0:
            print(f"❌ Failed to evaluate {failed_count} models")
    
    else:
        print("❌ No successful evaluations")

if __name__ == '__main__':
    main()