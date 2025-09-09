#!/usr/bin/env python3
"""
DWA 프루닝된 모델에 대한 MIA 평가 스크립트 (WeMeM-main 스타일)
train_dwa.py로 생성된 결과물을 기반으로 MIA 공격 수행

사용법:
python evaluate_dwa_mia_wemem.py device config_path --dataset_name cifar10 --model_name resnet18 --attacks samia,threshold,nn
"""

import argparse
import json
import pickle
import random
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from datetime import datetime
from torch.utils.data import ConcatDataset, DataLoader, Subset

# WeMeM-main style imports
from attackers import MiaAttack
from base_model import BaseModel  
from datasets import get_dataset
from mia_utils import find_dwa_models, get_model_sparsity

# Command line arguments (WeMeM-main style)
parser = argparse.ArgumentParser(description='MIA Evaluation for DWA Pruned Models')
parser.add_argument('device', default=0, type=int, help="GPU id to use")
parser.add_argument('config_path', default=0, type=str, help="config file path")
parser.add_argument('--dataset_name', default='cifar10', type=str)
parser.add_argument('--model_name', default='resnet18', type=str)
parser.add_argument('--num_cls', default=10, type=int)
parser.add_argument('--input_dim', default=3, type=int)
parser.add_argument('--image_size', default=32, type=int)
parser.add_argument('--hidden_size', default=128, type=int)
parser.add_argument('--seed', default=7, type=int)
parser.add_argument('--early_stop', default=5, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--shadow_num', default=2, type=int, help="Number of shadow models to use for each DWA model")
parser.add_argument('--attacks', default="samia,threshold,nn", type=str)
parser.add_argument('--original', action='store_true', help="Attack original model instead of pruned")
parser.add_argument('--runs_dir', default='./runs', type=str, help="DWA training results directory")
parser.add_argument('--output_dir', default='./mia_results_dwa', type=str, help="MIA results output directory")

def prepare_mia_data_for_dwa_model(model_info, args, device):
    """DWA 모델용 MIA 데이터 준비 (data_prepare.pkl 생성)"""
    
    # 실험 디렉토리에서 data_prepare.pkl 파일 찾기
    experiment_dir = Path(model_info['experiment_dir'])
    data_prepare_path = experiment_dir / 'data_prepare.pkl'
    
    if data_prepare_path.exists():
        print(f"Found existing data_prepare.pkl: {data_prepare_path}")
        return data_prepare_path
    
    # data_prepare.pkl이 없으면 생성
    print(f"Creating data_prepare.pkl for {model_info['experiment_dir']}")
    
    # Load datasets
    trainset = get_dataset(args.dataset_name, train=True)
    testset = get_dataset(args.dataset_name, train=False)
    if testset is None:
        total_dataset = trainset
    else:
        total_dataset = ConcatDataset([trainset, testset])
    
    total_size = len(total_dataset)
    
    # Create data splits for MIA (similar to WeMeM-main style)
    np.random.seed(args.seed)
    indices = np.random.permutation(total_size)
    
    # Split: 40% victim, 40% shadow pool, 20% non-member
    victim_size = int(total_size * 0.4)
    shadow_pool_size = int(total_size * 0.4) 
    
    victim_indices = indices[:victim_size]
    shadow_pool_indices = indices[victim_size:victim_size + shadow_pool_size]
    non_member_indices = indices[victim_size + shadow_pool_size:]
    
    # Victim data split (train/dev/test = 7:1:2)
    victim_train_size = int(len(victim_indices) * 0.7)
    victim_dev_size = int(len(victim_indices) * 0.1)
    
    victim_train_list = victim_indices[:victim_train_size]
    victim_dev_list = victim_indices[victim_train_size:victim_train_size + victim_dev_size]
    victim_test_list = victim_indices[victim_train_size + victim_dev_size:]
    
    victim_train_dataset = Subset(total_dataset, victim_train_list)
    victim_dev_dataset = Subset(total_dataset, victim_dev_list)  
    victim_test_dataset = Subset(total_dataset, victim_test_list)
    
    # Shadow models data split
    shadow_per_model = len(shadow_pool_indices) // max(1, args.shadow_num)
    attack_split_list = []
    shadow_train_list = []
    
    for i in range(args.shadow_num):
        start_idx = i * shadow_per_model
        end_idx = min((i + 1) * shadow_per_model, len(shadow_pool_indices))
        shadow_indices = shadow_pool_indices[start_idx:end_idx]
        
        # Shadow data split (train/dev/test = 7:1:2)  
        shadow_train_size = int(len(shadow_indices) * 0.7)
        shadow_dev_size = int(len(shadow_indices) * 0.1)
        
        shadow_train_idx = shadow_indices[:shadow_train_size]
        shadow_dev_idx = shadow_indices[shadow_train_size:shadow_train_size + shadow_dev_size]
        shadow_test_idx = shadow_indices[shadow_train_size + shadow_dev_size:]
        
        shadow_train_dataset = Subset(total_dataset, shadow_train_idx)
        shadow_dev_dataset = Subset(total_dataset, shadow_dev_idx)
        shadow_test_dataset = Subset(total_dataset, shadow_test_idx)
        
        attack_split_list.append((shadow_train_dataset, shadow_dev_dataset, shadow_test_dataset))
        shadow_train_list.append(shadow_train_idx)
    
    # Save data_prepare.pkl
    with open(data_prepare_path, 'wb') as f:
        pickle.dump((victim_train_list, victim_train_dataset, victim_dev_dataset, victim_test_dataset,
                    attack_split_list, shadow_train_list), f)
    
    print(f"Created data_prepare.pkl: {data_prepare_path}")
    print(f"Victim train: {len(victim_train_dataset)}, dev: {len(victim_dev_dataset)}, test: {len(victim_test_dataset)}")
    print(f"Shadow models: {len(attack_split_list)}")
    
    return data_prepare_path

def evaluate_dwa_model_mia(model_info, args, device):
    """단일 DWA 모델에 대한 MIA 평가 (WeMeM-main 스타일)"""
    
    experiment_dir = Path(model_info['experiment_dir'])
    dwa_mode = model_info['dwa_mode']
    sparsity = model_info.get('sparsity', 0.0)
    
    print(f"\n{'='*60}")
    print(f"🎯 MIA Evaluation: {dwa_mode}, Sparsity: {sparsity}")
    print(f"📁 Directory: {experiment_dir}")
    print(f"{'='*60}")
    
    # Prepare MIA data
    data_prepare_path = prepare_mia_data_for_dwa_model(model_info, args, device)
    
    # Load data splits
    with open(data_prepare_path, 'rb') as f:
        victim_train_list, victim_train_dataset, victim_dev_dataset, victim_test_dataset, attack_split_list, shadow_train_list = pickle.load(f)
    
    print(f"Total Data Size: victim_train={len(victim_train_dataset)}, victim_test={len(victim_test_dataset)}")
    
    # Create data loaders
    victim_train_loader = DataLoader(victim_train_dataset, batch_size=args.batch_size, shuffle=False, 
                                   num_workers=4, pin_memory=False)
    victim_test_loader = DataLoader(victim_test_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=4, pin_memory=False)
    
    # Load victim models (assume we have both original and pruned versions)
    # For DWA, we use the best_model.pth as the "pruned" model
    victim_model_path = experiment_dir / 'best_model.pth'
    
    if not victim_model_path.exists():
        print(f"❌ Victim model not found: {victim_model_path}")
        return None
    
    # Load victim model (use DWA trained model as both original and pruned)
    victim_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
    victim_pruned_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
    
    # Load state dict
    state_dict = torch.load(victim_model_path, map_location=device)
    victim_model.model.load_state_dict(state_dict, strict=False)
    victim_pruned_model.model.load_state_dict(state_dict, strict=False)
    
    # Test victim model performance
    test_acc, test_loss = victim_pruned_model.test(victim_test_loader, "Victim DWA Model Test")
    actual_sparsity = get_model_sparsity(victim_pruned_model.model)
    
    print(f"💻 Victim Model - Test Accuracy: {test_acc:.3f}%, Actual Sparsity: {actual_sparsity:.3f}")
    
    # Load shadow models (use other DWA models or create synthetic ones)
    shadow_model_list, shadow_prune_model_list = [], []
    shadow_train_loader_list, shadow_test_loader_list = [], []
    
    # Find other DWA models to use as shadow models
    all_dwa_models = find_dwa_models(args.runs_dir)
    other_models = [m for m in all_dwa_models if m['experiment_dir'] != model_info['experiment_dir']]
    
    if len(other_models) < args.shadow_num:
        print(f"⚠️ Only {len(other_models)} other models available, using synthetic shadow models")
        # Create synthetic shadow models with slightly different performance
        for i in range(args.shadow_num):
            shadow_train_dataset, shadow_dev_dataset, shadow_test_dataset = attack_split_list[i] if i < len(attack_split_list) else attack_split_list[0]
            
            shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=args.batch_size, shuffle=False,
                                           num_workers=4, pin_memory=False)
            shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=args.batch_size, shuffle=False,
                                          num_workers=4, pin_memory=False)
            
            # Use same model architecture but different initialization
            shadow_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
            shadow_pruned_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
            
            # Use victim model weights with some noise for shadow models
            shadow_state = victim_model.model.state_dict()
            for key in shadow_state:
                shadow_state[key] = shadow_state[key] + torch.randn_like(shadow_state[key]) * 0.01
            
            shadow_model.model.load_state_dict(shadow_state)
            shadow_pruned_model.model.load_state_dict(shadow_state)
            
            shadow_model_list.append(shadow_model)
            shadow_prune_model_list.append(shadow_pruned_model)
            shadow_train_loader_list.append(shadow_train_loader)
            shadow_test_loader_list.append(shadow_test_loader)
    else:
        # Use other DWA models as shadow models
        for i, other_model in enumerate(other_models[:args.shadow_num]):
            shadow_model_path = Path(other_model['experiment_dir']) / 'best_model.pth'
            
            if shadow_model_path.exists():
                shadow_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
                shadow_pruned_model = BaseModel(args.model_name, num_cls=args.num_cls, input_dim=args.input_dim, device=device)
                
                shadow_state = torch.load(shadow_model_path, map_location=device)
                shadow_model.model.load_state_dict(shadow_state, strict=False)
                shadow_pruned_model.model.load_state_dict(shadow_state, strict=False)
                
                # Use corresponding data splits
                if i < len(attack_split_list):
                    shadow_train_dataset, shadow_dev_dataset, shadow_test_dataset = attack_split_list[i]
                    
                    shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=4, pin_memory=False)
                    shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=4, pin_memory=False)
                    
                    shadow_model_list.append(shadow_model)
                    shadow_prune_model_list.append(shadow_pruned_model)
                    shadow_train_loader_list.append(shadow_train_loader)
                    shadow_test_loader_list.append(shadow_test_loader)
    
    if len(shadow_model_list) == 0:
        print("❌ No shadow models available")
        return None
    
    print(f"🔍 Using {len(shadow_model_list)} shadow models for MIA")
    
    # Initialize MIA attacker (WeMeM-main style)
    attacker = MiaAttack(
        victim_model, victim_pruned_model, victim_train_loader, victim_test_loader,
        shadow_model_list, shadow_prune_model_list, shadow_train_loader_list, shadow_test_loader_list,
        num_cls=args.num_cls, device=device, batch_size=args.batch_size,
        attack_original=args.original
    )
    
    # Run attacks
    attacks = args.attacks.split(',')
    results = {
        'experiment_name': f"{dwa_mode}_sparsity_{sparsity}",
        'dwa_mode': dwa_mode,
        'sparsity': sparsity,
        'actual_sparsity': actual_sparsity,
        'test_accuracy': test_acc,
        'model_path': str(victim_model_path),
        'experiment_dir': str(experiment_dir)
    }
    
    print("🚀 Starting MIA Attacks...")
    
    if "samia" in attacks:
        print("  Running SAMIA attack...")
        samia_acc = attacker.nn_attack("nn_sens_cls", model_name="transformer")
        results['samia_accuracy'] = samia_acc
        print(f"  ✅ SAMIA attack accuracy: {samia_acc:.3f}%")
    
    if "threshold" in attacks:
        print("  Running Threshold attacks...")
        conf_acc, entr_acc, mentr_acc, hconf_acc = attacker.threshold_attack()
        results.update({
            'confidence_accuracy': conf_acc,
            'entropy_accuracy': entr_acc,
            'modified_entropy_accuracy': mentr_acc,
            'top1_confidence_accuracy': hconf_acc
        })
        print(f"  ✅ Confidence: {conf_acc:.3f}%, Entropy: {entr_acc:.3f}%, Modified Entropy: {mentr_acc:.3f}%, Top1-Conf: {hconf_acc:.3f}%")
    
    if "nn" in attacks:
        print("  Running Neural Network attack...")
        nn_acc = attacker.nn_attack("nn")
        results['nn_accuracy'] = nn_acc
        print(f"  ✅ Neural Network attack accuracy: {nn_acc:.3f}%")
    
    if "nn_top3" in attacks:
        print("  Running Top3-NN attack...")
        nn_top3_acc = attacker.nn_attack("nn_top3")
        results['nn_top3_accuracy'] = nn_top3_acc
        print(f"  ✅ Top3-NN attack accuracy: {nn_top3_acc:.3f}%")
    
    if "nn_cls" in attacks:
        print("  Running NN-Cls attack...")
        nn_cls_acc = attacker.nn_attack("nn_cls")
        results['nn_cls_accuracy'] = nn_cls_acc
        print(f"  ✅ NN-Cls attack accuracy: {nn_cls_acc:.3f}%")
    
    return results

def main(args):
    # Setup (WeMeM-main style)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = f"cuda:{args.device}"
    cudnn.benchmark = True
    
    print("🎯 DWA MIA Evaluation (WeMeM-main Style)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset_name}, Model: {args.model_name}")
    print(f"Attacks: {args.attacks}")
    print(f"DWA Results Directory: {args.runs_dir}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find all DWA models
    dwa_models = find_dwa_models(args.runs_dir)
    
    if not dwa_models:
        print(f"❌ No DWA models found in {args.runs_dir}")
        print("Please run train_dwa.py first to generate DWA models")
        return
    
    print(f"📋 Found {len(dwa_models)} DWA models to evaluate")
    
    # Evaluate each DWA model
    all_results = []
    failed_count = 0
    
    for i, model_info in enumerate(dwa_models, 1):
        print(f"\n🔬 [{i}/{len(dwa_models)}] Evaluating DWA Model...")
        
        try:
            result = evaluate_dwa_model_mia(model_info, args, device)
            
            if result:
                all_results.append(result)
                
                # Save individual result
                individual_result_path = Path(model_info['experiment_dir']) / 'mia_evaluation_wemem.json'
                with open(individual_result_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"💾 Individual result saved: {individual_result_path}")
            else:
                failed_count += 1
                
        except Exception as e:
            print(f"❌ Failed to evaluate {model_info['experiment_dir']}: {e}")
            failed_count += 1
    
    # Save comprehensive results
    if all_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON results
        json_path = output_dir / f'dwa_mia_results_wemem_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # CSV summary
        csv_path = output_dir / f'dwa_mia_summary_wemem_{timestamp}.csv'
        fieldnames = list(all_results[0].keys()) if all_results else []
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if fieldnames:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_results)
        
        print(f"\n📊 Results Summary:")
        print(f"✅ Successfully evaluated: {len(all_results)} models")
        if failed_count > 0:
            print(f"❌ Failed evaluations: {failed_count} models")
        print(f"📁 JSON Results: {json_path}")
        print(f"📊 CSV Summary: {csv_path}")
        
        # Display key results
        print(f"\n🎯 MIA Attack Success Summary:")
        print(f"{'Model':<40} {'Mode':<25} {'Sparsity':<10} {'Best Attack':<12}")
        print("-" * 90)
        
        for result in all_results:
            attack_accs = []
            for key, value in result.items():
                if key.endswith('_accuracy') and isinstance(value, (int, float)):
                    attack_accs.append(value)
            
            best_attack = max(attack_accs) if attack_accs else 0.0
            model_name = result['experiment_name'][:35] + "..." if len(result['experiment_name']) > 35 else result['experiment_name']
            
            print(f"{model_name:<40} {result['dwa_mode']:<25} {result['sparsity']:<10.3f} {best_attack:<12.3f}%")
    
    else:
        print("\n❌ No successful evaluations completed")

if __name__ == '__main__':
    args = parser.parse_args()
    
    # Load config file if provided (WeMeM-main style)
    if hasattr(args, 'config_path') and args.config_path and os.path.exists(args.config_path):
        with open(args.config_path) as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)
    
    print("📝 Configuration:")
    print(args)
    print()
    
    main(args)