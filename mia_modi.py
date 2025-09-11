"""
This code is modified from https://github.com/Machine-Learning-Security-Lab/mia_prune
"""
import argparse
import json
import pickle
import random
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from attackers import MiaAttack
from base_model import BaseModel
from mia_utils import load_dwa_model
from datasets import get_dataset
from torch.utils.data import ConcatDataset, DataLoader, Subset

try:
    from mia_metrics import compute_mia_metrics, print_mia_metrics
    HAS_METRICS = True
except ImportError:
    print("Warning: mia_metrics not found, using basic metrics only")
    HAS_METRICS = False


parser = argparse.ArgumentParser(description='Membership inference Attacks on Network Pruning')
parser.add_argument('--device', default=0, type=int, help="GPU id to use")
parser.add_argument('--config_path', default=None, type=str, help="config file path")
parser.add_argument('--dataset_name', default='cifar10', type=str)
parser.add_argument('--model_name', default='resnet18', type=str)
parser.add_argument('--num_cls', default=10, type=int)
parser.add_argument('--input_dim', default=3, type=int)
parser.add_argument('--image_size', default=32, type=int)
parser.add_argument('--hidden_size', default=128, type=int)
parser.add_argument('--seed', default=7, type=int)
parser.add_argument('--early_stop', default=5, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--pruner_name', default='l1unstructure', type=str, help="prune method for victim model")
parser.add_argument('--sparsity', default=0.9, type=float, help="sparsity level (same for all models)")
parser.add_argument('--victim_seed', default=42, type=int, help="victim model seed")
parser.add_argument('--shadow_seeds', default=[43,44,45,46,47,48,49,50], nargs='+', type=int, help="shadow model seeds")
parser.add_argument('--alpha', default=5.0, type=float)
parser.add_argument('--beta', default=5.0, type=float)
parser.add_argument('--prune_method', default='dwa', type=str)
parser.add_argument('--prune_type', default='reactivate_only', type=str)
parser.add_argument('--defend', default='', type=str)
parser.add_argument('--defend_arg', default=4, type=float)
parser.add_argument('--attacks', default="samia,threshold", type=str)  
parser.add_argument('--original', action='store_true', help="Attack original models instead of pruned models")
parser.add_argument('--threshold_strategy', default='youden', choices=['youden', 'max_accuracy', 'fpr_1pct', 'equal_error_rate'], 
                   help="Threshold selection strategy for attacks")
parser.add_argument('--forward_mode', default='standard', choices=['standard', 'dwa_adaptive', 'scaling', 'dpf'], 
                   help="Forward pass mode for model inference")


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = f"cuda:{args.device}"
    cudnn.benchmark = True
    print(f"Running MIA attack:")
    print(f"Victim seed: {args.victim_seed}")
    print(f"Shadow seeds: {args.shadow_seeds}")
    print(f"Sparsity: {args.sparsity}")
    print(f"Alpha: {args.alpha}, Beta: {args.beta}")

    base_path = "runs"
    result_dir = f"mia_results/{args.prune_method}_{args.prune_type}"
    os.makedirs(result_dir, exist_ok=True)
    result_file = f"{result_dir}/sparsity_{args.sparsity}_alpha{args.alpha}_beta{args.beta}_victim{args.victim_seed}.json"
    os.makedirs(f'log/{args.dataset_name}_{args.model_name}', exist_ok=True)

    # Load fixed data splits (WeMeM-main style)
    print("Loading fixed data splits...")
    
    data_split_path = f"mia_data_splits/{args.dataset_name}_seed{args.seed}_victim{args.victim_seed}.pkl"
    
    if not os.path.exists(data_split_path):
        print(f"❌ Data split file not found: {data_split_path}")
        print("Please run: python create_fixed_data_splits.py --dataset {args.dataset_name} --victim_seed {args.victim_seed}")
        raise FileNotFoundError(f"Data split file not found: {data_split_path}")
    
    with open(data_split_path, 'rb') as f:
        data_splits = pickle.load(f)
    
    print(f"✅ Loaded fixed splits from {data_split_path}")
    print(f"Victim {data_splits['victim_seed']}: {len(data_splits['victim']['train_indices'])} members, {len(data_splits['victim']['test_indices'])} non-members")
    
    # Load full dataset to create subsets
    trainset = get_dataset(args.dataset_name, train=True)
    testset = get_dataset(args.dataset_name, train=False)
    if testset is None:
        total_dataset = trainset
    else:
        total_dataset = ConcatDataset([trainset, testset])
    
    # Create victim datasets using fixed indices
    victim_train_indices = data_splits['victim']['train_indices']  # members
    victim_test_indices = data_splits['victim']['test_indices']    # non-members
    
    victim_train_dataset = Subset(total_dataset, victim_train_indices)
    victim_test_dataset = Subset(total_dataset, victim_test_indices)
    
    victim_train_loader = DataLoader(victim_train_dataset, batch_size=args.batch_size, 
                                   shuffle=False, num_workers=4, pin_memory=False)
    victim_test_loader = DataLoader(victim_test_dataset, batch_size=args.batch_size, 
                                  shuffle=False, num_workers=4, pin_memory=False)

    # Load victim model
    def load_model_from_seed_folder(base_path, seed, dataset_name, model_name, sparsity, alpha, beta, 
                                   prune_method, prune_type, device, forward_mode='standard', num_cls=10, input_dim=3):
        model_dir = f"{base_path}/{prune_method}/{prune_type}/sparsity_{sparsity}/{dataset_name}/alpha{alpha}_beta{beta}/seed{seed}"
        model_path = f"{model_dir}/best_model.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        print(f"Loading model with forward_mode: {forward_mode}")

        # Prefer DWA-aware loader when prune_method is DWA/static/DPF
        if prune_method.lower() in {"dwa", "static", "dpf"}:
            # Try to locate a config.json near the seed folder
            candidate_cfgs = [
                os.path.join(model_dir, 'config.json'),
                os.path.join(os.path.dirname(model_dir), 'config.json'),
                os.path.join(os.path.dirname(os.path.dirname(model_dir)), 'config.json')
            ]
            config_path = None
            for c in candidate_cfgs:
                if os.path.exists(c):
                    config_path = c
                    break
            try:
                loaded_model, _ = load_dwa_model(model_path, config_path=config_path, device=device)
                # Wrap into BaseModel interface for downstream predict_target_sensitivity
                wrapper = BaseModel(model_name, num_cls=num_cls, input_dim=input_dim, device=device)
                wrapper.model = loaded_model.to(device)

                # Best-effort: set optional forward modes if supported
                if forward_mode == 'dwa_adaptive' and hasattr(wrapper.model, 'set_dwa_params'):
                    wrapper.model.set_dwa_params(alpha=alpha, beta=beta, mode=prune_type)
                elif forward_mode == 'scaling' and hasattr(wrapper.model, 'set_scaling_mode'):
                    wrapper.model.set_scaling_mode(True)
                elif forward_mode == 'dpf' and hasattr(wrapper.model, 'set_dp_mode'):
                    wrapper.model.set_dp_mode(True)

                return wrapper
            except Exception as e:
                print(f"Warning: DWA-aware loading failed ({e}); falling back to generic model loader.")

        # Fallback: generic model + state_dict (may be partial)
        wrapper = BaseModel(model_name, num_cls=num_cls, input_dim=input_dim, device=device)
        try:
            state = torch.load(model_path, map_location=device)
            # Attempt robust load via BaseModel.load for relaxed matching
            wrapper.load(model_path, verbose=True)
        except Exception as e:
            print(f"Warning: Fallback load failed ({e}); using randomly initialized weights.")
        return wrapper
    
    print(f"Loading victim model (seed {args.victim_seed}) with forward_mode={args.forward_mode}...")
    victim_model = load_model_from_seed_folder(
        base_path, args.victim_seed, args.dataset_name, args.model_name, 
        args.sparsity, args.alpha, args.beta, args.prune_method, args.prune_type,
        device, args.forward_mode, args.num_cls, args.input_dim
    )
    
    victim_model.test(victim_train_loader, "Victim Model Train")
    test_acc, loss = victim_model.test(victim_test_loader, "Victim Model Test")
    print(f"Victim model test accuracy: {test_acc:.3f}")

    # Load shadow models with fixed data splits
    shadow_model_list = []
    shadow_train_loader_list = []
    shadow_test_loader_list = []
    
    for shadow_seed in args.shadow_seeds:
        if shadow_seed not in data_splits['shadows']:
            print(f"⚠️ Warning: Shadow seed {shadow_seed} not in data splits, skipping...")
            continue
            
        print(f"Loading shadow model (seed {shadow_seed}) with forward_mode={args.forward_mode}...")
        shadow_model = load_model_from_seed_folder(
            base_path, shadow_seed, args.dataset_name, args.model_name,
            args.sparsity, args.alpha, args.beta, args.prune_method, args.prune_type,
            device, args.forward_mode, args.num_cls, args.input_dim
        )
        
        # Use fixed shadow data splits
        shadow_data = data_splits['shadows'][shadow_seed]
        shadow_train_indices = shadow_data['train_indices']  # members
        shadow_test_indices = shadow_data['test_indices']    # non-members
        
        shadow_train_dataset = Subset(total_dataset, shadow_train_indices)
        shadow_test_dataset = Subset(total_dataset, shadow_test_indices)
        
        shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=args.batch_size,
                                       shuffle=False, num_workers=4, pin_memory=False)
        shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=args.batch_size,
                                      shuffle=False, num_workers=4, pin_memory=False)
        
        print(f"Shadow {shadow_seed}: {len(shadow_train_indices)} members, {len(shadow_test_indices)} non-members")
        shadow_model.test(shadow_train_loader, f"Shadow Model {shadow_seed} Train (Members)")
        shadow_model.test(shadow_test_loader, f"Shadow Model {shadow_seed} Test (Non-members)")
        
        shadow_model_list.append(shadow_model)
        shadow_train_loader_list.append(shadow_train_loader)
        shadow_test_loader_list.append(shadow_test_loader)

    print("Start Membership Inference Attacks")

    if hasattr(args, 'original') and args.original:
        attack_original = True
    else:
        attack_original = False
    
    print(f"Attack mode: {'Original models' if attack_original else 'Pruned models'}")
    
    attacker = MiaAttack(
        victim_model, victim_model, victim_train_loader, victim_test_loader,
        shadow_model_list, shadow_model_list, shadow_train_loader_list, shadow_test_loader_list,
        num_cls=args.num_cls, device=device, batch_size=args.batch_size,
        attack_original=attack_original  # 🔥 이제 제대로 전달됨
    )

    attacks = args.attacks.split(',')
    results = {}
    
    if "samia" in attacks:
        nn_trans_acc = attacker.nn_attack("nn_sens_cls", model_name="transformer")
        results['samia'] = nn_trans_acc
        print(f"SAMIA attack accuracy: {nn_trans_acc:.3f}")
    
    if "threshold" in attacks:
        conf, xent, mentr, top1_conf = attacker.threshold_attack()
        results['confidence'] = conf
        results['entropy'] = xent
        results['modified_entropy'] = mentr
        results['top1_conf'] = top1_conf
        
        print(f"Confidence attack accuracy: {conf:.3f}")
        print(f"Entropy attack accuracy: {xent:.3f}")
        print(f"Modified entropy attack accuracy: {mentr:.3f}")
        print(f"Top1 confidence attack accuracy: {top1_conf:.3f}")
        
        # Extended metrics for threshold attacks
        if HAS_METRICS:
            print(f"\n🔍 Computing extended threshold metrics (strategy: {args.threshold_strategy})...")
            
            try:
                from mia_metrics import select_threshold_strategy
                
                # Confidence-based extended metrics with custom strategy
                victim_in_conf = attacker.victim_in_predicts.max(dim=1)[0].numpy()
                victim_out_conf = attacker.victim_out_predicts.max(dim=1)[0].numpy()
                
                # Use selected threshold strategy
                optimal_threshold = select_threshold_strategy(
                    victim_in_conf, victim_out_conf, strategy=args.threshold_strategy
                )
                
                conf_metrics = compute_mia_metrics(
                    victim_in_conf, victim_out_conf, threshold=optimal_threshold
                )
                results['confidence_extended'] = conf_metrics
                results['threshold_strategy'] = args.threshold_strategy
                print_mia_metrics(conf_metrics, f"Confidence Attack ({args.threshold_strategy.upper()})")
                
                # Compare different strategies
                print(f"\n📊 Threshold Strategy Comparison:")
                strategies = ['youden', 'max_accuracy', 'fpr_1pct', 'equal_error_rate']
                strategy_results = {}
                
                for strategy in strategies:
                    try:
                        thresh = select_threshold_strategy(victim_in_conf, victim_out_conf, strategy)
                        metrics = compute_mia_metrics(victim_in_conf, victim_out_conf, thresh)
                        strategy_results[strategy] = {
                            'auroc': metrics['auroc'],
                            'accuracy': metrics['accuracy'],
                            'advantage': metrics['advantage'],
                            'threshold': thresh
                        }
                        print(f"  {strategy:15s}: AUROC={metrics['auroc']:.4f}, Acc={metrics['accuracy']:.4f}, Adv={metrics['advantage']:.4f}")
                    except Exception as e:
                        print(f"  {strategy:15s}: Failed ({e})")
                
                results['strategy_comparison'] = strategy_results
                
            except Exception as e:
                print(f"Could not compute extended metrics: {e}")
                print("Extended metrics require access to prediction scores")
    
    if "nn" in attacks:
        nn_acc = attacker.nn_attack("nn")
        results['nn'] = nn_acc
        print(f"NN attack accuracy: {nn_acc:.3f}")
    
    if "nn_top3" in attacks:
        nn_top3_acc = attacker.nn_attack("nn_top3")
        results['nn_top3'] = nn_top3_acc
        print(f"Top3-NN attack accuracy: {nn_top3_acc:.3f}")
    
    if "nn_cls" in attacks:
        nn_cls_acc = attacker.nn_attack("nn_cls")
        results['nn_cls'] = nn_cls_acc
        print(f"NN-Cls attack accuracy: {nn_cls_acc:.3f}")
    
    # Save results with data split info
    import json
    with open(result_file, 'w') as f:
        json.dump({
            'config': vars(args),
            'results': results,
            'victim_test_acc': test_acc,
            'experiment_info': {
                'forward_mode': args.forward_mode,
                'threshold_strategy': args.threshold_strategy,
                'attack_mode': 'original' if (hasattr(args, 'original') and args.original) else 'pruned',
                'dwa_params': {
                    'alpha': args.alpha,
                    'beta': args.beta,
                    'prune_type': args.prune_type,
                    'sparsity': args.sparsity
                }
            },
            'data_splits_info': {
                'victim_members': len(data_splits['victim']['train_indices']),
                'victim_nonmembers': len(data_splits['victim']['test_indices']),
                'shadow_counts': {str(k): len(v['train_indices']) for k, v in data_splits['shadows'].items()},
                'split_file': data_split_path
            }
        }, f, indent=2)
    
    print(f"Results saved to {result_file}")

if __name__ == '__main__':
    args = parser.parse_args()
    
    # config 파일이 지정되면 로드
    if args.config_path and os.path.exists(args.config_path):
        print(f"Loading config from {args.config_path}")
        with open(args.config_path) as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)
    
    print(f"Arguments: {args}")
    main(args)
