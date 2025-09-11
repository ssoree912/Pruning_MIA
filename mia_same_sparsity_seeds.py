"""
MIA attack script for same sparsity different seeds
Only supports attacks between models with identical sparsity levels
Victim: seed42, Shadow models: other seeds (43-50)
"""
import argparse
import json
import pickle
import random
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from WeMeM-main.attackers import MiaAttack
from WeMeM-main.base_model import BaseModel
from WeMeM-main.datasets import get_dataset
from torch.utils.data import ConcatDataset, DataLoader, Subset

parser = argparse.ArgumentParser(description='MIA on Same Sparsity Different Seeds')
parser.add_argument('--device', default=0, type=int, help="GPU id to use")
parser.add_argument('--dataset_name', default='cifar10', type=str)
parser.add_argument('--model_name', default='resnet', type=str)
parser.add_argument('--num_cls', default=10, type=int)
parser.add_argument('--input_dim', default=3, type=int)
parser.add_argument('--image_size', default=32, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--sparsity', default=0.9, type=float, help="sparsity level")
parser.add_argument('--alpha', default=5.0, type=float)
parser.add_argument('--beta', default=5.0, type=float)
parser.add_argument('--victim_seed', default=42, type=int, help="victim model seed")
parser.add_argument('--shadow_seeds', default=[43,44,45,46,47,48,49,50], nargs='+', type=int, help="shadow model seeds")
parser.add_argument('--prune_method', default='dwa', type=str)
parser.add_argument('--prune_type', default='reactivate_only', type=str)
parser.add_argument('--attacks', default="samia,threshold", type=str)
parser.add_argument('--seed', default=7, type=int, help="random seed for attack")

def load_model_from_seed_folder(base_path, seed, dataset_name, model_name, sparsity, alpha, beta, 
                               prune_method, prune_type, device, num_cls=10, input_dim=3):
    """Load model from seed folder structure"""
    model_path = f"{base_path}/{prune_method}/{prune_type}/sparsity_{sparsity}/{dataset_name}/alpha{alpha}_beta{beta}/seed{seed}/best_model.pth"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = BaseModel(model_name, num_cls=num_cls, input_dim=input_dim, device=device)
    model.model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def create_data_splits(dataset, train_ratio=0.5):
    """Create train/test splits from dataset"""
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size
    
    indices = torch.randperm(total_size)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, test_dataset

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
    
    # Base path for models (will be runs/ in actual deployment)
    base_path = "runs"
    
    # Load datasets
    trainset = get_dataset(args.dataset_name, train=True)
    testset = get_dataset(args.dataset_name, train=False)
    if testset is None:
        total_dataset = trainset
    else:
        total_dataset = ConcatDataset([trainset, testset])
    
    # Create victim data splits
    victim_train_dataset, victim_test_dataset = create_data_splits(total_dataset, train_ratio=0.5)
    victim_train_loader = DataLoader(victim_train_dataset, batch_size=args.batch_size, 
                                   shuffle=False, num_workers=4, pin_memory=False)
    victim_test_loader = DataLoader(victim_test_dataset, batch_size=args.batch_size, 
                                  shuffle=False, num_workers=4, pin_memory=False)
    
    # Load victim model
    print(f"Loading victim model (seed {args.victim_seed})...")
    victim_model = load_model_from_seed_folder(
        base_path, args.victim_seed, args.dataset_name, args.model_name, 
        args.sparsity, args.alpha, args.beta, args.prune_method, args.prune_type,
        device, args.num_cls, args.input_dim
    )
    
    # Test victim model
    victim_model.test(victim_train_loader, "Victim Model Train")
    test_acc, loss = victim_model.test(victim_test_loader, "Victim Model Test")
    print(f"Victim model test accuracy: {test_acc:.3f}")
    
    # Load shadow models
    shadow_model_list = []
    shadow_train_loader_list = []
    shadow_test_loader_list = []
    
    for shadow_seed in args.shadow_seeds:
        print(f"Loading shadow model (seed {shadow_seed})...")
        shadow_model = load_model_from_seed_folder(
            base_path, shadow_seed, args.dataset_name, args.model_name,
            args.sparsity, args.alpha, args.beta, args.prune_method, args.prune_type,
            device, args.num_cls, args.input_dim
        )
        
        # Create shadow data splits
        shadow_train_dataset, shadow_test_dataset = create_data_splits(total_dataset, train_ratio=0.5)
        shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=args.batch_size,
                                       shuffle=False, num_workers=4, pin_memory=False)
        shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=args.batch_size,
                                      shuffle=False, num_workers=4, pin_memory=False)
        
        shadow_model.test(shadow_train_loader, f"Shadow Model {shadow_seed} Train")
        shadow_model.test(shadow_test_loader, f"Shadow Model {shadow_seed} Test")
        
        shadow_model_list.append(shadow_model)
        shadow_train_loader_list.append(shadow_train_loader)
        shadow_test_loader_list.append(shadow_test_loader)
    
    print("Starting Membership Inference Attacks...")
    
    # Initialize MIA attacker (using pruned models only, no original models)
    attacker = MiaAttack(
        victim_model, victim_model,  # Using same model for both (only pruned version)
        victim_train_loader, victim_test_loader,
        shadow_model_list, shadow_model_list,  # Using same models for both
        shadow_train_loader_list, shadow_test_loader_list,
        num_cls=args.num_cls, device=device, batch_size=args.batch_size,
        attack_original=False  # Attack pruned model
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
    
    # Save results
    result_dir = f"mia_results/{args.prune_method}_{args.prune_type}"
    os.makedirs(result_dir, exist_ok=True)
    result_file = f"{result_dir}/sparsity_{args.sparsity}_alpha{args.alpha}_beta{args.beta}_victim{args.victim_seed}.json"
    
    with open(result_file, 'w') as f:
        json.dump({
            'config': vars(args),
            'results': results,
            'victim_test_acc': test_acc
        }, f, indent=2)
    
    print(f"Results saved to {result_file}")

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)