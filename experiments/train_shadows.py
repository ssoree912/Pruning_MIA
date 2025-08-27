#!/usr/bin/env python3
"""
Training script for shadow models for LiRA calibration
Trains multiple shadow models for each target model configuration
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models
import pruning
from data import DataLoader
from mia import create_shadow_datasets

# Import common utilities
from common_utils import AverageMeter, accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='Shadow Model Training for LiRA')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--arch', default='resnet', type=str)
    parser.add_argument('--layers', default=18, type=int)
    parser.add_argument('--model-type', required=True, 
                        choices=['dense', 'static', 'dpf'],
                        help='Type of model to train shadows for')
    parser.add_argument('--sparsity', type=float, default=None,
                        help='Sparsity for pruned models (required for static/dpf)')
    parser.add_argument('--pretrained-target', type=str, default=None,
                        help='Path to pretrained target model (for static pruning)')
    
    # Training parameters
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    
    # Shadow model parameters
    parser.add_argument('--num-shadows', default=64, type=int,
                        help='Number of shadow models to train')
    parser.add_argument('--start-shadow', default=0, type=int,
                        help='Starting shadow model index (for parallel training)')
    parser.add_argument('--end-shadow', default=None, type=int,
                        help='Ending shadow model index (for parallel training)')
    
    # System parameters
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed-base', default=1000, type=int,
                        help='Base seed for shadow models')
    parser.add_argument('--save-dir', default='./runs/shadows', type=str)
    parser.add_argument('--datapath', default='../data', type=str)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--print-freq', default=200, type=int)
    
    # DPF specific parameters
    parser.add_argument('--prune-freq', default=16, type=int)
    parser.add_argument('--target-epoch', default=75, type=int)
    
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_shadow_model(args, shadow_id, shadow_data):
    """Train a single shadow model"""
    
    # Set unique seed for this shadow model
    shadow_seed = args.seed_base + shadow_id
    set_seed(shadow_seed)
    
    # Create save directory for this shadow
    model_type_dir = f"{args.model_type}"
    if args.sparsity is not None:
        model_type_dir += f"_sparsity{args.sparsity}"
    
    save_path = os.path.join(args.save_dir, model_type_dir, f'shadow{shadow_id:03d}')
    os.makedirs(save_path, exist_ok=True)
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.cuda.set_device(0)
    
    print(f"Training shadow model {shadow_id} for {args.model_type}")
    if args.sparsity is not None:
        print(f"Sparsity: {args.sparsity:.2%}")
    print(f"Save directory: {save_path}")
    
    # Create model
    if args.model_type == 'dense':
        model, image_size = models.__dict__[args.arch](
            data=args.dataset, 
            num_layers=args.layers
        )
    else:  # static or dpf
        pruner = pruning.dcil
        model, image_size = pruning.models.__dict__[args.arch](
            data=args.dataset, 
            num_layers=args.layers,
            mnn=pruner.mnn
        )
    
    model = model.cuda()
    
    # For static pruning, load pretrained model and apply pruning
    if args.model_type == 'static':
        if args.pretrained_target:
            print(f"Loading pretrained model from {args.pretrained_target}")
            checkpoint = torch.load(args.pretrained_target)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        # Apply static pruning
        from experiments.train_static import magnitude_pruning
        magnitude_pruning(model, args.sparsity)
    
    # For DPF, initialize masks and set dynamic masking
    elif args.model_type == 'dpf':
        for name, module in model.named_modules():
            if isinstance(module, pruning.dcil.mnn.MaskConv2d):
                module.mask.data.fill_(1.0)
                module.type_value = 6  # Use MaskerDynamic
    
    model = nn.DataParallel(model)
    cudnn.benchmark = True
    
    # Unpack shadow data
    shadow_train_loader, shadow_test_loader = shadow_data
    
    # Setup training
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, 
        momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1
    )
    
    # Training loop
    best_acc1 = 0
    train_history = []
    val_history = []
    
    iteration = [0]  # For DPF iteration tracking
    
    for epoch in range(args.epochs):
        if epoch % 50 == 0:
            print(f'Shadow {shadow_id} - Epoch: {epoch}, lr = {optimizer.param_groups[0]["lr"]}')
        
        # Train
        if args.model_type == 'dpf':
            train_acc1, train_acc5, train_loss = train_epoch_dpf(
                model, shadow_train_loader, criterion, optimizer, epoch, args, iteration
            )
        else:
            train_acc1, train_acc5, train_loss = train_epoch_standard(
                model, shadow_train_loader, criterion, optimizer, epoch, args
            )
        
        # Validate (less frequent for efficiency)
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            val_acc1, val_acc5, val_loss = validate(
                model, shadow_test_loader, criterion, args
            )
        else:
            val_acc1, val_acc5, val_loss = train_acc1, train_acc5, train_loss
        
        scheduler.step()
        
        # Save best model
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)
        
        # Record history
        train_history.append([epoch, train_acc1.item(), train_acc5.item(), train_loss])
        val_history.append([epoch, val_acc1.item(), val_acc5.item(), val_loss])
    
    # Save final model and history
    torch.save({
        'shadow_id': shadow_id,
        'epoch': args.epochs - 1,
        'arch': args.arch,
        'layers': args.layers,
        'model_type': args.model_type,
        'sparsity': args.sparsity,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'args': args,
    }, os.path.join(save_path, 'final_model.pth'))
    
    np.save(os.path.join(save_path, 'train_history.npy'), np.array(train_history))
    np.save(os.path.join(save_path, 'val_history.npy'), np.array(val_history))
    
    print(f'Shadow {shadow_id} completed. Best accuracy: {best_acc1:.3f}')
    
    return shadow_id, best_acc1

def train_epoch_standard(model, train_loader, criterion, optimizer, epoch, args):
    """Standard training epoch"""
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    model.train()
    
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()
        
        output = model(input)
        loss = criterion(output, target)
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return top1.avg, top5.avg, losses.avg

def train_epoch_dpf(model, train_loader, criterion, optimizer, epoch, args, iteration):
    """DPF training epoch with dynamic pruning"""
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    model.train()
    
    from experiments.train_dpf import get_current_sparsity, dynamic_pruning
    
    for i, (input, target) in enumerate(train_loader):
        
        # Dynamic pruning update
        if iteration[0] % args.prune_freq == 0 and args.sparsity is not None:
            current_sparsity = get_current_sparsity(
                model, args.sparsity, epoch, args.target_epoch
            )
            dynamic_pruning(model, current_sparsity, iteration[0])
        
        input = input.cuda()
        target = target.cuda()
        
        output = model(input)
        loss = criterion(output, target)
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        iteration[0] += 1
    
    return top1.avg, top5.avg, losses.avg

def validate(model, val_loader, criterion, args):
    """Validation function"""
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            
            output = model(input)
            loss = criterion(output, target)
            
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
    
    return top1.avg, top5.avg, losses.avg

def main():
    args = parse_args()
    
    # Validate arguments
    if args.model_type in ['static', 'dpf'] and args.sparsity is None:
        raise ValueError("Sparsity must be specified for pruned models")
    
    if args.model_type == 'static' and args.pretrained_target is None:
        print("Warning: No pretrained target model specified for static pruning")
    
    # Set end shadow if not specified
    if args.end_shadow is None:
        args.end_shadow = args.num_shadows
    
    print(f"Training shadow models {args.start_shadow}-{args.end_shadow-1} for {args.model_type}")
    
    # Create original data loaders for creating shadow datasets
    if args.model_type == 'dense':
        model, image_size = models.__dict__[args.arch](
            data=args.dataset, num_layers=args.layers
        )
    else:
        pruner = pruning.dcil
        model, image_size = pruning.models.__dict__[args.arch](
            data=args.dataset, num_layers=args.layers, mnn=pruner.mnn
        )
    
    # Get original data
    train_loader, val_loader = DataLoader(
        args.batch_size, args.dataset, args.workers, 
        args.datapath, image_size, True
    )
    
    # Create shadow datasets
    print("Creating shadow datasets...")
    shadow_datasets = create_shadow_datasets(
        train_loader, val_loader, 
        num_shadows=args.num_shadows, 
        seed=args.seed_base
    )
    
    # Save shadow dataset splits for reproducibility
    model_type_dir = f"{args.model_type}"
    if args.sparsity is not None:
        model_type_dir += f"_sparsity{args.sparsity}"
    
    shadow_dir = os.path.join(args.save_dir, model_type_dir)
    os.makedirs(shadow_dir, exist_ok=True)
    
    # Save dataset information
    with open(os.path.join(shadow_dir, 'shadow_datasets_info.pkl'), 'wb') as f:
        pickle.dump({
            'num_shadows': args.num_shadows,
            'seed_base': args.seed_base,
            'total_samples': len(train_loader.dataset) + len(val_loader.dataset),
            'train_samples': len(train_loader.dataset),
        }, f)
    
    # Train shadow models
    results = []
    
    for shadow_id in range(args.start_shadow, args.end_shadow):
        shadow_data = shadow_datasets[shadow_id]
        
        try:
            result = train_shadow_model(args, shadow_id, shadow_data)
            results.append(result)
            print(f"Completed shadow {shadow_id}: acc={result[1]:.3f}")
            
        except Exception as e:
            print(f"Error training shadow {shadow_id}: {e}")
            continue
    
    # Summary
    if results:
        accuracies = [acc for _, acc in results]
        print(f"\nShadow training completed:")
        print(f"  Trained {len(results)} shadow models")
        print(f"  Mean accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
        print(f"  Min accuracy: {np.min(accuracies):.3f}")
        print(f"  Max accuracy: {np.max(accuracies):.3f}")
    else:
        print("No shadow models completed successfully")

if __name__ == '__main__':
    main()