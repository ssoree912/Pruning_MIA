#!/usr/bin/env python3
"""
Training script for Static pruning experiments
Supports sparsity levels: 50%, 70%, 80%, 90%, 95%
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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pruning
from data import DataLoader
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='Static Pruning Training')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--arch', default='resnet', type=str)
    parser.add_argument('--layers', default=18, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--sparsity', required=True, type=float, 
                        help='Target sparsity (0.5, 0.7, 0.8, 0.9, 0.95)')
    parser.add_argument('--pretrained', required=True, type=str,
                        help='Path to pretrained dense model')
    parser.add_argument('--save-dir', default='./runs/static', type=str)
    parser.add_argument('--datapath', default='../data', type=str)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--print-freq', default=100, type=int)
    parser.add_argument('--prune-method', default='magnitude', 
                        choices=['magnitude', 'omp'], help='Pruning method')
    
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def magnitude_pruning(model, sparsity):
    """Magnitude-based pruning"""
    all_weights = []
    
    # Collect all weights
    for name, module in model.named_modules():
        if isinstance(module, pruning.dcil.mnn.MaskConv2d):
            weights = module.weight.data.abs().view(-1)
            all_weights.append(weights)
    
    # Calculate threshold
    all_weights = torch.cat(all_weights)
    threshold = torch.quantile(all_weights, sparsity)
    
    # Apply masks
    total_params = 0
    pruned_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, pruning.dcil.mnn.MaskConv2d):
            weights = module.weight.data.abs()
            mask = (weights > threshold).float()
            module.mask.data = mask
            module.type_value = 5  # Use MaskerStatic
            
            total_params += module.weight.numel()
            pruned_params += (mask == 0).sum().item()
    
    actual_sparsity = pruned_params / total_params
    print(f"Target sparsity: {sparsity:.2%}, Actual sparsity: {actual_sparsity:.2%}")
    return actual_sparsity

def omp_pruning(model, sparsity):
    """Orthogonal Matching Pursuit based pruning"""
    # For simplicity, using magnitude-based. Can be extended to OMP.
    return magnitude_pruning(model, sparsity)

def train_epoch(model, train_loader, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    model.train()
    end = time.time()
    
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
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
        
        batch_time.update(time.time() - end)
        
        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}] '
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  f'Loss {losses.val:.4f} ({losses.avg:.4f}) '
                  f'Acc@1 {top1.val:.3f} ({top1.avg:.3f}) '
                  f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')
        
        end = time.time()
    
    return top1.avg, top5.avg, losses.avg

def validate(model, val_loader, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            
            output = model(input)
            loss = criterion(output, target)
            
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            
            batch_time.update(time.time() - end)
            
            if i % args.print_freq == 0:
                print(f'Test: [{i}/{len(val_loader)}] '
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      f'Loss {losses.val:.4f} ({losses.avg:.4f}) '
                      f'Acc@1 {top1.val:.3f} ({top1.avg:.3f}) '
                      f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')
            
            end = time.time()
    
    print(f'Test Results: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    return top1.avg, top5.avg, losses.avg

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f'sparsity{args.sparsity}', f'seed{args.seed}')
    os.makedirs(save_path, exist_ok=True)
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.cuda.set_device(0)
    
    print(f"Training Static Pruned {args.arch}-{args.layers} on {args.dataset}")
    print(f"Sparsity: {args.sparsity:.2%}, Save directory: {save_path}")
    
    # Create model with pruning support
    pruner = pruning.dcil
    model, image_size = pruning.models.__dict__[args.arch](
        data=args.dataset, 
        num_layers=args.layers,
        mnn=pruner.mnn
    )
    
    model = model.cuda()
    
    # Load pretrained dense model
    print(f"Loading pretrained model from {args.pretrained}")
    checkpoint = torch.load(args.pretrained)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    model = nn.DataParallel(model)
    cudnn.benchmark = True
    
    # Apply static pruning
    print(f"Applying static pruning with {args.sparsity:.2%} sparsity...")
    if args.prune_method == 'magnitude':
        actual_sparsity = magnitude_pruning(model, args.sparsity)
    elif args.prune_method == 'omp':
        actual_sparsity = omp_pruning(model, args.sparsity)
    
    # Create data loaders
    train_loader, val_loader = DataLoader(
        args.batch_size, args.dataset, args.workers, 
        args.datapath, image_size, True
    )
    
    # Setup training
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, 
        momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1
    )
    
    # Fine-tuning loop
    best_acc1 = 0
    train_history = []
    val_history = []
    
    for epoch in range(args.epochs):
        print(f'\nEpoch: {epoch}, lr = {optimizer.param_groups[0]["lr"]}')
        
        # Train
        train_acc1, train_acc5, train_loss = train_epoch(
            model, train_loader, criterion, optimizer, epoch, args
        )
        
        # Validate
        val_acc1, val_acc5, val_loss = validate(
            model, val_loader, criterion, args
        )
        
        scheduler.step()
        
        # Save best model
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)
        
        if is_best:
            torch.save({
                'epoch': epoch,
                'arch': args.arch,
                'layers': args.layers,
                'sparsity': actual_sparsity,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'args': args,
            }, os.path.join(save_path, 'best_model.pth'))
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'arch': args.arch,
            'layers': args.layers,
            'sparsity': actual_sparsity,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'args': args,
        }, os.path.join(save_path, 'checkpoint.pth'))
        
        # Record history
        train_history.append([epoch, train_acc1.item(), train_acc5.item(), train_loss])
        val_history.append([epoch, val_acc1.item(), val_acc5.item(), val_loss])
        
        # Save training history
        np.save(os.path.join(save_path, 'train_history.npy'), np.array(train_history))
        np.save(os.path.join(save_path, 'val_history.npy'), np.array(val_history))
    
    print(f'\nStatic pruning training completed. Best accuracy: {best_acc1:.3f}')
    print(f'Final sparsity: {actual_sparsity:.2%}')

if __name__ == '__main__':
    main()