#!/usr/bin/env python3
"""
Training script for Dynamic Pruning (DPF) experiments
Supports sparsity levels: 50%, 70%, 80%, 90%, 95%
"""

import os
import sys
import time
import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pruning
from data import DataLoader
from utils import AverageMeter, accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='Dynamic Pruning (DPF) Training')
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
    parser.add_argument('--save-dir', default='./runs/dpf', type=str)
    parser.add_argument('--datapath', default='../data', type=str)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--print-freq', default=100, type=int)
    parser.add_argument('--prune-freq', default=16, type=int,
                        help='Frequency of mask updates')
    parser.add_argument('--target-epoch', default=75, type=int,
                        help='Epoch to reach target sparsity')
    
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_current_sparsity(model, target_sparsity, epoch, target_epoch):
    """Calculate current sparsity using polynomial schedule"""
    if epoch >= target_epoch:
        return target_sparsity
    
    # Polynomial sparsity schedule: s_t = s_f * (1 - (1 - t/T)^3)
    progress = epoch / target_epoch
    current_sparsity = target_sparsity * (1 - (1 - progress) ** 3)
    return current_sparsity

def dynamic_pruning(model, sparsity, iteration):
    """Apply dynamic pruning with mask updates"""
    all_weights = []
    modules_to_prune = []
    
    # Collect weights and modules
    for name, module in model.named_modules():
        if isinstance(module, pruning.dcil.mnn.MaskConv2d):
            weights = module.weight.data.abs().view(-1)
            all_weights.append(weights)
            modules_to_prune.append(module)
    
    # Calculate threshold
    all_weights = torch.cat(all_weights)
    threshold = torch.quantile(all_weights, sparsity)
    
    # Update masks and set dynamic masker
    total_params = 0
    pruned_params = 0
    reactivations = 0
    
    for module in modules_to_prune:
        weights = module.weight.data.abs()
        old_mask = module.mask.data.clone()
        new_mask = (weights > threshold).float()
        
        # Count reactivations (0 -> 1 transitions)
        reactivations += ((old_mask == 0) & (new_mask == 1)).sum().item()
        
        module.mask.data = new_mask
        module.type_value = 6  # Use MaskerDynamic
        
        total_params += module.weight.numel()
        pruned_params += (new_mask == 0).sum().item()
    
    actual_sparsity = pruned_params / total_params
    reactivation_rate = reactivations / total_params
    
    return actual_sparsity, reactivation_rate

def train_epoch(model, train_loader, criterion, optimizer, epoch, args, iteration_ref):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    model.train()
    end = time.time()
    
    sparsity_updates = []
    reactivation_updates = []
    
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        # Dynamic pruning update
        if iteration_ref[0] % args.prune_freq == 0:
            current_sparsity = get_current_sparsity(
                model, args.sparsity, epoch, args.target_epoch
            )
            actual_sparsity, reactivation_rate = dynamic_pruning(
                model, current_sparsity, iteration_ref[0]
            )
            sparsity_updates.append(actual_sparsity)
            reactivation_updates.append(reactivation_rate)
            
            if i % args.print_freq == 0:
                print(f'Iteration {iteration_ref[0]}: Target sparsity: {current_sparsity:.3f}, '
                      f'Actual sparsity: {actual_sparsity:.3f}, Reactivation rate: {reactivation_rate:.4f}')
        
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
        
        iteration_ref[0] += 1
        end = time.time()
    
    avg_sparsity = np.mean(sparsity_updates) if sparsity_updates else 0
    avg_reactivation = np.mean(reactivation_updates) if reactivation_updates else 0
    
    return top1.avg, top5.avg, losses.avg, avg_sparsity, avg_reactivation

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
    
    print(f"Training DPF {args.arch}-{args.layers} on {args.dataset}")
    print(f"Target sparsity: {args.sparsity:.2%}, Save directory: {save_path}")
    
    # Create model with pruning support
    pruner = pruning.dcil
    model, image_size = pruning.models.__dict__[args.arch](
        data=args.dataset, 
        num_layers=args.layers,
        mnn=pruner.mnn
    )
    
    model = model.cuda()
    model = nn.DataParallel(model)
    cudnn.benchmark = True
    
    # Initialize all masks to 1 (dense)
    for name, module in model.named_modules():
        if isinstance(module, pruning.dcil.mnn.MaskConv2d):
            module.mask.data.fill_(1.0)
            module.type_value = 6  # Use MaskerDynamic
    
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
    
    # Training loop
    best_acc1 = 0
    train_history = []
    val_history = []
    sparsity_history = []
    reactivation_history = []
    
    iteration = [0]  # Use list to allow modification in nested function
    
    for epoch in range(args.epochs):
        print(f'\nEpoch: {epoch}, lr = {optimizer.param_groups[0]["lr"]}')
        
        # Train
        train_acc1, train_acc5, train_loss, avg_sparsity, avg_reactivation = train_epoch(
            model, train_loader, criterion, optimizer, epoch, args, iteration
        )
        
        # Validate
        val_acc1, val_acc5, val_loss = validate(
            model, val_loader, criterion, args
        )
        
        scheduler.step()
        
        # Calculate current sparsity for logging
        current_target_sparsity = get_current_sparsity(
            model, args.sparsity, epoch, args.target_epoch
        )
        
        print(f'Epoch {epoch}: Target sparsity: {current_target_sparsity:.3f}, '
              f'Avg actual sparsity: {avg_sparsity:.3f}, Avg reactivation: {avg_reactivation:.4f}')
        
        # Save best model
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)
        
        if is_best:
            torch.save({
                'epoch': epoch,
                'arch': args.arch,
                'layers': args.layers,
                'target_sparsity': args.sparsity,
                'current_sparsity': avg_sparsity,
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
            'target_sparsity': args.sparsity,
            'current_sparsity': avg_sparsity,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'iteration': iteration[0],
            'args': args,
        }, os.path.join(save_path, 'checkpoint.pth'))
        
        # Record history
        train_history.append([epoch, train_acc1.item(), train_acc5.item(), train_loss])
        val_history.append([epoch, val_acc1.item(), val_acc5.item(), val_loss])
        sparsity_history.append([epoch, current_target_sparsity, avg_sparsity])
        reactivation_history.append([epoch, avg_reactivation])
        
        # Save training history
        np.save(os.path.join(save_path, 'train_history.npy'), np.array(train_history))
        np.save(os.path.join(save_path, 'val_history.npy'), np.array(val_history))
        np.save(os.path.join(save_path, 'sparsity_history.npy'), np.array(sparsity_history))
        np.save(os.path.join(save_path, 'reactivation_history.npy'), np.array(reactivation_history))
    
    print(f'\nDPF training completed. Best accuracy: {best_acc1:.3f}')
    print(f'Final target sparsity: {args.sparsity:.2%}')

if __name__ == '__main__':
    main()