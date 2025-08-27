#!/usr/bin/env python3
"""
Main experiment runner with comprehensive configuration support
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.experiment_config import parse_config_args, setup_reproducibility
from utils.logger import ExperimentLogger, get_system_info
from utils import AverageMeter, ProgressMeter, accuracy, set_scheduler, set_arch_name
import models
import pruning
from data import DataLoader

def create_model(config):
    """Create model based on configuration"""
    if config.pruning.enabled:
        # Create pruned model
        pruner = pruning.dcil
        model, image_size = pruning.models.__dict__[config.model.arch](
            data=config.data.dataset,
            num_layers=config.model.layers,
            width_mult=config.model.width_mult,
            depth_mult=config.model.depth_mult,
            model_mult=config.model.model_mult,
            mnn=pruner.mnn
        )
    else:
        # Create dense model
        model, image_size = models.__dict__[config.model.arch](
            data=config.data.dataset,
            num_layers=config.model.layers,
            width_mult=config.model.width_mult,
            depth_mult=config.model.depth_mult,
            model_mult=config.model.model_mult
        )
    
    # Check if model creation failed
    if model is None:
        raise ValueError(f"Failed to create model: {config.model.arch} with {config.model.layers} layers for {config.data.dataset}")
    
    return model, image_size

def setup_training(model, config):
    """Setup training components"""
    
    # Loss function
    criterion = nn.CrossEntropyLoss().cuda()
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.training.lr,
        momentum=config.training.momentum,
        weight_decay=config.training.weight_decay,
        nesterov=config.training.nesterov
    )
    
    # Learning rate scheduler
    if config.training.scheduler == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.training.milestones,
            gamma=config.training.gamma
        )
    elif config.training.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.training.step_size,
            gamma=config.training.gamma
        )
    elif config.training.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs
        )
    elif config.training.scheduler == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.training.gamma
        )
    else:
        scheduler = None
    
    return criterion, optimizer, scheduler

def apply_static_pruning(model, config, logger):
    """Apply static pruning to model"""
    logger.logger.info(f"Applying static pruning with {config.pruning.sparsity:.2%} sparsity")
    
    # Magnitude-based pruning
    all_weights = []
    
    # Collect all weights
    for name, module in model.named_modules():
        if isinstance(module, pruning.dcil.mnn.MaskConv2d):
            weights = module.weight.data.abs().view(-1)
            all_weights.append(weights)
    
    # Calculate threshold
    all_weights = torch.cat(all_weights)
    threshold = torch.quantile(all_weights, config.pruning.sparsity)
    
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
    logger.log_pruning_info(actual_sparsity)
    
    return actual_sparsity

def apply_dynamic_pruning(model, config, epoch, iteration, logger):
    """Apply dynamic pruning during training"""
    
    if iteration % config.pruning.prune_freq != 0:
        return 0.0, 0.0
    
    # Calculate current target sparsity using polynomial schedule
    if epoch >= config.pruning.target_epoch:
        current_sparsity = config.pruning.sparsity
    else:
        progress = epoch / config.pruning.target_epoch
        current_sparsity = config.pruning.sparsity * (1 - (1 - progress) ** 3)
    
    # Collect all weights
    all_weights = []
    modules_to_prune = []
    
    for name, module in model.named_modules():
        if isinstance(module, pruning.dcil.mnn.MaskConv2d):
            weights = module.weight.data.abs().view(-1)
            all_weights.append(weights)
            modules_to_prune.append(module)
    
    # Calculate threshold
    all_weights = torch.cat(all_weights)
    threshold = torch.quantile(all_weights, current_sparsity)
    
    # Update masks
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
    
    if iteration % (config.pruning.prune_freq * 10) == 0:  # Log less frequently
        logger.log_pruning_info(actual_sparsity, reactivation_rate)
    
    return actual_sparsity, reactivation_rate

def train_epoch(model, train_loader, criterion, optimizer, epoch, config, logger, iteration_counter):
    """Train for one epoch"""
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
        if config.pruning.enabled and config.pruning.method == 'dpf':
            sparsity, reactivation = apply_dynamic_pruning(
                model, config, epoch, iteration_counter[0], logger
            )
            if sparsity > 0:
                sparsity_updates.append(sparsity)
                reactivation_updates.append(reactivation)
        
        input = input.cuda()
        target = target.cuda()
        
        # Forward pass
        if config.pruning.enabled and config.pruning.method == 'dcil':
            # DCIL with dual outputs
            output = model(input, 0)  # Sparse output
            output_full = model(input, 1)  # Full output
            
            # DCIL loss with KL divergence
            if epoch < config.training.warmup_loss_epoch:
                loss = criterion(output, target) + criterion(output_full, target)
            else:
                from run_dcil import KLLoss
                criterion_kl = KLLoss().cuda()
                loss = (criterion(output, target) + criterion(output_full, target) + 
                       criterion_kl(output, output_full) + criterion_kl(output_full, output))
        else:
            output = model(input)
            loss = criterion(output, target)
        
        # Measure accuracy
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        
        if i % config.system.print_freq == 0:
            logger.logger.info(
                f'Epoch: [{epoch}][{i}/{len(train_loader)}] '
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                f'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                f'Loss {losses.val:.4f} ({losses.avg:.4f}) '
                f'Acc@1 {top1.val:.3f} ({top1.avg:.3f}) '
                f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'
            )
        
        iteration_counter[0] += 1
        end = time.time()
    
    # Calculate average sparsity and reactivation for this epoch
    avg_sparsity = np.mean(sparsity_updates) if sparsity_updates else 0
    avg_reactivation = np.mean(reactivation_updates) if reactivation_updates else 0
    
    metrics = {
        'acc1': top1.avg.item(),
        'acc5': top5.avg.item(),
        'loss': losses.avg
    }
    
    if avg_sparsity > 0:
        metrics['sparsity'] = avg_sparsity
        metrics['reactivation_rate'] = avg_reactivation
    
    return metrics

def validate(model, val_loader, criterion, config, logger):
    """Validate model"""
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
            
            if config.pruning.enabled and config.pruning.method == 'dcil':
                output = model(input, 0)  # Use sparse output for evaluation
            else:
                output = model(input)
            
            loss = criterion(output, target)
            
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            
            batch_time.update(time.time() - end)
            
            if i % config.system.print_freq == 0:
                logger.logger.info(
                    f'Test: [{i}/{len(val_loader)}] '
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    f'Loss {losses.val:.4f} ({losses.avg:.4f}) '
                    f'Acc@1 {top1.val:.3f} ({top1.avg:.3f}) '
                    f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'
                )
            
            end = time.time()
    
    return {
        'acc1': top1.avg.item(),
        'acc5': top5.avg.item(),
        'loss': losses.avg
    }

def main():
    # Parse configuration
    config = parse_config_args()
    
    # Setup reproducibility
    setup_reproducibility(config.system)
    
    # Create save directory
    save_path = config.get_save_path()
    os.makedirs(save_path, exist_ok=True)
    
    # Setup GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.system.gpu)
    torch.cuda.set_device(0)
    
    # Initialize logger
    logger = ExperimentLogger(config.name, save_path)
    
    # Log configuration and system info
    logger.log_hyperparameters(config.to_dict())
    logger.log_system_info(get_system_info())
    
    # Save configuration
    config.to_yaml(os.path.join(save_path, 'config.yaml'))
    config.to_json(os.path.join(save_path, 'config.json'))
    
    logger.logger.info(f"Starting experiment: {config.name}")
    logger.logger.info(f"Model: {config.model.arch}-{config.model.layers}")
    logger.logger.info(f"Dataset: {config.data.dataset}")
    if config.pruning.enabled:
        logger.logger.info(f"Pruning: {config.pruning.method} ({config.pruning.sparsity:.2%})")
    
    # Create model
    model, image_size = create_model(config)
    model = model.cuda()
    
    # Log model info
    logger.log_model_info(model, 
                         "pruned" if config.pruning.enabled else "dense",
                         config.pruning.sparsity if config.pruning.enabled else None)
    
    # Apply static pruning if needed
    if config.pruning.enabled and config.pruning.method == 'static':
        apply_static_pruning(model, config, logger)
    elif config.pruning.enabled and config.pruning.method in ['dpf', 'dcil']:
        # Initialize all masks to 1 for dynamic methods
        for name, module in model.named_modules():
            if isinstance(module, pruning.dcil.mnn.MaskConv2d):
                module.mask.data.fill_(1.0)
                if config.pruning.method == 'dpf':
                    module.type_value = 6  # MaskerDynamic
                elif config.pruning.method == 'dcil':
                    module.type_value = 0  # Original DCIL masking
    
    model = nn.DataParallel(model)
    cudnn.benchmark = config.system.benchmark
    
    # Create data loaders
    train_loader, val_loader = DataLoader(
        config.data.batch_size, config.data.dataset, config.data.workers,
        config.data.datapath, image_size, True
    )
    
    # Setup training
    criterion, optimizer, scheduler = setup_training(model, config)
    
    # Training loop
    best_acc1 = 0.0
    iteration_counter = [0]  # Use list for modification in nested function
    
    logger.logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(config.training.epochs):
        logger.logger.info(f'\nEpoch: {epoch}, lr = {optimizer.param_groups[0]["lr"]}')
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, epoch, 
            config, logger, iteration_counter
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, config, logger)
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Log epoch results
        logger.log_epoch(epoch, train_metrics, val_metrics, optimizer.param_groups[0]["lr"])
        
        # Save best model
        is_best = val_metrics['acc1'] > best_acc1
        best_acc1 = max(val_metrics['acc1'], best_acc1)
        
        if is_best or epoch % config.system.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'config': config.to_dict(),
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'iteration': iteration_counter[0],
            }
            
            if is_best:
                torch.save(checkpoint, os.path.join(save_path, 'best_model.pth'))
                logger.save_checkpoint_info(epoch, best_acc1, 'best_model.pth')
            
            torch.save(checkpoint, os.path.join(save_path, 'checkpoint.pth'))
    
    # Training completed
    training_time = time.time() - start_time
    logger.log_timing('training', training_time)
    
    logger.logger.info(f'\nTraining completed!')
    logger.logger.info(f'Best accuracy: {best_acc1:.4f}')
    logger.logger.info(f'Total training time: {training_time/3600:.2f} hours')
    
    # Finalize logging
    logger.finalize()
    
    return best_acc1

if __name__ == '__main__':
    main()