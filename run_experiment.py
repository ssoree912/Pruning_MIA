#!/usr/bin/env python3
"""
Main experiment runner (dense-only).

This branch no longer runs static/dynamic pruning in training.
"""

import os
import sys
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import parse_config_args, setup_reproducibility
from data import DataLoader
from utils.logger import ExperimentLogger, get_system_info
from utils.utils import TrainAverageMeter as AverageMeter
from utils.utils import accuracy
import models


def create_model(config):
    """Create dense model based on configuration."""
    model, image_size = models.__dict__[config.model.arch](
        data=config.data.dataset,
        num_layers=config.model.layers,
        width_mult=config.model.width_mult,
        depth_mult=config.model.depth_mult,
        model_mult=config.model.model_mult,
    )
    if model is None:
        raise ValueError(
            f"Failed to create model: {config.model.arch} with {config.model.layers} layers for {config.data.dataset}"
        )
    return model, image_size


def setup_training(model, config, start_epoch: int = 0):
    """Setup loss/optimizer/scheduler."""
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.SGD(
        model.parameters(),
        lr=config.training.lr,
        momentum=config.training.momentum,
        weight_decay=config.training.weight_decay,
        nesterov=config.training.nesterov,
    )
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])

    last_epoch = start_epoch - 1
    if config.training.scheduler == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.training.milestones,
            gamma=config.training.gamma,
            last_epoch=last_epoch,
        )
    elif config.training.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.training.step_size,
            gamma=config.training.gamma,
            last_epoch=last_epoch,
        )
    elif config.training.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs,
            last_epoch=last_epoch,
        )
    elif config.training.scheduler == "exp":
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.training.gamma,
            last_epoch=last_epoch,
        )
    else:
        scheduler = None

    return criterion, optimizer, scheduler


def train_epoch(model, train_loader, criterion, optimizer, epoch, config, logger, iteration_counter):
    """Train for one epoch."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

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

        if i % config.system.print_freq == 0:
            logger.logger.info(
                f"Epoch: [{epoch}][{i}/{len(train_loader)}] "
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                f"Loss {losses.val:.4f} ({losses.avg:.4f}) "
                f"Acc@1 {top1.val:.3f} ({top1.avg:.3f}) "
                f"Acc@5 {top5.val:.3f} ({top5.avg:.3f})"
            )

        iteration_counter[0] += 1
        end = time.time()

    return {
        "acc1": top1.avg.item(),
        "acc5": top5.avg.item(),
        "loss": losses.avg,
    }


def validate(model, val_loader, criterion, config, logger):
    """Validate model."""
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

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

            if i % config.system.print_freq == 0:
                logger.logger.info(
                    f"Test: [{i}/{len(val_loader)}] "
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                    f"Loss {losses.val:.4f} ({losses.avg:.4f}) "
                    f"Acc@1 {top1.val:.3f} ({top1.avg:.3f}) "
                    f"Acc@5 {top5.val:.3f} ({top5.avg:.3f})"
                )

            end = time.time()

    return {
        "acc1": top1.avg.item(),
        "acc5": top5.avg.item(),
        "loss": losses.avg,
    }


def main():
    config = parse_config_args()

    # Setup reproducibility (optionally decouple init/data seeds for model merging)
    init_seed = config.system.init_seed if config.system.init_seed is not None else config.system.seed
    setup_reproducibility(config.system, seed_override=init_seed)

    save_path = config.get_save_path()
    os.makedirs(save_path, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.system.gpu)
    torch.cuda.set_device(0)

    logger = ExperimentLogger(config.name, save_path)
    logger.log_hyperparameters(config.to_dict())
    logger.log_system_info(get_system_info())

    config.to_yaml(os.path.join(save_path, "config.yaml"))
    config.to_json(os.path.join(save_path, "config.json"))

    logger.logger.info(f"Starting experiment: {config.name}")
    logger.logger.info(f"Model: {config.model.arch}-{config.model.layers}")
    logger.logger.info(f"Dataset: {config.data.dataset}")
    logger.logger.info("Mode: dense-only (pruning disabled in this branch)")

    model, image_size = create_model(config)
    model = model.cuda()

    logger.log_model_info(model, "dense", None)
    torch.save({"state_dict": model.state_dict()}, os.path.join(save_path, "init_model.pth"))

    is_resuming = getattr(config.system, "resume", None) is not None
    resume_path = Path(config.system.resume) if is_resuming else None

    # Reseed before data loader / training to allow shared init but different SGD noise
    data_seed = config.system.data_seed if config.system.data_seed is not None else config.system.seed
    if data_seed != init_seed:
        setup_reproducibility(config.system, seed_override=data_seed)

    model = nn.DataParallel(model)
    cudnn.benchmark = config.system.benchmark

    train_loader, val_loader = DataLoader(
        config.data.batch_size,
        config.data.dataset,
        config.data.workers,
        config.data.datapath,
        image_size,
        True,
    )

    start_epoch = getattr(config.training, "start_epoch", 0)
    criterion, optimizer, scheduler = setup_training(model, config, start_epoch=start_epoch)

    best_acc1 = 0.0
    iteration_counter = [0]

    if is_resuming:
        assert resume_path is not None and resume_path.exists(), f"resume not found: {resume_path}"
        ckpt = torch.load(str(resume_path), map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=True)
        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if start_epoch == 0 and "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        if "best_acc1" in ckpt:
            best_acc1 = float(ckpt["best_acc1"])
        if "iteration" in ckpt:
            iteration_counter[0] = int(ckpt["iteration"])
        logger.logger.info(
            f"[RESUME] loaded={resume_path} start_epoch={start_epoch} "
            f"best_acc1={best_acc1:.4f} iter={iteration_counter[0]}"
        )

    logger.logger.info("Starting training...")
    start_time = time.time()
    save_split_epoch = getattr(config.system, "save_split_ckpt_epoch", None)

    for epoch in range(start_epoch, config.training.epochs):
        logger.logger.info(f"\nEpoch: {epoch}, lr = {optimizer.param_groups[0]['lr']}")

        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            epoch,
            config,
            logger,
            iteration_counter,
        )
        val_metrics = validate(model, val_loader, criterion, config, logger)

        if scheduler:
            scheduler.step()

        logger.log_epoch(epoch, train_metrics, val_metrics, optimizer.param_groups[0]["lr"])

        remaining = config.training.epochs - (epoch + 1)
        logger.logger.info(
            f"Remaining epochs: {remaining} | "
            f"Train acc1: {train_metrics.get('acc1', 0):.3f}, loss: {train_metrics.get('loss', 0):.4f} | "
            f"Val acc1: {val_metrics.get('acc1', 0):.3f}, loss: {val_metrics.get('loss', 0):.4f}"
        )

        is_best = val_metrics["acc1"] > best_acc1
        best_acc1 = max(val_metrics["acc1"], best_acc1)

        if is_best or epoch % config.system.save_freq == 0:
            checkpoint = {
                "epoch": epoch,
                "config": config.to_dict(),
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "optimizer": optimizer.state_dict(),
                "iteration": iteration_counter[0],
            }

            if is_best:
                torch.save(checkpoint, os.path.join(save_path, "best_model.pth"))
                logger.save_checkpoint_info(epoch, best_acc1, "best_model.pth")

            torch.save(checkpoint, os.path.join(save_path, "checkpoint.pth"))

        if save_split_epoch is not None and epoch == int(save_split_epoch):
            split_path = os.path.join(save_path, f"split_ckpt_epoch{epoch}.pth")
            checkpoint = {
                "epoch": epoch,
                "config": config.to_dict(),
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "optimizer": optimizer.state_dict(),
                "iteration": iteration_counter[0],
            }
            torch.save(checkpoint, split_path)
            logger.logger.info(f"[SPLIT] saved split ckpt at {split_path} and exiting")
            logger.finalize()
            return best_acc1

    training_time = time.time() - start_time
    logger.log_timing("training", training_time)

    logger.logger.info("\nTraining completed!")
    logger.logger.info(f"Best accuracy: {best_acc1:.4f}")
    logger.logger.info(f"Total training time: {training_time / 3600:.2f} hours")

    logger.finalize()
    return best_acc1


if __name__ == "__main__":
    main()

