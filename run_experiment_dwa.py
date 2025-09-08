#!/usr/bin/env python3
"""
DWA Experiment Runner (official-style loop)
- 3 modes: reactivate_only, kill_active_plain_dead, kill_and_reactivate
- Gradual sparsity + prune_freq + threshold/mask update via pruning.*
"""
import os, time, pathlib
from os.path import isfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import models
import pruning
from utils import *
from common_utils import *
from data import DataLoader

from configs.config import parse_config_args, setup_reproducibility

# --------- helpers ---------
def _iter_mask_convs(model):
    net = model.module if hasattr(model, 'module') else model
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d) and hasattr(module, 'mask'):
            yield name, module

# --------- core ---------
def main():
    cfg = parse_config_args()
    setup_reproducibility(cfg.system)

    if cfg.system.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(cfg.system.gpu)

    # logger
    arch_name = set_arch_name(argize(cfg))  # util이 요구하는 형태로 변환
    logger = SummaryLogger(os.path.join(str(cfg.pruning.sparsity), cfg.name))

    # txt log
    log_file_path = os.path.join("txt_logs", str(cfg.pruning.sparsity),
                                 str(cfg.training.warmup_lr_epoch), f"{cfg.name}_acc_log.txt")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    if not os.path.exists(log_file_path):
        with open(log_file_path, "w") as f:
            f.write("epoch\tacc1_train\tacc1_valid\tbest_acc1\n")

    # model
    print("\n=> creating model '{}'".format(arch_name))
    if not cfg.pruning.enabled:
        model, image_size = models.__dict__[cfg.model.arch](
            data=cfg.data.dataset, num_layers=cfg.model.layers,
            width_mult=cfg.model.width_mult, depth_mult=cfg.model.depth_mult,
            model_mult=cfg.model.model_mult,
        )
    else:
        pruner = pruning.__dict__['dcil'] if cfg.pruning.method == 'dcil' else pruning.__dict__[cfg.pruning.method]
        # pruner.mnn 안에 MaskConv2d 가 있고, forward_type 문자열을 인식해야 함
        model, image_size = pruning.models.__dict__[cfg.model.arch](
            data=cfg.data.dataset, num_layers=cfg.model.layers,
            width_mult=cfg.model.width_mult, depth_mult=cfg.model.depth_mult,
            model_mult=cfg.model.model_mult, mnn=pruner.mnn
        )
    assert model is not None, "Unavailable model parameters!! exit...\n"

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.training.lr,
                          momentum=cfg.training.momentum,
                          weight_decay=cfg.training.weight_decay,
                          nesterov=cfg.training.nesterov)
    scheduler = set_scheduler(optimizer, argize(cfg))

    if torch.cuda.is_available():
        with torch.cuda.device(cfg.system.gpu):
            model = model.cuda(); criterion = criterion.cuda()
        model = nn.DataParallel(model, device_ids=[cfg.system.gpu])  # 단일 GPU면 자동 1장
        cudnn.benchmark = cfg.system.benchmark

    # data
    print("==> Load data..")
    t0 = time.time()
    train_loader, val_loader = DataLoader(
        cfg.data.batch_size, cfg.data.dataset,
        cfg.data.workers, cfg.data.datapath,
        image_size, torch.cuda.is_available()
    )
    print("===> Data loading time: {:,}m {:.2f}s".format(int((time.time()-t0)//60),(time.time()-t0)%60))
    print("===> Data loaded..")

    # optionally resume
    # (공식 스크립트 기준으로 생략: cfg에 별도 load옵션 없으므로)

    # train
    if True:
        best_acc1 = _run_train(cfg, model, train_loader, val_loader,
                               criterion, optimizer, scheduler, arch_name, logger, log_file_path)
        return best_acc1

def _run_train(cfg, model, train_loader, val_loader, criterion, optimizer, scheduler, arch_name, logger, log_file_path):
    start_epoch = 0
    best_acc1 = 0.0
    train_time = 0.0
    validate_time = 0.0
    global iterations
    iterations = 0

    for epoch in range(start_epoch, cfg.training.epochs):
        print("\n==> {}/{} training".format(arch_name, cfg.data.dataset))
        print("==> Epoch: {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

        # ---- Train ----
        print("===> [ Training ]")
        t0 = time.time()
        acc1_train, acc5_train = train_one_epoch(cfg, train_loader, epoch, model, criterion, optimizer)
        train_time += (time.time() - t0)
        print("====> {:.2f} seconds to train this epoch\n".format(time.time()-t0))

        # ---- Validate ----
        print("===> [ Validation ]")
        t0 = time.time()
        acc1_valid, acc5_valid = validate(cfg, val_loader, epoch, model, criterion)
        validate_time += (time.time() - t0)
        print("====> {:.2f} seconds to validate this epoch".format(time.time()-t0))

        # ---- LR schedule ----
        if scheduler is not None:
            scheduler.step()

        # ---- Logging / Save ----
        acc1_train, acc5_train = round(acc1_train.item(),4), round(acc5_train.item(),4)
        acc1_valid, acc5_valid = round(acc1_valid.item(),4), round(acc5_valid.item(),4)

        if acc1_valid > best_acc1 and (epoch >= cfg.pruning.target_epoch or cfg.pruning.sparsity == 0):
            best_acc1 = acc1_valid
            save_model(arch_name, cfg.data.dataset, model.state_dict(), cfg.name)

        train_log = {"acc1": acc1_train, "acc5": acc5_train}
        valid_log = {"best": best_acc1, "acc1": acc1_valid, "acc5": acc5_valid, "lr": optimizer.param_groups[0]["lr"]}
        logger.add_scalar_group("train", train_log, epoch)
        logger.add_scalar_group("test", valid_log, epoch)
        with open(log_file_path, "a") as f:
            f.write(f"{epoch + 1}\t{acc1_train}\t{acc1_valid}\t{best_acc1}\n")

        if cfg.pruning.enabled:
            num_total, num_zero, sparsity = pruning.cal_sparsity(model)
            print("\n====> sparsity: {:.2f}% || num_zero/num_total: {}/{}".format(sparsity, num_zero, num_total))
        print()

    # ---- time summary ----
    avg_train_time = train_time / (cfg.training.epochs - start_epoch)
    avg_valid_time = validate_time / (cfg.training.epochs - start_epoch)
    total_train_time = train_time + validate_time
    print("====> average training time each epoch: {:,}m {:.2f}s".format(int(avg_train_time//60), avg_train_time%60))
    print("====> average validation time each epoch: {:,}m {:.2f}s".format(int(avg_valid_time//60), avg_valid_time%60))
    print("====> training time: {}h {}m {:.2f}s".format(int(train_time//3600), int((train_time%3600)//60), train_time%60))
    print("====> validation time: {}h {}m {:.2f}s".format(int(validate_time//3600), int((validate_time%3600)//60), validate_time%60))
    print("====> total training time: {}h {}m {:.2f}s".format(int(total_train_time//3600), int((total_train_time%3600)//60), total_train_time%60))
    return best_acc1

def train_one_epoch(cfg, train_loader, epoch, model, criterion, optimizer):
    global iterations
    th = None

    batch_time = AverageMeter("Time", ":6.3f")
    data_time  = AverageMeter("Data", ":6.3f")
    losses     = AverageMeter("Loss", ":.4e")
    top1       = AverageMeter("Acc@1", ":6.2f")
    top5       = AverageMeter("Acc@5", ":6.2f")
    progress   = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1, top5, prefix=f"Epoch: [{epoch}]")

    model.train()
    end = time.time()

    for i, (inp, tgt) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            tgt = tgt.cuda(non_blocking=True)

        # ---- pruning schedule + frequency (공식) ----
        if cfg.pruning.enabled:
            target_sparsity = (
                cfg.pruning.sparsity if epoch > cfg.pruning.target_epoch
                else cfg.pruning.sparsity - cfg.pruning.sparsity * (1 - epoch / cfg.pruning.target_epoch) ** 3
            )
            if epoch < cfg.pruning.freeze_epoch:
                if iterations % cfg.pruning.prune_freq == 0:
                    if cfg.pruning.prune_type == "unstructured":
                        threshold = pruning.get_weight_threshold(model, target_sparsity, argize(cfg))
                        # 방어: 마스크 레이어 없으면 skip
                        if threshold is not None:
                            pruning.weight_prune(model, threshold, argize(cfg))
                            th = threshold

        # ---- forward_type 결정 ----
        if cfg.pruning.enabled:
            if (th is None) or (epoch <= cfg.training.warmup_lr_epoch) or (epoch >= cfg.pruning.freeze_epoch):
                forward_type = "DPF"  # dead도 gradient 살려둠
            else:
                forward_type = cfg.pruning.dwa_mode or "scaling"
        else:
            forward_type = None  # dense

        # ---- forward 호출 ----
        if forward_type is None:  # dense
            out = model(inp)
        elif forward_type == "DPF":
            out = model(inp, "DPF")
        elif forward_type == "static":
            out = model(inp, "static")
        else:
            # scaling 류는 τ 필요, DWA 3모드는 α/β/τ 필요
            if forward_type == "scaling":
                out = model(inp, forward_type, th)
            else:
                out = model(inp, forward_type, cfg.pruning.dwa_alpha, cfg.pruning.dwa_beta, th)

        loss = criterion(out, tgt)

        acc1, acc5 = accuracy(out, tgt, topk=(1, 5))
        losses.update(loss.item(), inp.size(0))
        top1.update(acc1[0], inp.size(0))
        top5.update(acc5[0], inp.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        if i % argize(cfg).print_freq == 0:
            progress.print(i)
        end = time.time()
        iterations += 1

    print("====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5))
    return top1.avg, top5.avg

def validate(cfg, val_loader, epoch, model, criterion):
    batch_time = AverageMeter("Time", ":6.3f")
    losses     = AverageMeter("Loss", ":.4e")
    top1       = AverageMeter("Acc@1", ":6.2f")
    top5       = AverageMeter("Acc@5", ":6.2f")
    progress   = ProgressMeter(len(val_loader), batch_time, losses, top1, top5, prefix="Test: ")

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (inp, tgt) in enumerate(val_loader):
            if torch.cuda.is_available():
                tgt = tgt.cuda(non_blocking=True)
            # 공식 평가는 static 경로
            out = model(inp, "static") if cfg.pruning.enabled else model(inp)
            loss = criterion(out, tgt)
            acc1, acc5 = accuracy(out, tgt, topk=(1, 5))
            losses.update(loss.item(), inp.size(0))
            top1.update(acc1[0], inp.size(0))
            top5.update(acc5[0], inp.size(0))
            batch_time.update(time.time() - end)
            if i % argize(cfg).print_freq == 0:
                progress.print(i)
            end = time.time()
        print("====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5))
    return top1.avg, top5.avg

# ---- utils: 기존 utils.*가 argparse.Namespace를 기대하므로 간단 변환 ----
class _NS:
    def __init__(self, **kw): self.__dict__.update(kw)
def argize(cfg):
    # utils.set_scheduler, pruning.* 등에서 접근하는 필드 최소 셋만 변환
    return _NS(
        # training
        epochs=cfg.training.epochs, lr=cfg.training.lr, momentum=cfg.training.momentum,
        weight_decay=cfg.training.weight_decay, nesterov=cfg.training.nesterov,
        scheduler=cfg.training.scheduler, milestones=cfg.training.milestones,
        gamma=cfg.training.gamma, step_size=cfg.training.step_size,
        warmup_epoch=cfg.training.warmup_lr_epoch,
        # pruning
        prune=cfg.pruning.enabled, prune_rate=cfg.pruning.sparsity,
        prune_freq=cfg.pruning.prune_freq, target_epoch=cfg.pruning.target_epoch,
        freeze_epoch=cfg.pruning.freeze_epoch, prune_type=cfg.pruning.prune_type,
        # system
        print_freq=cfg.system.print_freq,
        # misc (utils가 참조할 수 있는 필드 대비)
        dataset=cfg.data.dataset, arch=cfg.model.arch, layers=cfg.model.layers,
        width_mult=cfg.model.width_mult, depth_mult=cfg.model.depth_mult, model_mult=cfg.model.model_mult,
    )

if __name__ == "__main__":
    main()