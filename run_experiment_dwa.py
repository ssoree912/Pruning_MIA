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
import numpy as np

# Import wandb for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

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

def apply_dynamic_pruning(model, config, epoch, iteration, logger):
    """
    Dynamic pruning during training
    Returns:
        actual_sparsity, reactivation_rate, global_threshold(float), target_sparsity
    Also logs step-level metrics to wandb (if enabled).
    """
    if iteration % config.pruning.prune_freq != 0:
        return 0.0, 0.0, None, None

    # Gradual (polynomial) sparsity schedule (Algorithm 1의 pe)
    if epoch >= config.pruning.target_epoch:
        target_sparsity = config.pruning.sparsity
    else:
        progress = epoch / max(1, config.pruning.target_epoch)
        target_sparsity = config.pruning.sparsity * (1 - (1 - progress) ** 3)

    # 1) 모든 가중치 수집
    all_weights = []
    modules_to_prune = []
    net = model.module if hasattr(model, 'module') else model
    for name, module in net.named_modules():
        if isinstance(module, torch.nn.Conv2d) and hasattr(module, 'mask'):
            all_weights.append(module.weight.data.abs().view(-1))
            modules_to_prune.append(module)

    if not all_weights:
        if iteration % (config.pruning.prune_freq * 10) == 0:
            logger.logger.warning("No masked Conv2d layers found for dynamic pruning — skipping.")
        return 0.0, 0.0, None, target_sparsity

    # 2) Global threshold 계산 (Eq.2 의 τ_t 역할; 여기서는 전-모듈 기준 quantile)
    all_weights = torch.cat(all_weights)
    global_threshold = torch.quantile(all_weights, target_sparsity)

    # 3) 마스크 업데이트 (Eq.4)
    total_params = 0
    pruned_params = 0
    reactivations = 0

    for module in modules_to_prune:
        weights = module.weight.data.abs()
        old_mask = module.mask.data.clone()
        new_mask = (weights > global_threshold).float()  # magnitude 기준
        reactivations += ((old_mask == 0) & (new_mask == 1)).sum().item()
        module.mask.data = new_mask
        total_params += module.weight.numel()
        pruned_params += (new_mask == 0).sum().item()

    actual_sparsity = pruned_params / total_params
    reactivation_rate = reactivations / total_params

    # (선택) DWA threshold(τ)도 주기적으로 갱신: 각 레이어 퍼센타일 기반
    try:
        from pruning.dcil.mnn_dwa import MaskConv2dDWA
        tau_list = []
        for name, module in net.named_modules():
            if isinstance(module, MaskConv2dDWA):
                # 논문식 weight alignment에 쓰일 τ는 레이어별 percentile로 갱신
                module.update_threshold(config.pruning.dwa_threshold_percentile)
                tau_list.append(module.threshold.item())
        # 평균/표준편차도 참고용으로 남길 수 있음(원하면 wandb에 추가)
        tau_mean = float(np.mean(tau_list)) if tau_list else None
        tau_std  = float(np.std(tau_list)) if tau_list else None
    except Exception:
        tau_mean = None
        tau_std  = None

    # 로그 (너무 자주 찍지 않게, prune step에서만)
    if WANDB_AVAILABLE and hasattr(config, 'wandb') and getattr(config.wandb, 'enabled', False):
        log_payload = {
            'prune/iteration': iteration,
            'prune/target_sparsity': float(target_sparsity),   # 스케줄 값(pe)
            'prune/actual_sparsity': float(actual_sparsity),   # 실제 적용 결과
            'prune/global_threshold': float(global_threshold.item()),
            'prune/reactivation_rate': float(reactivation_rate),
            'prune/reactivated_params': int(reactivations),
            'prune/total_params_masked_layers': int(total_params),
        }
        if tau_mean is not None:
            log_payload['dwa/threshold_mean'] = tau_mean
            log_payload['dwa/threshold_std']  = tau_std
        try:
            wandb.log(log_payload, step=iteration)
        except Exception:
            pass

    if iteration % (config.pruning.prune_freq * 10) == 0:
        print(f"Dynamic pruning: actual_sparsity={actual_sparsity:.3f}, reactivation_rate={reactivation_rate:.3f}")

    return actual_sparsity, reactivation_rate, float(global_threshold.item()), float(target_sparsity)

# --------- core ---------
def main():
    cfg = parse_config_args()
    setup_reproducibility(cfg.system)

    if cfg.system.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(cfg.system.gpu)

    # Create experiment directory structure (like existing runs)
    arch_name = set_arch_name(argize(cfg))  # util이 요구하는 형태로 변환
    
    # Use save_dir from config (set by train_dwa.py orchestrator)
    if hasattr(cfg, 'save_dir') and cfg.save_dir:
        experiment_dir = pathlib.Path(cfg.save_dir)
    else:
        # Fallback to default structure
        experiment_dir = pathlib.Path("runs") / "dwa" / f"sparsity_{cfg.pruning.sparsity}" / cfg.data.dataset
    
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Experiment directory: {experiment_dir}")
    
    # logger (tensorboard logs go to experiment_dir/logs)
    logs_dir = experiment_dir / "logs"
    logger = SummaryLogger(str(logs_dir))
    
    # Initialize wandb if enabled
    if WANDB_AVAILABLE and cfg.wandb.enabled:
        try:
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=cfg.wandb.name or cfg.name,
                tags=cfg.wandb.tags,
                notes=cfg.wandb.notes,
                config=cfg.to_dict()
            )
            print(f"✅ Wandb initialized: project={cfg.wandb.project}, name={cfg.wandb.name or cfg.name}")
        except Exception as e:
            print(f"❌ Wandb initialization failed: {e}")
            cfg.wandb.enabled = False
    else:
        if not WANDB_AVAILABLE:
            print("⚠️ Wandb not available - install with: pip install wandb")
        else:
            print("ℹ️ Wandb disabled")
        cfg.wandb.enabled = False

    # txt log (goes to experiment_dir)
    log_file_path = experiment_dir / f"{cfg.name}_acc_log.txt"
    if not log_file_path.exists():
        with open(log_file_path, "w") as f:
            f.write("epoch\tacc1_train\tacc1_valid\tbest_acc1\n")
    
    # Save config files to experiment directory
    cfg.to_json(str(experiment_dir / "config.json"))
    cfg.to_yaml(str(experiment_dir / "config.yaml"))

    # model
    print("\n=> creating model '{}'".format(arch_name))
    if not cfg.pruning.enabled:
        model, image_size = models.__dict__[cfg.model.arch](
            data=cfg.data.dataset, num_layers=cfg.model.layers,
            width_mult=cfg.model.width_mult, depth_mult=cfg.model.depth_mult,
            model_mult=cfg.model.model_mult,
        )
    else:
        pruner_key = (cfg.pruning.method or '').lower()
        # dpf/dwa/static 모두 dcil 백엔드의 MaskConv2d를 공유해도 무방 (forward_type으로 동작 분기)
        if pruner_key in ('dpf', 'dwa', 'static'):
            pruner_key = 'dcil'
        try:
            pruner = pruning.__dict__[pruner_key]
        except KeyError as e:
            raise KeyError(
                f"Unknown pruner '{cfg.pruning.method}'. "
                f"Use one of: {', '.join(sorted(k for k in pruning.__dict__.keys() if not k.startswith('_')))}"
            ) from e

        model, image_size = pruning.models.__dict__[cfg.model.arch](
            data=cfg.data.dataset,
            num_layers=cfg.model.layers,
            width_mult=cfg.model.width_mult,
            depth_mult=cfg.model.depth_mult,
            model_mult=cfg.model.model_mult,
            mnn=pruner.mnn,   # 여기로 dcil의 mnn 주입
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
                               criterion, optimizer, scheduler, arch_name, logger, log_file_path, experiment_dir)
        
        # Finish wandb run
        if WANDB_AVAILABLE and cfg.wandb.enabled:
            try:
                wandb.finish()
                print("✅ Wandb run finished")
            except Exception as e:
                print(f"⚠️ Wandb finish failed: {e}")
        
        return best_acc1

def _run_train(cfg, model, train_loader, val_loader, criterion, optimizer, scheduler, arch_name, logger, log_file_path, experiment_dir):
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
        train_metrics = train_one_epoch(cfg, train_loader, epoch, model, criterion, optimizer, logger)
        train_time += (time.time() - t0)
        print("====> {:.2f} seconds to train this epoch\n".format(time.time()-t0))

        # ---- Validate ----
        print("===> [ Validation ]")
        t0 = time.time()
        val_metrics = validate(cfg, val_loader, epoch, model, criterion)
        validate_time += (time.time() - t0)
        print("====> {:.2f} seconds to validate this epoch".format(time.time()-t0))

        # ---- LR schedule ----
        if scheduler is not None:
            scheduler.step()

        # ---- Logging / Save ----
        acc1_train, acc5_train = round(train_metrics['acc1'],4), round(train_metrics['acc5'],4)
        acc1_valid, acc5_valid = round(val_metrics['acc1'],4), round(val_metrics['acc5'],4)

        if acc1_valid > best_acc1 and (epoch >= cfg.pruning.target_epoch or cfg.pruning.sparsity == 0):
            best_acc1 = acc1_valid
            # Save model to experiment directory
            model_path = experiment_dir / "best_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f"💾 Best model saved: {model_path}")

        train_log = {"acc1": acc1_train, "acc5": acc5_train}
        valid_log = {"best": best_acc1, "acc1": acc1_valid, "acc5": acc5_valid, "lr": optimizer.param_groups[0]["lr"]}
        logger.add_scalar_group("train", train_log, epoch)
        logger.add_scalar_group("test", valid_log, epoch)
        
        # ---- Wandb logging ----
        if hasattr(cfg, 'wandb') and getattr(cfg.wandb, 'enabled', False) and WANDB_AVAILABLE:
            wandb_log = {
                'epoch': epoch,
                'learning_rate': optimizer.param_groups[0]["lr"],
                # 학습/검증
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **{f'val/{k}': v for k, v in val_metrics.items()},
                # prune 관련 에폭 평균(가독성 위해 별도 네임스페이스도 추가)
                'prune/epoch_avg_actual_sparsity': train_metrics.get('sparsity', 0.0),
                'prune/epoch_avg_target_sparsity': train_metrics.get('target_sparsity', 0.0),
                'prune/epoch_avg_threshold': train_metrics.get('global_threshold', 0.0),
                'prune/epoch_avg_sparsity_delta': train_metrics.get('sparsity_delta', 0.0),
                'prune/epoch_avg_reactivation_rate': train_metrics.get('reactivation_rate', 0.0),
            }
            try:
                wandb.log(wandb_log, step=iterations)
            except Exception as e:
                print(f"Warning: wandb logging failed: {e}")
        
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
    
    # Save experiment summary
    summary = {
        "best_accuracy": float(best_acc1),
        "total_epochs": cfg.training.epochs,
        "total_training_time_seconds": float(total_train_time),
        "avg_epoch_time_seconds": float(avg_train_time + avg_valid_time),
        "experiment_name": cfg.name,
        "dataset": cfg.data.dataset,
        "architecture": arch_name,
        "sparsity": cfg.pruning.sparsity if cfg.pruning.enabled else 0.0,
        "dwa_mode": getattr(cfg.pruning, 'dwa_mode', None),
    }
    
    import json
    with open(experiment_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"📊 Experiment summary saved: {experiment_dir / 'experiment_summary.json'}")
    return best_acc1

def train_one_epoch(cfg, train_loader, epoch, model, criterion, optimizer, logger):
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

    # prune step 관측값 축적
    sparsity_updates = []
    sparsity_deltas = []
    reactivation_updates = []
    threshold_updates = []
    target_sparsity_updates = []
    
    last_actual_sparsity = None

    for i, (inp, tgt) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            tgt = tgt.cuda(non_blocking=True)

        # ---- Dynamic pruning update (마스크 미-프리즈 시) ----
        if (cfg.pruning.enabled and 
            (cfg.pruning.freeze_epoch < 0 or epoch < cfg.pruning.freeze_epoch)):

            actual_sparsity, reactivation, g_thr, tgt_s = apply_dynamic_pruning(
                model, cfg, epoch, iterations, logger
            )
            if tgt_s is not None:
                target_sparsity_updates.append(tgt_s)
            if g_thr is not None:
                threshold_updates.append(g_thr)
            if actual_sparsity > 0:
                sparsity_updates.append(actual_sparsity)
                reactivation_updates.append(reactivation)
                if last_actual_sparsity is not None:
                    sparsity_deltas.append(actual_sparsity - last_actual_sparsity)
                last_actual_sparsity = actual_sparsity

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

    # 에폭 요약(평균)
    avg_sparsity = float(np.mean(sparsity_updates)) if sparsity_updates else 0.0
    avg_reactivation = float(np.mean(reactivation_updates)) if reactivation_updates else 0.0
    avg_threshold = float(np.mean(threshold_updates)) if threshold_updates else 0.0
    avg_target_sparsity = float(np.mean(target_sparsity_updates)) if target_sparsity_updates else 0.0
    avg_sparsity_delta = float(np.mean(sparsity_deltas)) if sparsity_deltas else 0.0

    metrics = {
        'acc1': top1.avg.item(),
        'acc5': top5.avg.item(),
        'loss': losses.avg,
        'sparsity': avg_sparsity,
        'reactivation_rate': avg_reactivation,
        'global_threshold': avg_threshold,
        'target_sparsity': avg_target_sparsity,
        'sparsity_delta': avg_sparsity_delta,
    }

    print("====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5))
    return metrics

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
    
    return {
        'acc1': top1.avg.item(),
        'acc5': top5.avg.item(),
        'loss': losses.avg
    }

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