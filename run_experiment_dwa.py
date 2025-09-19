#!/usr/bin/env python3
"""
DWA Experiment Runner (official-style loop)
- 3 modes: reactivate_only, kill_active_plain_dead, kill_and_reactivate
- Gradual sparsity + prune_freq + threshold/mask update via pruning.*
"""
import os, time, pathlib, json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np

# wandb (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import models
import pruning
from utils.utils import *    # AverageMeter/ProgressMeter/accuracy/schedulers/SummaryLogger
# Compatibility aliases: older code expects AverageMeter/ProgressMeter symbols
try:
    AverageMeter  # type: ignore[name-defined]
except NameError:
    from utils.utils import LightAverageMeter as AverageMeter  # fallback
try:
    ProgressMeter  # type: ignore[name-defined]
except NameError:
    from utils.utils import LightProgressMeter as ProgressMeter  # fallback
from data import DataLoader

from configs.config import parse_config_args, setup_reproducibility


# ---------- helpers ----------
def _iter_mask_convs(model):
    net = model.module if hasattr(model, 'module') else model
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d) and hasattr(module, 'mask'):
            yield name, module

def _apply_dwa_to_modules(model, forward_type, alpha=None, beta=None, global_threshold=None):
    """모델 내부 MaskConv2d(DWA) 모듈들에 모드/하이퍼파라미터 주입"""
    net = model.module if hasattr(model, 'module') else model
    for m in net.modules(): #모듈 설정 주입
        if isinstance(m, nn.Conv2d) and hasattr(m, 'mask') and hasattr(m, 'forward_type'):
            m.forward_type = forward_type
            if (alpha is not None) and hasattr(m, 'alpha'):
                m.alpha = float(alpha)
            if (beta is not None) and hasattr(m, 'beta'):
                m.beta = float(beta)
            if (global_threshold is not None) and hasattr(m, 'threshold'):
                with torch.no_grad():
                    m.threshold.data = torch.as_tensor(
                        float(global_threshold), dtype=m.threshold.dtype, device=m.threshold.device
                    )

def _update_dwa_tau(model, percentile):
    """레이어별 DWA threshold(τ) 갱신; 평균/표준편차 반환"""
    net = model.module if hasattr(model, 'module') else model
    tau_list = []
    for _, m in net.named_modules():
        if hasattr(m, "update_threshold") and hasattr(m, "threshold"):
            try:
                m.update_threshold(percentile)
                if m.threshold is not None:
                    tau_list.append(float(m.threshold))
            except Exception:
                pass
    if not tau_list:
        return None, None
    return float(np.mean(tau_list)), float(np.std(tau_list))


# ---------- core ----------
def main():
    cfg = parse_config_args()
    setup_reproducibility(cfg.system)

    if cfg.system.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(cfg.system.gpu)

    # exp dir
    arch_name = set_arch_name(argize(cfg))
    if hasattr(cfg, 'save_dir') and cfg.save_dir:
        experiment_dir = pathlib.Path(cfg.save_dir)
    else:
        experiment_dir = pathlib.Path("runs") / "dwa" / f"sparsity_{cfg.pruning.sparsity}" / cfg.data.dataset
    experiment_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Experiment directory: {experiment_dir}")

    # tensorboard logger
    logs_dir = experiment_dir / "logs"
    logger = SummaryLogger(str(logs_dir))

    # wandb
    if WANDB_AVAILABLE and getattr(cfg.wandb, 'enabled', False):
        try:
            wandb.init(
                project=cfg.wandb.project, entity=cfg.wandb.entity,
                name=cfg.wandb.name or cfg.name, tags=cfg.wandb.tags, notes=cfg.wandb.notes,
                config=cfg.to_dict()
            )
            # 축 정의: prune/* 은 iteration, train/val/* 은 epoch
            wandb.define_metric("iteration")
            wandb.define_metric("prune/*", step_metric="iteration")
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("val/*",   step_metric="epoch")
            print(f"✅ Wandb initialized: project={cfg.wandb.project}, name={cfg.wandb.name or cfg.name}")
        except Exception as e:
            print(f"❌ Wandb initialization failed: {e}")
            cfg.wandb.enabled = False
    else:
        if not WANDB_AVAILABLE:
            print("⚠️ Wandb not available - pip install wandb")
        else:
            print("ℹ️ Wandb disabled")
        cfg.wandb.enabled = False

    # text log
    log_file_path = experiment_dir / f"{cfg.name}_acc_log.txt"
    if not log_file_path.exists():
        with open(log_file_path, "w") as f:
            f.write("epoch\tacc1_train\tacc1_valid\tbest_acc1\n")
    
    # detailed training log
    detailed_log_path = experiment_dir / f"{cfg.name}_training.log"
    
    def log_message(message):
        """Helper function to log messages to both console and file"""
        print(message)
        with open(detailed_log_path, "a") as f:
            f.write(f"{message}\n")

    # save config
    cfg.to_json(str(experiment_dir / "config.json"))
    cfg.to_yaml(str(experiment_dir / "config.yaml"))

    # model
    print(f"\n=> creating model '{arch_name}'")
    if not cfg.pruning.enabled: #model init
        model, image_size = models.__dict__[cfg.model.arch](
            data=cfg.data.dataset, num_layers=cfg.model.layers,
            width_mult=cfg.model.width_mult, depth_mult=cfg.model.depth_mult,
            model_mult=cfg.model.model_mult,
        )
    else:
        pruner_key = (cfg.pruning.method or '').lower() #프루닝 분기
        if pruner_key in ('dpf', 'dwa', 'static'):
            pruner_key = 'dcil'  # 동일 백엔드 사용
        try:
            pruner = pruning.__dict__[pruner_key]
        except KeyError as e:
            raise KeyError(
                f"Unknown pruner '{cfg.pruning.method}'. "
                f"Available: {', '.join(sorted(k for k in pruning.__dict__.keys() if not k.startswith('_')))}"
            ) from e

        model, image_size = pruning.models.__dict__[cfg.model.arch](
            data=cfg.data.dataset, num_layers=cfg.model.layers,
            width_mult=cfg.model.width_mult, depth_mult=cfg.model.depth_mult,
            model_mult=cfg.model.model_mult, mnn=pruner.mnn
        )
    assert model is not None, "Unavailable model parameters!"

    criterion = nn.CrossEntropyLoss() #분류 손실함수
    optimizer = optim.SGD(model.parameters(), lr=cfg.training.lr, #옵티마이저
                          momentum=cfg.training.momentum,
                          weight_decay=cfg.training.weight_decay,
                          nesterov=cfg.training.nesterov)
    scheduler = set_scheduler(optimizer, argize(cfg))

    if torch.cuda.is_available():
        with torch.cuda.device(cfg.system.gpu):
            model = model.cuda(); criterion = criterion.cuda()
        model = nn.DataParallel(model, device_ids=[cfg.system.gpu])
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

    # Initial log
    log_message(f"🚀 Starting DWA experiment: {cfg.name}")
    log_message(f"📊 Dataset: {cfg.data.dataset}, Architecture: {cfg.model.arch}")
    log_message(f"🎯 Sparsity: {cfg.pruning.sparsity}, DWA Mode: {getattr(cfg.pruning, 'dwa_mode', 'None')}")
    log_message(f"⚙️ Alpha: {getattr(cfg.pruning, 'dwa_alpha', 1.0)}, Beta: {getattr(cfg.pruning, 'dwa_beta', 1.0)}")
    
    # train
    best_acc1 = _run_train(cfg, model, train_loader, val_loader,
                           criterion, optimizer, scheduler, arch_name, logger, log_file_path, experiment_dir, log_message)

    if WANDB_AVAILABLE and getattr(cfg.wandb, 'enabled', False):
        try: wandb.finish()
        except Exception: pass

    return best_acc1


def _run_train(cfg, model, train_loader, val_loader, criterion, optimizer, scheduler, arch_name, logger, log_file_path, experiment_dir, log_message):
    start_epoch = 0
    best_acc1 = 0.0
    best_loss = float('inf')
    train_time = 0.0
    validate_time = 0.0
    global iterations
    iterations = 0
    last_threshold = None
    validation_history = []

    for epoch in range(start_epoch, cfg.training.epochs):
        print("\n==> {}/{} training".format(arch_name, cfg.data.dataset))
        print("==> Epoch: {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

        # ---- Train ----
        print("===> [ Training ]")
        t0 = time.time()
        train_metrics, last_threshold = train_one_epoch(
            cfg, train_loader, epoch, model, criterion, optimizer, logger, last_threshold
        )
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

        # Track validation history
        validation_entry = {
            'epoch': epoch,
            'acc1': acc1_valid,
            'acc5': acc5_valid,
            'loss': val_metrics['loss'],
            'train_acc1': acc1_train,
            'train_acc5': acc5_train,
            'train_loss': train_metrics['loss'],
            'lr': optimizer.param_groups[0]["lr"]
        }
        validation_history.append(validation_entry)

        if acc1_valid > best_acc1 and (epoch >= cfg.pruning.target_epoch or cfg.pruning.sparsity == 0):
            best_acc1 = acc1_valid
            best_loss = val_metrics['loss']
            model_path = experiment_dir / "best_model.pth"
            torch.save(model.state_dict(), model_path)
            log_message(f"💾 Best model saved at epoch {epoch}: {model_path} (acc1: {best_acc1:.4f})")

        logger.add_scalar_group("train", {"acc1": acc1_train, "acc5": acc5_train}, epoch)
        logger.add_scalar_group("test",  {"best": best_acc1, "acc1": acc1_valid, "acc5": acc5_valid, "lr": optimizer.param_groups[0]["lr"]}, epoch)

        if getattr(cfg.wandb, 'enabled', False) and WANDB_AVAILABLE:
            wandb_log = {
                'epoch': epoch,
                'learning_rate': optimizer.param_groups[0]["lr"],
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **{f'val/{k}': v for k, v in val_metrics.items()},
                'prune/epoch_avg_actual_sparsity': train_metrics.get('sparsity', 0.0),
                'prune/epoch_avg_target_sparsity': train_metrics.get('target_sparsity', 0.0),
                'prune/epoch_avg_threshold': train_metrics.get('global_threshold', 0.0),
                'prune/epoch_avg_sparsity_delta': train_metrics.get('sparsity_delta', 0.0),
                'prune/epoch_avg_reactivation_rate': train_metrics.get('reactivation_rate', 0.0),
            }
            try: wandb.log(wandb_log, commit=True)
            except Exception as e: print(f"Warning: wandb logging failed: {e}")

        with open(log_file_path, "a") as f:
            f.write(f"{epoch + 1}\t{acc1_train}\t{acc1_valid}\t{best_acc1}\n")

        if cfg.pruning.enabled:
            num_total, num_zero, sparsity = pruning.cal_sparsity(model)
            sparsity_msg = "\n====> sparsity: {:.2f}% || num_zero/num_total: {}/{}".format(sparsity, num_zero, num_total)
            print(sparsity_msg)
            log_message(sparsity_msg.strip())
        print()

    # ---- time summary & save summary ----
    avg_train_time = train_time / (cfg.training.epochs - start_epoch)
    avg_valid_time = validate_time / (cfg.training.epochs - start_epoch)
    total_train_time = train_time + validate_time
    print("====> average training time each epoch: {:,}m {:.2f}s".format(int(avg_train_time//60), avg_train_time%60))
    print("====> average validation time each epoch: {:,}m {:.2f}s".format(int(avg_valid_time//60), avg_valid_time%60))
    print("====> training time: {}h {}m {:.2f}s".format(int(train_time//3600), int((train_time%3600)//60), train_time%60))
    print("====> validation time: {}h {}m {:.2f}s".format(int(validate_time//3600), int((validate_time%3600)//60), validate_time%60))
    print("====> total training time: {}h {}m {:.2f}s".format(int(total_train_time//3600), int((total_train_time%3600)//60), total_train_time%60))

    # Save validation_history.json and experiment_summary.json for aggregation
    try:
        import json as _json
        (experiment_dir / 'validation_history.json').write_text(
            _json.dumps(validation_history, indent=2)
        )
        last_val = validation_history[-1] if validation_history else {}
        summary = {
            'best_metrics': {
                'best_acc1': best_acc1,
                'best_loss': best_loss,
            },
            'final_metrics': {
                'acc1': last_val.get('acc1'),
                'acc5': last_val.get('acc5'),
                'loss': last_val.get('loss'),
            },
            'total_duration': (train_time + validate_time),
            'epochs': getattr(cfg.training, 'epochs', None),
        }
        (experiment_dir / 'experiment_summary.json').write_text(
            _json.dumps(summary, indent=2)
        )
    except Exception as e:
        log_message(f"⚠️ Failed to write history/summary JSON: {e}")

    return best_acc1


def train_one_epoch(cfg, train_loader, epoch, model, criterion, optimizer, logger, last_threshold):
    """
    - 마스크 갱신은 '공식 블록' 한 번만 수행
    - 프루닝 스텝마다: target_sparsity(pe), threshold, actual sparsity, reactivation 등을 W&B에 iteration 축으로 로깅
    """
    global iterations 

    batch_time = AverageMeter("Time", ":6.3f")
    data_time  = AverageMeter("Data", ":6.3f")
    losses     = AverageMeter("Loss", ":.4e")
    top1       = AverageMeter("Acc@1", ":6.2f")
    top5       = AverageMeter("Acc@5", ":6.2f")
    progress   = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1, top5, prefix=f"Epoch: [{epoch}]")

    model.train()
    end = time.time()

    # epoch-avg accumulation
    sparsity_updates, sparsity_deltas = [], []
    reactivation_updates, threshold_updates, target_sparsity_updates = [], [], []
    last_actual_sparsity = None

    num_batches = len(train_loader)

    for i, (inp, tgt) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            tgt = tgt.cuda(non_blocking=True)

        # ---- pruning schedule + frequency (공식) ----
        if cfg.pruning.enabled and (cfg.pruning.freeze_epoch < 0 or epoch < cfg.pruning.freeze_epoch):
            # 점진적 sparsity 증가
            progress_ratio = min(1.0, (epoch + (i + 1) / num_batches) / max(1, cfg.pruning.target_epoch))
            target_sparsity = cfg.pruning.sparsity * (1 - (1 - progress_ratio) ** 3)
            target_sparsity_updates.append(target_sparsity)

            if iterations % cfg.pruning.prune_freq == 0:
                # 1. 기존 마스크 백업
                old_masks = {id(m): m.mask.data.clone() for _, m in _iter_mask_convs(model)}

                # threshold 계산 & 프루닝  
                threshold = pruning.get_weight_threshold(model, target_sparsity, argize(cfg))
                if threshold is not None:
                    pruning.weight_prune(model, threshold, argize(cfg))
                    last_threshold = threshold

                    # 통계: actual sparsity & reactivation
                    total, pruned, reactivated = 0, 0, 0
                    for _, m in _iter_mask_convs(model):
                        new_mask = m.mask.data
                        old_mask = old_masks[id(m)]
                        reactivated += ((old_mask == 0) & (new_mask == 1)).sum().item()
                        pruned     += (new_mask == 0).sum().item()
                        total      += new_mask.numel()

                    actual_sparsity   = (pruned / total) if total else 0.0
                    reactivation_rate = (reactivated / total) if total else 0.0

                    sparsity_updates.append(actual_sparsity)
                    reactivation_updates.append(reactivation_rate)
                    threshold_updates.append(float(threshold))

                    if last_actual_sparsity is not None:
                        sparsity_deltas.append(actual_sparsity - last_actual_sparsity)
                    last_actual_sparsity = actual_sparsity

                    # τ 갱신(레이어별) & 스텝 로깅
                    tau_mean, tau_std = _update_dwa_tau(model, getattr(cfg.pruning, 'dwa_threshold_percentile', 50))

                    if WANDB_AVAILABLE and getattr(cfg.wandb, 'enabled', False):
                        log_payload = {
                            'iteration': iterations,
                            'prune/target_sparsity': float(target_sparsity),
                            'prune/actual_sparsity': float(actual_sparsity),
                            'prune/global_threshold': float(threshold),
                            'prune/reactivation_rate': float(reactivation_rate),
                            'prune/reactivated_params': int(reactivated),
                            'prune/total_params_masked_layers': int(total),
                        }
                        wandb.log(log_payload, commit=False) 
                        if tau_mean is not None:
                            log_payload['dwa/threshold_mean'] = float(tau_mean)
                            log_payload['dwa/threshold_std']  = float(tau_std)
                        try:
                            wandb.log(log_payload, step=iterations, commit=False)
                        except Exception:
                            pass

        # ---- forward_type ----
        if cfg.pruning.enabled:
            # 프루닝이 활성화된 상태에서만 DWA 모드 사용
            freeze_epoch = cfg.pruning.freeze_epoch
            if (last_threshold is None) or (epoch <= cfg.training.warmup_lr_epoch) or ((freeze_epoch >= 0) and (epoch >= freeze_epoch)):
                forward_type = "DPF" #oneshot
            else:
                forward_type = cfg.pruning.dwa_mode or "DPF" #DWA 
        else:
            forward_type = None #dense

        # ---- forward ----
        if forward_type is None:                # dense
            out = model(inp)
        elif forward_type == "DPF":
            out = model(inp, "DPF")             # 기존 레거시 경로 유지
        elif forward_type == "static":
            out = model(inp, "static")          # 검증 등에서 사용
        else:
            # DWA 3모드만 허용
            ALLOWED = {"reactivate_only", "kill_active_plain_dead", "kill_and_reactivate"}
            assert forward_type in ALLOWED, f"Unknown DWA mode: {forward_type}"
            _apply_dwa_to_modules( #모듈 설정 후 
                model, forward_type,
                getattr(cfg.pruning, 'dwa_alpha', 1.0), #default 1.0
                getattr(cfg.pruning, 'dwa_beta', 1.0), #default 1.0
                last_threshold
            )
            # Debug log for DWA activation (첫 번째 배치에서만)
            if i == 0:
                print(f"[DWA ON] mode={forward_type}, alpha={cfg.pruning.dwa_alpha}, beta={cfg.pruning.dwa_beta}, τ≈{threshold_updates[-1] if threshold_updates else last_threshold}")
            out = model(inp,"DPF") #dwa 모드로 호출

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

    # epoch summary
    avg_sparsity        = float(np.mean(sparsity_updates))        if sparsity_updates        else 0.0
    avg_reactivation    = float(np.mean(reactivation_updates))    if reactivation_updates    else 0.0
    avg_threshold       = float(np.mean(threshold_updates))       if threshold_updates       else 0.0
    avg_target_sparsity = float(np.mean(target_sparsity_updates)) if target_sparsity_updates else 0.0
    avg_sparsity_delta  = float(np.mean(sparsity_deltas))         if sparsity_deltas         else 0.0

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

    # Note: Persisting validation history/summary is handled in _run_train() after the loop.
    
    return metrics, last_threshold


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

    return {'acc1': top1.avg.item(), 'acc5': top5.avg.item(), 'loss': losses.avg}


# ---- utils: argparse.Namespace shim ----
class _NS:
    def __init__(self, **kw): self.__dict__.update(kw)

def argize(cfg):
    return _NS(
        # training
        epochs=cfg.training.epochs, lr=cfg.training.lr, momentum=cfg.training.momentum,
        weight_decay=cfg.training.weight_decay, nesterov=cfg.training.nesterov,
        scheduler=cfg.training.scheduler, milestones=cfg.training.milestones,
        gamma=cfg.training.gamma, step_size=cfg.training.step_size,
        warmup_epoch=cfg.training.warmup_lr_epoch,

        # pruning (DWA-only essentials + 필요한 필드 복원)
        prune=cfg.pruning.enabled,
        prune_rate=cfg.pruning.sparsity,
        prune_freq=cfg.pruning.prune_freq,
        target_epoch=cfg.pruning.target_epoch,
        freeze_epoch=cfg.pruning.freeze_epoch,
        
        # 내부 유틸에서 참조
        prune_type="unstructured",  # 명시적으로 고정
        prune_imp=getattr(cfg.pruning, 'importance_method', 'L1'),  # 기본값 L1

        # DWA parameters
        dwa_mode=getattr(cfg.pruning, 'dwa_mode', None),
        dwa_alpha=getattr(cfg.pruning, 'dwa_alpha', 1.0),
        dwa_beta=getattr(cfg.pruning, 'dwa_beta', 1.0),
        dwa_threshold_percentile=getattr(cfg.pruning, 'dwa_threshold_percentile', 50),

        # system
        print_freq=cfg.system.print_freq,

        # model
        dataset=cfg.data.dataset, arch=cfg.model.arch, layers=cfg.model.layers,
        width_mult=cfg.model.width_mult, depth_mult=cfg.model.depth_mult, model_mult=cfg.model.model_mult,
    )
if __name__ == "__main__":
    main()
