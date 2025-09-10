#!/usr/bin/env python3
"""
DWA (Dynamic Weight Adjustment) 실험 스크립트 (오케스트레이터)
- 3가지 실험 모드 반복 실행
"""
import os, sys, json, subprocess, pandas as pd
from pathlib import Path
import argparse, shutil
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 스크립트 경로 안정성
SCRIPT_DIR = Path(__file__).resolve().parent
RUN_SCRIPT = str(SCRIPT_DIR / 'run_experiment_dwa.py')

def run_training(config_params):
    cmd = [sys.executable, RUN_SCRIPT]
    for k, v in config_params.items():
        if v is None: continue
        if isinstance(v, bool):
            if v: cmd.append(f'--{k}')
        elif isinstance(v, (list, tuple)):
            # 리스트는 콤마로 (wandb_tags 등)
            cmd.extend([f'--{k}', ",".join(map(str, v))])
        else:
            cmd.extend([f'--{k}', str(v)])
    print(f"Running training command: {' '.join(cmd)}")
    try:
        # 타임아웃 제거 & 실시간 로그 스트리밍
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"Training failed with return code: {res.returncode}")
            return False, f"Process failed with return code: {res.returncode}"
        return True, "Training completed successfully"
    except Exception as e:
        return False, str(e)

def collect_results(results_dir: Path):
    results = {}
    
    # Config file
    cfg = results_dir / 'config.json'
    if cfg.exists():
        with open(cfg, 'r') as f: 
            config = json.load(f)
        results['config'] = config
        # DWA 메타 동봉
        p = config.get('pruning', {})
        results['dwa_mode'] = p.get('dwa_mode')
        results['dwa_alpha'] = p.get('dwa_alpha')
        results['dwa_beta'] = p.get('dwa_beta')
    
    # Log files (now includes training.log)
    log_files = list(results_dir.glob('*.log'))
    if log_files:
        results['log_file'] = str(log_files[0])
        results['log_files'] = [str(f) for f in log_files]  # All log files
    
    # Experiment summary (now with structured metrics)
    exp_sum = results_dir / 'experiment_summary.json'
    if exp_sum.exists():
        with open(exp_sum, 'r') as f: 
            results['training'] = json.load(f)
    
    # Validation history (new)
    val_hist = results_dir / 'validation_history.json'
    if val_hist.exists():
        with open(val_hist, 'r') as f: 
            results['validation_history'] = json.load(f)
    
    
    # Accuracy log file
    acc_log = results_dir / f"*_acc_log.txt"
    acc_log_files = list(results_dir.glob("*_acc_log.txt"))
    if acc_log_files:
        results['acc_log_file'] = str(acc_log_files[0])
    
    return results

def create_training_summary_csv(all_results, experiment_prefix='dwa_experiments'):
    results_dir = Path('results'); results_dir.mkdir(exist_ok=True)
    rows = []
    for exp_name, r in all_results.items():
        row = {'experiment': exp_name}
        if 'config' in r:
            c = r['config']
            row.update({
                'method': c.get('pruning', {}).get('method', 'dcil'),
                'sparsity': c.get('pruning', {}).get('sparsity', 0.0),
                'dataset': c.get('data', {}).get('dataset', 'cifar10'),
                'arch': c.get('model', {}).get('arch', 'resnet'),
                'layers': c.get('model', {}).get('layers', 20),
                'epochs': c.get('training', {}).get('epochs', 200),
                'lr': c.get('training', {}).get('lr', 0.1),
                # DWA 메타 추가
                'dwa_mode': r.get('dwa_mode'),
                'dwa_alpha': r.get('dwa_alpha'),
                'dwa_beta': r.get('dwa_beta'),
            })
        if 'training' in r:
            t = r['training']
            row.update({
                'best_acc1': t.get('best_metrics', {}).get('best_acc1'),
                'best_loss': t.get('best_metrics', {}).get('best_loss'),
                'final_acc1': t.get('final_metrics', {}).get('acc1'),
                'final_acc5': t.get('final_metrics', {}).get('acc5'),
                'final_loss': t.get('final_metrics', {}).get('loss'),
                'total_training_time_hours': (t.get('total_duration', 0) or 0)/3600,
            })
        if 'validation_history' in r:
            vh = r['validation_history']
            if vh:
                best_val = max((e.get('acc1', 0) for e in vh if 'acc1' in e), default=None)
                row.update({
                    'val_best_acc1': best_val,
                    'val_final_acc1': vh[-1].get('acc1', None),
                    'val_final_loss': vh[-1].get('loss', None),
                    'total_epochs_logged': len(vh),
                })
        
        rows.append(row)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    if all_results:
        first = next(iter(all_results.values()))
        if 'config' in first:
            c = first['config']
            dataset = c.get('data', {}).get('dataset', 'cifar10')
            arch = c.get('model', {}).get('arch', 'resnet')
            epochs = c.get('training', {}).get('epochs', 50)
            suffix = f"{dataset}_{arch}_e{epochs}_{ts}"
        else:
            suffix = f"{experiment_prefix}_{ts}"
    else:
        suffix = f"{experiment_prefix}_{ts}"

    out = Path("results")/f"training_results_{suffix}.csv"
    df = pd.DataFrame(rows); df.to_csv(out, index=False)
    print(f"Summary results saved to {out}")
    return df

def main():
    p = argparse.ArgumentParser(description='DWA (Dynamic Weight Adjustment) Experiments')
    p.add_argument('--dwa-modes', nargs='+',
                   default=['reactivate_only','kill_active_plain_dead','kill_and_reactivate'],
                   choices=['reactivate_only','kill_active_plain_dead','kill_and_reactivate'])
    p.add_argument('--dwa-alphas', nargs='+', type=float, default=[1.0])
    p.add_argument('--dwa-betas', nargs='+', type=float, default=[1.0])
    p.add_argument('--dwa-threshold-percentile', type=int, default=50)
    p.add_argument('--target-epoch', type=int, default=75)
    p.add_argument('--prune-freq', type=int, default=16)
    p.add_argument('--freeze-epoch', type=int, default=-1)
    p.add_argument('--sparsities', nargs='+', type=float, default=[0.5,0.8,0.9])
    p.add_argument('--dataset', default='cifar10', choices=['cifar10','cifar100'])
    p.add_argument('--arch', default='resnet', choices=['resnet','wideresnet'])
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--skip-existing', action='store_true')
    p.add_argument('--reorganize-only', action='store_true')
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb-project', default='dwa-experiments')
    p.add_argument('--wandb-entity', default=None)
    p.add_argument('--wandb-tags', nargs='*', default=[])
    p.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    p.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    args = p.parse_args()

    # 필요시 과거 runs 재정리 (생략 가능)
    # ...

    all_results = {}
    failed = []

    print(f"🚀 Starting DWA experiments")
    print(f"DWA Modes: {args.dwa_modes}")
    print(f"Alphas: {args.dwa_alphas}, Betas: {args.dwa_betas}")
    print(f"Sparsities: {args.sparsities}")
    print(f"Dataset: {args.dataset}, Architecture: {args.arch}")

    for mode in args.dwa_modes:
        for alpha in args.dwa_alphas:
            for beta in args.dwa_betas:
                for sp in args.sparsities:
                    exp_name = f"dwa_{mode}"
                    if alpha != 1.0: exp_name += f"_alpha{alpha}"
                    if beta  != 1.0: exp_name += f"_beta{beta}"
                    exp_name += f"_sparsity_{sp}_{args.dataset}_{args.arch}"

                    print("\n" + "="*50)
                    print(f"Running experiment: {exp_name}")
                    print(f"DWA Mode: {mode}, Alpha: {alpha}, Beta: {beta}, Sparsity: {sp}")
                    print("="*50)

                    save_path = Path('./runs/dwa')/mode/f'sparsity_{sp}'/args.dataset
                    if alpha != 1.0 or beta != 1.0:
                        save_path = save_path / f'alpha{alpha}_beta{beta}'
                    save_path.mkdir(parents=True, exist_ok=True)

                    if args.skip_existing and ((save_path/'best_model.pth').exists() or (save_path/'experiment_summary.json').exists()):
                        print(f"Results already exist for {exp_name}, skipping...")
                        all_results[exp_name] = collect_results(save_path)
                        continue

                    cfg_kwargs = {
                        'name': exp_name,
                        'save-dir': str(save_path),

                        'dataset': args.dataset,
                        'arch': args.arch,
                        'epochs': args.epochs,

                        # ✅ DWA는 dcil 백엔드 사용 (dpf 말고 dcil)
                        'prune': True,
                        'prune-method': 'dcil',
                        'sparsity': sp,

                        # ✅ 스케줄/빈도 전달 (오케스트레이터 인자에서 받아와야 함)
                        'target-epoch': args.target_epoch,
                        'prune-freq': args.prune_freq,
                        'freeze-epoch': getattr(args, 'freeze_epoch', -1),

                        # DWA
                        'dwa-mode': mode,
                        'dwa-alpha': alpha,
                        'dwa-beta': beta,
                        'dwa-threshold-percentile': args.dwa_threshold_percentile,

                        # 새로 추가: seed와 gpu
                        'seed': args.seed,
                        'gpu': args.gpu,
                    }
                    if args.wandb:
                        cfg_kwargs.update({
                            'wandb': True,
                            'wandb_project': args.wandb_project,  # 하이픈 접근 버그 수정
                            'wandb_entity': args.wandb_entity,
                            'wandb_name': exp_name,
                            'wandb_tags': args.wandb_tags + ['dwa', mode, args.dataset, args.arch],
                        })

                    ok, out = run_training(cfg_kwargs)
                    if not ok:
                        print(f"Training failed for {exp_name}: {out}")
                        failed.append((exp_name, 'training', out))
                        continue

                    all_results[exp_name] = collect_results(save_path)
                    print(f"Experiment {exp_name} completed")

    print("\n" + "="*50)
    print("Creating training summary...")
    print("="*50)
    if all_results:
        tag = f"dwa_{'-'.join(args.dwa_modes)}_alpha{'-'.join(map(str,args.dwa_alphas))}_beta{'-'.join(map(str,args.dwa_betas))}"
        df = create_training_summary_csv(all_results, experiment_prefix=tag)
        print(f"Completed {len(all_results)} experiments")
        print(df.to_string())

    if failed:
        print(f"\n{'='*20} FAILED EXPERIMENTS {'='*20}")
        for name, stage, err in failed:
            print(f"FAILED: {name} ({stage}) - {err}")

    print("\nAll experiments completed!")

if __name__ == '__main__':
    main()