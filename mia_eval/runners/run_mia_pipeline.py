#!/usr/bin/env python3
"""
MIA pipeline over trained models under runs/ (DWA / Static / DPF / Dense).

1) Scan runs/ and group seeds per experiment
2) Ensure fixed MIA splits (pkl)
3) Run MIA per group (victim + shadows)
4) Summarize results
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Resolve project root and absolute script paths (no hardcoded names)
THIS_DIR = Path(__file__).resolve().parent

def _find_repo_root(start: Path) -> Path:
    for cand in [start] + list(start.parents):
        if (cand / '.git').exists():
            return cand
        if (cand / 'base_model.py').exists() and (cand / 'mia_eval').exists():
            return cand
    return start.parents[2]

REPO_ROOT = _find_repo_root(THIS_DIR)
RUN_SINGLE = REPO_ROOT / 'mia_eval' / 'runners' / 'run_single_mia.py'
CREATE_SPLITS = REPO_ROOT / 'mia_eval' / 'create_data' / 'create_fixed_data_splits.py'
from datetime import datetime

def run_command(cmd, cwd=None):
    """명령어 실행"""
    print(f"🔧 Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def _scan_runs(runs_dir: str, dataset: str):
    from contextlib import suppress
    runs = Path(runs_dir)
    groups = []
    total = 0

    # DWA
    dwa = runs / 'dwa'
    if dwa.exists():
        for mode in dwa.iterdir():
            if not mode.is_dir():
                continue
            for sp in mode.glob('sparsity_*'):
                try:
                    spv = float(sp.name.split('_',1)[1])
                except Exception:
                    continue
                for ds in sp.iterdir():
                    if not ds.is_dir() or ds.name != dataset:
                        continue
                    for ab in ds.glob('alpha*_beta*'):
                        if not ab.is_dir():
                            continue
                        parts = ab.name.split('_')
                        alpha = parts[0].replace('alpha','')
                        beta  = parts[1].replace('beta','')
                        seeds = []
                        for sdir in ab.glob('seed*'):
                            if (sdir / 'best_model.pth').exists():
                                with suppress(Exception):
                                    seeds.append(int(sdir.name.replace('seed','')))
                                    total += 1
                        if len(seeds) >= 2:
                            groups.append({'method':'dwa','mode':mode.name,'sparsity':spv,'dataset':ds.name,
                                           'alpha':alpha,'beta':beta,'seeds':sorted(seeds)})

    # Static
    st = runs / 'static'
    if st.exists():
        for sp in st.glob('sparsity_*'):
            with suppress(Exception):
                spv = float(sp.name.split('_',1)[1])
            for ds in sp.iterdir():
                if not ds.is_dir() or ds.name != dataset:
                    continue
                seeds = []
                for sdir in ds.glob('seed*'):
                    if (sdir / 'best_model.pth').exists():
                        with suppress(Exception):
                            seeds.append(int(sdir.name.replace('seed','')))
                            total += 1
                if len(seeds) >= 2:
                    groups.append({'method':'static','mode':'na','sparsity':spv,'dataset':ds.name,
                                   'alpha':None,'beta':None,'seeds':sorted(seeds)})

    # DPF
    dpf = runs / 'dpf'
    if dpf.exists():
        for sp in dpf.glob('sparsity_*'):
            rest = sp.name.split('sparsity_',1)[1]
            with suppress(Exception):
                parts = rest.split('_',1)
                spv = float(parts[0])
                tag = parts[1] if len(parts) > 1 else None
            for ds in sp.iterdir():
                if not ds.is_dir() or ds.name != dataset:
                    continue
                seeds = []
                for sdir in ds.glob('seed*'):
                    if (sdir / 'best_model.pth').exists():
                        with suppress(Exception):
                            seeds.append(int(sdir.name.replace('seed','')))
                            total += 1
                if len(seeds) >= 2:
                    groups.append({'method':'dpf','mode':'na','sparsity':spv,'dataset':ds.name,
                                   'alpha':None,'beta':None,'seeds':sorted(seeds),'freeze_tag':tag})

    # Dense
    de = runs / 'dense'
    if de.exists():
        ds = de / dataset
        if ds.exists():
            seeds = []
            for sdir in ds.glob('seed*'):
                if (sdir / 'best_model.pth').exists():
                    with suppress(Exception):
                        seeds.append(int(sdir.name.replace('seed','')))
                        total += 1
            if len(seeds) >= 2:
                groups.append({'method':'dense','mode':'na','sparsity':None,'dataset':dataset,
                               'alpha':None,'beta':None,'seeds':sorted(seeds)})

    print(f"✅ Found {total} checkpoints across {len(groups)} experiment groups")
    return groups

def main():
    parser = argparse.ArgumentParser(description='MIA Evaluation Pipeline (DWA / Static / DPF / Dense)')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'cifar100'], help='Dataset name')
    parser.add_argument('--runs_dir', type=str, default='./runs', 
                       help='DWA training results directory')
    parser.add_argument('--output_dir', type=str, default='./mia_results',
                       help='MIA results output directory')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--split_seed', type=int, default=7, help='Seed used for fixed MIA data splits')
    parser.add_argument('--debug', action='store_true', help='Enable debug prints inside per-run MIA evaluation')
    parser.add_argument('--forward_mode', type=str, default='standard', choices=['standard','dwa_adaptive','scaling','dpf'], help='Model forward mode to pass through')
    parser.add_argument('--attacks', default='samia,threshold,nn,nn_top3,nn_cls,lira', help='Comma-separated attacks to run')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--datapath', type=str, default='~/Datasets', help='Dataset path')
    parser.add_argument('--skip_data_prep', action='store_true', 
                       help='Skip MIA data preparation step')
    
    args = parser.parse_args()
    
    print("🚀 MIA Evaluation Pipeline")
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Runs dir: {args.runs_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print("=" * 50)
    
    # Step 1: Scan runs directory for all methods
    print("\n📋 Step 1: Scanning runs/ for experiments...")
    experiments = _scan_runs(args.runs_dir, args.dataset)
    if not experiments:
        print("❌ No trained checkpoints found under runs/.")
        print("   Make sure to train models or adjust --dataset/--runs_dir.")
        return
    print(f"Found {len(experiments)} experiment groups:")
    for exp in experiments:
        label = exp['method']
        if exp['method'] == 'dwa':
            desc = f"{exp['mode']}/sparsity_{exp['sparsity']}/{exp['dataset']}/alpha{exp['alpha']}_beta{exp['beta']}"
        elif exp['method'] == 'static':
            desc = f"static/sparsity_{exp['sparsity']}/{exp['dataset']}"
        elif exp['method'] == 'dpf':
            tag = exp.get('freeze_tag')
            tag_s = f"_{tag}" if tag else ''
            desc = f"dpf/sparsity_{exp['sparsity']}{tag_s}/{exp['dataset']}"
        else:
            desc = f"dense/{exp['dataset']}"
        print(f"  - [{label}] {desc} ({len(exp['seeds'])} seeds)")
    
    # Step 2: Ensure fixed data splits exist (auto-create if missing)
    print("\n🧩 Step 2: Ensuring fixed MIA data splits...")
    split_seed = args.split_seed
    
    # Step 3: MIA 평가 실행
    print(f"\n🎯 Step 3: Running MIA evaluation on {len(experiments)} experiment groups...")
    
    success_count = 0
    for i, exp in enumerate(experiments, 1):
        if exp['method'] == 'dwa':
            cur_desc = f"{exp['mode']}/sparsity_{exp['sparsity']}/{exp['dataset']}/alpha{exp['alpha']}_beta{exp['beta']}"
        elif exp['method'] == 'static':
            cur_desc = f"static/sparsity_{exp['sparsity']}/{exp['dataset']}"
        elif exp['method'] == 'dpf':
            tag = exp.get('freeze_tag')
            tag_s = f"_{tag}" if tag else ''
            cur_desc = f"dpf/sparsity_{exp['sparsity']}{tag_s}/{exp['dataset']}"
        else:
            cur_desc = f"dense/{exp['dataset']}"
        print(f"\n[{i}/{len(experiments)}] Processing [{exp['method']}] {cur_desc}...")
        
        if len(exp['seeds']) < 2:
            print(f"⏭️ Skipping - need at least 2 seeds, found {len(exp['seeds'])}")
            continue
            
        # victim은 첫 번째 seed, shadow는 나머지
        victim_seed = exp['seeds'][0]
        shadow_seeds = exp['seeds'][1:]
        
        # Ensure pkl exists per victim seed
        split_path = Path(f"mia_data_splits/{exp['dataset']}_seed{split_seed}_victim{victim_seed}.pkl")
        if not split_path.exists():
            print(f"  📦 Creating fixed splits: {split_path}")
            mk_cmd = [
                sys.executable, str(CREATE_SPLITS),
                '--dataset', exp['dataset'],
                '--seed', str(split_seed),
                '--victim_seed', str(victim_seed),
                '--shadow_seeds', *[str(s) for s in shadow_seeds]
            ]
            if not run_command(mk_cmd, cwd=str(REPO_ROOT)):
                print(f"  ❌ Failed to create data splits for victim_seed={victim_seed}. Skipping.")
                continue

        # Build command for single runner according to method
        sparsity = exp.get('sparsity')
        sparsity_str = str(sparsity if sparsity is not None else 0.0)
        eval_cmd = [
            sys.executable, str(RUN_SINGLE),
            '--dataset', exp['dataset'],
            '--sparsity', sparsity_str,
            '--prune_method', exp['method'],
            '--prune_type', exp.get('mode','na'),
            '--victim_seed', str(victim_seed),
            '--shadow_seeds'] + [str(s) for s in shadow_seeds] + [
            '--device', args.device.replace('cuda:', ''),
            '--split_seed', str(split_seed),
            '--forward_mode', args.forward_mode,
            '--attacks', args.attacks
        ]
        if exp['method'] == 'dwa':
            eval_cmd += ['--alpha', str(exp.get('alpha')), '--beta', str(exp.get('beta'))]
        if exp['method'] == 'dpf' and exp.get('freeze_tag'):
            eval_cmd += ['--freeze_tag', str(exp['freeze_tag'])]
        if args.debug:
            eval_cmd.append('--debug')
        
        if run_command(eval_cmd, cwd=str(REPO_ROOT)):
            success_count += 1
        else:
            print(f"❌ Failed to evaluate [{exp['method']}] {cur_desc}")
    
    print(f"\n📊 Completed {success_count}/{len(experiments)} evaluations")
    
    # Step 4: 결과 요약
    print(f"\n📈 Step 4: Results summary...")
    
    result_dir = Path('mia_results')
    if result_dir.exists():
        json_files = list(result_dir.glob('**/*.json'))
        if json_files:
            print(f"\n📁 Found {len(json_files)} result files:")
            
            # 간단한 요약 출력 + 구조화된 결과 수집
            import json
            all_results = []
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        config = data['config']
                        results = data['results']
                        
                        summary = {
                            'mode': config.get('prune_type', 'unknown'),
                            'sparsity': config.get('sparsity', 0),
                            'victim_seed': config.get('victim_seed', 0),
                            'victim_acc': data.get('victim_test_acc', 0),
                        }
                        # Flatten known nested metrics
                        def pull_attack(d, name):
                            v = results.get(name)
                            if isinstance(v, dict):
                                d[f'{name}_acc'] = v.get('accuracy')
                                d[f'{name}_auc'] = v.get('auc')
                                d[f'{name}_balacc'] = v.get('balanced_accuracy')
                                d[f'{name}_adv'] = v.get('advantage')
                            elif v is not None:
                                d[name] = v
                        # Threshold metrics (scalars)
                        for k in ['confidence','entropy','modified_entropy','top1_conf']:
                            if k in results:
                                summary[k] = results[k]
                        # Confidence-extended
                        ext = results.get('confidence_extended')
                        if isinstance(ext, dict):
                            summary['confidence_extended_auroc'] = ext.get('auroc')
                            summary['confidence_extended_balacc'] = ext.get('balanced_accuracy')
                            summary['confidence_extended_adv'] = ext.get('advantage')
                            summary['confidence_extended_thr'] = ext.get('threshold')
                        # Classifier-based
                        for name in ['samia','nn','nn_top3','nn_cls','lira']:
                            pull_attack(summary, name)
                        all_results.append(summary)
                        
                except Exception as e:
                    print(f"⚠️ Error reading {json_file}: {e}")
            
            if all_results:
                print(f"\n📊 MIA Attack Success Summary ({len(all_results)} experiments):")
                print("-" * 80)
                for result in all_results:
                    print(f"{result['mode']:20s} sparsity={result['sparsity']:4.2f} victim_seed={result['victim_seed']:2d} acc={result['victim_acc']:5.3f}")
                    if 'samia' in result:
                        print(f"{'':20s} SAMIA: {result['samia']:5.3f}")
                    if 'confidence' in result:
                        print(f"{'':20s} Conf: {result['confidence']:5.3f}")
                    print()

                # Write CSV summary
                import csv
                summary_file = result_dir / 'summary.csv'
                fieldnames = [
                    'mode', 'sparsity', 'victim_seed', 'victim_acc',
                    'confidence', 'entropy', 'modified_entropy', 'top1_conf',
                    'confidence_extended_auroc', 'confidence_extended_balacc', 'confidence_extended_adv', 'confidence_extended_thr',
                    'samia_acc','samia_auc','samia_balacc','samia_adv',
                    'nn_acc','nn_auc','nn_balacc','nn_adv',
                    'nn_top3_acc','nn_top3_auc','nn_top3_balacc','nn_top3_adv',
                    'nn_cls_acc','nn_cls_auc','nn_cls_balacc','nn_cls_adv',
                    'lira_acc','lira_auc','lira_balacc','lira_adv'
                ]
                with open(summary_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for r in all_results:
                        row = {k: r.get(k, '') for k in fieldnames}
                        writer.writerow(row)
                print(f"📄 Wrote CSV summary: {summary_file}")
        else:
            print("📄 No result files found")
    
    if success_count > 0:
        print(f"\n✅ MIA evaluation pipeline completed successfully!")
        print(f"📁 Results saved in: mia_results/")
    else:
        print(f"\n❌ No evaluations completed successfully")

if __name__ == '__main__':
    main()
