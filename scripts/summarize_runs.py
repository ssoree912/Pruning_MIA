#!/usr/bin/env python3
"""
Summarize models under runs/ into a flat CSV.

Extracts best-effort metadata from path patterns like:
  runs/dwa/<mode>/sparsity_<s>/<dataset>/alpha<a>_beta<b>/seed<seed>/best_model.pth
and optional files if present (config.json, experiment_summary.json).

Usage:
  python scripts/summarize_runs.py --runs ./runs --out results/runs_summary.csv
"""

import argparse
from pathlib import Path
import json
import csv
import re


def parse_path(p: Path):
    """Parse metadata from runs/ path. Handles dense/static/dpf/dwa layouts."""
    parts = p.parts
    meta = {
        'path': str(p.parent),
        'method': '',
        'mode': '',           # DWA only
        'sparsity': None,     # static/dpf/dwa
        'dataset': '',
        'alpha': None,        # DWA only
        'beta': None,         # DWA only
        'seed': None,
        'freeze_tag': '',     # DPF only (e.g., freeze180 / nofreeze)
    }

    if 'runs' not in parts:
        return meta

    i = parts.index('runs')
    method = parts[i+1] if i+1 < len(parts) else ''
    meta['method'] = method

    try:
        if method == 'dwa':
            # runs/dwa/<mode>/sparsity_<s>/<dataset>/alpha<a>_beta<b>/seed<seed>/best_model.pth
            mode = parts[i+2] if i+2 < len(parts) else ''
            meta['mode'] = mode
            sp_token = parts[i+3] if i+3 < len(parts) else ''
            if sp_token.startswith('sparsity_'):
                s_str = sp_token.split('sparsity_', 1)[1]
                try:
                    meta['sparsity'] = float(s_str)
                except Exception:
                    pass
            meta['dataset'] = parts[i+4] if i+4 < len(parts) else ''
            ab_token = parts[i+5] if i+5 < len(parts) else ''
            m = re.match(r'alpha([0-9.]+)_beta([0-9.]+)', ab_token)
            if m:
                meta['alpha'] = float(m.group(1))
                meta['beta'] = float(m.group(2))
            seed_token = parts[i+6] if i+6 < len(parts) else ''
            if seed_token.startswith('seed'):
                try:
                    meta['seed'] = int(seed_token.replace('seed', ''))
                except Exception:
                    pass

        elif method in ('static', 'dpf'):
            # static: runs/static/sparsity_<s>/<dataset>/seed<seed>/best_model.pth
            # dpf   : runs/dpf/sparsity_<s>_<tag>/<dataset>/seed<seed>/best_model.pth
            sp_token = parts[i+2] if i+2 < len(parts) else ''
            if sp_token.startswith('sparsity_'):
                rest = sp_token[len('sparsity_'):]
                m = re.match(r'([0-9.]+)(?:_(.*))?$', rest)
                if m:
                    try:
                        meta['sparsity'] = float(m.group(1))
                    except Exception:
                        pass
                    if m.group(2):
                        meta['freeze_tag'] = m.group(2)
            meta['dataset'] = parts[i+3] if i+3 < len(parts) else ''
            seed_token = parts[i+4] if i+4 < len(parts) else ''
            if seed_token.startswith('seed'):
                try:
                    meta['seed'] = int(seed_token.replace('seed', ''))
                except Exception:
                    pass

        elif method == 'dense':
            # runs/dense/<dataset>/seed<seed>/best_model.pth
            meta['dataset'] = parts[i+2] if i+2 < len(parts) else ''
            seed_token = parts[i+3] if i+3 < len(parts) else ''
            if seed_token.startswith('seed'):
                try:
                    meta['seed'] = int(seed_token.replace('seed', ''))
                except Exception:
                    pass

        else:
            # Best-effort generic: infer seed and dataset by proximity
            for idx, tok in enumerate(parts):
                if tok.startswith('seed'):
                    try:
                        meta['seed'] = int(tok.replace('seed', ''))
                    except Exception:
                        pass
                    if idx - 1 >= 0:
                        meta['dataset'] = parts[idx - 1]
                    break
            for tok in parts:
                if tok.startswith('sparsity_') and meta['sparsity'] is None:
                    try:
                        num = tok.split('sparsity_', 1)[1].split('_', 1)[0]
                        meta['sparsity'] = float(num)
                    except Exception:
                        pass
                    break
    except Exception:
        pass

    return meta


def optional_files(root: Path):
    acc = {}
    cfg = root / 'config.json'
    summ = root / 'experiment_summary.json'
    if cfg.exists():
        try:
            with open(cfg) as f:
                acc['config'] = json.load(f)
        except Exception:
            pass
    if summ.exists():
        try:
            with open(summ) as f:
                s = json.load(f)
                acc['best_acc1'] = s.get('best_metrics',{}).get('best_acc1')
                acc['final_acc1'] = s.get('final_metrics',{}).get('acc1')
                acc['final_loss'] = s.get('final_metrics',{}).get('loss')
        except Exception:
            pass
    # Fallback: parse *_acc_log.txt when summary is missing
    if 'best_acc1' not in acc or acc.get('best_acc1') is None:
        try:
            # Find acc log file
            logs = list(root.glob('*_acc_log.txt'))
            if logs:
                path = logs[0]
                last = None
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('epoch'):
                            continue
                        last = line
                if last:
                    # Format: epoch\tacc1_train\tacc1_valid\tbest_acc1
                    parts = last.split('\t')
                    if len(parts) >= 4:
                        try:
                            acc['final_acc1'] = float(parts[2])
                        except Exception:
                            pass
                        try:
                            acc['best_acc1'] = float(parts[3])
                        except Exception:
                            pass
        except Exception:
            pass
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='./runs')
    ap.add_argument('--out', default='results/runs_summary.csv')
    args = ap.parse_args()

    runs_dir = Path(args.runs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for best in runs_dir.rglob('best_model.pth'):
        meta = parse_path(best)
        extra = optional_files(best.parent)
        row = {
            'path': meta['path'],
            'method': meta['method'],
            'mode': meta['mode'],
            'sparsity': meta['sparsity'],
            'dataset': meta['dataset'],
            'alpha': meta['alpha'],
            'beta': meta['beta'],
            'seed': meta['seed'],
            'freeze_tag': meta['freeze_tag'],
            'best_acc1': extra.get('best_acc1'),
            'final_acc1': extra.get('final_acc1'),
            'final_loss': extra.get('final_loss'),
        }
        rows.append(row)

    # Write CSV
    cols = ['path','method','mode','sparsity','dataset','alpha','beta','seed','freeze_tag','best_acc1','final_acc1','final_loss']
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"✅ Wrote {len(rows)} rows to {out_path}")


if __name__ == '__main__':
    main()

# (no helpers)
