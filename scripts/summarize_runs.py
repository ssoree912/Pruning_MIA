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
    """Parse metadata from a standard DWA path; return dict."""
    meta = {
        'path': str(p.parent),
        'mode': '',
        'sparsity': None,
        'dataset': '',
        'alpha': None,
        'beta': None,
        'seed': None,
        'method': '',
    }
    parts = p.parts
    try:
        # .../runs/<method>/<mode>/sparsity_<s>/<dataset>/alpha<a>_beta<b>/seed<seed>/best_model.pth
        if 'runs' in parts:
            i = parts.index('runs')
            meta['method'] = parts[i+1] if i+1 < len(parts) else ''
            # mode
            if i+2 < len(parts):
                meta['mode'] = parts[i+2]
            # sparsity
            if i+3 < len(parts) and parts[i+3].startswith('sparsity_'):
                try:
                    meta['sparsity'] = float(parts[i+3].split('_',1)[1])
                except Exception:
                    pass
            # dataset
            if i+4 < len(parts):
                meta['dataset'] = parts[i+4]
            # alpha_beta
            if i+5 < len(parts):
                m = re.match(r'alpha([0-9.]+)_beta([0-9.]+)', parts[i+5])
                if m:
                    meta['alpha'] = float(m.group(1))
                    meta['beta'] = float(m.group(2))
            # seed
            if i+6 < len(parts) and parts[i+6].startswith('seed'):
                try:
                    meta['seed'] = int(parts[i+6].replace('seed',''))
                except Exception:
                    pass
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
            'best_acc1': extra.get('best_acc1'),
            'final_acc1': extra.get('final_acc1'),
            'final_loss': extra.get('final_loss'),
        }
        rows.append(row)

    # Write CSV
    cols = ['path','method','mode','sparsity','dataset','alpha','beta','seed','best_acc1','final_acc1','final_loss']
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"✅ Wrote {len(rows)} rows to {out_path}")


if __name__ == '__main__':
    main()

