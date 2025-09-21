#!/usr/bin/env python3
"""
Create per-victim pivot tables (and optional plots) from MIA summary CSV.

For each victim_seed, builds sparsity x mode tables for selected metrics so you
can compare modes across sparsity per victim.

Usage examples:
  python scripts/mia_victim_pivots.py \
    --csv results/mia_results_summary.csv \
    --dataset cifar10 \
    --methods dpf dwa static dense \
    --metrics confidence_extended_auroc,lira_auc,nn_auc,samia_auc \
    --out_dir results/mia_victim_pivots

  # Only DPF (e.g., nofreeze), victims 42,43, and also plot PNGs
  python scripts/mia_victim_pivots.py \
    --csv results/mia_results_summary.csv \
    --dataset cifar10 \
    --methods dpf \
    --metrics confidence_extended_auroc,lira_auc,nn_auc \
    --victims 42 43 \
    --plot \
    --out_dir results/mia_victim_pivots
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False


def coerce_numeric(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def main():
    ap = argparse.ArgumentParser(description='Per-victim MIA pivot tables')
    ap.add_argument('--csv', required=True, help='Input summary CSV (from summarize_mia_results.py)')
    ap.add_argument('--dataset', default=None, help='Filter by dataset (e.g., cifar10)')
    ap.add_argument('--methods', nargs='*', default=None, help='Filter by prune methods (e.g., dpf dwa static dense)')
    ap.add_argument('--victims', type=int, nargs='*', default=None, help='Victim seeds to include (default: all)')
    ap.add_argument('--metrics', default='confidence_extended_auroc,lira_auc,nn_auc,samia_auc', help='Comma-separated metrics to pivot')
    ap.add_argument('--out_dir', default='results/mia_victim_pivots', help='Output directory')
    ap.add_argument('--plot', action='store_true', help='Also generate line plots per victim/metric')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if args.dataset:
        df = df[df['dataset'] == args.dataset]
    if args.methods:
        df = df[df['method'].isin(args.methods)]

    # coerce sparsity numeric
    if 'sparsity' in df.columns:
        df['sparsity'] = pd.to_numeric(df['sparsity'], errors='coerce')
        df = df.dropna(subset=['sparsity'])

    victims = sorted(df['victim_seed'].unique()) if args.victims is None else args.victims
    metrics = [m.strip() for m in (args.metrics or '').split(',') if m.strip()]

    wrote = 0
    for v in victims:
        dv = df[df['victim_seed'] == v].copy()
        if dv.empty:
            continue
        for metric in metrics:
            if metric not in dv.columns:
                continue
            piv = dv.pivot_table(index='sparsity', columns='mode', values=metric, aggfunc='mean')
            out_csv = out_dir / f'victim_{v}_{metric}.csv'
            piv.to_csv(out_csv)
            wrote += 1

            if args.plot:
                if not HAS_PLOT:
                    continue
                plt.figure(figsize=(7, 4))
                sns.set(style='whitegrid')
                for mode in piv.columns:
                    plt.plot(piv.index, piv[mode], marker='o', label=str(mode))
                plt.xlabel('Sparsity')
                plt.ylabel(metric)
                plt.title(f'Victim {v}: {metric} vs Sparsity')
                plt.legend(fontsize=8)
                plt.tight_layout()
                out_png = out_dir / f'victim_{v}_{metric}.png'
                plt.savefig(out_png, dpi=150)
                plt.close()

    print(f'Wrote {wrote} victim pivot CSVs to {out_dir}')


if __name__ == '__main__':
    main()

