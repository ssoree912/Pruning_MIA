#!/usr/bin/env python3
"""
Plot MIA metrics by sparsity grouped by mode.

Reads the summarized CSV (e.g., results/mia_results_summary.csv) and produces
line plots of selected metrics vs sparsity, with a separate line per mode
(e.g., reactivate_only, kill_active_plain_dead, static, dpf:freeze180, dense).

Example:
  python scripts/plot_mia_metrics.py \
      --csv results/mia_results_summary.csv \
      --dataset cifar10 \
      --methods dwa static dpf dense \
      --metrics confidence_extended_auroc,lira_auc,nn_auc,samia_auc \
      --include_tpr \
      --save_dir plots
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def coerce_numeric(s):
    try:
        return float(s)
    except Exception:
        return np.nan


def agg_by_mode_sparsity(df: pd.DataFrame, value_col: str):
    g = df.groupby(['mode', 'sparsity'], as_index=False)[value_col].agg(['mean', 'std', 'count']).reset_index()
    g.rename(columns={'mean': 'mean', 'std': 'std', 'count': 'n'}, inplace=True)
    return g


def plot_metric(df: pd.DataFrame, metric: str, outdir: Path, title_suffix: str = "", mode_order=None):
    if metric not in df.columns:
        print(f"[skip] metric '{metric}' not in CSV columns")
        return
    sub = df[['mode', 'sparsity', metric]].dropna()
    if sub.empty:
        print(f"[skip] metric '{metric}' has no data after dropna")
        return
    sub['sparsity'] = sub['sparsity'].apply(coerce_numeric)
    sub = sub.dropna(subset=['sparsity'])
    agg = agg_by_mode_sparsity(sub, metric)

    # Plot
    plt.figure(figsize=(8, 5))
    sns.set(style='whitegrid')
    # Order modes if provided
    if mode_order:
        agg['mode'] = pd.Categorical(agg['mode'], categories=mode_order, ordered=True)
        agg = agg.sort_values(['mode', 'sparsity'])
    # Draw with error bars
    for mode, d in agg.groupby('mode'):
        if d.empty:
            continue
        plt.errorbar(d['sparsity'], d['mean'], yerr=d['std'], fmt='-o', capsize=3, label=str(mode))

    plt.xlabel('Sparsity')
    plt.ylabel(metric)
    ttl = f"{metric} vs Sparsity"
    if title_suffix:
        ttl += f" ({title_suffix})"
    plt.title(ttl)
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    outpath = outdir / f"metric_{metric}.png"
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"[saved] {outpath}")


def main():
    ap = argparse.ArgumentParser(description='Plot MIA metrics by sparsity grouped by mode')
    ap.add_argument('--csv', default='results/mia_results_summary.csv', help='Input summary CSV path')
    ap.add_argument('--dataset', default=None, help='Filter by dataset (e.g., cifar10)')
    ap.add_argument('--methods', nargs='*', default=None, help='Filter by methods (e.g., dwa static dpf dense)')
    ap.add_argument('--metrics', default='confidence_extended_auroc,lira_auc,nn_auc,samia_auc',
                   help='Comma-separated list of metrics to plot')
    ap.add_argument('--include_tpr', action='store_true', help='Also plot all TPR@FPR columns found')
    ap.add_argument('--save_dir', default='plots', help='Directory to save plots')
    ap.add_argument('--mode_order', default='reactivate_only,kill_active_plain_dead,kill_and_reactivate,static,dpf:freeze180,dpf:nofreeze,dense',
                   help='Comma-separated preferred mode order (missing entries are ignored)')
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[error] CSV not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)

    # Basic filtering
    if args.dataset:
        df = df[df['dataset'] == args.dataset]
    if args.methods:
        df = df[df['method'].isin(args.methods)]

    # Coerce sparsity numeric
    if 'sparsity' in df.columns:
        df['sparsity'] = pd.to_numeric(df['sparsity'], errors='coerce')

    # Determine metrics
    metrics = [m.strip() for m in (args.metrics or '').split(',') if m.strip()]
    if args.include_tpr:
        tpr_cols = [c for c in df.columns if c.startswith('tpr_at_fpr_')]
        metrics.extend(sorted(tpr_cols, key=lambda x: float(x.rsplit('_',1)[-1].replace('_','.')) if x.rsplit('_',1)[-1].replace('_','.').replace('.','',1).isdigit() else x))

    outdir = Path(args.save_dir)
    # Mode order (optional)
    mode_order = [m.strip() for m in (args.mode_order or '').split(',') if m.strip()]

    # Plot each metric
    for metric in metrics:
        plot_metric(df, metric, outdir, title_suffix=args.dataset or '', mode_order=mode_order)

    # Always plot victim accuracy as utility
    if 'victim_acc' in df.columns:
        plot_metric(df, 'victim_acc', outdir, title_suffix=(args.dataset or '') + ' (utility)', mode_order=mode_order)


if __name__ == '__main__':
    main()

