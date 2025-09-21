#!/usr/bin/env python3
"""
Aggregate MIA JSON results into tidy CSVs and produce standard plots.

Outputs
- results/results_raw_tidy.csv: one row per (experiment × attack × metric)
- results/summary_by_group.csv: grouped statistics with mean/std/CI/median/iqr
- results/plots/*.png: a small set of ready-to-use figures

Usage examples
  python scripts/mia_aggregate.py \
    --src mia_results \
    --out_dir results \
    --plots_dir results/plots \
    --metrics confidence_extended_auroc,lira_auc,nn_auc,samia_auc

  # DPF only, with accuracy-matching ±0.5pp, and selected victims
  python scripts/mia_aggregate.py --src mia_results \
    --methods dpf --victims 42 43 --acc_match_pp 0.5 --make_plots
"""

import argparse
import json
import math
import os
from pathlib import Path
import glob
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False


def parse_args():
    ap = argparse.ArgumentParser(description='Aggregate MIA results into tidy CSVs and plots')
    ap.add_argument('--src', default='mia_results', help='Root directory containing result JSONs (recursively scanned)')
    ap.add_argument('--out_dir', default='results', help='Base output directory for CSVs')
    ap.add_argument('--plots_dir', default='results/plots', help='Output directory for plots')
    ap.add_argument('--dataset', default=None, help='Filter dataset (e.g., cifar10)')
    ap.add_argument('--methods', nargs='*', default=None, help='Filter methods (e.g., dwa dpf static dense)')
    ap.add_argument('--victims', type=int, nargs='*', default=None, help='Restrict to specific victim seeds')
    ap.add_argument('--acc_match_pp', type=float, default=0.5, help='Accuracy-matching ±pp band around per-(method,sparsity) median')
    ap.add_argument('--make_plots', action='store_true', help='Generate standard figures (requires matplotlib)')
    ap.add_argument('--metrics', default='confidence_extended_auroc,lira_auc,nn_auc,samia_auc', help='Comma-separated metrics to consider for plots (CSV always has all)')
    ap.add_argument('--verbose', action='store_true', help='Print debug logs and sample rows')
    return ap.parse_args()


def derive_mode_label(cfg: dict) -> str:
    method = (cfg.get('prune_method') or 'unknown').lower()
    prune_type = cfg.get('prune_type', 'na')
    freeze_tag = cfg.get('freeze_tag')
    if method == 'dwa':
        return prune_type
    if method == 'dpf':
        tag = freeze_tag if freeze_tag else 'nofreeze'
        return f'dpf:{tag}'
    if method == 'static':
        return 'static'
    if method == 'dense':
        return 'dense'
    return f'{method}:{prune_type}'


def parse_one_json(fp: Path):
    try:
        data = json.loads(fp.read_text())
    except Exception:
        return []

    cfg = data.get('config', {})
    exp = data.get('experiment_info', {})
    res = data.get('results', {})

    # base metadata
    meta = {
        'file': str(fp),
        'dataset': cfg.get('dataset_name') or data.get('dataset') or '',
        'arch': (data.get('victim_config') or {}).get('model', {}).get('arch', ''),
        'method': (cfg.get('prune_method') or '').lower(),
        'mode': derive_mode_label(cfg),
        'forward_mode': exp.get('forward_mode') or cfg.get('forward_mode'),
        'sparsity': cfg.get('sparsity'),
        'victim_seed': cfg.get('victim_seed'),
        'victim_test_acc': data.get('victim_test_acc'),
        'use_temperature': data.get('use_temperature', None),
        'attack_mode': exp.get('attack_mode'),
        'alpha': cfg.get('alpha'),
        'beta': cfg.get('beta'),
    }

    rows = []

    def add_attack(name: str, block: dict):
        if not isinstance(block, dict):
            return
        # AUROC
        if 'auc' in block:
            rows.append({'attack': name, 'metric': 'auroc', 'value': block['auc']})
        if 'auroc' in block:
            rows.append({'attack': name, 'metric': 'auroc', 'value': block['auroc']})
        # Advantage
        if 'advantage' in block:
            rows.append({'attack': name, 'metric': 'advantage', 'value': block['advantage']})
        # TPR suite
        tprs = block.get('tpr_at_fprs') or {}
        if isinstance(tprs, dict):
            for k, v in tprs.items():
                rows.append({'attack': name, 'metric': f'tpr@{k}', 'value': v})
        # Back-compat 1% only
        if 'tpr_at_1fpr' in block and not any(r['metric'].startswith('tpr@') for r in rows if r['attack'] == name):
            rows.append({'attack': name, 'metric': 'tpr@1', 'value': block['tpr_at_1fpr']})

    # Pull from typical blocks
    add_attack('lira', res.get('lira'))
    add_attack('confidence', res.get('confidence_extended'))  # threshold variant with extended metrics
    add_attack('samia', res.get('samia'))
    add_attack('nn', res.get('nn'))
    add_attack('nn_top3', res.get('nn_top3'))
    add_attack('nn_cls', res.get('nn_cls'))

    for r in rows:
        r.update(meta)
    return rows


def accuracy_matching(df: pd.DataFrame, band_pp: float) -> pd.DataFrame:
    """Filter within ±band_pp around median accuracy per (method, sparsity).
    If a group's accuracy is missing/NaN, skip filtering for that group.
    """
    if df.empty or 'victim_test_acc' not in df.columns:
        return df
    keep = []
    for (method, sparsity), g in df.groupby(['method', 'sparsity'], dropna=False):
        acc = pd.to_numeric(g['victim_test_acc'], errors='coerce')
        med = float(acc.median()) if acc.notna().any() else float('nan')
        if math.isnan(med):
            keep.append(g)
            continue
        lo, hi = med - band_pp, med + band_pp
        mask = (acc >= lo) & (acc <= hi)
        keep.append(g[mask])
    return pd.concat(keep, ignore_index=True) if keep else df


def ci95(values: np.ndarray):
    vals = np.asarray([v for v in values if pd.notnull(v)], dtype=float)
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    m = float(vals.mean())
    sd = float(vals.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        rng = np.random.default_rng(0)
        boots = rng.choice(vals, (10000, n), replace=True).mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
    else:
        lo = hi = np.nan
    return m, sd, float(lo), float(hi), n


def plot_bar_ci(summary: pd.DataFrame, title: str, ylab: str, out_png: Path):
    if not HAS_PLOT or summary.empty:
        return
    methods = sorted(summary['method'].dropna().unique())
    sparsities = sorted(summary['sparsity'].dropna().unique())
    w = 0.8 / max(1, len(methods))
    import matplotlib.pyplot as plt  # lazy import for headless envs
    plt.figure(figsize=(10, 4 + 0.2 * len(sparsities)))
    for i, s in enumerate(sparsities):
        g = summary[summary['sparsity'] == s]
        for j, m in enumerate(methods):
            row = g[g['method'] == m]
            if row.empty:
                continue
            mu = float(row['mean'].iloc[0])
            lo = float(row['ci95_lo'].iloc[0])
            hi = float(row['ci95_hi'].iloc[0])
            x = i + (j - (len(methods) - 1) / 2) * w
            plt.bar(x, mu, width=w)
            if not math.isnan(lo) and not math.isnan(hi):
                plt.plot([x, x], [lo, hi], color='black')
    plt.xticks(range(len(sparsities)), [f's={s}' for s in sparsities])
    plt.ylabel(ylab)
    plt.title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    args = parse_args()
    src = Path(args.src)
    out_dir = Path(args.out_dir)
    plots_dir = Path(args.plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        if args.verbose:
            print(f"[mia_aggregate] {msg}")

    # 1) load all JSONs
    all_rows = []
    files = list(src.rglob('*.json'))
    log(f"Scanning {src} -> found {len(files)} JSON files")
    for fp in files:
        rows = parse_one_json(fp)
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    raw_path = out_dir / 'results_raw_tidy.csv'
    df.to_csv(raw_path, index=False)
    print(f'Wrote raw tidy CSV: {raw_path} ({len(df)} rows from {len(files)} files)')
    if args.verbose:
        log(f"Raw tidy columns: {list(df.columns)}")
        try:
            log("Raw tidy head:\n" + df.head(8).to_string(index=False))
        except Exception:
            pass

    # basic filtering
    if args.dataset:
        df = df[df['dataset'] == args.dataset]
        log(f"Filter dataset={args.dataset} -> {len(df)} rows")
    if args.methods:
        df = df[df['method'].isin(args.methods)]
        log(f"Filter methods={args.methods} -> {len(df)} rows")
    if args.victims:
        df = df[df['victim_seed'].isin(args.victims)]
        log(f"Filter victims={args.victims} -> {len(df)} rows")

    # coerce numeric sparsity
    if 'sparsity' in df.columns:
        df['sparsity'] = pd.to_numeric(df['sparsity'], errors='coerce')
    log(f"After filters: rows={len(df)}, unique victims={df['victim_seed'].nunique() if 'victim_seed' in df.columns else 'NA'}, unique methods={df['method'].nunique() if 'method' in df.columns else 'NA'}, unique sparsities={df['sparsity'].nunique() if 'sparsity' in df.columns else 'NA'}")

    # 2) accuracy matching band per (method, sparsity)
    before_rows = len(df)
    dfm = accuracy_matching(df, args.acc_match_pp)
    log(f"Accuracy matching ±{args.acc_match_pp}pp: kept {len(dfm)}/{before_rows} rows")
    if args.verbose and not dfm.empty:
        try:
            grp = dfm.groupby(['method','sparsity'], dropna=False)['victim_test_acc'].agg(['count','min','median','max']).reset_index()
            log("Post-match group stats (method,sparsity):\n" + grp.head(20).to_string(index=False))
        except Exception:
            pass

    # 3) grouped summary
    agg_rows = []
    group_keys = ['dataset', 'method', 'sparsity', 'attack', 'metric', 'use_temperature']
    for keys, g in dfm.groupby(group_keys):
        m, sd, lo, hi, n = ci95(g['value'].values)
        med = float(np.nanmedian(g['value'].values)) if len(g) else np.nan
        q75 = float(np.nanpercentile(g['value'].values, 75)) if len(g) else np.nan
        q25 = float(np.nanpercentile(g['value'].values, 25)) if len(g) else np.nan
        agg_rows.append({
            'dataset': keys[0], 'method': keys[1], 'sparsity': keys[2],
            'attack': keys[3], 'metric': keys[4], 'use_temperature': keys[5],
            'mean': m, 'std': sd, 'ci95_lo': lo, 'ci95_hi': hi, 'n': n,
            'median': med, 'iqr': (q75 - q25) if (not math.isnan(q75) and not math.isnan(q25)) else np.nan
        })
    summary = pd.DataFrame(agg_rows)
    if not summary.empty:
        summary = summary.sort_values(['dataset', 'attack', 'metric', 'sparsity', 'method'])
    else:
        # Ensure expected columns exist for downstream consumers
        for col in ['dataset','method','sparsity','attack','metric','use_temperature','mean','std','ci95_lo','ci95_hi','n','median','iqr']:
            if col not in summary.columns:
                summary[col] = []
    summ_path = out_dir / 'summary_by_group.csv'
    summary.to_csv(summ_path, index=False)
    print(f'Wrote grouped summary CSV: {summ_path} ({len(summary)} rows)')
    if args.verbose:
        try:
            log("Summary head:\n" + summary.head(12).to_string(index=False))
        except Exception:
            pass

    # 4) Standard plots (optional)
    if args.make_plots and HAS_PLOT:
        # LiRA TPR@1 (bar+CI)
        sub = summary[(summary.attack == 'lira') & (summary.metric == 'tpr@1')]
        plot_bar_ci(sub, 'LiRA TPR@1%FPR (↓ lower is better)', 'TPR@1%FPR', plots_dir / 'lira_tpr1_bar.png')
        # LiRA AUROC (bar+CI)
        sub = summary[(summary.attack == 'lira') & (summary.metric == 'auroc')]
        plot_bar_ci(sub, 'LiRA AUROC (↓ closer to 0.5 is better)', 'AUROC', plots_dir / 'lira_auroc_bar.png')

    if args.make_plots and not HAS_PLOT:
        print('matplotlib not available; skipped plot generation')


if __name__ == '__main__':
    main()
