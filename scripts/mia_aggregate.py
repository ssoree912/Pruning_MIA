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
import re
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
    ap.add_argument('--methods', nargs='*', default=None, help='Filter methods (e.g., dpf static dense)')
    ap.add_argument('--modes', nargs='*', default=None, help='Filter modes (e.g., static dpf:nofreeze kill_and_reactivate)')
    ap.add_argument('--victims', type=int, nargs='*', default=None, help='Restrict to specific victim seeds')
    ap.add_argument('--acc_match_pp', type=float, default=0.5, help='Accuracy-matching ±pp band around per-(method,sparsity) median')
    ap.add_argument('--make_plots', action='store_true', help='Generate standard figures (requires matplotlib)')
    ap.add_argument('--metrics', default='confidence_extended_auroc,lira_auc,nn_auc,samia_auc', help='Comma-separated metrics to consider for plots (CSV always has all)')
    ap.add_argument('--verbose', action='store_true', help='Print debug logs and sample rows')
    ap.add_argument('--split_by_temperature', action='store_true',
                    help='Split summaries by use_temperature (default: off)')
    ap.add_argument('--strict_paired', action='store_true',
                    help='Keep only victim seeds that exist for ALL (method,mode) within each (dataset,sparsity,attack,metric) block')
    ap.add_argument('--assert_equal_n', action='store_true',
                    help='Assert equal n across methods within each (dataset,sparsity,attack,metric) block after filtering')
    ap.add_argument('--pair_then_match', action='store_true',
                    help='First fix common seeds within each block, then apply joint accuracy matching across methods')
    ap.add_argument('--write_wide', action='store_true',
                    help='Also write a wide, per-experiment CSV (threshold scalars + attack metrics as columns)')
    ap.add_argument('--wide_filename', default='results_raw_wide.csv',
                    help='Filename for wide CSV (under --out_dir)')
    return ap.parse_args()


def derive_mode_label(cfg: dict) -> str:
    method = (cfg.get('prune_method') or 'unknown').lower()
    freeze_tag = cfg.get('freeze_tag')
    if method == 'dpf':
        tag = freeze_tag if freeze_tag else 'nofreeze'
        return f'dpf:{tag}'
    if method == 'static':
        return 'static'
    if method == 'dense':
        return 'dense'
    return method


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
    }

    # Fallbacks from filename when metadata missing
    try:
        m = re.match(r'(?P<ds>[^_]+)_sparsity_(?P<s>[0-9.]+)_victim(?P<seed>\d+)\.json', fp.name)
        if (not meta['dataset']) and m:
            meta['dataset'] = m.group('ds')
        if (meta['sparsity'] is None) and m:
            meta['sparsity'] = float(m.group('s'))
        if (meta['victim_seed'] is None) and m:
            meta['victim_seed'] = int(m.group('seed'))
    except Exception:
        pass
    # Method fallback from forward_mode
    if not meta.get('method'):
        fm = (meta.get('forward_mode') or '').lower()
        if 'dpf' in fm:
            meta['method'] = 'dpf'
        elif 'standard' in fm or 'static' in fm:
            meta['method'] = 'static'
        else:
            meta['method'] = 'unknown'
    # Mode fallback if unclear
    if (not meta.get('mode')) or meta['mode'] in ('unknown:na', ''):
        meta['mode'] = meta.get('forward_mode') or meta.get('method')
    # Normalize use_temperature to {0,1}
    ut = meta.get('use_temperature')
    if ut is None or (isinstance(ut, float) and math.isnan(ut)):
        meta['use_temperature'] = 0
    elif isinstance(ut, bool):
        meta['use_temperature'] = int(ut)
    else:
        try:
            meta['use_temperature'] = int(ut)
        except Exception:
            meta['use_temperature'] = 0

    rows = []

    def add_attack(name: str, block: dict):
        if not isinstance(block, dict):
            return
        # AUROC (single)
        auroc = block.get('auc', block.get('auroc', None))
        if auroc is not None:
            rows.append({'attack': name, 'metric': 'auroc', 'value': auroc})
        # Average Precision (PR-AUC)
        ap = block.get('ap', block.get('average_precision', None))
        if ap is not None:
            rows.append({'attack': name, 'metric': 'ap', 'value': ap})
        # Advantage
        if 'advantage' in block:
            rows.append({'attack': name, 'metric': 'advantage', 'value': block['advantage']})
        # Accuracy / Balanced Accuracy
        if 'accuracy' in block:
            rows.append({'attack': name, 'metric': 'accuracy', 'value': block['accuracy']})
        if 'balanced_accuracy' in block:
            rows.append({'attack': name, 'metric': 'balanced_accuracy', 'value': block['balanced_accuracy']})
        # Precision/Recall/F1 if present
        for m in ('precision', 'recall', 'f1'):
            if m in block:
                rows.append({'attack': name, 'metric': m, 'value': block[m]})
        # TPR suite normalization
        tprs = block.get('tpr_at_fprs') or {}
        if not tprs and ('tpr_at_1fpr' in block):
            tprs = {'1': block['tpr_at_1fpr']}
        def _norm_fpr_key(k):
            try:
                x = float(k)
                if abs(x - round(x)) < 1e-9:
                    return str(int(round(x)))
                return str(x).rstrip('0').rstrip('.')
            except Exception:
                return str(k)
        if isinstance(tprs, dict):
            for k, v in tprs.items():
                rows.append({'attack': name, 'metric': f'tpr@{_norm_fpr_key(k)}', 'value': v})

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


def parse_one_json_wide(fp: Path):
    """Parse a single JSON into a single-row wide dict with useful columns.
    Includes threshold scalars, confidence-extended metrics (AUROC/adv/TPRs),
    and classifier-based metrics for samia/nn/nn_top3/nn_cls/lira.
    """
    try:
        data = json.loads(fp.read_text())
    except Exception:
        return None

    cfg = data.get('config', {})
    exp = data.get('experiment_info', {})
    res = data.get('results', {})

    row = {
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
    }

    # Shadow count for convenience
    sc_map = (exp or {}).get('shadow_configs') or {}
    if isinstance(sc_map, dict) and sc_map:
        row['shadow_count'] = len(sc_map)
    else:
        ss = cfg.get('shadow_seeds')
        if isinstance(ss, list):
            row['shadow_count'] = len(ss)

    # Threshold scalars (if present at top-level results)
    for k in ['confidence', 'entropy', 'modified_entropy', 'top1_conf']:
        if isinstance(res, dict) and (k in res):
            row[k] = res[k]

    # Confidence-extended block
    ext = res.get('confidence_extended') if isinstance(res, dict) else None
    if isinstance(ext, dict):
        row['confidence_extended_auroc'] = ext.get('auroc')
        row['confidence_extended_balacc'] = ext.get('balanced_accuracy')
        row['confidence_extended_adv'] = ext.get('advantage')
        row['confidence_extended_thr'] = ext.get('threshold')
        # PR metrics
        if 'ap' in ext:
            row['confidence_extended_ap'] = ext.get('ap')
        if 'precision' in ext:
            row['confidence_extended_precision'] = ext.get('precision')
        if 'recall' in ext:
            row['confidence_extended_recall'] = ext.get('recall')
        if 'f1' in ext:
            row['confidence_extended_f1'] = ext.get('f1')
        # TPR@FPRs
        tprs = ext.get('tpr_at_fprs') or {}
        if isinstance(tprs, dict):
            for fk, fv in tprs.items():
                key = f"tpr_at_fpr_{str(fk).replace('.', '_')}"
                row[key] = fv
        if 'tpr_at_1fpr' in ext and 'tpr_at_fpr_1' not in row:
            row['tpr_at_fpr_1'] = ext.get('tpr_at_1fpr')

    # Helper to pull classifier blocks
    def pull_attack_block(prefix: str, block: dict):
        if not isinstance(block, dict):
            return
        # Try common keys with fallbacks
        acc = block.get('acc', block.get('accuracy'))
        auc = block.get('auc', block.get('auroc'))
        bal = block.get('balanced_accuracy', block.get('balacc'))
        adv = block.get('advantage')
        ap  = block.get('ap', block.get('average_precision'))
        prec = block.get('precision')
        rec  = block.get('recall')
        f1   = block.get('f1')
        if acc is not None:
            row[f'{prefix}_acc'] = acc
        if auc is not None:
            row[f'{prefix}_auc'] = auc
        if bal is not None:
            row[f'{prefix}_balacc'] = bal
        if adv is not None:
            row[f'{prefix}_adv'] = adv
        if ap is not None:
            row[f'{prefix}_ap'] = ap
        if prec is not None:
            row[f'{prefix}_precision'] = prec
        if rec is not None:
            row[f'{prefix}_recall'] = rec
        if f1 is not None:
            row[f'{prefix}_f1'] = f1

    pull_attack_block('samia', res.get('samia'))
    pull_attack_block('nn', res.get('nn'))
    pull_attack_block('nn_top3', res.get('nn_top3'))
    pull_attack_block('nn_cls', res.get('nn_cls'))
    pull_attack_block('lira', res.get('lira'))

    # Fallbacks from filename for key fields
    try:
        m = re.match(r'(?P<ds>[^_]+)_sparsity_(?P<s>[0-9.]+)_victim(?P<seed>\d+)\.json', fp.name)
        if (not row.get('dataset')) and m:
            row['dataset'] = m.group('ds')
        if (row.get('sparsity') is None) and m:
            row['sparsity'] = float(m.group('s'))
        if (row.get('victim_seed') is None) and m:
            row['victim_seed'] = int(m.group('seed'))
    except Exception:
        pass

    # Method/mode fallbacks mirroring tidy parser
    if not row.get('method'):
        fm = (row.get('forward_mode') or '').lower()
        if 'dpf' in fm:
            row['method'] = 'dpf'
        elif 'standard' in fm or 'static' in fm:
            row['method'] = 'static'
        else:
            row['method'] = 'unknown'
    if (not row.get('mode')) or row['mode'] in ('unknown:na', ''):
        row['mode'] = row.get('forward_mode') or row.get('method')

    return row


def accuracy_matching(df: pd.DataFrame, band_pp: float) -> pd.DataFrame:
    """Filter within ±band_pp around median accuracy per (dataset, method, mode, sparsity).
    If a group's accuracy is missing/NaN, skip filtering for that group.
    """
    if df.empty or 'victim_test_acc' not in df.columns:
        return df
    keep = []
    for keys, g in df.groupby(['dataset', 'method', 'mode', 'sparsity'], dropna=False):
        acc = pd.to_numeric(g['victim_test_acc'], errors='coerce')
        med = float(acc.median()) if acc.notna().any() else float('nan')
        if math.isnan(med):
            keep.append(g)
            continue
        lo, hi = med - band_pp, med + band_pp
        mask = (acc >= lo) & (acc <= hi)
        keep.append(g[mask])
    return pd.concat(keep, ignore_index=True) if keep else df


def strict_pair_first(df: pd.DataFrame) -> pd.DataFrame:
    """Within each (dataset,sparsity,attack,metric) block, keep only the
    victim seeds that are present for all (method,mode) combos in that block.
    Drops blocks whose common seed intersection is empty.
    """
    if df.empty or 'victim_seed' not in df.columns:
        return df
    kept = []
    for keys, sub in df.groupby(['dataset', 'sparsity', 'attack', 'metric'], dropna=False):
        combos = sub[['method', 'mode']].drop_duplicates()
        if len(combos) <= 1:
            kept.append(sub); continue
        seeds_by_combo = {}
        for r in combos.itertuples(index=False):
            mask = (sub['method'] == r.method) & (sub['mode'] == r.mode)
            S = set(sub.loc[mask, 'victim_seed'].dropna().astype(int).tolist())
            seeds_by_combo[(r.method, r.mode)] = S
        if not seeds_by_combo:
            continue
        common = set.intersection(*seeds_by_combo.values())
        if not common:
            # No common seeds; skip this block
            continue
        kept.append(sub[sub['victim_seed'].isin(common)])
    return pd.concat(kept, ignore_index=True) if kept else df


def joint_accuracy_match(df: pd.DataFrame, band_pp: float) -> pd.DataFrame:
    """After fixing common seeds, require each seed to pass accuracy matching
    simultaneously for all (method,mode) within a block.
    """
    if df.empty or 'victim_test_acc' not in df.columns:
        return df
    df2 = df.copy()
    med = df2.groupby(['dataset','method','mode','sparsity'], dropna=False)['victim_test_acc'].transform('median')
    df2['acc_pass'] = (df2['victim_test_acc'] >= (med - band_pp)) & (df2['victim_test_acc'] <= (med + band_pp))
    pass_seeds = (df2.groupby(['dataset','sparsity','attack','metric','victim_seed'], dropna=False)['acc_pass']
                     .all().reset_index())
    pass_seeds = pass_seeds[pass_seeds['acc_pass']][['dataset','sparsity','attack','metric','victim_seed']]
    out = df2.merge(pass_seeds, on=['dataset','sparsity','attack','metric','victim_seed'], how='inner')
    return out.drop(columns=['acc_pass'])


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


def plot_bar_ci(summary: pd.DataFrame, title: str, ylab: str, out_png: Path, group_key: str = 'mode'):
    if not HAS_PLOT or summary.empty:
        return
    label_key = group_key if group_key in summary.columns else 'method'
    labels = sorted(summary[label_key].dropna().unique())
    sparsities = sorted(summary['sparsity'].dropna().unique())
    w = 0.8 / max(1, len(labels))
    import matplotlib.pyplot as plt  # lazy import for headless envs
    # Matplotlib 3.7+ deprecates cm.get_cmap; use colormaps API when available
    try:
        import matplotlib as mpl
        palette = [mpl.colormaps.get_cmap('tab10')(i % mpl.colormaps.get_cmap('tab10').N)
                   for i in range(max(1, len(labels)))]
    except Exception:
        import matplotlib.cm as cm
        cmap_fallback = cm.get_cmap('tab10', max(3, len(labels)))
        palette = [cmap_fallback(i) for i in range(max(1, len(labels)))]
    # stable colors per label across sparsities
    colors = {lab: palette[i] for i, lab in enumerate(labels)}
    legend_handles = {}
    plt.figure(figsize=(10, 4 + 0.2 * len(sparsities)))
    for i, s in enumerate(sparsities):
        g = summary[summary['sparsity'] == s]
        for j, lab in enumerate(labels):
            row = g[g[label_key] == lab]
            if row.empty:
                continue
            mu = float(row['mean'].iloc[0])
            lo = float(row['ci95_lo'].iloc[0])
            hi = float(row['ci95_hi'].iloc[0])
            x = i + (j - (len(labels) - 1) / 2) * w
            bar = plt.bar(x, mu, width=w, color=colors[lab])
            # remember a single handle per label for legend
            if lab not in legend_handles and len(bar) > 0:
                legend_handles[lab] = bar[0]
            if not math.isnan(lo) and not math.isnan(hi):
                plt.plot([x, x], [lo, hi], color='black')
    plt.xticks(range(len(sparsities)), [f's={s}' for s in sparsities])
    plt.ylabel(ylab)
    plt.title(title)
    # legend: map color → label (mode/method)
    if legend_handles:
        plt.legend(legend_handles.values(), legend_handles.keys(), title=label_key, fontsize=8)
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
    wide_rows = []
    files = list(src.rglob('*.json'))
    log(f"Scanning {src} -> found {len(files)} JSON files")
    for fp in files:
        rows = parse_one_json(fp)
        all_rows.extend(rows)
        if args.write_wide:
            wr = parse_one_json_wide(fp)
            if wr is not None:
                wide_rows.append(wr)
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.drop_duplicates(subset=['file', 'attack', 'metric'])
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
    if args.modes:
        df = df[df['mode'].isin(args.modes)]
        log(f"Filter modes={args.modes} -> {len(df)} rows")
    if args.victims:
        df = df[df['victim_seed'].isin(args.victims)]
        log(f"Filter victims={args.victims} -> {len(df)} rows")

    # coerce numeric sparsity
    if 'sparsity' in df.columns:
        df['sparsity'] = pd.to_numeric(df['sparsity'], errors='coerce')
    log(f"After filters: rows={len(df)}, unique victims={df['victim_seed'].nunique() if 'victim_seed' in df.columns else 'NA'}, unique methods={df['method'].nunique() if 'method' in df.columns else 'NA'}, unique sparsities={df['sparsity'].nunique() if 'sparsity' in df.columns else 'NA'}")

    # 2) Pair-then-match flow (optional) or legacy flow
    if args.pair_then_match:
        before_rows = len(df)
        dfm = strict_pair_first(df)
        log(f"strict_pair_first: {before_rows} -> {len(dfm)} rows")
        before_rows2 = len(dfm)
        dfm = joint_accuracy_match(dfm, args.acc_match_pp)
        log(f"joint_accuracy_match ±{args.acc_match_pp}pp: {before_rows2} -> {len(dfm)} rows")
    else:
        before_rows = len(df)
        dfm = accuracy_matching(df, args.acc_match_pp)
        log(f"Accuracy matching ±{args.acc_match_pp}pp: kept {len(dfm)}/{before_rows} rows")
    if args.verbose and not dfm.empty:
        try:
            grp = dfm.groupby(['method','sparsity'], dropna=False)['victim_test_acc'].agg(['count','min','median','max']).reset_index()
            log("Post-match group stats (method,sparsity):\n" + grp.head(20).to_string(index=False))
        except Exception:
            pass

    # 2.5) strict paired filter: keep only seeds present for all (method,mode) combos per block
    if (not args.pair_then_match) and args.strict_paired and not dfm.empty and 'victim_seed' in dfm.columns:
        kept_blocks = []
        total_blocks = 0
        changed_blocks = 0
        for keys, sub in dfm.groupby(['dataset', 'sparsity', 'attack', 'metric'], dropna=False):
            total_blocks += 1
            mm = sub[['method', 'mode']].drop_duplicates()
            need = len(mm)
            if need <= 1:
                kept_blocks.append(sub)
                continue
            counts = (sub[['victim_seed', 'method', 'mode']]
                        .drop_duplicates()
                        .groupby('victim_seed').size())
            ok_seeds = counts[counts == need].index
            kept = sub[sub['victim_seed'].isin(ok_seeds)]
            if kept['victim_seed'].nunique() != sub['victim_seed'].nunique():
                changed_blocks += 1
            kept_blocks.append(kept)
        new_dfm = pd.concat(kept_blocks, ignore_index=True) if kept_blocks else dfm
        log(f"[strict_paired] methods per block matched; blocks={total_blocks}, reduced={changed_blocks}, rows {len(dfm)} -> {len(new_dfm)}")
        dfm = new_dfm
    else:
        log("[strict_paired] OFF or no victim_seed column")

    # === Debug: detect blocks with unequal n across methods and save diagnostics ===
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Pre-scan common seeds per block (before making summary) for quick visibility
    if not df.empty and 'victim_seed' in df.columns:
        dbg_rows = []
        for keys, sub0 in df.groupby(['dataset','sparsity','attack','metric'], dropna=False):
            combos0 = sub0[['method','mode']].drop_duplicates().sort_values(['method','mode'])
            seeds_by_combo0 = {}
            for r in combos0.itertuples(index=False):
                mask0 = (sub0['method']==r.method) & (sub0['mode']==r.mode)
                S0 = set(sub0.loc[mask0, 'victim_seed'].dropna().astype(int).tolist())
                seeds_by_combo0[(r.method, r.mode)] = S0
            if not seeds_by_combo0:
                continue
            inter0 = set.intersection(*seeds_by_combo0.values()) if len(seeds_by_combo0)>1 else next(iter(seeds_by_combo0.values()))
            row = dict(dataset=keys[0], sparsity=keys[1], attack=keys[2], metric=keys[3],
                       n_combos=len(seeds_by_combo0), n_inter=len(inter0))
            for (m0, mo0), S0 in seeds_by_combo0.items():
                row[f"{m0}|{mo0}#seeds"] = len(S0)
            dbg_rows.append(row)
        import pandas as _pd
        _pd.DataFrame(dbg_rows).to_csv(debug_dir/"common_seed_scan.csv", index=False)
        log(f"[debug] common seed scan -> {debug_dir/'common_seed_scan.csv'}")

    if not dfm.empty and 'victim_seed' in dfm.columns:
        gseed = (dfm.groupby(['dataset','sparsity','attack','metric','method','mode'], dropna=False)
                    ['victim_seed'].nunique().reset_index(name='n_seeds'))
        span = (gseed.groupby(['dataset','sparsity','attack','metric'], dropna=False)
                    .agg(n_min=('n_seeds','min'), n_max=('n_seeds','max'), n_methods=('method','nunique'))
                    .reset_index())
        bad = span[span['n_min'] != span['n_max']].copy()
        bad_path = debug_dir / 'mismatch_blocks.csv'
        bad.to_csv(bad_path, index=False)
        log(f"[debug] mismatch blocks saved to {bad_path} (rows={len(bad)})")

        if not bad.empty:
            b = bad.iloc[0]
            mask = (
                (dfm['dataset'] == b['dataset']) &
                (dfm['sparsity'] == b['sparsity']) &
                (dfm['attack']   == b['attack']) &
                (dfm['metric']   == b['metric'])
            )
            sub = dfm[mask].copy()
            pres = (sub[['victim_seed','method','mode']]
                        .drop_duplicates()
                        .assign(present=1)
                        .pivot_table(index='victim_seed', columns=['method','mode'], values='present', fill_value=0)
                        .sort_index())
            pres_path = debug_dir / f"presence_{b['dataset']}_s{b['sparsity']}_{b['attack']}_{b['metric']}.csv"
            pres.to_csv(pres_path)
            log(f"[debug] presence matrix saved to {pres_path}")

            miss_rows = []
            for col in pres.columns:
                miss = pres.index[pres[col] == 0].tolist()
                miss_rows.append({
                    'dataset': b['dataset'],
                    'sparsity': b['sparsity'],
                    'attack': b['attack'],
                    'metric': b['metric'],
                    'method': col[0],
                    'mode': col[1],
                    'missing_seeds': ' '.join(map(str, miss)),
                    'n_missing': len(miss)
                })
            import pandas as _pd
            _pd.DataFrame(miss_rows).to_csv(debug_dir / f"missing_seeds_{b['dataset']}_s{b['sparsity']}_{b['attack']}_{b['metric']}.csv", index=False)
            log("[debug] missing seeds per method saved")

        if args.assert_equal_n and not bad.empty:
            raise RuntimeError(f"Equal-n assertion failed for {len(bad)} blocks. See {bad_path}")

    # 3) grouped summary (by dataset, method, mode, sparsity, attack, metric[, use_temperature])
    agg_rows = []
    base_keys = ['dataset', 'method', 'mode', 'sparsity', 'attack', 'metric']
    include_temp = (args.split_by_temperature and ('use_temperature' in dfm.columns) and dfm['use_temperature'].notna().any())
    group_keys = base_keys + (['use_temperature'] if include_temp else [])
    # Group with dropna=False so NaN keys don't drop all rows
    for keys, g in dfm.groupby(group_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        keymap = {k: v for k, v in zip(group_keys, keys)}
        # Aggregate per victim to ensure one sample per victim
        if 'victim_seed' in g.columns:
            vals = g.groupby('victim_seed')['value'].mean().values
            n_victims = g['victim_seed'].nunique()
        else:
            vals = g['value'].values
            n_victims = len(vals)
        m, sd, lo, hi, _ = ci95(vals)
        med = float(np.nanmedian(vals)) if len(vals) else np.nan
        q75 = float(np.nanpercentile(vals, 75)) if len(vals) else np.nan
        q25 = float(np.nanpercentile(vals, 25)) if len(vals) else np.nan
        agg_rows.append({
            'dataset': keymap.get('dataset'),
            'method': keymap.get('method'),
            'mode': keymap.get('mode'),
            'sparsity': keymap.get('sparsity'),
            'attack': keymap.get('attack'),
            'metric': keymap.get('metric'),
            'use_temperature': keymap.get('use_temperature', None),
            'mean': m, 'std': sd, 'ci95_lo': lo, 'ci95_hi': hi, 'n': int(n_victims),
            'median': med,
            'iqr': (q75 - q25) if (not math.isnan(q75) and not math.isnan(q25)) else np.nan
        })
    summary = pd.DataFrame(agg_rows)
    if not summary.empty:
        if 'mode' not in summary.columns:
            summary['mode'] = None
        summary = summary.sort_values(['dataset', 'attack', 'metric', 'sparsity', 'method', 'mode'])
    else:
        # Ensure expected columns exist for downstream consumers
        for col in ['dataset','method','mode','sparsity','attack','metric','use_temperature','mean','std','ci95_lo','ci95_hi','n','median','iqr']:
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

    # 3.5) Optional wide CSV per experiment
    if args.write_wide:
        wide_df = pd.DataFrame(wide_rows)
        # Coerce sparsity to numeric
        if 'sparsity' in wide_df.columns:
            wide_df['sparsity'] = pd.to_numeric(wide_df['sparsity'], errors='coerce')
        # Stable column ordering: metadata first, then metrics sorted
        meta_cols = [
            'file','dataset','arch','method','mode','forward_mode','sparsity','victim_seed','victim_test_acc','use_temperature','shadow_count'
        ]
        metric_cols = sorted([c for c in wide_df.columns if c not in meta_cols])
        cols = [c for c in meta_cols if c in wide_df.columns] + metric_cols
        wide_df = wide_df[cols]
        wide_path = out_dir / args.wide_filename
        wide_df.to_csv(wide_path, index=False)
        print(f"Wrote wide CSV: {wide_path} ({len(wide_df)} rows, {len(wide_df.columns)} columns)")

    # 4) Standard plots (optional)
    if args.make_plots and HAS_PLOT:
        # LiRA TPR@1 (bar+CI)
        sub = summary[(summary.attack == 'lira') & (summary.metric == 'tpr@1')]
        plot_bar_ci(sub, 'LiRA TPR@1%FPR (↓ lower is better)', 'TPR@1%FPR', plots_dir / 'lira_tpr1_bar.png', group_key='mode')
        # LiRA AUROC (bar+CI)
        sub = summary[(summary.attack == 'lira') & (summary.metric == 'auroc')]
        plot_bar_ci(sub, 'LiRA AUROC (↓ closer to 0.5 is better)', 'AUROC', plots_dir / 'lira_auroc_bar.png', group_key='mode')

    if args.make_plots and not HAS_PLOT:
        print('matplotlib not available; skipped plot generation')


if __name__ == '__main__':
    main()
