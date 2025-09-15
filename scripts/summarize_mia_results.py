#!/usr/bin/env python3
"""
Summarize existing MIA JSON results in mia_results/ into a single CSV.

- Scans mia_results/**.json
- Extracts config and flatten metrics
- Derives readable mode labels (dwa mode, static, dpf:<tag>, dense)
"""

import os, json, argparse, csv
from pathlib import Path


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


def flatten_metrics(data: dict) -> dict:
    cfg = data.get('config', {})
    res = data.get('results', {})
    row = {
        'method': cfg.get('prune_method'),
        'mode': derive_mode_label(cfg),
        'dataset': cfg.get('dataset_name'),
        'sparsity': cfg.get('sparsity'),
        'alpha': cfg.get('alpha'),
        'beta': cfg.get('beta'),
        'freeze_tag': cfg.get('freeze_tag'),
        'victim_seed': cfg.get('victim_seed'),
        'victim_acc': data.get('victim_test_acc'),
    }

    # threshold-style scalars
    for k in ['confidence','entropy','modified_entropy','top1_conf']:
        if k in res:
            row[k] = res[k]

    # confidence_extended
    ext = res.get('confidence_extended')
    if isinstance(ext, dict):
        row['confidence_extended_auroc'] = ext.get('auroc')
        row['confidence_extended_balacc'] = ext.get('balanced_accuracy')
        row['confidence_extended_adv'] = ext.get('advantage')
        row['confidence_extended_thr'] = ext.get('threshold')

    # helper for classifier-based attacks
    def pull_attack(prefix: str):
        v = res.get(prefix)
        if isinstance(v, dict):
            row[f'{prefix}_acc'] = v.get('accuracy')
            row[f'{prefix}_auc'] = v.get('auc')
            row[f'{prefix}_balacc'] = v.get('balanced_accuracy')
            row[f'{prefix}_adv'] = v.get('advantage')

    for name in ['samia','nn','nn_top3','nn_cls','lira']:
        pull_attack(name)

    return row


def main():
    ap = argparse.ArgumentParser(description='Summarize existing MIA results into CSV')
    ap.add_argument('--results_dir', default='mia_results', help='Root of MIA results JSONs')
    ap.add_argument('--out', default='results/mia_results_summary.csv', help='Output CSV path')
    args = ap.parse_args()

    root = Path(args.results_dir)
    if not root.exists():
        print(f'No results directory found: {root}')
        return

    files = list(root.rglob('*.json'))
    if not files:
        print(f'No JSON results found under {root}')
        return

    rows = []
    for jf in files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
            rows.append(flatten_metrics(data))
        except Exception as e:
            print(f'⚠️ skip {jf}: {e}')

    # determine fieldnames
    fieldnames = [
        'method','mode','dataset','sparsity','alpha','beta','freeze_tag','victim_seed','victim_acc',
        'confidence','entropy','modified_entropy','top1_conf',
        'confidence_extended_auroc','confidence_extended_balacc','confidence_extended_adv','confidence_extended_thr',
        'samia_acc','samia_auc','samia_balacc','samia_adv',
        'nn_acc','nn_auc','nn_balacc','nn_adv',
        'nn_top3_acc','nn_top3_auc','nn_top3_balacc','nn_top3_adv',
        'nn_cls_acc','nn_cls_auc','nn_cls_balacc','nn_cls_adv',
        'lira_acc','lira_auc','lira_balacc','lira_adv'
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})
    print(f'📄 Wrote CSV: {out_path} ({len(rows)} rows from {len(files)} files)')


if __name__ == '__main__':
    main()

