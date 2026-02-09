#!/usr/bin/env python3
"""Export MIA JSON results into a single wide CSV sorted by victim seed."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


def derive_mode_label(cfg: Dict[str, object]) -> str:
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


def sanitize_key(value: object) -> str:
    return str(value).replace(' ', '_').replace('.', '_').replace('%', 'pct')


def add_confidence_block(row: Dict[str, object], block: Dict[str, object]) -> None:
    if not isinstance(block, dict):
        return
    row['confidence_extended_auroc'] = block.get('auroc', block.get('auc'))
    row['confidence_extended_balacc'] = block.get('balanced_accuracy')
    row['confidence_extended_adv'] = block.get('advantage')
    row['confidence_extended_thr'] = block.get('threshold')
    row['confidence_extended_ap'] = block.get('ap')
    row['confidence_extended_precision'] = block.get('precision')
    row['confidence_extended_recall'] = block.get('recall')
    row['confidence_extended_f1'] = block.get('f1')
    tprs = block.get('tpr_at_fprs')
    if isinstance(tprs, dict):
        for fpr, tpr in tprs.items():
            row[f'tpr_at_fpr_{sanitize_key(fpr)}'] = tpr
    if 'tpr_at_1fpr' in block and 'tpr_at_fpr_1' not in row:
        row['tpr_at_fpr_1'] = block['tpr_at_1fpr']


def add_attack_block(prefix: str, row: Dict[str, object], block: Dict[str, object]) -> None:
    if not isinstance(block, dict):
        return
    row[f'{prefix}_acc'] = block.get('accuracy', block.get('acc'))
    row[f'{prefix}_auc'] = block.get('auc', block.get('auroc'))
    row[f'{prefix}_balacc'] = block.get('balanced_accuracy', block.get('balacc'))
    row[f'{prefix}_adv'] = block.get('advantage')
    row[f'{prefix}_ap'] = block.get('ap', block.get('average_precision'))
    row[f'{prefix}_precision'] = block.get('precision')
    row[f'{prefix}_recall'] = block.get('recall')
    row[f'{prefix}_f1'] = block.get('f1')
    if prefix == 'lira':
        row[f'{prefix}_member_mean'] = block.get('member_mean')
        row[f'{prefix}_member_std'] = block.get('member_std')
        row[f'{prefix}_nonmember_mean'] = block.get('nonmember_mean')
        row[f'{prefix}_nonmember_std'] = block.get('nonmember_std')
        row[f'{prefix}_tpr_at_1fpr'] = block.get('tpr_at_1fpr')
        tprs = block.get('tpr_at_fprs')
        if isinstance(tprs, dict):
            for fpr, tpr in tprs.items():
                row[f'lira_tpr_at_fpr_{sanitize_key(fpr)}'] = tpr


def parse_json_file(fp: Path) -> Dict[str, object]:
    data = json.loads(fp.read_text())
    cfg = data.get('config', {})
    exp = data.get('experiment_info', {})
    results = data.get('results', {})

    row: Dict[str, object] = {
        'file': str(fp),
        'dataset': cfg.get('dataset_name') or data.get('dataset'),
        'method': (cfg.get('prune_method') or '').lower() if cfg.get('prune_method') else None,
        'mode': derive_mode_label(cfg),
        'sparsity': cfg.get('sparsity'),
        'victim_seed': cfg.get('victim_seed'),
        'seed': cfg.get('seed'),
        'victim_test_acc': data.get('victim_test_acc'),
    }

    if not row.get('method'):
        fm = (exp.get('forward_mode') or cfg.get('forward_mode') or '').lower()
        if 'dpf' in fm:
            row['method'] = 'dpf'
        elif 'static' in fm or 'standard' in fm:
            row['method'] = 'static'
        else:
            row['method'] = 'unknown'

    # Shadow model count convenience column
    shadow_cfgs = exp.get('shadow_configs') if isinstance(exp, dict) else None
    if isinstance(shadow_cfgs, dict) and shadow_cfgs:
        row['shadow_count'] = len(shadow_cfgs)
    else:
        seeds = cfg.get('shadow_seeds')
        if isinstance(seeds, list):
            row['shadow_count'] = len(seeds)

    for scalar_key in ('confidence', 'entropy', 'modified_entropy', 'top1_conf'):
        if isinstance(results, dict) and scalar_key in results:
            row[scalar_key] = results[scalar_key]

    add_confidence_block(row, results.get('confidence_extended'))
    add_attack_block('samia', row, results.get('samia'))
    add_attack_block('nn', row, results.get('nn'))
    add_attack_block('nn_top3', row, results.get('nn_top3'))
    add_attack_block('nn_cls', row, results.get('nn_cls'))
    add_attack_block('lira', row, results.get('lira'))

    # Victim seed fallback from filename pattern when missing
    if not row.get('victim_seed'):
        stem = fp.stem
        for part in stem.split('_'):
            if part.startswith('victim'):
                try:
                    row['victim_seed'] = int(part.replace('victim', ''))
                except Exception:
                    pass

    return row


def gather_json_files(src_dir: Path, recursive: bool) -> List[Path]:
    pattern = '**/*.json' if recursive else '*.json'
    return sorted(src_dir.glob(pattern))


def sort_key(row: Dict[str, object]):
    seed = row.get('victim_seed')
    try:
        seed_val = int(seed) if seed is not None else float('inf')
    except Exception:
        seed_val = float('inf')
    return (seed_val, row.get('file', ''))


def pick_field_order(all_keys: Iterable[str]) -> List[str]:
    preferred = [
        'victim_seed', 'file', 'dataset', 'method', 'mode', 'sparsity', 'alpha', 'beta',
        'victim_test_acc', 'shadow_count',
        'confidence', 'entropy', 'modified_entropy', 'top1_conf',
        'confidence_extended_auroc', 'confidence_extended_balacc',
        'confidence_extended_adv', 'confidence_extended_thr',
        'confidence_extended_ap', 'confidence_extended_precision',
        'confidence_extended_recall', 'confidence_extended_f1',
        'tpr_at_fpr_0_1', 'tpr_at_fpr_1', 'tpr_at_fpr_5',
        'samia_acc', 'samia_auc', 'samia_balacc', 'samia_adv', 'samia_ap',
        'samia_precision', 'samia_recall', 'samia_f1',
        'nn_acc', 'nn_auc', 'nn_balacc', 'nn_adv', 'nn_ap', 'nn_precision',
        'nn_recall', 'nn_f1',
        'nn_top3_acc', 'nn_top3_auc', 'nn_top3_balacc', 'nn_top3_adv',
        'nn_top3_ap', 'nn_top3_precision', 'nn_top3_recall', 'nn_top3_f1',
        'nn_cls_acc', 'nn_cls_auc', 'nn_cls_balacc', 'nn_cls_adv',
        'nn_cls_ap', 'nn_cls_precision', 'nn_cls_recall', 'nn_cls_f1',
        'lira_acc', 'lira_auc', 'lira_adv', 'lira_ap', 'lira_precision',
        'lira_recall', 'lira_f1', 'lira_balacc', 'lira_member_mean',
        'lira_member_std', 'lira_nonmember_mean', 'lira_nonmember_std',
        'lira_tpr_at_1fpr'
    ]
    ordered: List[str] = []
    for key in preferred:
        if key in all_keys and key not in ordered:
            ordered.append(key)
    for key in sorted(all_keys):
        if key not in ordered:
            ordered.append(key)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description='Aggregate MIA JSONs into a CSV sorted by victim seed')
    parser.add_argument('--src', default='mia_results', help='Directory containing result JSON files')
    parser.add_argument('--output', default='results/mia_results_summary.csv',
                        help='Path to write the aggregated CSV')
    parser.add_argument('--recursive', action='store_true',
                        help='Recursively search for JSON files under --src')
    args = parser.parse_args()

    src_dir = Path(args.src).expanduser()
    if not src_dir.exists():
        raise SystemExit(f'Source directory not found: {src_dir}')

    files = gather_json_files(src_dir, args.recursive)
    if not files:
        raise SystemExit(f'No JSON files found under {src_dir}')

    rows: List[Dict[str, object]] = []
    for fp in files:
        try:
            row = parse_json_file(fp)
        except Exception as exc:
            print(f'Skipped {fp}: {exc}')
            continue
        try:
            relative = fp.relative_to(src_dir)
            row['file'] = str(relative)
        except ValueError:
            row['file'] = str(fp)
        rows.append(row)

    if not rows:
        raise SystemExit('No valid result rows parsed; aborting')

    rows.sort(key=sort_key)

    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    fieldnames = pick_field_order(all_keys)

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})

    print(f'Wrote {len(rows)} rows to {output_path}')


if __name__ == '__main__':
    main()
