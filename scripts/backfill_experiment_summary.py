#!/usr/bin/env python3
"""
Backfill experiment_summary.json for existing runs that are missing it.

Heuristics:
- If validation_history.json exists: use it to compute best_acc1, best_loss, final_acc1, final_loss.
- Else if *_acc_log.txt exists: parse last line to fill best_acc1 and final_acc1.
- total_duration is left empty unless provided elsewhere.

Usage:
  python scripts/backfill_experiment_summary.py --runs ./runs
"""

import argparse
from pathlib import Path
import json


def parse_acc_log(path: Path):
    last = None
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('epoch'):
                    continue
                last = line
        if last:
            parts = last.split('\t')
            if len(parts) >= 4:
                # epoch, acc1_train, acc1_valid, best_acc1
                return {
                    'final_acc1': float(parts[2]),
                    'best_acc1': float(parts[3])
                }
    except Exception:
        pass
    return {}


def build_summary_from_history(hist_path: Path):
    try:
        with open(hist_path) as f:
            hist = json.load(f)
        if not isinstance(hist, list) or not hist:
            return {}
        # Each entry: {'epoch': int, 'acc1': float, 'acc5': float, 'loss': float, ...}
        best_acc1 = max((e.get('acc1', 0.0) for e in hist if isinstance(e, dict)), default=None)
        best_loss = min((e.get('loss', float('inf')) for e in hist if isinstance(e, dict)), default=None)
        last = hist[-1]
        summary = {
            'best_metrics': {
                'best_acc1': best_acc1,
                'best_loss': best_loss,
            },
            'final_metrics': {
                'acc1': last.get('acc1'),
                'acc5': last.get('acc5'),
                'loss': last.get('loss'),
            },
        }
        return summary
    except Exception:
        return {}


def backfill_one(run_dir: Path) -> bool:
    exp_summary = run_dir / 'experiment_summary.json'
    if exp_summary.exists():
        return False
    # Try validation_history.json first
    hist = run_dir / 'validation_history.json'
    if hist.exists():
        summary = build_summary_from_history(hist)
        if summary:
            exp_summary.write_text(json.dumps(summary, indent=2))
            return True
    # Fallback: acc log
    acc_logs = list(run_dir.glob('*_acc_log.txt'))
    if acc_logs:
        acc = parse_acc_log(acc_logs[0])
        if acc:
            summary = {
                'best_metrics': {
                    'best_acc1': acc.get('best_acc1'),
                },
                'final_metrics': {
                    'acc1': acc.get('final_acc1'),
                }
            }
            exp_summary.write_text(json.dumps(summary, indent=2))
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='./runs', help='Root runs directory')
    args = ap.parse_args()

    runs_root = Path(args.runs)
    created = 0
    scanned = 0
    for pth in runs_root.rglob('best_model.pth'):
        run_dir = pth.parent
        scanned += 1
        if backfill_one(run_dir):
            print(f"Created: {run_dir}/experiment_summary.json")
            created += 1

    print(f"Backfill complete. Scanned {scanned} runs. Created {created} summaries.")


if __name__ == '__main__':
    main()

