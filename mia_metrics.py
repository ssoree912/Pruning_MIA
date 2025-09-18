#!/usr/bin/env python3
"""
Lightweight MIA metrics helper.

Provides compute_mia_metrics() and print_mia_metrics() so callers can
compute AUROC, Accuracy, Balanced Accuracy, Youden-threshold, Advantage,
and TPR@X%FPR style diagnostics from a score vector.

This file is intentionally dependency-light (uses scikit-learn if present).
"""

from typing import Dict, Sequence, Optional
import numpy as np

try:
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
    HAS_SK = True
except Exception:
    HAS_SK = False


def _youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Evaluate at unique score values to find argmax(TPR - FPR)
    vals = np.unique(y_score)
    best_adv, best_thr = -1.0, 0.5
    for thr in vals:
        y_pred = (y_score >= thr).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum(); fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum(); fp = ((y_pred == 1) & (y_true == 0)).sum()
        tpr = tp / (tp + fn + 1e-8); fpr = fp / (fp + tn + 1e-8)
        adv = tpr - fpr
        if adv > best_adv:
            best_adv, best_thr = adv, thr
    return float(best_thr)


def compute_mia_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold_strategy: str = 'youden',
    tpr_fprs: Optional[Sequence[float]] = None,
) -> Dict:
    """Compute core MIA metrics from labels (1=member, 0=non-member) and scores.

    Args:
      y_true: binary labels (1=member, 0=non-member)
      y_score: continuous scores (higher ⇒ more likely member)
      threshold_strategy: currently only 'youden'
      tpr_fprs: list of FPR percentages at which to report TPR
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if tpr_fprs is None:
        tpr_fprs = [0.1, 1.0, 5.0]

    # AUROC
    if HAS_SK and len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_score))
    else:
        auc = 0.0

    # Threshold selection
    if threshold_strategy == 'youden':
        thr = _youden_threshold(y_true, y_score)
    else:
        thr = _youden_threshold(y_true, y_score)

    y_pred = (y_score >= thr).astype(int)

    # Accuracy / Balanced Accuracy
    if HAS_SK:
        acc = float(accuracy_score(y_true, y_pred))
        bal = float(balanced_accuracy_score(y_true, y_pred))
    else:
        acc = float((y_pred == y_true).mean())
        # simple balanced accuracy proxy if sklearn missing
        pos = (y_true == 1); neg = (y_true == 0)
        tpr = float((y_pred[pos] == 1).mean()) if pos.any() else 0.0
        tnr = float((y_pred[neg] == 0).mean()) if neg.any() else 0.0
        bal = (tpr + tnr) / 2.0

    # Advantage
    tp = ((y_pred == 1) & (y_true == 1)).sum(); fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum(); fp = ((y_pred == 1) & (y_true == 0)).sum()
    tpr = tp / (tp + fn + 1e-8); fpr = fp / (fp + tn + 1e-8)
    advantage = float(tpr - fpr)

    # TPR@X%FPR
    tprs = {}
    non_member = y_score[y_true == 0]
    member = y_score[y_true == 1]
    if non_member.size > 0 and member.size > 0:
        for fpr_pct in tpr_fprs:
            q = max(0.0, min(1.0, 1.0 - (fpr_pct/100.0)))
            tau = float(np.quantile(non_member, q))
            tprs[f"{fpr_pct:g}"] = float((member >= tau).mean())

    return {
        'auroc': auc,
        'accuracy': acc,
        'balanced_accuracy': bal,
        'advantage': advantage,
        'threshold': float(thr),
        'tpr_at_fprs': tprs,
        'tpr_at_1fpr': tprs.get('1', None),
        'threshold_strategy': threshold_strategy,
    }


def print_mia_metrics(metrics: Dict, prefix: str = "📊 MIA"):
    parts = [
        f"AUROC={metrics.get('auroc', 0.0):.4f}",
        f"Acc={metrics.get('accuracy', 0.0):.4f}",
        f"BalAcc={metrics.get('balanced_accuracy', 0.0):.4f}",
        f"Adv={metrics.get('advantage', 0.0):.4f}",
        f"Thr={metrics.get('threshold', 0.5):.4f}",
    ]
    tprs = metrics.get('tpr_at_fprs') or {}
    if tprs:
        tpr_msg = ", ".join([f"TPR@{k}%FPR={v:.4f}" for k, v in sorted(tprs.items(), key=lambda x: float(x[0]))])
        parts.append(tpr_msg)
    print(f"{prefix}: " + " | ".join(parts))

