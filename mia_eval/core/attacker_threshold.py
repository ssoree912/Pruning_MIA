"""
Threshold-based MIA attacker used by attackers.py.

This version keeps the original class name/API, but fixes a few evaluation issues:
- threshold selection is performed on shadow data only (never on victim scores),
- threshold strategy is configurable,
- top1_conf really uses max-confidence instead of true-label confidence,
- returned metrics are on a consistent 0-1 scale.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ThresholdAttacker:
    def __init__(
        self,
        shadow_train_performance,
        shadow_test_performance,
        target_train_performance,
        target_test_performance,
        num_classes,
        threshold_strategy='max_accuracy',
        tpr_fprs=None,
    ):
        self.num_classes = num_classes
        self.threshold_strategy = threshold_strategy
        self.tpr_fprs = list(tpr_fprs) if tpr_fprs is not None else [0.1, 1.0, 5.0]

        self.s_tr_outputs, self.s_tr_labels = shadow_train_performance
        self.s_te_outputs, self.s_te_labels = shadow_test_performance
        self.t_tr_outputs, self.t_tr_labels = target_train_performance
        self.t_te_outputs, self.t_te_labels = target_test_performance

        self.s_tr_corr = (np.argmax(self.s_tr_outputs, axis=1) == self.s_tr_labels).astype(int)
        self.s_te_corr = (np.argmax(self.s_te_outputs, axis=1) == self.s_te_labels).astype(int)
        self.t_tr_corr = (np.argmax(self.t_tr_outputs, axis=1) == self.t_tr_labels).astype(int)
        self.t_te_corr = (np.argmax(self.t_te_outputs, axis=1) == self.t_te_labels).astype(int)

        self.s_tr_conf = np.array([self.s_tr_outputs[i, self.s_tr_labels[i]] for i in range(len(self.s_tr_labels))])
        self.s_te_conf = np.array([self.s_te_outputs[i, self.s_te_labels[i]] for i in range(len(self.s_te_labels))])
        self.t_tr_conf = np.array([self.t_tr_outputs[i, self.t_tr_labels[i]] for i in range(len(self.t_tr_labels))])
        self.t_te_conf = np.array([self.t_te_outputs[i, self.t_te_labels[i]] for i in range(len(self.t_te_labels))])

        self.s_tr_top1_conf = np.max(self.s_tr_outputs, axis=1)
        self.s_te_top1_conf = np.max(self.s_te_outputs, axis=1)
        self.t_tr_top1_conf = np.max(self.t_tr_outputs, axis=1)
        self.t_te_top1_conf = np.max(self.t_te_outputs, axis=1)

        self.s_tr_entr = self._entr_comp(self.s_tr_outputs)
        self.s_te_entr = self._entr_comp(self.s_te_outputs)
        self.t_tr_entr = self._entr_comp(self.t_tr_outputs)
        self.t_te_entr = self._entr_comp(self.t_te_outputs)

        self.s_tr_m_entr = self._m_entr_comp(self.s_tr_outputs, self.s_tr_labels)
        self.s_te_m_entr = self._m_entr_comp(self.s_te_outputs, self.s_te_labels)
        self.t_tr_m_entr = self._m_entr_comp(self.t_tr_outputs, self.t_tr_labels)
        self.t_te_m_entr = self._m_entr_comp(self.t_te_outputs, self.t_te_labels)

    def _log_value(self, probs, small_value=1e-20):
        return -np.log(np.maximum(probs, small_value))

    def _entr_comp(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)), axis=1)

    def _m_entr_comp(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1 - probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(true_labels.size), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)

    @staticmethod
    def _candidate_thresholds(member_scores, nonmember_scores):
        values = np.unique(np.concatenate([member_scores, nonmember_scores]).astype(float))
        if values.size == 0:
            return np.asarray([0.0], dtype=float)
        eps = 1e-12
        if values.size == 1:
            return np.asarray([values[0] - eps, values[0], values[0] + eps], dtype=float)
        mids = (values[:-1] + values[1:]) / 2.0
        return np.concatenate(([values.min() - eps], values, mids, [values.max() + eps]))

    @staticmethod
    def _confusion_at_threshold(member_scores, nonmember_scores, threshold):
        member_pred = member_scores >= threshold
        nonmember_pred = nonmember_scores >= threshold
        tp = int(member_pred.sum())
        fn = int(member_scores.shape[0] - tp)
        fp = int(nonmember_pred.sum())
        tn = int(nonmember_scores.shape[0] - fp)
        return tp, fp, tn, fn

    def _choose_threshold(self, member_scores, nonmember_scores, strategy=None):
        strategy = strategy or self.threshold_strategy
        member_scores = np.asarray(member_scores, dtype=float)
        nonmember_scores = np.asarray(nonmember_scores, dtype=float)
        if member_scores.size == 0 or nonmember_scores.size == 0:
            return 0.0

        best_threshold = 0.0
        best_rank = None
        for threshold in self._candidate_thresholds(member_scores, nonmember_scores):
            tp, fp, tn, fn = self._confusion_at_threshold(member_scores, nonmember_scores, threshold)
            tpr = tp / (tp + fn + 1e-12)
            fpr = fp / (fp + tn + 1e-12)
            tnr = tn / (tn + fp + 1e-12)
            fnr = 1.0 - tpr
            bal_acc = 0.5 * (tpr + tnr)

            if strategy == 'youden':
                rank = (tpr - fpr, bal_acc, -fpr)
            elif strategy == 'max_accuracy':
                rank = (bal_acc, tpr - fpr, -fpr)
            elif strategy == 'fpr_1pct':
                feasible = 1 if fpr <= 0.01 + 1e-12 else 0
                rank = (feasible, tpr if feasible else -fpr, bal_acc, -fpr)
            elif strategy == 'equal_error_rate':
                rank = (-abs(fpr - fnr), bal_acc, tpr - fpr)
            else:
                raise ValueError(f'Unknown threshold strategy: {strategy}')

            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_threshold = float(threshold)
        return best_threshold

    @staticmethod
    def _tpr_at_fprs(y_true, y_score, fprs):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        non_member = y_score[y_true == 0]
        member = y_score[y_true == 1]
        out = {}
        if non_member.size == 0 or member.size == 0:
            return out
        for fpr_pct in fprs:
            try:
                q = max(0.0, min(1.0, 1.0 - (float(fpr_pct) / 100.0)))
                tau = float(np.quantile(non_member, q))
                out[f"{float(fpr_pct):g}"] = float((member >= tau).mean())
            except Exception:
                continue
        return out

    def _metric_arrays(self, metric_name):
        if metric_name == 'confidence':
            return self.s_tr_conf, self.s_te_conf, self.t_tr_conf, self.t_te_conf, True
        if metric_name == 'entropy':
            return -self.s_tr_entr, -self.s_te_entr, -self.t_tr_entr, -self.t_te_entr, True
        if metric_name == 'modified_entropy':
            return -self.s_tr_m_entr, -self.s_te_m_entr, -self.t_tr_m_entr, -self.t_te_m_entr, True
        if metric_name == 'top1_conf':
            return self.s_tr_top1_conf, self.s_te_top1_conf, self.t_tr_top1_conf, self.t_te_top1_conf, False
        raise ValueError(f'Unknown threshold metric: {metric_name}')

    def evaluate(self, metric_name, strategy=None, classwise=None):
        strategy = strategy or self.threshold_strategy
        s_tr_scores, s_te_scores, t_tr_scores, t_te_scores, default_classwise = self._metric_arrays(metric_name)
        if classwise is None:
            classwise = default_classwise

        s_tr_scores = np.asarray(s_tr_scores, dtype=float)
        s_te_scores = np.asarray(s_te_scores, dtype=float)
        t_tr_scores = np.asarray(t_tr_scores, dtype=float)
        t_te_scores = np.asarray(t_te_scores, dtype=float)

        global_threshold = self._choose_threshold(s_tr_scores, s_te_scores, strategy=strategy)

        if classwise:
            thresholds = {}
            for cls in range(self.num_classes):
                tr_mask = self.s_tr_labels == cls
                te_mask = self.s_te_labels == cls
                if np.any(tr_mask) and np.any(te_mask):
                    thresholds[int(cls)] = float(
                        self._choose_threshold(s_tr_scores[tr_mask], s_te_scores[te_mask], strategy=strategy)
                    )
                else:
                    thresholds[int(cls)] = float(global_threshold)

            train_pred = np.zeros_like(t_tr_scores, dtype=int)
            test_pred = np.zeros_like(t_te_scores, dtype=int)
            for cls in range(self.num_classes):
                threshold = thresholds[int(cls)]
                tr_mask = self.t_tr_labels == cls
                te_mask = self.t_te_labels == cls
                if np.any(tr_mask):
                    train_pred[tr_mask] = (t_tr_scores[tr_mask] >= threshold).astype(int)
                if np.any(te_mask):
                    test_pred[te_mask] = (t_te_scores[te_mask] >= threshold).astype(int)
            threshold_info = {str(k): float(v) for k, v in thresholds.items()}
        else:
            train_pred = (t_tr_scores >= global_threshold).astype(int)
            test_pred = (t_te_scores >= global_threshold).astype(int)
            threshold_info = {'global': float(global_threshold)}

        y_true = np.concatenate([
            np.ones(t_tr_scores.shape[0], dtype=int),
            np.zeros(t_te_scores.shape[0], dtype=int),
        ])
        y_score = np.concatenate([t_tr_scores, t_te_scores])
        y_pred = np.concatenate([train_pred, test_pred])

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tpr = tp / (tp + fn + 1e-12)
        fpr = fp / (fp + tn + 1e-12)
        try:
            auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.0
        except Exception:
            auc = 0.0
        try:
            ap = float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.0
        except Exception:
            ap = 0.0

        result = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'auc': auc,
            'ap': ap,
            'advantage': float(tpr - fpr),
            'threshold_strategy': strategy,
            'classwise_thresholds': bool(classwise),
            'thresholds': threshold_info,
            'tpr_at_1fpr': None,
            'tpr_at_fprs': self._tpr_at_fprs(y_true, y_score, self.tpr_fprs),
        }
        if '1' in result['tpr_at_fprs']:
            result['tpr_at_1fpr'] = result['tpr_at_fprs']['1']
        return result, y_true, y_score

    def _mem_inf_via_corr(self):
        t_tr_acc = np.sum(self.t_tr_corr) / (len(self.t_tr_corr) + 0.0)
        t_te_acc = np.sum(self.t_te_corr) / (len(self.t_te_corr) + 0.0)
        mem_inf_acc = 0.5 * (t_tr_acc + 1 - t_te_acc)
        return mem_inf_acc

    def _mem_inf_benchmarks(self):
        confidence = self.evaluate('confidence')[0]['accuracy']
        entropy = self.evaluate('entropy')[0]['accuracy']
        modentr = self.evaluate('modified_entropy')[0]['accuracy']
        return confidence, entropy, modentr

    def _mem_inf_benchmarks_non_cls(self):
        confidence = self.evaluate('top1_conf', classwise=False)[0]['accuracy']
        entropy = self.evaluate('entropy', classwise=False)[0]['accuracy']
        modentr = self.evaluate('modified_entropy', classwise=False)[0]['accuracy']
        return confidence, entropy, modentr
