#!/usr/bin/env python3
"""
Extended MIA evaluation metrics
논문 표준 지표들 (AUROC, Balanced Acc, TPR@FPR, Advantage 등)
"""

import numpy as np

def roc_auc_score_numpy(y_true, y_scores):
    """Numpy implementation of ROC AUC"""
    # Sort by scores (descending)
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    # Calculate TPR and FPR at each threshold
    tpr, fpr = [], []
    
    for i in range(len(y_true_sorted) + 1):
        if i == 0:
            tp = fp = 0
        else:
            tp = np.sum(y_true_sorted[:i] == 1)
            fp = np.sum(y_true_sorted[:i] == 0)
        
        total_pos = np.sum(y_true == 1)
        total_neg = np.sum(y_true == 0)
        
        tpr_val = tp / total_pos if total_pos > 0 else 0
        fpr_val = fp / total_neg if total_neg > 0 else 0
        
        tpr.append(tpr_val)
        fpr.append(fpr_val)
    
    # Calculate AUC using trapezoidal rule
    tpr, fpr = np.array(tpr), np.array(fpr)
    try:
        auc = np.trapezoid(tpr, fpr)  # New numpy version
    except AttributeError:
        auc = np.trapz(tpr, fpr)  # Legacy numpy version
    return auc

def roc_curve_numpy(y_true, y_scores):
    """Numpy implementation of ROC curve"""
    # Get unique thresholds
    thresholds = np.unique(y_scores)[::-1]
    
    tpr_list, fpr_list = [], []
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    return np.array(fpr_list), np.array(tpr_list), thresholds

def compute_mia_metrics(member_scores, nonmember_scores, threshold=None):
    """
    Complete MIA evaluation metrics
    
    Args:
        member_scores: 멤버 샘플들의 스코어 (높을수록 멤버일 가능성 높음)
        nonmember_scores: 비멤버 샘플들의 스코어  
        threshold: 임계값 (None이면 최적 임계값 자동 선택)
    
    Returns:
        dict: 모든 MIA 지표들
    """
    # 라벨 생성 (멤버=1, 비멤버=0)
    y_true = np.concatenate([
        np.ones(len(member_scores)),  # 멤버
        np.zeros(len(nonmember_scores))  # 비멤버
    ])
    
    y_scores = np.concatenate([member_scores, nonmember_scores])
    
    # AUROC
    auroc = roc_auc_score_numpy(y_true, y_scores)
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve_numpy(y_true, y_scores)
    
    # 임계값 선택
    if threshold is None:
        # Youden's J statistic (TPR - FPR를 최대화)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        threshold = thresholds[optimal_idx]
        optimal_strategy = "youden"
    else:
        optimal_strategy = "provided"
        optimal_idx = np.argmax(thresholds >= threshold) if threshold <= thresholds.max() else -1
    
    # 예측
    y_pred = (y_scores >= threshold).astype(int)
    
    # Basic metrics
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))  
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Standard metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # TPR = Recall
    specificity_tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR = Specificity
    fpr_rate = fp / (fp + tn) if (fp + tn) > 0 else 0  # FPR = 1 - TNR
    
    balanced_acc = (recall_tpr + specificity_tnr) / 2  # Balanced accuracy = (TPR + TNR) / 2
    
    # MIA specific metrics
    advantage = recall_tpr - fpr_rate  # TPR - FPR (공격 우위성)
    
    # TPR at specific FPR levels (low FPR에서의 성능)
    tpr_at_fpr_1pct = np.interp(0.01, fpr, tpr)  # TPR at 1% FPR
    tpr_at_fpr_0_1pct = np.interp(0.001, fpr, tpr)  # TPR at 0.1% FPR
    tpr_at_fpr_5pct = np.interp(0.05, fpr, tpr)  # TPR at 5% FPR
    
    # Privacy loss (KL divergence approximation)
    member_mean = np.mean(member_scores)
    nonmember_mean = np.mean(nonmember_scores)
    privacy_loss = abs(member_mean - nonmember_mean)
    
    return {
        # Core metrics
        'auroc': float(auroc),
        'accuracy': float(accuracy), 
        'balanced_accuracy': float(balanced_acc),
        'advantage': float(advantage),
        
        # Confusion matrix
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        
        # Rates  
        'tpr': float(recall_tpr),
        'tnr': float(specificity_tnr), 
        'fpr': float(fpr_rate),
        'precision': float(precision),
        
        # Low FPR performance
        'tpr_at_1pct_fpr': float(tpr_at_fpr_1pct),
        'tpr_at_0.1pct_fpr': float(tpr_at_fpr_0_1pct),
        'tpr_at_5pct_fpr': float(tpr_at_fpr_5pct),
        
        # Other
        'privacy_loss': float(privacy_loss),
        'threshold': float(threshold),
        'optimal_strategy': optimal_strategy,
        
        # Sample statistics
        'member_count': len(member_scores),
        'nonmember_count': len(nonmember_scores),
        'member_score_mean': float(np.mean(member_scores)),
        'member_score_std': float(np.std(member_scores)),
        'nonmember_score_mean': float(np.mean(nonmember_scores)),
        'nonmember_score_std': float(np.std(nonmember_scores))
    }

def print_mia_metrics(metrics, title="MIA Metrics"):
    """Pretty print MIA metrics"""
    print(f"\n📊 {title}")
    print("=" * 50)
    
    # Core metrics
    print(f"🎯 AUROC: {metrics['auroc']:.4f}")
    print(f"🎯 Accuracy: {metrics['accuracy']:.4f}")
    print(f"🎯 Balanced Acc: {metrics['balanced_accuracy']:.4f}")
    print(f"🎯 Advantage (TPR-FPR): {metrics['advantage']:.4f}")
    
    # Rates
    print(f"\n📈 TPR (Recall): {metrics['tpr']:.4f}")
    print(f"📉 FPR: {metrics['fpr']:.4f}")
    print(f"🔒 TNR (Specificity): {metrics['tnr']:.4f}")
    print(f"🎯 Precision: {metrics['precision']:.4f}")
    
    # Low FPR performance  
    print(f"\n🔍 TPR @ 1% FPR: {metrics['tpr_at_1pct_fpr']:.4f}")
    print(f"🔍 TPR @ 0.1% FPR: {metrics['tpr_at_0.1pct_fpr']:.4f}")
    print(f"🔍 TPR @ 5% FPR: {metrics['tpr_at_5pct_fpr']:.4f}")
    
    # Statistics
    print(f"\n📊 Privacy Loss: {metrics['privacy_loss']:.4f}")
    print(f"⚖️ Threshold: {metrics['threshold']:.4f} ({metrics['optimal_strategy']})")
    print(f"👥 Samples: {metrics['member_count']} members, {metrics['nonmember_count']} non-members")

# Threshold selection strategies
def select_threshold_strategy(member_scores, nonmember_scores, strategy="youden"):
    """
    Different threshold selection strategies
    """
    y_true = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    y_scores = np.concatenate([member_scores, nonmember_scores])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    if strategy == "youden":
        # Youden's J = TPR - FPR 최대화
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        return thresholds[optimal_idx]
    
    elif strategy == "max_accuracy":
        # Accuracy 최대화  
        accuracies = []
        for thresh in thresholds:
            y_pred = (y_scores >= thresh).astype(int)
            acc = np.mean(y_true == y_pred)
            accuracies.append(acc)
        optimal_idx = np.argmax(accuracies)
        return thresholds[optimal_idx]
    
    elif strategy == "fpr_1pct":
        # FPR을 1%로 고정
        target_fpr = 0.01
        optimal_idx = np.argmax(fpr >= target_fpr)
        return thresholds[optimal_idx] if optimal_idx > 0 else thresholds[0]
    
    elif strategy == "equal_error_rate":
        # FPR = FNR인 지점
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        return thresholds[eer_idx]
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

if __name__ == "__main__":
    # Test with dummy data
    np.random.seed(42)
    member_scores = np.random.normal(0.7, 0.2, 1000)
    nonmember_scores = np.random.normal(0.3, 0.2, 1000)
    
    metrics = compute_mia_metrics(member_scores, nonmember_scores)
    print_mia_metrics(metrics, "Test MIA Metrics")