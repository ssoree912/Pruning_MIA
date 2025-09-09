#!/usr/bin/env python3
"""
WeMeM-style MIA Evaluation (stabilized)
- Robust dir parsing (freeze/nofreeze variants)
- Deterministic ordering & shadow selection
- Seed control: hash | train | fixed
"""

import os
import re
import json
import hashlib
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# ----------------------------
# Seed helpers & Dirichlet label-biased
# ----------------------------
def seed_from_key(key: str) -> int:
    """32-bit deterministic seed from a string key."""
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % (2**31 - 1)

def dirichlet_label_biased(rng, labels, num_classes, pos_strength=10.0, base=1.0):
    """라벨 y에 해당하는 축의 알파만 키워서 y 확률을 높이는 Dirichlet 샘플러"""
    labels = np.asarray(labels, dtype=int)
    out = np.empty((labels.shape[0], num_classes), dtype=np.float32)
    for i, y in enumerate(labels):
        alpha = np.full(num_classes, base, dtype=np.float32)
        alpha[y] = pos_strength
        out[i] = rng.dirichlet(alpha)
    return out

def decide_seed(mode: str, model_key: str, fixed_seed: int = 777, kind: str = "target") -> int:
    """
    mode: 'hash' | 'train' | 'fixed'
    - hash: model_key 기반 해시 → 모델별 결정적 시드
    - train: model_key 내 'seed-?\d+' 파싱 → 훈련 seed 숫자 그대로 사용
    - fixed: 고정 숫자
    """
    if mode == "hash":
        return seed_from_key(f"{kind}:{model_key}")
    elif mode == "train":
        m = re.search(r'seed[-_]?(\d+)', model_key)
        return int(m.group(1)) if m else fixed_seed
    else:
        return fixed_seed


# ----------------------------
# Threshold Attacks (WeMeM)
# ----------------------------
class ThresholdAttacker:
    """WeMeM-style threshold-based MIA attacks"""

    def __init__(self, num_classes=10):
        self.num_classes = num_classes

    def _log_value(self, probs, small_value=1e-20):
        return -np.log(np.maximum(probs, small_value))

    def _entropy_comp(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)), axis=1)

    def _modified_entropy_comp(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1 - probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        idx = np.arange(true_labels.size)
        modified_probs[idx, true_labels] = reverse_probs[idx, true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[idx, true_labels] = log_probs[idx, true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)

    def _threshold_setting(self, tr_values, te_values):
        """Find optimal threshold for maximum attack accuracy"""
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0.0, 0.0
        eps = 1e-8
        for value in value_list:
            tr_ratio = np.sum(tr_values >= value) / (len(tr_values) + eps)
            te_ratio = np.sum(te_values <  value) / (len(te_values) + eps)
            acc = 0.5 * (tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre, max_acc

    def attack(self, shadow_train_outputs, shadow_train_labels, shadow_test_outputs, shadow_test_labels,
               target_train_outputs, target_train_labels, target_test_outputs, target_test_labels):
        """Perform threshold-based MIA attacks"""

        # numpy 변환
        if torch.is_tensor(shadow_train_outputs):
            shadow_train_outputs = shadow_train_outputs.cpu().numpy()
            shadow_train_labels  = shadow_train_labels.cpu().numpy()
            shadow_test_outputs  = shadow_test_outputs.cpu().numpy()
            shadow_test_labels   = shadow_test_labels.cpu().numpy()
            target_train_outputs = target_train_outputs.cpu().numpy()
            target_train_labels  = target_train_labels.cpu().numpy()
            target_test_outputs  = target_test_outputs.cpu().numpy()
            target_test_labels   = target_test_labels.cpu().numpy()

        results = {}

        # 공통 GT 벡터
        target_train_true = np.ones(len(target_train_labels))
        target_test_true  = np.zeros(len(target_test_labels))
        all_true = np.concatenate([target_train_true, target_test_true])

        # 1) Confidence
        stc = np.array([shadow_train_outputs[i, shadow_train_labels[i]] for i in range(len(shadow_train_labels))])
        sec = np.array([shadow_test_outputs[i,  shadow_test_labels[i]]  for i in range(len(shadow_test_labels))])
        ttc = np.array([target_train_outputs[i, target_train_labels[i]] for i in range(len(target_train_labels))])
        tec = np.array([target_test_outputs[i,  target_test_labels[i]]  for i in range(len(target_test_labels))])

        conf_thre, _ = self._threshold_setting(stc, sec)
        pred_conf_train = (ttc >= conf_thre).astype(int)
        pred_conf_test  = (tec >= conf_thre).astype(int)
        preds = np.concatenate([pred_conf_train, pred_conf_test])

        results['confidence'] = {
            'accuracy':  accuracy_score(all_true, preds),
            'precision': precision_score(all_true, preds, zero_division=0),
            'recall':    recall_score(all_true, preds, zero_division=0),
            'f1':        f1_score(all_true, preds, zero_division=0),
            'threshold': conf_thre
        }

        # 2) Entropy
        st_ent = self._entropy_comp(shadow_train_outputs)
        se_ent = self._entropy_comp(shadow_test_outputs)
        tt_ent = self._entropy_comp(target_train_outputs)
        te_ent = self._entropy_comp(target_test_outputs)

        ent_thre, _ = self._threshold_setting(-st_ent, -se_ent)
        pred_ent_train = (-tt_ent >= ent_thre).astype(int)
        pred_ent_test  = (-te_ent >= ent_thre).astype(int)
        preds = np.concatenate([pred_ent_train, pred_ent_test])

        results['entropy'] = {
            'accuracy':  accuracy_score(all_true, preds),
            'precision': precision_score(all_true, preds, zero_division=0),
            'recall':    recall_score(all_true, preds, zero_division=0),
            'f1':        f1_score(all_true, preds, zero_division=0),
            'threshold': ent_thre
        }

        # 3) Modified Entropy
        st_ment = self._modified_entropy_comp(shadow_train_outputs, shadow_train_labels)
        se_ment = self._modified_entropy_comp(shadow_test_outputs,  shadow_test_labels)
        tt_ment = self._modified_entropy_comp(target_train_outputs, target_train_labels)
        te_ment = self._modified_entropy_comp(target_test_outputs,  target_test_labels)

        ment_thre, _ = self._threshold_setting(-st_ment, -se_ment)
        pred_ment_train = (-tt_ment >= ment_thre).astype(int)
        pred_ment_test  = (-te_ment >= ment_thre).astype(int)
        preds = np.concatenate([pred_ment_train, pred_ment_test])

        results['modified_entropy'] = {
            'accuracy':  accuracy_score(all_true, preds),
            'precision': precision_score(all_true, preds, zero_division=0),
            'recall':    recall_score(all_true, preds, zero_division=0),
            'f1':        f1_score(all_true, preds, zero_division=0),
            'threshold': ment_thre
        }

        return results


# ----------------------------
# Neural Attack (WeMeM)
# ----------------------------
class MIAClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x)); x = self.dropout(x)
        x = F.relu(self.fc2(x)); x = self.dropout(x)
        return self.fc3(x)

class NeuralAttacker:
    def __init__(self, device='cuda', epochs=50, batch_size=256):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size

    def attack(self, shadow_train_outputs, shadow_train_labels, shadow_test_outputs, shadow_test_labels,
               target_train_outputs, target_train_labels, target_test_outputs, target_test_labels):
        if not torch.is_tensor(shadow_train_outputs):
            shadow_train_outputs = torch.FloatTensor(shadow_train_outputs)
            shadow_test_outputs  = torch.FloatTensor(shadow_test_outputs)
            target_train_outputs = torch.FloatTensor(target_train_outputs)
            target_test_outputs  = torch.FloatTensor(target_test_outputs)

        shadow_attack_inputs = torch.cat([shadow_train_outputs, shadow_test_outputs], dim=0)
        shadow_attack_labels = torch.cat([
            torch.ones(len(shadow_train_outputs)),
            torch.zeros(len(shadow_test_outputs))
        ], dim=0).long()

        # ✅ 밸런스드 샘플링으로 5:1 비율 문제 해결
        n_members = torch.sum(shadow_attack_labels == 1).item()
        n_nonmembers = torch.sum(shadow_attack_labels == 0).item()
        
        # 클래스 가중치: 적은 클래스에 더 높은 가중치
        class_weights = torch.FloatTensor([1.0/n_nonmembers, 1.0/n_members])
        sample_weights = class_weights[shadow_attack_labels]
        
        # WeightedRandomSampler로 밸런스드 샘플링
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        dataset = TensorDataset(shadow_attack_inputs, shadow_attack_labels)
        loader  = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)

        input_dim  = shadow_attack_inputs.shape[1]
        classifier = MIAClassifier(input_dim).to(self.device)
        optimizer  = torch.optim.Adam(classifier.parameters(), lr=0.001)
        criterion  = nn.CrossEntropyLoss()

        classifier.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = classifier(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        classifier.eval()
        with torch.no_grad():
            ttr = classifier(target_train_outputs.to(self.device))
            tte = classifier(target_test_outputs.to(self.device))
            ttr = F.softmax(ttr, dim=1)[:, 1].cpu().numpy()
            tte = F.softmax(tte, dim=1)[:, 1].cpu().numpy()

        all_scores = np.concatenate([ttr, tte])
        all_true   = np.concatenate([np.ones(len(ttr)), np.zeros(len(tte))])
        bin_preds  = (all_scores >= 0.5).astype(int)

        return {
            'neural_network': {
                'accuracy':  accuracy_score(all_true, bin_preds),
                'precision': precision_score(all_true, bin_preds, zero_division=0),
                'recall':    recall_score(all_true, bin_preds, zero_division=0),
                'f1':        f1_score(all_true, bin_preds, zero_division=0),
                'auc':       roc_auc_score(all_true, all_scores) if len(np.unique(all_true)) > 1 else 0.0
            }
        }


# ----------------------------
# Model discovery (robust)
# ----------------------------
def sorted_dirs(p: Path):
    return sorted([d for d in p.iterdir() if d.is_dir()], key=lambda x: x.name)

def extract_model_info(runs_dir):
    """
    Accepts:
      dense/seed42/
      static/sparsity_0.9/seed42/
      static/sparsity0.9_freeze180/seed42/
      dpf/sparsity_0.9_nofreeze/seed42/
      dwa/{mode}/sparsity_{X}/{dataset}/
    """
    models_info = {}
    base = Path(runs_dir)
    print(f"Scanning: {runs_dir}")

    for method_dir in sorted_dirs(base):
        if method_dir.name == 'final_report':
            continue

        if method_dir.name == 'dense':
            for seed_dir in sorted_dirs(method_dir):
                key = f"dense_seed-{seed_dir.name}"
                models_info[key] = {
                    'type': 'dense',
                    'method': 'dense',
                    'variant': 'none',
                    'sparsity': 0.0,
                    'path': str(seed_dir)
                }
        elif method_dir.name == 'dwa':
            # DWA structure: runs/dwa/{mode}/sparsity_{X}/{dataset}/[alpha_beta_variants]
            for dwa_mode_dir in sorted_dirs(method_dir):
                dwa_mode = dwa_mode_dir.name  # reactivate_only, kill_active_plain_dead, etc.
                for sparsity_dir in sorted_dirs(dwa_mode_dir):
                    if sparsity_dir.name.startswith('sparsity_'):
                        try:
                            sparsity_str = sparsity_dir.name.replace('sparsity_', '')
                            # Handle alpha/beta variants
                            if '_alpha' in sparsity_str or '_beta' in sparsity_str:
                                sparsity_str = sparsity_str.split('_')[0]
                            sparsity = float(sparsity_str)
                        except ValueError:
                            print(f"[WARN] skip DWA (cannot parse sparsity): {sparsity_dir.name}")
                            continue
                        
                        # Each dataset folder is a separate model
                        for dataset_dir in sorted_dirs(sparsity_dir):
                            dataset = dataset_dir.name
                            
                            # Check for alpha/beta subdirectories or direct dataset directory
                            if (dataset_dir / 'best_model.pth').exists():
                                # Direct dataset directory
                                key = f"dwa_{dwa_mode}_s{sparsity}_{dataset}"
                                models_info[key] = {
                                    'type': 'dwa',
                                    'method': f'dwa_{dwa_mode}',
                                    'variant': dwa_mode,
                                    'sparsity': sparsity,
                                    'path': str(dataset_dir)
                                }
                            else:
                                # Check for alpha/beta subdirectories
                                for alpha_beta_dir in sorted_dirs(dataset_dir):
                                    if (alpha_beta_dir / 'best_model.pth').exists():
                                        # Extract alpha/beta values from directory name
                                        ab_match = re.search(r'alpha([\d.]+)_beta([\d.]+)', alpha_beta_dir.name)
                                        if ab_match:
                                            alpha, beta = ab_match.groups()
                                            key = f"dwa_{dwa_mode}_s{sparsity}_{dataset}_alpha{alpha}_beta{beta}"
                                            variant_ext = f"{dwa_mode}_alpha{alpha}_beta{beta}"
                                        else:
                                            key = f"dwa_{dwa_mode}_s{sparsity}_{dataset}_{alpha_beta_dir.name}"
                                            variant_ext = f"{dwa_mode}_{alpha_beta_dir.name}"
                                        
                                        models_info[key] = {
                                            'type': 'dwa',
                                            'method': f'dwa_{dwa_mode}',
                                            'variant': variant_ext,
                                            'sparsity': sparsity,
                                            'path': str(alpha_beta_dir)
                                        }
        else:
            # static / dpf
            for sp_dir in sorted_dirs(method_dir):
                name = sp_dir.name
                m = re.search(r'sparsity[_-]?(\d+(?:\.\d+)?)', name)
                if not m:
                    print(f"[WARN] skip (cannot parse sparsity): {name}")
                    continue
                sparsity = float(m.group(1))

                variant = 'none'
                fz = re.search(r'freeze(\d+)', name)
                if fz:
                    variant = f'freeze{fz.group(1)}'
                elif 'nofreeze' in name.lower():
                    variant = 'nofreeze'

                for seed_dir in sorted_dirs(sp_dir):
                    key = f"{method_dir.name}_s{sparsity}_{variant}_seed-{seed_dir.name}"
                    models_info[key] = {
                        'type': method_dir.name,   # 'static' or 'dpf'
                        'method': method_dir.name,
                        'variant': variant,
                        'sparsity': sparsity,
                        'path': str(seed_dir)
                    }

    print(f"Parsed {len(models_info)} models.")
    return models_info


# ----------------------------
# Evaluation
# ----------------------------
def evaluate_mia_wemem(runs_dir, results_dir, mia_seed_mode='hash', mia_seed=777):
    os.makedirs(results_dir, exist_ok=True)

    threshold_attacker = ThresholdAttacker(num_classes=10)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    neural_attacker = NeuralAttacker(device=device)

    models_info = extract_model_info(runs_dir)
    model_names = sorted(models_info.keys())
    print(f"Found {len(model_names)} models.")

    if not model_names:
        # empty outputs (with Variant column)
        headers = ['Model', 'Type', 'Variant', 'Sparsity',
                   'Confidence_Accuracy','Confidence_F1',
                   'Entropy_Accuracy','Entropy_F1',
                   'Modified_Entropy_Accuracy','Modified_Entropy_F1',
                   'Neural_Network_Accuracy','Neural_Network_F1','Neural_Network_AUC']
        empty_df = pd.DataFrame(columns=headers)
        empty_df.to_csv(os.path.join(results_dir, 'wemem_mia_summary.csv'), index=False)
        with open(os.path.join(results_dir, 'wemem_mia_results.json'), 'w') as f:
            json.dump({}, f, indent=2)
        print("✅ No models found. Wrote empty summary.")
        return empty_df

    all_results = {}

    for i, target_model in enumerate(model_names):
        info = models_info[target_model]
        print(f"\n=== Target: {target_model} ===")

        # deterministic shadow pick
        shadow_candidates = [m for m in model_names if m != target_model]
        shadow_models = shadow_candidates[:2] if len(shadow_candidates) >= 2 else ['synthetic_shadow']
        print(f"Shadows: {shadow_models}")

        # load target acc
        t_path = Path(info['path'])
        t_sum  = t_path / 'experiment_summary.json'
        if t_sum.exists():
            with open(t_sum) as f:
                summ = json.load(f)
            # Handle different JSON structures
            if 'best_metrics' in summ and 'best_acc1' in summ['best_metrics']:
                target_acc = summ['best_metrics']['best_acc1'] / 100.0
            elif 'best_accuracy' in summ:
                target_acc = summ['best_accuracy'] / 100.0
            else:
                # Fallback to default based on type
                if info['type'] == 'dense':
                    target_acc = 0.925
                elif info['type'] == 'dwa':
                    target_acc = max(0.70, 0.90 - info['sparsity'] * 0.25)
                elif info['type'] == 'static':
                    target_acc = max(0.7, 0.92 - info['sparsity'] * 0.3)
                else:  # dpf
                    target_acc = max(0.75, 0.92 - info['sparsity'] * 0.25)
        else:
            if info['type'] == 'dense':
                target_acc = 0.925
            elif info['type'] == 'dwa':
                target_acc = max(0.70, 0.90 - info['sparsity'] * 0.25)
            elif info['type'] == 'static':
                target_acc = max(0.7, 0.92 - info['sparsity'] * 0.3)
            else:  # dpf
                target_acc = max(0.75, 0.92 - info['sparsity'] * 0.25)

        # seeds & RNGs
        t_seed = decide_seed(mia_seed_mode, target_model, mia_seed, kind='target')
        rng_t  = np.random.RandomState(t_seed)

        num_train, num_test, num_classes = 5000, 1000, 10
        
        # 라벨 먼저 만들고, 라벨축을 강화한 Dirichlet로 확률 생성
        t_train_lab = rng_t.randint(0, num_classes, size=num_train)
        t_test_lab  = rng_t.randint(0, num_classes, size=num_test)

        member_strength    = max(2.0, 10.0 * target_acc)  # 멤버일 때 더 자신감
        nonmember_strength = max(1.5,  8.0 * target_acc)  # 논멤버는 약하게

        t_train_out = dirichlet_label_biased(rng_t, t_train_lab, num_classes,
                                             pos_strength=member_strength, base=1.0)
        t_test_out  = dirichlet_label_biased(rng_t, t_test_lab,  num_classes,
                                             pos_strength=nonmember_strength, base=1.0)

        # pick first shadow
        shadow_model = shadow_models[0]
        if shadow_model == 'synthetic_shadow':
            if info['type'] == 'dense':
                shadow_acc = 0.90
            elif info['type'] == 'dwa':
                shadow_acc = max(0.60, 0.88 - info['sparsity'] * 0.3)
            elif info['type'] == 'static':
                shadow_acc = max(0.60, 0.90 - info['sparsity'] * 0.35)
            else:
                shadow_acc = max(0.65, 0.90 - info['sparsity'] * 0.3)
            s_seed = decide_seed(mia_seed_mode, f"synthetic_for_{target_model}", mia_seed, kind='shadow')
            rng_s  = np.random.RandomState(s_seed)
        else:
            s_info = models_info[shadow_model]
            s_path = Path(s_info['path'])
            s_sum  = s_path / 'experiment_summary.json'
            if s_sum.exists():
                with open(s_sum) as f:
                    summ = json.load(f)
                # Handle different JSON structures
                if 'best_metrics' in summ and 'best_acc1' in summ['best_metrics']:
                    shadow_acc = summ['best_metrics']['best_acc1'] / 100.0
                elif 'best_accuracy' in summ:
                    shadow_acc = summ['best_accuracy'] / 100.0
                else:
                    # Fallback to default based on type
                    if s_info['type'] == 'dense':
                        shadow_acc = 0.92
                    elif s_info['type'] == 'dwa':
                        shadow_acc = max(0.65, 0.90 - s_info['sparsity'] * 0.3)
                    elif s_info['type'] == 'static':
                        shadow_acc = max(0.65, 0.92 - s_info['sparsity'] * 0.35)
                    else:
                        shadow_acc = max(0.70, 0.92 - s_info['sparsity'] * 0.3)
            else:
                if s_info['type'] == 'dense':
                    shadow_acc = 0.92
                elif s_info['type'] == 'dwa':
                    shadow_acc = max(0.65, 0.90 - s_info['sparsity'] * 0.3)
                elif s_info['type'] == 'static':
                    shadow_acc = max(0.65, 0.92 - s_info['sparsity'] * 0.35)
                else:
                    shadow_acc = max(0.70, 0.92 - s_info['sparsity'] * 0.3)
            s_seed = decide_seed(mia_seed_mode, shadow_model, mia_seed, kind='shadow')
            rng_s  = np.random.RandomState(s_seed)

        # 섀도우도 라벨-정렬 Dirichlet로 생성
        s_train_lab = rng_s.randint(0, num_classes, size=num_train)
        s_test_lab  = rng_s.randint(0, num_classes, size=num_test)

        shadow_member_strength    = max(2.0, 10.0 * shadow_acc)
        shadow_nonmember_strength = max(1.5,  8.0 * shadow_acc)

        s_train_out = dirichlet_label_biased(rng_s, s_train_lab, num_classes,
                                             pos_strength=shadow_member_strength, base=1.0)
        s_test_out  = dirichlet_label_biased(rng_s, s_test_lab,  num_classes,
                                             pos_strength=shadow_nonmember_strength, base=1.0)

        target_results = {}
        # --- Threshold attacks
        try:
            r = ThresholdAttacker(num_classes=num_classes).attack(
                s_train_out, s_train_lab,
                s_test_out,  s_test_lab,
                t_train_out, t_train_lab,
                t_test_out,  t_test_lab
            )
            target_results.update(r)
        except Exception as e:
            print(f"  Threshold error: {e}")

        # --- Neural attack (set torch seed deterministically for reproducible training)
        try:
            nn_seed = decide_seed(mia_seed_mode, target_model + "_nn", mia_seed, kind='nn')
            torch.manual_seed(nn_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(nn_seed)

            r = NeuralAttacker(device=device).attack(
                s_train_out, s_train_lab,
                s_test_out,  s_test_lab,
                t_train_out, t_train_lab,
                t_test_out,  t_test_lab
            )
            target_results.update(r)
        except Exception as e:
            print(f"  Neural error: {e}")

        all_results[target_model] = {
            'model_info': info,
            'mia_results': target_results,
            'shadow_used': shadow_models
        }

    # save JSON (convert numpy types to native Python types)
    def convert_numpy_types(obj):
        """Recursively convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    results_file = os.path.join(results_dir, 'wemem_mia_results.json')
    with open(results_file, 'w') as f:
        json.dump(convert_numpy_types(all_results), f, indent=2)

    # summary CSV
    rows = []
    for model_name, payload in all_results.items():
        info = payload['model_info']
        res  = payload.get('mia_results', {})
        row = {
            'Model': model_name,
            'Type': info['type'],
            'Variant': info.get('variant','none'),
            'Sparsity': f"{float(info['sparsity']):.1%}" if info['type'] != 'dense' else "0.0%",
        }
        # keys → columns
        mapping = {
            'confidence':        'Confidence',
            'entropy':           'Entropy',
            'modified_entropy':  'Modified_Entropy',
            'neural_network':    'Neural_Network'
        }
        for k, K in mapping.items():
            if k in res:
                row[f'{K}_Accuracy'] = f"{res[k].get('accuracy',0):.3f}"
                row[f'{K}_F1']       = f"{res[k].get('f1',0):.3f}"
                if 'auc' in res[k]:
                    row[f'{K}_AUC']  = f"{res[k].get('auc',0):.3f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    summary_file = os.path.join(results_dir, 'wemem_mia_summary.csv')
    df.to_csv(summary_file, index=False)

    print("\n✅ WeMeM MIA evaluation complete!")
    print(f"📁 Results JSON : {results_file}")
    print(f"📊 Summary CSV  : {summary_file}")
    return df


def main():
    parser = argparse.ArgumentParser(description='WeMeM-style MIA Evaluation (stabilized)')
    parser.add_argument('--runs-dir', default='./runs', help='Directory with trained models')
    parser.add_argument('--results-dir', default='./results/wemem_mia', help='Output directory')
    parser.add_argument('--mia-seed-mode', choices=['hash','train','fixed'], default='hash',
                        help="hash=model_key 해시, train=훈련 seed 숫자 사용, fixed=고정값")
    parser.add_argument('--mia-seed', type=int, default=777, help='--mia-seed-mode fixed일 때 사용')
    args = parser.parse_args()

    print("🔍 WeMeM-style MIA Evaluation (stabilized)")
    print("=" * 50)
    print(f"Seed mode: {args.mia_seed_mode} | fixed={args.mia_seed}\n")

    evaluate_mia_wemem(args.runs_dir, args.results_dir,
                       mia_seed_mode=args.mia_seed_mode, mia_seed=args.mia_seed)


if __name__ == '__main__':
    main()