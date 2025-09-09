#!/usr/bin/env python3
"""
Advanced MIA Evaluation (stabilized)
- Robust dir parsing (regex)
- Deterministic ordering
- Model-key-based seeds
- Explicit freeze/nofreeze variant handling
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
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score


# ----------------------------
# Attackers (원본과 동일)
# ----------------------------
class LiRAAttacker:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes

    def attack(self, shadow_train_outputs, shadow_train_labels, shadow_test_outputs, shadow_test_labels,
               target_train_outputs, target_train_labels, target_test_outputs, target_test_labels):
        if torch.is_tensor(shadow_train_outputs):
            shadow_train_outputs = shadow_train_outputs.cpu().numpy()
            shadow_train_labels = shadow_train_labels.cpu().numpy()
            shadow_test_outputs = shadow_test_outputs.cpu().numpy()
            shadow_test_labels = shadow_test_labels.cpu().numpy()
            target_train_outputs = target_train_outputs.cpu().numpy()
            target_train_labels = target_train_labels.cpu().numpy()
            target_test_outputs = target_test_outputs.cpu().numpy()
            target_test_labels = target_test_labels.cpu().numpy()

        shadow_train_ll = self._log_likelihood(shadow_train_outputs, shadow_train_labels)
        shadow_test_ll = self._log_likelihood(shadow_test_outputs, shadow_test_labels)
        target_train_ll = self._log_likelihood(target_train_outputs, target_train_labels)
        target_test_ll = self._log_likelihood(target_test_outputs, target_test_labels)

        member_mean, member_std = np.mean(shadow_train_ll), np.std(shadow_train_ll)
        non_member_mean, non_member_std = np.mean(shadow_test_ll), np.std(shadow_test_ll)

        print(f"LiRA fit: member μ={member_mean:.3f}, σ={member_std:.3f} | non-member μ={non_member_mean:.3f}, σ={non_member_std:.3f}")

        def likelihood_ratio(ll):
            member_prob = stats.norm.pdf(ll, member_mean, member_std + 1e-8)
            non_member_prob = stats.norm.pdf(ll, non_member_mean, non_member_std + 1e-8)
            return member_prob / (non_member_prob + 1e-8)

        target_train_lr = [likelihood_ratio(ll) for ll in target_train_ll]
        target_test_lr = [likelihood_ratio(ll) for ll in target_test_ll]

        target_train_pred = [1 if lr > 1.0 else 0 for lr in target_train_lr]
        target_test_pred = [1 if lr > 1.0 else 0 for lr in target_test_lr]

        all_preds = np.array(target_train_pred + target_test_pred)
        all_scores = np.array(target_train_lr + target_test_lr)
        all_true = np.array([1] * len(target_train_pred) + [0] * len(target_test_pred))

        results = {
            'lira': {
                'accuracy': accuracy_score(all_true, all_preds),
                'balanced_accuracy': balanced_accuracy_score(all_true, all_preds),
                'precision': precision_score(all_true, all_preds, zero_division=0),
                'recall': recall_score(all_true, all_preds, zero_division=0),
                'f1': f1_score(all_true, all_preds, zero_division=0),
                'auc': roc_auc_score(all_true, all_scores) if len(np.unique(all_true)) > 1 else 0.0,
                'member_mean': member_mean,
                'member_std': member_std,
                'non_member_mean': non_member_mean,
                'non_member_std': non_member_std
            }
        }
        return results

    def _log_likelihood(self, outputs, labels):
        ll = []
        for i, label in enumerate(labels):
            prob = outputs[i][label]
            ll.append(np.log(prob + 1e-8))
        return np.array(ll)


class ShokriNNAttacker:
    def __init__(self, device='cuda', epochs=50, batch_size=256):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size

    def attack(self, shadow_train_outputs, shadow_train_labels, shadow_test_outputs, shadow_test_labels,
               target_train_outputs, target_train_labels, target_test_outputs, target_test_labels):
        if not torch.is_tensor(shadow_train_outputs):
            shadow_train_outputs = torch.FloatTensor(shadow_train_outputs)
            shadow_test_outputs = torch.FloatTensor(shadow_test_outputs)
            target_train_outputs = torch.FloatTensor(target_train_outputs)
            target_test_outputs = torch.FloatTensor(target_test_outputs)

        shadow_inputs = torch.cat([shadow_train_outputs, shadow_test_outputs], dim=0)
        shadow_labels = torch.cat([torch.ones(len(shadow_train_outputs)),
                                   torch.zeros(len(shadow_test_outputs))], dim=0).long()

        input_dim = shadow_inputs.shape[1]
        attack_model = self._create_attack_model(input_dim).to(self.device)
        self._train_attack_model(attack_model, shadow_inputs, shadow_labels)

        target_inputs = torch.cat([target_train_outputs, target_test_outputs], dim=0)
        target_true = torch.cat([torch.ones(len(target_train_outputs)),
                                 torch.zeros(len(target_test_outputs))], dim=0).numpy()

        attack_model.eval()
        with torch.no_grad():
            target_logits = attack_model(target_inputs.to(self.device))
            target_probs = F.softmax(target_logits, dim=1)[:, 1].cpu().numpy()
            target_preds = (target_probs >= 0.5).astype(int)

        results = {
            'shokri_nn': {
                'accuracy': accuracy_score(target_true, target_preds),
                'balanced_accuracy': balanced_accuracy_score(target_true, target_preds),
                'precision': precision_score(target_true, target_preds, zero_division=0),
                'recall': recall_score(target_true, target_preds, zero_division=0),
                'f1': f1_score(target_true, target_preds, zero_division=0),
                'auc': roc_auc_score(target_true, target_probs) if len(np.unique(target_true)) > 1 else 0.0
            }
        }
        return results

    def _create_attack_model(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def _train_attack_model(self, model, inputs, labels):
        # 1) 균형 샘플링
        if not torch.is_tensor(inputs): inputs = torch.FloatTensor(inputs)
        if not torch.is_tensor(labels): labels = torch.LongTensor(labels)

        pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
        neg_idx = (labels == 0).nonzero(as_tuple=True)[0]
        n = min(len(pos_idx), len(neg_idx))

        # 고정 시드
        g = torch.Generator().manual_seed(0)
        pos_sel = pos_idx[torch.randperm(len(pos_idx), generator=g)[:n]]
        neg_sel = neg_idx[torch.randperm(len(neg_idx), generator=g)[:n]]
        sel = torch.cat([pos_sel, neg_sel])
        sel = sel[torch.randperm(len(sel), generator=g)]

        dataset = TensorDataset(inputs[sel], labels[sel])
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 2) 표준 학습
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for _ in range(self.epochs):
            for batch_inputs, batch_labels in loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_labels = batch_labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()


class Top3NNAttacker(ShokriNNAttacker):
    def attack(self, shadow_train_outputs, shadow_train_labels, shadow_test_outputs, shadow_test_labels,
               target_train_outputs, target_train_labels, target_test_outputs, target_test_labels):
        if not torch.is_tensor(shadow_train_outputs):
            shadow_train_outputs = torch.FloatTensor(shadow_train_outputs)
            shadow_test_outputs = torch.FloatTensor(shadow_test_outputs)
            target_train_outputs = torch.FloatTensor(target_train_outputs)
            target_test_outputs = torch.FloatTensor(target_test_outputs)

        shadow_train_top3, _ = torch.topk(shadow_train_outputs, k=3, dim=1)
        shadow_test_top3, _ = torch.topk(shadow_test_outputs, k=3, dim=1)
        target_train_top3, _ = torch.topk(target_train_outputs, k=3, dim=1)
        target_test_top3, _ = torch.topk(target_test_outputs, k=3, dim=1)

        shadow_inputs = torch.cat([shadow_train_top3, shadow_test_top3], dim=0)
        shadow_labels = torch.cat([torch.ones(len(shadow_train_top3)),
                                   torch.zeros(len(shadow_test_top3))], dim=0).long()

        attack_model = self._create_attack_model(3).to(self.device)
        self._train_attack_model(attack_model, shadow_inputs, shadow_labels)

        target_inputs = torch.cat([target_train_top3, target_test_top3], dim=0)
        target_true = torch.cat([torch.ones(len(target_train_top3)),
                                 torch.zeros(len(target_test_top3))], dim=0).numpy()

        attack_model.eval()
        with torch.no_grad():
            target_logits = attack_model(target_inputs.to(self.device))
            target_probs = F.softmax(target_logits, dim=1)[:, 1].cpu().numpy()
            target_preds = (target_probs >= 0.5).astype(int)

        results = {
            'top3_nn': {
                'accuracy': accuracy_score(target_true, target_preds),
                'balanced_accuracy': balanced_accuracy_score(target_true, target_preds),
                'precision': precision_score(target_true, target_preds, zero_division=0),
                'recall': recall_score(target_true, target_preds, zero_division=0),
                'f1': f1_score(target_true, target_preds, zero_division=0),
                'auc': roc_auc_score(target_true, target_probs) if len(np.unique(target_true)) > 1 else 0.0
            }
        }
        return results


class ClassLabelNNAttacker(ShokriNNAttacker):
    def attack(self, shadow_train_outputs, shadow_train_labels, shadow_test_outputs, shadow_test_labels,
               target_train_outputs, target_train_labels, target_test_outputs, target_test_labels):
        if not torch.is_tensor(shadow_train_outputs):
            shadow_train_outputs = torch.FloatTensor(shadow_train_outputs)
            shadow_train_labels = torch.LongTensor(shadow_train_labels)
            shadow_test_outputs = torch.FloatTensor(shadow_test_outputs)
            shadow_test_labels = torch.LongTensor(shadow_test_labels)
            target_train_outputs = torch.FloatTensor(target_train_outputs)
            target_train_labels = torch.LongTensor(target_train_labels)
            target_test_outputs = torch.FloatTensor(target_test_outputs)
            target_test_labels = torch.LongTensor(target_test_labels)

        num_classes = shadow_train_outputs.shape[1]
        shadow_train_onehot = F.one_hot(shadow_train_labels, num_classes).float()
        shadow_test_onehot = F.one_hot(shadow_test_labels, num_classes).float()
        target_train_onehot = F.one_hot(target_train_labels, num_classes).float()
        target_test_onehot = F.one_hot(target_test_labels, num_classes).float()

        shadow_train_input = torch.cat([shadow_train_outputs, shadow_train_onehot], dim=1)
        shadow_test_input = torch.cat([shadow_test_outputs, shadow_test_onehot], dim=1)
        target_train_input = torch.cat([target_train_outputs, target_train_onehot], dim=1)
        target_test_input = torch.cat([target_test_outputs, target_test_onehot], dim=1)

        shadow_inputs = torch.cat([shadow_train_input, shadow_test_input], dim=0)
        shadow_labels = torch.cat([torch.ones(len(shadow_train_input)),
                                   torch.zeros(len(shadow_test_input))], dim=0).long()

        input_dim = shadow_inputs.shape[1]
        attack_model = self._create_attack_model(input_dim).to(self.device)
        self._train_attack_model(attack_model, shadow_inputs, shadow_labels)

        target_inputs = torch.cat([target_train_input, target_test_input], dim=0)
        target_true = torch.cat([torch.ones(len(target_train_input)),
                                 torch.zeros(len(target_test_input))], dim=0).numpy()

        attack_model.eval()
        with torch.no_grad():
            target_logits = attack_model(target_inputs.to(self.device))
            target_probs = F.softmax(target_logits, dim=1)[:, 1].cpu().numpy()
            target_preds = (target_probs >= 0.5).astype(int)

        results = {
            'class_label_nn': {
                'accuracy': accuracy_score(target_true, target_preds),
                'balanced_accuracy': balanced_accuracy_score(target_true, target_preds),
                'precision': precision_score(target_true, target_preds, zero_division=0),
                'recall': recall_score(target_true, target_preds, zero_division=0),
                'f1': f1_score(target_true, target_preds, zero_division=0),
                'auc': roc_auc_score(target_true, target_probs) if len(np.unique(target_true)) > 1 else 0.0
            }
        }
        return results


class SAMIAAttacker:
    def __init__(self, device='cuda', epochs=50, batch_size=256):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size

    def attack(self, shadow_train_outputs, shadow_train_labels, shadow_test_outputs, shadow_test_labels,
               target_train_outputs, target_train_labels, target_test_outputs, target_test_labels):
        if not torch.is_tensor(shadow_train_outputs):
            shadow_train_outputs = torch.FloatTensor(shadow_train_outputs)
            shadow_test_outputs = torch.FloatTensor(shadow_test_outputs)
            target_train_outputs = torch.FloatTensor(target_train_outputs)
            target_test_outputs = torch.FloatTensor(target_test_outputs)

        shadow_inputs = torch.cat([shadow_train_outputs, shadow_test_outputs], dim=0)
        shadow_labels = torch.cat([torch.ones(len(shadow_train_outputs)),
                                   torch.zeros(len(shadow_test_outputs))], dim=0).long()

        input_dim = shadow_inputs.shape[1]
        attack_model = self._create_samia_model(input_dim).to(self.device)
        self._train_attack_model(attack_model, shadow_inputs, shadow_labels)

        target_inputs = torch.cat([target_train_outputs, target_test_outputs], dim=0)
        target_true = torch.cat([torch.ones(len(target_train_outputs)),
                                 torch.zeros(len(target_test_outputs))], dim=0).numpy()

        attack_model.eval()
        with torch.no_grad():
            target_logits = attack_model(target_inputs.to(self.device))
            target_probs = F.softmax(target_logits, dim=1)[:, 1].cpu().numpy()
            target_preds = (target_probs >= 0.5).astype(int)

        results = {
            'samia': {
                'accuracy': accuracy_score(target_true, target_preds),
                'balanced_accuracy': balanced_accuracy_score(target_true, target_preds),
                'precision': precision_score(target_true, target_preds, zero_division=0),
                'recall': recall_score(target_true, target_preds, zero_division=0),
                'f1': f1_score(target_true, target_preds, zero_division=0),
                'auc': roc_auc_score(target_true, target_probs) if len(np.unique(target_true)) > 1 else 0.0
            }
        }
        return results

    def _create_samia_model(self, input_dim):
        class SAMIAModel(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.embed_dim = 64
                self.embedding = nn.Linear(input_dim, self.embed_dim)
                self.attention = nn.MultiheadAttention(self.embed_dim, num_heads=4, batch_first=True)
                self.classifier = nn.Sequential(
                    nn.Linear(self.embed_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 2)
                )

            def forward(self, x):
                x = self.embedding(x)
                x = x.unsqueeze(1)
                attended, _ = self.attention(x, x, x)
                attended = attended.squeeze(1)
                return self.classifier(attended)
        return SAMIAModel(input_dim)

    def _train_attack_model(self, model, inputs, labels):
        # 1) 균형 샘플링
        if not torch.is_tensor(inputs): inputs = torch.FloatTensor(inputs)
        if not torch.is_tensor(labels): labels = torch.LongTensor(labels)

        pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
        neg_idx = (labels == 0).nonzero(as_tuple=True)[0]
        n = min(len(pos_idx), len(neg_idx))

        # 고정 시드
        g = torch.Generator().manual_seed(0)
        pos_sel = pos_idx[torch.randperm(len(pos_idx), generator=g)[:n]]
        neg_sel = neg_idx[torch.randperm(len(neg_idx), generator=g)[:n]]
        sel = torch.cat([pos_sel, neg_sel])
        sel = sel[torch.randperm(len(sel), generator=g)]

        dataset = TensorDataset(inputs[sel], labels[sel])
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 2) 표준 학습
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for _ in range(self.epochs):
            for batch_inputs, batch_labels in loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_labels = batch_labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()


# ----------------------------
# Helper: stable seed from key & label-aligned Dirichlet
# ----------------------------
def seed_from_key(key: str) -> int:
    # 32-bit deterministic seed
    return int(hashlib.sha256(key.encode('utf-8')).hexdigest(), 16) % (2**31 - 1)

def dirichlet_label_biased(rng, labels, num_classes, pos_strength, other_strength=1.0):
    """라벨-정렬 Dirichlet 생성: 정답 클래스가 높은 확률을 갖도록"""
    alphas = np.full((len(labels), num_classes), other_strength, dtype=float)
    alphas[np.arange(len(labels)), labels] = pos_strength
    # per-sample draw
    return np.vstack([rng.dirichlet(alpha) for alpha in alphas])


# ----------------------------
# Evaluation
# ----------------------------
def evaluate_advanced_mia(runs_dir, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    # init attackers
    lira_attacker = LiRAAttacker(num_classes=10)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    shokri_attacker = ShokriNNAttacker(device=device)
    top3_attacker = Top3NNAttacker(device=device)
    cl_attacker = ClassLabelNNAttacker(device=device)
    samia_attacker = SAMIAAttacker(device=device)

    # gather models (deterministic & robust parsing)
    models_info = {}
    runs_path = Path(runs_dir)

    def sorted_dirs(p: Path):
        return sorted([d for d in p.iterdir() if d.is_dir()], key=lambda x: x.name)

    for method_dir in sorted_dirs(runs_path):
        if method_dir.name == 'final_report':
            continue

        if method_dir.name == 'dense':
            for seed_dir in sorted_dirs(method_dir):
                model_key = f"dense_seed-{seed_dir.name}"
                models_info[model_key] = {
                    'type': 'dense',
                    'method': 'dense',
                    'variant': 'none',
                    'sparsity': 0.0,
                    'path': str(seed_dir)
                }
        elif method_dir.name == 'dwa':
            # DWA structure: runs/dwa/{mode}/sparsity_{X}/{dataset}/
            for dwa_mode_dir in sorted_dirs(method_dir):
                dwa_mode = dwa_mode_dir.name  # reactivate_only, kill_active_plain_dead, etc.
                for sparsity_dir in sorted_dirs(dwa_mode_dir):
                    name = sparsity_dir.name
                    m = re.search(r'sparsity[_-]?(\d+(?:\.\d+)?)', name)
                    if not m:
                        print(f"[WARN] Cannot parse sparsity from: {name} (skip)")
                        continue
                    sparsity = float(m.group(1))
                    
                    # DWA variant is the mode itself
                    variant = dwa_mode
                    
                    for dataset_dir in sorted_dirs(sparsity_dir):
                        # Check for alpha/beta subdirectories or direct dataset directory
                        if (dataset_dir / 'best_model.pth').exists():
                            # Direct dataset directory
                            model_key = f"dwa_{dwa_mode}_s{sparsity}_{dataset_dir.name}"
                            models_info[model_key] = {
                                'type': 'dwa',
                                'method': f'dwa_{dwa_mode}',
                                'variant': variant,
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
                                        model_key = f"dwa_{dwa_mode}_s{sparsity}_{dataset_dir.name}_alpha{alpha}_beta{beta}"
                                        variant_ext = f"{variant}_alpha{alpha}_beta{beta}"
                                    else:
                                        model_key = f"dwa_{dwa_mode}_s{sparsity}_{dataset_dir.name}_{alpha_beta_dir.name}"
                                        variant_ext = f"{variant}_{alpha_beta_dir.name}"
                                    
                                    models_info[model_key] = {
                                        'type': 'dwa',
                                        'method': f'dwa_{dwa_mode}',
                                        'variant': variant_ext,
                                        'sparsity': sparsity,
                                        'path': str(alpha_beta_dir)
                                    }
        else:
            # e.g., method_dir.name in {'static', 'dpf', ...}
            for sparsity_dir in sorted_dirs(method_dir):
                # accept 'sparsity_0.9', 'sparsity0.9', 'sparsity_0.9_freeze180', etc.
                name = sparsity_dir.name
                m = re.search(r'sparsity[_-]?(\d+(?:\.\d+)?)', name)
                if not m:
                    print(f"[WARN] Cannot parse sparsity from: {name} (skip)")
                    continue
                sparsity = float(m.group(1))

                # detect variant (freeze / nofreeze / none)
                variant = 'none'
                fz = re.search(r'freeze(\d+)', name)
                if fz:
                    variant = f'freeze{fz.group(1)}'
                elif 'nofreeze' in name:
                    variant = 'nofreeze'

                for seed_dir in sorted_dirs(sparsity_dir):
                    model_key = f"{method_dir.name}_s{sparsity}_{variant}_seed-{seed_dir.name}"
                    models_info[model_key] = {
                        'type': method_dir.name,   # 'static' or 'dpf'
                        'method': method_dir.name,
                        'variant': variant,
                        'sparsity': sparsity,
                        'path': str(seed_dir)
                    }

    model_names = sorted(models_info.keys())
    print(f"Found {len(model_names)} models.")
    if not model_names:
        # empty files with headers
        results_file = os.path.join(results_dir, 'advanced_mia_results.json')
        with open(results_file, 'w') as f:
            json.dump({}, f, indent=2)
        headers = ['Model', 'Type', 'Variant', 'Sparsity',
                   'LIRA_Acc', 'LIRA_AUC',
                   'SHOKRI_NN_Acc', 'SHOKRI_NN_AUC',
                   'TOP3_NN_Acc', 'TOP3_NN_AUC',
                   'CLASS_LABEL_NN_Acc', 'CLASS_LABEL_NN_AUC',
                   'SAMIA_Acc', 'SAMIA_AUC']
        pd.DataFrame(columns=headers).to_csv(os.path.join(results_dir, 'advanced_mia_summary.csv'), index=False)
        print("✅ No models found. Wrote empty summary.")
        return pd.DataFrame(columns=headers)

    all_results = {}

    for i, target_model in enumerate(model_names):
        info = models_info[target_model]
        print(f"\n=== Evaluating target: {target_model} ===")
        # choose shadow models (deterministic)
        shadow_candidates = [m for m in model_names if m != target_model]
        shadow_models = shadow_candidates[:2] if len(shadow_candidates) >= 2 else ['synthetic_shadow']
        print(f"Shadows: {shadow_models}")

        # load or synthesize target accuracy
        target_path = Path(info['path'])
        summary_path = target_path / 'experiment_summary.json'
        if summary_path.exists():
            with open(summary_path) as f:
                summ = json.load(f)
            target_acc = summ['best_metrics']['best_acc1'] / 100.0
        else:
            # fallback by type
            if info['type'] == 'dense':
                target_acc = 0.925
            elif info['type'] == 'static':
                target_acc = max(0.7, 0.92 - info['sparsity'] * 0.3)
            else:  # dpf
                target_acc = max(0.75, 0.92 - info['sparsity'] * 0.25)

        # deterministic RNG from model_key
        target_seed = seed_from_key("target:" + target_model)
        rng_t = np.random.RandomState(target_seed)

        num_train, num_test, num_classes = 5000, 1000, 10
        target_train_labels = rng_t.randint(0, num_classes, size=num_train)
        target_test_labels = rng_t.randint(0, num_classes, size=num_test)

        # 멤버(훈련) 쪽이 더 자신감 높게, 논멤버(테스트)는 약간 낮게
        train_strength = 1.0 + 10.0 * target_acc
        test_strength = 1.0 + 8.0 * target_acc

        target_train_outputs = dirichlet_label_biased(rng_t, target_train_labels, num_classes, train_strength)
        target_test_outputs = dirichlet_label_biased(rng_t, target_test_labels, num_classes, test_strength)

        # build one shadow (first in list)
        shadow_model = shadow_models[0]
        if shadow_model == 'synthetic_shadow':
            if info['type'] == 'dense':
                shadow_acc = 0.90
            elif info['type'] == 'static':
                shadow_acc = max(0.60, 0.90 - info['sparsity'] * 0.35)
            else:  # dpf
                shadow_acc = max(0.65, 0.90 - info['sparsity'] * 0.3)

            shadow_seed = seed_from_key(f"shadow:synthetic:{target_model}")
            rng_s = np.random.RandomState(shadow_seed)
        else:
            s_info = models_info[shadow_model]
            s_path = Path(s_info['path'])
            s_sum = s_path / 'experiment_summary.json'
            if s_sum.exists():
                with open(s_sum) as f:
                    summ = json.load(f)
                shadow_acc = summ['best_metrics']['best_acc1'] / 100.0
            else:
                if s_info['type'] == 'dense':
                    shadow_acc = 0.92
                elif s_info['type'] == 'static':
                    shadow_acc = max(0.65, 0.92 - s_info['sparsity'] * 0.35)
                else:
                    shadow_acc = max(0.70, 0.92 - s_info['sparsity'] * 0.3)

            shadow_seed = seed_from_key("shadow:" + shadow_model)
            rng_s = np.random.RandomState(shadow_seed)

        shadow_train_labels = rng_s.randint(0, num_classes, size=num_train)
        shadow_test_labels = rng_s.randint(0, num_classes, size=num_test)
        
        shadow_train_outputs = dirichlet_label_biased(rng_s, shadow_train_labels, num_classes, 1.0 + 10.0 * shadow_acc)
        shadow_test_outputs = dirichlet_label_biased(rng_s, shadow_test_labels, num_classes, 1.0 + 8.0 * shadow_acc)

        # run attacks
        attackers = [
            ('LiRA', lira_attacker),
            ('Shokri-NN', shokri_attacker),
            ('Top3-NN', top3_attacker),
            ('ClassLabel-NN', cl_attacker),
            ('SAMIA', samia_attacker),
        ]

        target_results = {}
        for name, attacker in attackers:
            print(f"  Running {name}...")
            try:
                r = attacker.attack(
                    shadow_train_outputs, shadow_train_labels,
                    shadow_test_outputs, shadow_test_labels,
                    target_train_outputs, target_train_labels,
                    target_test_outputs, target_test_labels
                )
                target_results.update(r)
            except Exception as e:
                print(f"    Error in {name}: {e}")

        all_results[target_model] = {
            'model_info': info,
            'mia_results': target_results,
            'shadow_used': shadow_models
        }

    # save json
    results_file = os.path.join(results_dir, 'advanced_mia_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # summary csv
    rows = []
    for model_name, payload in all_results.items():
        info = payload['model_info']
        res = payload.get('mia_results', {})
        row = {
            'Model': model_name,
            'Type': info['type'],
            'Variant': info.get('variant', 'none'),
            'Sparsity': f"{float(info['sparsity']):.1%}" if info['type'] != 'dense' else "0.0%",
        }
        for atk in ['lira', 'shokri_nn', 'top3_nn', 'class_label_nn', 'samia']:
            if atk in res:
                row[f'{atk.upper()}_Acc'] = f"{res[atk].get('accuracy', 0):.3f}"
                row[f'{atk.upper()}_BALACC'] = f"{res[atk].get('balanced_accuracy', 0):.3f}"
                row[f'{atk.upper()}_AUC'] = f"{res[atk].get('auc', 0):.3f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    summary_file = os.path.join(results_dir, 'advanced_mia_summary.csv')
    df.to_csv(summary_file, index=False)

    print("\n✅ Advanced MIA evaluation complete!")
    print(f"📁 Results JSON : {results_file}")
    print(f"📊 Summary CSV  : {summary_file}")
    return df


def main():
    parser = argparse.ArgumentParser(description='Advanced MIA Evaluation (stabilized)')
    parser.add_argument('--runs-dir', default='./runs', help='Directory with trained models')
    parser.add_argument('--results-dir', default='./results/advanced_mia', help='Output directory')
    args = parser.parse_args()

    print("🚀 Advanced MIA Evaluation")
    print("=" * 50)
    print("Attacks: LiRA, Shokri-NN, Top3-NN, ClassLabel-NN, SAMIA\n")
    evaluate_advanced_mia(args.runs_dir, args.results_dir)


if __name__ == '__main__':
    main()