import torch
import torch.nn.functional as F
from pathlib import Path
import sys
try:
    from mia_eval.core.attacker_threshold import ThresholdAttacker
except ImportError:
    from attacker_threshold import ThresholdAttacker

# Ensure repo root import path for base_model
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from base_model import BaseModel
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from utils.utils import seed_worker



class MiaAttack:
    def __init__(self, victim_model, victim_pruned_model, victim_train_loader, victim_test_loader,
                 shadow_model_list, shadow_pruned_model_list, shadow_train_loader_list, shadow_test_loader_list,
                 num_cls=10, batch_size=128,  device="cuda",
                 lr=0.001, optimizer="sgd", epochs=100, weight_decay=5e-4,
                 # lr=0.001, optimizer="adam", epochs=100, weight_decay=5e-4,
                 attack_original=False
                 ):
        self.victim_model = victim_model
        self.victim_pruned_model = victim_pruned_model
        self.victim_train_loader = victim_train_loader
        self.victim_test_loader = victim_test_loader
        self.shadow_model_list = shadow_model_list
        self.shadow_pruned_model_list = shadow_pruned_model_list
        self.shadow_train_loader_list = shadow_train_loader_list
        self.shadow_test_loader_list = shadow_test_loader_list
        self.num_cls = num_cls
        self.device = device
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.attack_original = attack_original
        self._prepare()

    def _prepare(self):
        print("[MIA] Preparing attack data ...")
        attack_in_predicts_list, attack_in_targets_list, attack_in_sens_list = [], [], []
        attack_out_predicts_list, attack_out_targets_list, attack_out_sens_list = [], [], []
        total = len(self.shadow_model_list)
        for idx, (shadow_model, shadow_pruned_model, shadow_train_loader, shadow_test_loader) in enumerate(zip(
                self.shadow_model_list, self.shadow_pruned_model_list, self.shadow_train_loader_list,
                self.shadow_test_loader_list), start=1):
            print(f"[MIA] Shadow {idx}/{total}: collecting train/test outputs ...")

            if self.attack_original:
                attack_in_predicts, attack_in_targets, attack_in_sens = \
                    shadow_model.predict_target_sensitivity(shadow_train_loader)
                attack_out_predicts, attack_out_targets, attack_out_sens = \
                    shadow_model.predict_target_sensitivity(shadow_test_loader)
            else:
                attack_in_predicts, attack_in_targets, attack_in_sens = \
                    shadow_pruned_model.predict_target_sensitivity(shadow_train_loader)
                attack_out_predicts, attack_out_targets, attack_out_sens = \
                    shadow_pruned_model.predict_target_sensitivity(shadow_test_loader)

            attack_in_predicts_list.append(attack_in_predicts)
            attack_in_targets_list.append(attack_in_targets)
            attack_in_sens_list.append(attack_in_sens)
            attack_out_predicts_list.append(attack_out_predicts)
            attack_out_targets_list.append(attack_out_targets)
            attack_out_sens_list.append(attack_out_sens)

        self.attack_in_predicts = torch.cat(attack_in_predicts_list, dim=0)
        self.attack_in_targets = torch.cat(attack_in_targets_list, dim=0)
        self.attack_in_sens = torch.cat(attack_in_sens_list, dim=0)
        self.attack_out_predicts = torch.cat(attack_out_predicts_list, dim=0)
        self.attack_out_targets = torch.cat(attack_out_targets_list, dim=0)
        self.attack_out_sens = torch.cat(attack_out_sens_list, dim=0)

        print("[MIA] Collecting victim outputs ...")
        if self.attack_original:
            self.victim_in_predicts, self.victim_in_targets, self.victim_in_sens = \
                self.victim_model.predict_target_sensitivity(self.victim_train_loader)
            self.victim_out_predicts, self.victim_out_targets, self.victim_out_sens = \
                self.victim_model.predict_target_sensitivity(self.victim_test_loader)
        else:
            self.victim_in_predicts, self.victim_in_targets, self.victim_in_sens = \
                self.victim_pruned_model.predict_target_sensitivity(self.victim_train_loader)
            self.victim_out_predicts, self.victim_out_targets, self.victim_out_sens = \
                self.victim_pruned_model.predict_target_sensitivity(self.victim_test_loader)
        print("[MIA] Attack data ready.")

    def nn_attack(self, mia_type="nn_sens_cls", model_name="mia_fc"):
        attack_predicts = torch.cat([self.attack_in_predicts, self.attack_out_predicts], dim=0)
        attack_sens = torch.cat([self.attack_in_sens, self.attack_out_sens], dim=0)
        attack_targets = torch.cat([self.attack_in_targets, self.attack_out_targets], dim=0)
        attack_targets = F.one_hot(attack_targets, num_classes=self.num_cls).float()
        attack_labels = torch.cat([torch.ones(self.attack_in_targets.size(0)),
                                   torch.zeros(self.attack_out_targets.size(0))], dim=0).long()

        victim_predicts = torch.cat([self.victim_in_predicts, self.victim_out_predicts], dim=0)
        victim_sens = torch.cat([self.victim_in_sens, self.victim_out_sens], dim=0)
        victim_targets = torch.cat([self.victim_in_targets, self.victim_out_targets], dim=0)
        victim_targets = F.one_hot(victim_targets, num_classes=self.num_cls).float()
        victim_labels = torch.cat([torch.ones(self.victim_in_targets.size(0)),
                                   torch.zeros(self.victim_out_targets.size(0))], dim=0).long()

        if mia_type == "nn_cls":
            new_attack_data = torch.cat([attack_predicts, attack_targets], dim=1)
            new_victim_data = torch.cat([victim_predicts, victim_targets], dim=1)
        elif mia_type == "nn_top3":
            new_attack_data, _ = torch.topk(attack_predicts, k=3, dim=-1)
            new_victim_data, _ = torch.topk(victim_predicts, k=3, dim=-1)
        elif mia_type == "nn_sens_cls":
            new_attack_data = torch.cat([attack_predicts, attack_sens, attack_targets], dim=1)
            new_victim_data = torch.cat([victim_predicts, victim_sens, victim_targets], dim=1)
        else:
            new_attack_data = attack_predicts
            new_victim_data = victim_predicts

        attack_train_dataset = TensorDataset(new_attack_data, attack_labels)
        # Balanced sampling to mitigate member/non-member imbalance
        with torch.no_grad():
            class_counts = torch.bincount(attack_labels, minlength=2).float()
            # Avoid div by zero; if a class is missing, fall back to uniform
            if (class_counts == 0).any():
                sample_weights = torch.ones_like(attack_labels, dtype=torch.float)
            else:
                class_weights = 1.0 / class_counts
                sample_weights = class_weights[attack_labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        attack_train_dataloader = DataLoader(
            attack_train_dataset, batch_size=self.batch_size, sampler=sampler, shuffle=False,
            num_workers=4, pin_memory=True, worker_init_fn=seed_worker)
        attack_test_dataset = TensorDataset(new_victim_data, victim_labels)
        attack_test_dataloader = DataLoader(
            attack_test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True,
            worker_init_fn=seed_worker)

        attack_model = BaseModel(
            model_name, device=self.device, num_cls=new_victim_data.size(1), optimizer=self.optimizer, lr=self.lr,
            weight_decay=self.weight_decay, epochs=self.epochs)

        for epoch in range(self.epochs):
            train_acc, train_loss = attack_model.train(attack_train_dataloader)
            test_acc, test_loss = attack_model.test(attack_test_dataloader)
        # Compute detailed metrics on victim set
        attack_model.model.eval()
        with torch.no_grad():
            logits = attack_model.model(new_victim_data.to(self.device))
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        y_true = victim_labels.cpu().numpy()
        y_pred = (probs >= 0.5).astype(int)
        # TPR@1%FPR helper
        def _tpr_at_fpr(y_true, y_score, fpr_target=0.01):
            y_true = np.asarray(y_true)
            y_score = np.asarray(y_score)
            non_member = y_score[y_true == 0]
            if non_member.size == 0:
                return 0.0
            tau = np.quantile(non_member, 1.0 - fpr_target)
            member = y_score[y_true == 1]
            if member.size == 0:
                return 0.0
            return float((member >= tau).mean())
        # Advantage = TPR - FPR
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tpr = tp / (tp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)
        result = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'auc': float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.0,
            'advantage': float(tpr - fpr),
            'tpr_at_1fpr': _tpr_at_fpr(y_true, probs, 0.01),
        }
        return result

    def lira_attack(self):
        """LiRA-style Gaussian likelihood ratio attack using shadow outputs.
        Estimates member/non-member distributions from shadow in/out and applies LR to victim.
        Returns dict with accuracy/balanced_accuracy/auc/advantage.
        """
        # Convert to numpy
        s_in = self.attack_in_predicts.cpu().numpy()
        s_in_y = self.attack_in_targets.cpu().numpy()
        s_out = self.attack_out_predicts.cpu().numpy()
        s_out_y = self.attack_out_targets.cpu().numpy()
        v_in = self.victim_in_predicts.cpu().numpy()
        v_in_y = self.victim_in_targets.cpu().numpy()
        v_out = self.victim_out_predicts.cpu().numpy()
        v_out_y = self.victim_out_targets.cpu().numpy()

        # Log-likelihood proxy: log p(true_class)
        def true_logp(probs, labels):
            idx = np.arange(labels.shape[0])
            return np.log(np.clip(probs[idx, labels], 1e-12, 1.0))

        s_in_ll = true_logp(s_in, s_in_y)
        s_out_ll = true_logp(s_out, s_out_y)
        v_in_ll = true_logp(v_in, v_in_y)
        v_out_ll = true_logp(v_out, v_out_y)

        # Fit Gaussians
        mu_in,  std_in  = float(np.mean(s_in_ll)),  float(np.std(s_in_ll) + 1e-8)
        mu_out, std_out = float(np.mean(s_out_ll)), float(np.std(s_out_ll) + 1e-8)

        # Gaussian pdf
        def norm_pdf(x, mu, std):
            z = (x - mu) / std
            return np.exp(-0.5 * z * z) / (std * np.sqrt(2.0 * np.pi))

        def lr_scores(ll):
            pin  = norm_pdf(ll, mu_in, std_in)
            pout = norm_pdf(ll, mu_out, std_out)
            return pin / (pout + 1e-12)

        v_train_lr = lr_scores(v_in_ll)
        v_test_lr  = lr_scores(v_out_ll)
        y_true = np.concatenate([np.ones_like(v_train_lr), np.zeros_like(v_test_lr)])
        y_score = np.concatenate([v_train_lr, v_test_lr])
        y_pred = (y_score > 1.0).astype(int)  # LR>1 ⇒ member
        # TPR@1%FPR helper
        def _tpr_at_fpr(y_true, y_score, fpr_target=0.01):
            y_true = np.asarray(y_true)
            y_score = np.asarray(y_score)
            non_member = y_score[y_true == 0]
            if non_member.size == 0:
                return 0.0
            tau = np.quantile(non_member, 1.0 - fpr_target)
            member = y_score[y_true == 1]
            if member.size == 0:
                return 0.0
            return float((member >= tau).mean())

        # Metrics
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tpr = tp / (tp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)

        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
            'auc': float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.0,
            'advantage': float(tpr - fpr),
            'member_mean': mu_in,
            'member_std': std_in,
            'nonmember_mean': mu_out,
            'nonmember_std': std_out,
            'tpr_at_1fpr': _tpr_at_fpr(y_true, y_score, 0.01),
        }

    def threshold_attack(self):
        victim_in_predicts = self.victim_in_predicts.numpy()
        victim_out_predicts = self.victim_out_predicts.numpy()

        attack_in_predicts = self.attack_in_predicts.numpy()
        attack_out_predicts = self.attack_out_predicts.numpy()
        attacker = ThresholdAttacker((attack_in_predicts, self.attack_in_targets.numpy()),
                                 (attack_out_predicts, self.attack_out_targets.numpy()),
                                 (victim_in_predicts, self.victim_in_targets.numpy()),
                                 (victim_out_predicts, self.victim_out_targets.numpy()),
                                 self.num_cls)
        confidence, entropy, modified_entropy = attacker._mem_inf_benchmarks()
        top1_conf, _, _ = attacker._mem_inf_benchmarks_non_cls()
        return confidence * 100., entropy * 100., modified_entropy * 100., \
               top1_conf * 100.
