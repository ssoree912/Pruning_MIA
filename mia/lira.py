#!/usr/bin/env python3
"""
LiRA (Likelihood Ratio Attack) implementation for membership inference
Based on "Membership inference attacks from first principles" by Carlini et al.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import pickle
import json
from tqdm import tqdm

class LiRAAttacker:
    def __init__(self, num_shadow_models: int = 64, confidence_threshold: float = 0.95):
        """
        Initialize LiRA attacker
        
        Args:
            num_shadow_models: Number of shadow models for calibration
            confidence_threshold: Confidence threshold for attack
        """
        self.num_shadow_models = num_shadow_models
        self.confidence_threshold = confidence_threshold
        
        # Stores for shadow model statistics
        self.shadow_in_stats = []  # Loss statistics when data is IN training set
        self.shadow_out_stats = []  # Loss statistics when data is OUT of training set
        
        # Fitted distributions
        self.in_distribution = None
        self.out_distribution = None
        
    def collect_shadow_statistics(self, shadow_models: List[nn.Module], 
                                shadow_datasets: List[Tuple], 
                                device: str = 'cuda'):
        """
        Collect statistics from shadow models for calibration
        
        Args:
            shadow_models: List of shadow models
            shadow_datasets: List of (train_loader, test_loader) tuples for each shadow model
            device: Device to run inference on
        """
        print(f"Collecting statistics from {len(shadow_models)} shadow models...")
        
        all_in_losses = []
        all_out_losses = []
        
        for i, (model, (train_loader, test_loader)) in enumerate(tqdm(zip(shadow_models, shadow_datasets), 
                                                                      total=len(shadow_models), 
                                                                      desc="Processing shadow models")):
            model.eval()
            model.to(device)
            
            # Collect IN losses (from training data)
            in_losses = []
            with torch.no_grad():
                for batch_idx, (data, targets) in enumerate(train_loader):
                    if batch_idx >= 10:  # Limit samples for efficiency
                        break
                        
                    data, targets = data.to(device), targets.to(device)
                    outputs = model(data)
                    
                    # Calculate per-sample loss
                    loss_fn = nn.CrossEntropyLoss(reduction='none')
                    losses = loss_fn(outputs, targets)
                    in_losses.extend(losses.cpu().numpy())
            
            # Collect OUT losses (from test data) 
            out_losses = []
            with torch.no_grad():
                for batch_idx, (data, targets) in enumerate(test_loader):
                    if batch_idx >= 10:  # Limit samples for efficiency
                        break
                        
                    data, targets = data.to(device), targets.to(device)
                    outputs = model(data)
                    
                    # Calculate per-sample loss
                    loss_fn = nn.CrossEntropyLoss(reduction='none')
                    losses = loss_fn(outputs, targets)
                    out_losses.extend(losses.cpu().numpy())
            
            all_in_losses.extend(in_losses)
            all_out_losses.extend(out_losses)
        
        self.shadow_in_stats = np.array(all_in_losses)
        self.shadow_out_stats = np.array(all_out_losses)
        
        print(f"Collected {len(self.shadow_in_stats)} IN samples and {len(self.shadow_out_stats)} OUT samples")
        
    def fit_distributions(self):
        """
        Fit Gaussian distributions to shadow model statistics
        """
        print("Fitting distributions to shadow statistics...")
        
        # Fit Gaussian to IN distribution
        in_mean = np.mean(self.shadow_in_stats)
        in_std = np.std(self.shadow_in_stats)
        self.in_distribution = stats.norm(loc=in_mean, scale=in_std)
        
        # Fit Gaussian to OUT distribution  
        out_mean = np.mean(self.shadow_out_stats)
        out_std = np.std(self.shadow_out_stats)
        self.out_distribution = stats.norm(loc=out_mean, scale=out_std)
        
        print(f"IN distribution: μ={in_mean:.4f}, σ={in_std:.4f}")
        print(f"OUT distribution: μ={out_mean:.4f}, σ={out_std:.4f}")
        
    def calculate_likelihood_ratio(self, losses: np.ndarray) -> np.ndarray:
        """
        Calculate likelihood ratio for given losses
        
        Args:
            losses: Array of loss values
            
        Returns:
            Array of likelihood ratios
        """
        if self.in_distribution is None or self.out_distribution is None:
            raise ValueError("Distributions not fitted. Call fit_distributions() first.")
        
        # Calculate likelihood under each distribution
        p_in = self.in_distribution.pdf(losses)
        p_out = self.out_distribution.pdf(losses)
        
        # Calculate likelihood ratio (avoiding division by zero)
        likelihood_ratio = np.divide(p_in, p_out, 
                                   out=np.zeros_like(p_in), 
                                   where=(p_out != 0))
        
        return likelihood_ratio
    
    def attack(self, target_model: nn.Module, 
               target_train_loader: torch.utils.data.DataLoader,
               target_test_loader: torch.utils.data.DataLoader,
               device: str = 'cuda') -> Dict:
        """
        Perform LiRA attack on target model
        
        Args:
            target_model: Target model to attack
            target_train_loader: Target model's training data
            target_test_loader: Target model's test data  
            device: Device to run inference on
            
        Returns:
            Dictionary with attack results
        """
        print("Performing LiRA attack on target model...")
        
        target_model.eval()
        target_model.to(device)
        
        # Collect losses from target model
        member_losses = []  # Losses on training data (members)
        non_member_losses = []  # Losses on test data (non-members)
        
        # Get member losses (training data)
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(tqdm(target_train_loader, 
                                                            desc="Collecting member losses")):
                if batch_idx >= 50:  # Limit for computational efficiency
                    break
                    
                data, targets = data.to(device), targets.to(device)
                outputs = target_model(data)
                
                loss_fn = nn.CrossEntropyLoss(reduction='none')
                losses = loss_fn(outputs, targets)
                member_losses.extend(losses.cpu().numpy())
        
        # Get non-member losses (test data)
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(tqdm(target_test_loader, 
                                                            desc="Collecting non-member losses")):
                if batch_idx >= 50:  # Limit for computational efficiency  
                    break
                    
                data, targets = data.to(device), targets.to(device)
                outputs = target_model(data)
                
                loss_fn = nn.CrossEntropyLoss(reduction='none')
                losses = loss_fn(outputs, targets)
                non_member_losses.extend(losses.cpu().numpy())
        
        member_losses = np.array(member_losses)
        non_member_losses = np.array(non_member_losses)
        
        print(f"Collected {len(member_losses)} member and {len(non_member_losses)} non-member samples")
        
        # Calculate likelihood ratios
        member_lr = self.calculate_likelihood_ratio(member_losses)
        non_member_lr = self.calculate_likelihood_ratio(non_member_losses)
        
        # Create labels (1 for member, 0 for non-member)
        labels = np.concatenate([np.ones(len(member_lr)), np.zeros(len(non_member_lr))])
        scores = np.concatenate([member_lr, non_member_lr])
        
        # Calculate AUC
        auc = roc_auc_score(labels, scores)
        
        # Calculate TPR at specific FPR thresholds
        fpr, tpr, thresholds = roc_curve(labels, scores)
        
        # Find TPR at common FPR values
        target_fprs = [0.001, 0.01, 0.05, 0.1]
        tpr_at_fpr = {}
        
        for target_fpr in target_fprs:
            # Find closest FPR
            idx = np.argmin(np.abs(fpr - target_fpr))
            tpr_at_fpr[f'TPR@{target_fpr}'] = tpr[idx]
        
        results = {
            'auc': auc,
            'tpr_at_fpr': tpr_at_fpr,
            'member_losses': member_losses,
            'non_member_losses': non_member_losses,
            'member_lr': member_lr,
            'non_member_lr': non_member_lr,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'num_members': len(member_losses),
            'num_non_members': len(non_member_losses)
        }
        
        print(f"Attack Results:")
        print(f"  AUC: {auc:.4f}")
        for key, value in tpr_at_fpr.items():
            print(f"  {key}: {value:.4f}")
        
        return results
    
    def save_calibration(self, save_path: str):
        """Save calibration data"""
        calibration_data = {
            'shadow_in_stats': self.shadow_in_stats,
            'shadow_out_stats': self.shadow_out_stats,
            'in_distribution': {
                'loc': self.in_distribution.kwds['loc'] if self.in_distribution else None,
                'scale': self.in_distribution.kwds['scale'] if self.in_distribution else None,
            },
            'out_distribution': {
                'loc': self.out_distribution.kwds['loc'] if self.out_distribution else None,
                'scale': self.out_distribution.kwds['scale'] if self.out_distribution else None,
            },
            'num_shadow_models': self.num_shadow_models
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(calibration_data, f)
        print(f"Calibration data saved to {save_path}")
    
    def load_calibration(self, load_path: str):
        """Load calibration data"""
        with open(load_path, 'rb') as f:
            calibration_data = pickle.load(f)
        
        self.shadow_in_stats = calibration_data['shadow_in_stats']
        self.shadow_out_stats = calibration_data['shadow_out_stats']
        self.num_shadow_models = calibration_data['num_shadow_models']
        
        # Restore distributions
        if calibration_data['in_distribution']['loc'] is not None:
            self.in_distribution = stats.norm(
                loc=calibration_data['in_distribution']['loc'],
                scale=calibration_data['in_distribution']['scale']
            )
        
        if calibration_data['out_distribution']['loc'] is not None:
            self.out_distribution = stats.norm(
                loc=calibration_data['out_distribution']['loc'],
                scale=calibration_data['out_distribution']['scale']
            )
        
        print(f"Calibration data loaded from {load_path}")

    def plot_distributions(self, save_path: Optional[str] = None):
        """Plot the fitted distributions"""
        if self.shadow_in_stats is None or self.shadow_out_stats is None:
            raise ValueError("No statistics available. Collect shadow statistics first.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot histograms
        ax1.hist(self.shadow_in_stats, bins=50, alpha=0.7, label='IN (Members)', density=True)
        ax1.hist(self.shadow_out_stats, bins=50, alpha=0.7, label='OUT (Non-members)', density=True)
        
        # Plot fitted distributions
        x_range = np.linspace(min(np.min(self.shadow_in_stats), np.min(self.shadow_out_stats)),
                             max(np.max(self.shadow_in_stats), np.max(self.shadow_out_stats)), 100)
        
        if self.in_distribution:
            ax1.plot(x_range, self.in_distribution.pdf(x_range), 'b--', 
                    label=f'IN fit (μ={self.in_distribution.mean():.3f})')
        if self.out_distribution:
            ax1.plot(x_range, self.out_distribution.pdf(x_range), 'r--', 
                    label=f'OUT fit (μ={self.out_distribution.mean():.3f})')
        
        ax1.set_xlabel('Loss')
        ax1.set_ylabel('Density')
        ax1.set_title('Loss Distributions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot likelihood ratio
        if self.in_distribution and self.out_distribution:
            lr = self.in_distribution.pdf(x_range) / self.out_distribution.pdf(x_range)
            ax2.plot(x_range, lr, 'g-', linewidth=2)
            ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Loss')
            ax2.set_ylabel('Likelihood Ratio')
            ax2.set_title('Likelihood Ratio vs Loss')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution plot saved to {save_path}")
        
        plt.show()

def create_shadow_datasets(original_train_loader, original_test_loader, 
                          num_shadows: int = 64, seed: int = 42):
    """
    Create shadow datasets by randomly splitting available data
    
    Args:
        original_train_loader: Original training data loader
        original_test_loader: Original test data loader  
        num_shadows: Number of shadow models to create datasets for
        seed: Random seed for reproducibility
        
    Returns:
        List of (shadow_train_loader, shadow_test_loader) tuples
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Combine all available data
    all_data = []
    all_targets = []
    
    # Extract data from loaders
    for data, targets in original_train_loader:
        all_data.append(data)
        all_targets.append(targets)
    
    for data, targets in original_test_loader:
        all_data.append(data)
        all_targets.append(targets)
    
    # Concatenate all data
    all_data = torch.cat(all_data, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Create shadow datasets
    shadow_datasets = []
    total_samples = len(all_data)
    train_size = len(original_train_loader.dataset)
    
    for i in range(num_shadows):
        # Random permutation for each shadow model
        perm = torch.randperm(total_samples)
        
        # Split into train/test for shadow model
        shadow_train_indices = perm[:train_size]
        shadow_test_indices = perm[train_size:train_size*2]  # Take same amount for test
        
        shadow_train_data = all_data[shadow_train_indices]
        shadow_train_targets = all_targets[shadow_train_indices]
        shadow_test_data = all_data[shadow_test_indices]  
        shadow_test_targets = all_targets[shadow_test_indices]
        
        # Create datasets and loaders
        shadow_train_dataset = torch.utils.data.TensorDataset(shadow_train_data, shadow_train_targets)
        shadow_test_dataset = torch.utils.data.TensorDataset(shadow_test_data, shadow_test_targets)
        
        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train_dataset, batch_size=original_train_loader.batch_size, shuffle=True
        )
        shadow_test_loader = torch.utils.data.DataLoader(
            shadow_test_dataset, batch_size=original_test_loader.batch_size, shuffle=False
        )
        
        shadow_datasets.append((shadow_train_loader, shadow_test_loader))
    
    return shadow_datasets