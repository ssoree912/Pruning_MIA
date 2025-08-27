#!/usr/bin/env python3
"""
Comprehensive logging and result collection system
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
import csv
import pickle

class ExperimentLogger:
    """Comprehensive experiment logging system"""
    
    def __init__(self, experiment_name: str, save_dir: str, 
                 log_level: int = logging.INFO):
        """
        Initialize experiment logger
        
        Args:
            experiment_name: Name of the experiment
            save_dir: Directory to save logs and results
            log_level: Logging level
        """
        self.experiment_name = experiment_name
        self.save_dir = save_dir
        self.start_time = time.time()
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize logging
        self.log_file = os.path.join(save_dir, f'{experiment_name}.log')
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(experiment_name)
        
        # Initialize result storage
        self.metrics = {}
        self.hyperparameters = {}
        self.system_info = {}
        self.training_history = []
        self.validation_history = []
        
        # Record experiment start
        self.logger.info(f"Starting experiment: {experiment_name}")
        self.logger.info(f"Save directory: {save_dir}")
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        self.hyperparameters.update(params)
        self.logger.info("Hyperparameters:")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")
        
        # Save to JSON
        with open(os.path.join(self.save_dir, 'hyperparameters.json'), 'w') as f:
            json.dump(self.hyperparameters, f, indent=2)
    
    def log_system_info(self, info: Dict[str, Any]):
        """Log system information"""
        self.system_info.update(info)
        self.logger.info("System Information:")
        for key, value in info.items():
            self.logger.info(f"  {key}: {value}")
        
        # Save to JSON
        with open(os.path.join(self.save_dir, 'system_info.json'), 'w') as f:
            json.dump(self.system_info, f, indent=2)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        if step is not None:
            self.logger.info(f"Step {step} metrics:")
        else:
            self.logger.info("Metrics:")
        
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value:.6f}")
            
            # Store in metrics dict
            if key not in self.metrics:
                self.metrics[key] = []
            
            metric_entry = {'value': value, 'step': step, 'timestamp': time.time()}
            self.metrics[key].append(metric_entry)
        
        # Save to JSON
        with open(os.path.join(self.save_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def log_epoch(self, epoch: int, train_metrics: Dict[str, float], 
                  val_metrics: Dict[str, float], lr: float = None):
        """Log epoch results"""
        
        # Log training metrics
        train_entry = {'epoch': epoch, 'timestamp': time.time()}
        train_entry.update(train_metrics)
        if lr is not None:
            train_entry['learning_rate'] = lr
        self.training_history.append(train_entry)
        
        # Log validation metrics
        val_entry = {'epoch': epoch, 'timestamp': time.time()}
        val_entry.update(val_metrics)
        self.validation_history.append(val_entry)
        
        # Log to console
        self.logger.info(f"Epoch {epoch}:")
        self.logger.info(f"  Train: " + " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()]))
        self.logger.info(f"  Val:   " + " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()]))
        if lr is not None:
            self.logger.info(f"  LR: {lr:.6f}")
        
        # Save histories
        self.save_training_history()
    
    def save_training_history(self):
        """Save training and validation history"""
        # Save as JSON
        with open(os.path.join(self.save_dir, 'training_history.json'), 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        with open(os.path.join(self.save_dir, 'validation_history.json'), 'w') as f:
            json.dump(self.validation_history, f, indent=2)
        
        # Save as CSV for easy analysis
        if self.training_history:
            df_train = pd.DataFrame(self.training_history)
            df_train.to_csv(os.path.join(self.save_dir, 'training_history.csv'), index=False)
        
        if self.validation_history:
            df_val = pd.DataFrame(self.validation_history)
            df_val.to_csv(os.path.join(self.save_dir, 'validation_history.csv'), index=False)
        
        # Save as NumPy arrays for fast loading
        if self.training_history:
            train_array = np.array([[entry['epoch']] + [entry.get(k, 0) for k in ['acc1', 'acc5', 'loss']] 
                                   for entry in self.training_history])
            np.save(os.path.join(self.save_dir, 'train_history.npy'), train_array)
        
        if self.validation_history:
            val_array = np.array([[entry['epoch']] + [entry.get(k, 0) for k in ['acc1', 'acc5', 'loss']] 
                                 for entry in self.validation_history])
            np.save(os.path.join(self.save_dir, 'val_history.npy'), val_array)
    
    def log_pruning_info(self, sparsity: float, reactivation_rate: float = None, 
                        mask_changes: int = None):
        """Log pruning-specific information"""
        pruning_info = {
            'sparsity': sparsity,
            'timestamp': time.time()
        }
        
        if reactivation_rate is not None:
            pruning_info['reactivation_rate'] = reactivation_rate
        
        if mask_changes is not None:
            pruning_info['mask_changes'] = mask_changes
        
        # Store in metrics
        if 'pruning_history' not in self.metrics:
            self.metrics['pruning_history'] = []
        
        self.metrics['pruning_history'].append(pruning_info)
        
        # Log to console
        self.logger.info(f"Pruning info: sparsity={sparsity:.3f}")
        if reactivation_rate is not None:
            self.logger.info(f"  Reactivation rate: {reactivation_rate:.4f}")
        if mask_changes is not None:
            self.logger.info(f"  Mask changes: {mask_changes}")
    
    def log_mia_results(self, attack_name: str, results: Dict[str, Any]):
        """Log membership inference attack results"""
        
        # Store results
        if 'mia_results' not in self.metrics:
            self.metrics['mia_results'] = {}
        
        self.metrics['mia_results'][attack_name] = results
        
        # Log summary
        self.logger.info(f"{attack_name} Results:")
        if 'auc' in results:
            self.logger.info(f"  AUC: {results['auc']:.4f}")
        
        if 'tpr_at_fpr' in results:
            for fpr_level, tpr in results['tpr_at_fpr'].items():
                self.logger.info(f"  {fpr_level}: {tpr:.4f}")
        
        # Save detailed results
        results_file = os.path.join(self.save_dir, f'{attack_name}_results.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        # Save summary as JSON
        summary = {
            'attack_name': attack_name,
            'auc': results.get('auc', None),
            'tpr_at_fpr': results.get('tpr_at_fpr', {}),
            'num_samples': {
                'members': results.get('num_members', 0),
                'non_members': results.get('num_non_members', 0)
            },
            'timestamp': time.time()
        }
        
        summary_file = os.path.join(self.save_dir, f'{attack_name}_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def log_model_info(self, model, model_type: str, sparsity: float = None):
        """Log model information"""
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            'model_type': model_type,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'sparsity': sparsity,
            'timestamp': time.time()
        }
        
        self.system_info['model_info'] = model_info
        
        # Log to console
        self.logger.info(f"Model Information:")
        self.logger.info(f"  Type: {model_type}")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        if sparsity is not None:
            self.logger.info(f"  Sparsity: {sparsity:.2%}")
    
    def save_checkpoint_info(self, epoch: int, best_metric: float, 
                           checkpoint_path: str):
        """Log checkpoint information"""
        checkpoint_info = {
            'epoch': epoch,
            'best_metric': best_metric,
            'checkpoint_path': checkpoint_path,
            'timestamp': time.time()
        }
        
        if 'checkpoints' not in self.metrics:
            self.metrics['checkpoints'] = []
        
        self.metrics['checkpoints'].append(checkpoint_info)
        
        self.logger.info(f"Checkpoint saved: epoch={epoch}, metric={best_metric:.4f}")
    
    def log_timing(self, phase: str, elapsed_time: float):
        """Log timing information"""
        if 'timing' not in self.metrics:
            self.metrics['timing'] = {}
        
        self.metrics['timing'][phase] = elapsed_time
        
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60
        
        self.logger.info(f"{phase} time: {hours:02d}h {minutes:02d}m {seconds:05.2f}s")
    
    def finalize(self):
        """Finalize experiment logging"""
        total_time = time.time() - self.start_time
        self.log_timing('total_experiment', total_time)
        
        # Create final summary
        summary = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time,
            'end_time': time.time(),
            'total_duration': total_time,
            'hyperparameters': self.hyperparameters,
            'system_info': self.system_info,
            'final_metrics': {},
            'best_metrics': {}
        }
        
        # Extract final and best metrics from training history
        if self.validation_history:
            last_epoch = self.validation_history[-1]
            summary['final_metrics'] = {k: v for k, v in last_epoch.items() if k != 'epoch'}
            
            # Find best metrics
            if 'acc1' in last_epoch:
                best_acc = max(entry.get('acc1', 0) for entry in self.validation_history)
                summary['best_metrics']['best_acc1'] = best_acc
            
            if 'loss' in last_epoch:
                best_loss = min(entry.get('loss', float('inf')) for entry in self.validation_history)
                summary['best_metrics']['best_loss'] = best_loss
        
        # Save final summary
        with open(os.path.join(self.save_dir, 'experiment_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info("Experiment completed successfully")
        self.logger.info(f"Total time: {total_time/3600:.2f} hours")
        
        # Close logging handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

class ResultsCollector:
    """Collect and aggregate results across multiple experiments"""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        
    def collect_experiment_summaries(self) -> List[Dict]:
        """Collect experiment summaries from all subdirectories"""
        summaries = []
        
        for root, dirs, files in os.walk(self.results_dir):
            if 'experiment_summary.json' in files:
                summary_path = os.path.join(root, 'experiment_summary.json')
                try:
                    with open(summary_path, 'r') as f:
                        summary = json.load(f)
                    summary['experiment_path'] = root
                    summaries.append(summary)
                except Exception as e:
                    print(f"Error loading {summary_path}: {e}")
        
        return summaries
    
    def create_comparison_table(self, summaries: List[Dict]) -> pd.DataFrame:
        """Create comparison table from experiment summaries"""
        
        rows = []
        for summary in summaries:
            row = {
                'experiment_name': summary.get('experiment_name', ''),
                'total_duration_hours': summary.get('total_duration', 0) / 3600,
            }
            
            # Add hyperparameters
            hyperparams = summary.get('hyperparameters', {})
            row.update(hyperparams)
            
            # Add best metrics
            best_metrics = summary.get('best_metrics', {})
            row.update(best_metrics)
            
            # Add final metrics
            final_metrics = summary.get('final_metrics', {})
            for key, value in final_metrics.items():
                row[f'final_{key}'] = value
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def generate_report(self, output_path: str):
        """Generate comprehensive comparison report"""
        
        summaries = self.collect_experiment_summaries()
        
        if not summaries:
            print("No experiment summaries found")
            return
        
        # Create comparison table
        df = self.create_comparison_table(summaries)
        
        # Save as CSV
        csv_path = os.path.join(output_path, 'experiments_comparison.csv')
        df.to_csv(csv_path, index=False)
        
        # Create summary statistics
        report = {
            'generation_time': datetime.now().isoformat(),
            'total_experiments': len(summaries),
            'summary_statistics': {},
            'experiments': summaries
        }
        
        # Calculate summary statistics for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in df.columns:
                report['summary_statistics'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'count': int(df[col].count())
                }
        
        # Save comprehensive report
        report_path = os.path.join(output_path, 'comprehensive_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report generated: {report_path}")
        print(f"Comparison table saved: {csv_path}")
        
        return df, report

def get_system_info() -> Dict[str, Any]:
    """Get system information for logging"""
    import platform
    import torch
    import psutil
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'timestamp': time.time()
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_names'] = [torch.cuda.get_device_name(i) 
                           for i in range(torch.cuda.device_count())]
    
    return info