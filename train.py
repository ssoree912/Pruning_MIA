#!/usr/bin/env python3
"""
통합 훈련 및 결과 수집 스크립트
훈련 완료 후 자동으로 MIA 평가 수행하고 결과 수집
"""

import os
import sys
import json
import subprocess
import pandas as pd
from pathlib import Path
import argparse
import shutil
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_organized_save_path(method, sparsity=None, dataset='cifar10', freeze_epoch=None, total_epochs=None):
    """Create organized save path structure"""
    base_path = Path('./runs')
    
    if method == 'dense':
        save_path = base_path / 'dense' / dataset
    elif method == 'static':
        if sparsity is None:
            raise ValueError(f"Sparsity must be specified for {method} method")
        save_path = base_path / method / f'sparsity_{sparsity}' / dataset
    elif method == 'dpf':
        if sparsity is None:
            raise ValueError(f"Sparsity must be specified for {method} method")
        
        # Add freeze info to path
        if freeze_epoch is not None and total_epochs is not None:
            if freeze_epoch >= 0 and freeze_epoch < total_epochs:
                freeze_suffix = f'_freeze{freeze_epoch}'
            elif freeze_epoch < 0:
                freeze_suffix = '_nofreeze'
            else:
                freeze_suffix = ''
        else:
            freeze_suffix = ''
            
        save_path = base_path / method / f'sparsity_{sparsity}{freeze_suffix}' / dataset
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return save_path

def run_training(config_params):
    """Run training with specified parameters"""
    cmd = ['python', 'run_experiment.py']
    
    # Add all config parameters to command
    for key, value in config_params.items():
        if value is not None:
            if isinstance(value, bool):
                # For boolean flags, only add the flag if True
                if value:
                    cmd.append(f'--{key}')
            else:
                cmd.extend([f'--{key}', str(value)])
    
    print(f"Running training command: {' '.join(cmd)}")
    
    try:
        # Show output in real-time instead of capturing
        # result = subprocess.run(cmd, text=True, timeout=7200)  # 2 hour timeout
        # if result.returncode != 0:
        #     print(f"Training failed with return code: {result.returncode}")
        #     return False, f"Return code: {result.returncode}"
        # return True, "Training completed"
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
        if result.returncode != 0:
                print(f"Training failed with error: {result.stderr}")
            return False, result.stderr
        return True, result.stdout


    except subprocess.TimeoutExpired:
        print("Training timed out after 2 hours")
        return False, "Training timeout"
    except Exception as e:
        print(f"Training failed with exception: {str(e)}")
        return False, str(e)

def run_comprehensive_mia_evaluation(runs_dir):
    """Run comprehensive MIA evaluation using advanced and WeMeM methods"""
    
    results_dir = Path('results/mia')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎯 Running Advanced MIA evaluation...")
    
    # 1. Run Advanced MIA (LiRA, Shokri-NN, Top3-NN, ClassLabel-NN, SAMIA)
    cmd_advanced = ['python', 'mia/mia_advanced.py', '--runs-dir', str(runs_dir), '--results-dir', str(results_dir / 'advanced')]
    print(f"Command: {' '.join(cmd_advanced)}")
    
    try:
        result = subprocess.run(cmd_advanced, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            print(f"Advanced MIA failed: {result.stderr}")
            advanced_success = False
        else:
            print("✅ Advanced MIA completed")
            advanced_success = True
    except subprocess.TimeoutExpired:
        print("❌ Advanced MIA timed out")
        advanced_success = False
    except Exception as e:
        print(f"❌ Advanced MIA error: {e}")
        advanced_success = False
    
    print("\n🔍 Running WeMeM MIA evaluation...")
    
    # 2. Run WeMeM MIA (Confidence, Entropy, Modified Entropy, Neural Network)
    cmd_wemem = ['python', 'mia/mia_classic.py', '--runs-dir', str(runs_dir), '--results-dir', str(results_dir / 'wemem')]
    print(f"Command: {' '.join(cmd_wemem)}")
    
    try:
        result = subprocess.run(cmd_wemem, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            print(f"WeMeM MIA failed: {result.stderr}")
            wemem_success = False
        else:
            print("✅ WeMeM MIA completed")
            wemem_success = True
    except subprocess.TimeoutExpired:
        print("❌ WeMeM MIA timed out")
        wemem_success = False
    except Exception as e:
        print(f"❌ WeMeM MIA error: {e}")
        wemem_success = False
    
    # 3. Combine results
    print("\n🔄 Combining MIA results...")
    try:
        combine_mia_results(results_dir, advanced_success, wemem_success)
        return True, "MIA evaluation completed"
    except Exception as e:
        return False, f"Failed to combine results: {e}"

def combine_mia_results(results_dir, advanced_success, wemem_success):
    """Combine Advanced and WeMeM MIA results into unified CSV"""
    
    combined_data = []
    
    # Load Advanced MIA results
    if advanced_success:
        advanced_csv = results_dir / 'advanced' / 'advanced_mia_summary.csv'
        if advanced_csv.exists():
            import pandas as pd
            advanced_df = pd.read_csv(advanced_csv)
            
            for _, row in advanced_df.iterrows():
                entry = {
                    'experiment': row['Model'],
                    'method': row['Type'],
                    'sparsity': float(row['Sparsity'].replace('%', '')) / 100 if isinstance(row['Sparsity'], str) else row['Sparsity'],
                    
                    # Advanced MIA results
                    'lira_accuracy': float(row.get('LIRA_Acc', '0').replace('%', '')) if isinstance(row.get('LIRA_Acc', 0), str) else row.get('LIRA_Acc', 0),
                    'lira_auc': float(row.get('LIRA_AUC', '0').replace('%', '')) if isinstance(row.get('LIRA_AUC', 0), str) else row.get('LIRA_AUC', 0),
                    'shokri_nn_accuracy': float(row.get('SHOKRI_NN_Acc', '0').replace('%', '')) if isinstance(row.get('SHOKRI_NN_Acc', 0), str) else row.get('SHOKRI_NN_Acc', 0),
                    'shokri_nn_auc': float(row.get('SHOKRI_NN_AUC', '0').replace('%', '')) if isinstance(row.get('SHOKRI_NN_AUC', 0), str) else row.get('SHOKRI_NN_AUC', 0),
                    'top3_nn_accuracy': float(row.get('TOP3_NN_Acc', '0').replace('%', '')) if isinstance(row.get('TOP3_NN_Acc', 0), str) else row.get('TOP3_NN_Acc', 0),
                    'top3_nn_auc': float(row.get('TOP3_NN_AUC', '0').replace('%', '')) if isinstance(row.get('TOP3_NN_AUC', 0), str) else row.get('TOP3_NN_AUC', 0),
                    'class_label_nn_accuracy': float(row.get('CLASS_LABEL_NN_Acc', '0').replace('%', '')) if isinstance(row.get('CLASS_LABEL_NN_Acc', 0), str) else row.get('CLASS_LABEL_NN_Acc', 0),
                    'class_label_nn_auc': float(row.get('CLASS_LABEL_NN_AUC', '0').replace('%', '')) if isinstance(row.get('CLASS_LABEL_NN_AUC', 0), str) else row.get('CLASS_LABEL_NN_AUC', 0),
                    'samia_accuracy': float(row.get('SAMIA_Acc', '0').replace('%', '')) if isinstance(row.get('SAMIA_Acc', 0), str) else row.get('SAMIA_Acc', 0),
                    'samia_auc': float(row.get('SAMIA_AUC', '0').replace('%', '')) if isinstance(row.get('SAMIA_AUC', 0), str) else row.get('SAMIA_AUC', 0),
                }
                combined_data.append(entry)
    
    # Load WeMeM MIA results  
    if wemem_success:
        wemem_csv = results_dir / 'wemem' / 'wemem_mia_summary.csv'
        if wemem_csv.exists():
            import pandas as pd
            try:
                wemem_df = pd.read_csv(wemem_csv)
                if wemem_df.empty:
                    print("⚠️ WeMeM CSV is empty, skipping WeMeM results")
                    wemem_df = None
            except pd.errors.EmptyDataError:
                print("⚠️ WeMeM CSV has no data, skipping WeMeM results")
                wemem_df = None
            
            # Match by experiment name and add WeMeM results
            if wemem_df is not None:
                for _, row in wemem_df.iterrows():
                    model_name = row['Model']
                    
                    # Find matching entry in combined_data
                    matching_entry = None
                    for entry in combined_data:
                        if entry['experiment'] == model_name:
                            matching_entry = entry
                            break
                    
                    if matching_entry is None:
                        # Create new entry if not found
                        matching_entry = {
                            'experiment': model_name,
                            'method': row['Type'],
                            'sparsity': row['Sparsity'],
                        }
                        combined_data.append(matching_entry)
                    
                    # Add WeMeM results - using correct column names
                    def safe_float(value):
                        try:
                            return float(value) if value is not None and str(value).strip() != '' and str(value) != '0' else 0
                        except (ValueError, TypeError):
                            return 0
                    
                    matching_entry.update({
                        'confidence_accuracy': safe_float(row.get('Confidence_Accuracy')),
                        'confidence_f1': safe_float(row.get('Confidence_F1')),
                        'entropy_accuracy': safe_float(row.get('Entropy_Accuracy')),
                        'entropy_f1': safe_float(row.get('Entropy_F1')),
                        'modified_entropy_accuracy': safe_float(row.get('Modified_Entropy_Accuracy')),
                        'modified_entropy_f1': safe_float(row.get('Modified_Entropy_F1')),
                        'neural_network_accuracy': safe_float(row.get('Neural_Network_Accuracy')),
                        'neural_network_f1': safe_float(row.get('Neural_Network_F1')),
                        'neural_network_auc': safe_float(row.get('Neural_Network_AUC')),
                    })
    
    # Save combined results
    if combined_data:
        import pandas as pd
        combined_df = pd.DataFrame(combined_data)
        
        # Sort by method and sparsity
        if 'method' in combined_df.columns and 'sparsity' in combined_df.columns:
            combined_df = combined_df.sort_values(['method', 'sparsity'])
        
        combined_file = results_dir / 'comprehensive_mia_results.csv'
        combined_df.to_csv(combined_file, index=False)
        
        print(f"📊 Combined MIA results saved: {combined_file}")
        print(f"📈 Total experiments: {len(combined_data)}")
        
        # Display summary
        print(f"\n🎯 MIA Attack Results Summary:")
        key_cols = ['experiment', 'method', 'sparsity', 'lira_auc', 'confidence_accuracy', 'neural_network_auc']
        available_cols = [col for col in key_cols if col in combined_df.columns]
        if available_cols:
            print(combined_df[available_cols].to_string(index=False))
    else:
        print("⚠️ No combined data to save")

    return True

def collect_results(results_dir):
    """Collect training and MIA results from directory"""
    results = {}
    
    # Look for training results
    config_file = results_dir / 'config.json'
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
            results['config'] = config
    
    # Look for training logs
    log_files = list(results_dir.glob('*.log'))
    if log_files:
        results['log_file'] = str(log_files[0])
    
    # Look for experiment summary (training results)
    experiment_summary_file = results_dir / 'experiment_summary.json'
    if experiment_summary_file.exists():
        with open(experiment_summary_file, 'r') as f:
            training_results = json.load(f)
            results['training'] = training_results
    
    # Look for validation history
    val_history_file = results_dir / 'validation_history.json'
    if val_history_file.exists():
        with open(val_history_file, 'r') as f:
            val_history = json.load(f)
            results['validation_history'] = val_history
    
    # Look for MIA results
    mia_results_file = results_dir / 'mia_results.json'
    if mia_results_file.exists():
        with open(mia_results_file, 'r') as f:
            mia_results = json.load(f)
            results['mia'] = mia_results
    
    return results

def create_training_summary_csv(all_results, experiment_prefix='experiments'):
    """Create summary CSV with all results"""
    # Ensure results directory exists
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    summary_data = []
    
    for exp_name, results in all_results.items():
        row = {'experiment': exp_name}
        
        # Add config info
        if 'config' in results:
            config = results['config']
            row.update({
                'method': config.get('pruning', {}).get('method', 'dense'),
                'sparsity': config.get('pruning', {}).get('sparsity', 0.0),
                'dataset': config.get('data', {}).get('dataset', 'cifar10'),
                'arch': config.get('model', {}).get('arch', 'resnet'),
                'layers': config.get('model', {}).get('layers', 20),
                'epochs': config.get('training', {}).get('epochs', 200),
                'lr': config.get('training', {}).get('lr', 0.1),
            })
        
        # Add training results if available
        if 'training' in results:
            training = results['training']
            row.update({
                'best_acc1': training.get('best_metrics', {}).get('best_acc1', None),
                'best_loss': training.get('best_metrics', {}).get('best_loss', None),
                'final_acc1': training.get('final_metrics', {}).get('acc1', None),
                'final_acc5': training.get('final_metrics', {}).get('acc5', None),
                'final_loss': training.get('final_metrics', {}).get('loss', None),
                'total_training_time_hours': training.get('total_duration', 0) / 3600 if training.get('total_duration') else None,
            })
        
        # Add validation history summary if available
        if 'validation_history' in results:
            val_history = results['validation_history']
            if val_history:
                # Get best and final metrics from validation history
                best_val_acc1 = max(epoch.get('acc1', 0) for epoch in val_history if 'acc1' in epoch) if val_history else None
                final_val_acc1 = val_history[-1].get('acc1', None) if val_history else None
                final_val_loss = val_history[-1].get('loss', None) if val_history else None
                
                row.update({
                    'val_best_acc1': best_val_acc1,
                    'val_final_acc1': final_val_acc1,
                    'val_final_loss': final_val_loss,
                })
        
        # MIA 결과는 별도 파일로 분리
        
        summary_data.append(row)
    
    # 실험 설정 기반으로 파일명 생성
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 첫 번째 실험의 설정으로 파일명 생성
    if all_results:
        first_result = next(iter(all_results.values()))
        if 'config' in first_result:
            config = first_result['config']
            dataset = config.get('data', {}).get('dataset', 'cifar10')
            arch = config.get('model', {}).get('arch', 'resnet')
            epochs = config.get('training', {}).get('epochs', 200)
            methods = set()
            for result in all_results.values():
                if 'config' in result:
                    method = result['config'].get('pruning', {}).get('method', 'dense')
                    methods.add(method)
            methods_str = '-'.join(sorted(methods))
            file_suffix = f"{methods_str}_{dataset}_{arch}_e{epochs}_{timestamp}"
        else:
            file_suffix = f"{experiment_prefix}_{timestamp}"
    else:
        file_suffix = f"{experiment_prefix}_{timestamp}"
    
    output_file = f"results/training_results_{file_suffix}.csv"
    
    df = pd.DataFrame(summary_data)
    df.to_csv(output_file, index=False)
    print(f"Summary results saved to {output_file}")
    return df

def log_mia_results_to_wandb(args):
    """Log MIA evaluation results to Weights & Biases"""
    try:
        import wandb
        
        # Check if MIA results file exists
        mia_results_file = Path('results/mia/comprehensive_mia_results.csv')
        if not mia_results_file.exists():
            print("⚠️ MIA results file not found, skipping wandb logging")
            return
        
        # Read MIA results
        mia_df = pd.read_csv(mia_results_file)
        
        print("📊 Logging MIA results to Weights & Biases...")
        
        # Initialize wandb for MIA results
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            job_type="mia_evaluation",
            name=f"MIA_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags=args.wandb_tags + ['mia', 'evaluation'],
            reinit=True
        )
        
        # Create summary table
        wandb_table = wandb.Table(dataframe=mia_df)
        wandb.log({"MIA_Results_Table": wandb_table})
        
        # Log key metrics for each experiment
        for _, row in mia_df.iterrows():
            experiment_name = row['experiment']
            method = row['method']
            sparsity = row.get('sparsity', 0)
            
            # Create metrics dict
            mia_metrics = {}
            
            # Advanced MIA metrics
            for metric in ['lira_accuracy', 'lira_auc', 'shokri_nn_accuracy', 'shokri_nn_auc', 
                          'top3_nn_accuracy', 'top3_nn_auc', 'class_label_nn_accuracy', 'class_label_nn_auc',
                          'samia_accuracy', 'samia_auc']:
                if metric in row and pd.notna(row[metric]):
                    mia_metrics[f"mia/{metric}"] = float(row[metric])
            
            # WeMeM metrics
            for metric in ['confidence_accuracy', 'confidence_f1', 'entropy_accuracy', 'entropy_f1',
                          'modified_entropy_accuracy', 'modified_entropy_f1', 'neural_network_accuracy',
                          'neural_network_f1', 'neural_network_auc']:
                if metric in row and pd.notna(row[metric]):
                    mia_metrics[f"mia/{metric}"] = float(row[metric])
            
            # Add experiment info
            mia_metrics.update({
                "mia/experiment": experiment_name,
                "mia/method": method,
                "mia/sparsity": float(sparsity) if sparsity else 0.0,
            })
            
            # Log metrics
            wandb.log(mia_metrics)
        
        # Create visualizations
        if len(mia_df) > 1:
            # Plot MIA attack success rates by method and sparsity
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('MIA Attack Success Rates', fontsize=16)
            
            # LiRA AUC
            if 'lira_auc' in mia_df.columns:
                ax1 = axes[0, 0]
                for method in mia_df['method'].unique():
                    method_data = mia_df[mia_df['method'] == method]
                    if len(method_data) > 1:
                        ax1.plot(method_data['sparsity'], method_data['lira_auc'], 'o-', label=method)
                    else:
                        ax1.scatter(method_data['sparsity'], method_data['lira_auc'], label=method, s=100)
                ax1.set_title('LiRA AUC')
                ax1.set_xlabel('Sparsity')
                ax1.set_ylabel('AUC')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # Neural Network AUC (WeMeM)
            if 'neural_network_auc' in mia_df.columns:
                ax2 = axes[0, 1]
                for method in mia_df['method'].unique():
                    method_data = mia_df[mia_df['method'] == method]
                    if len(method_data) > 1:
                        ax2.plot(method_data['sparsity'], method_data['neural_network_auc'], 'o-', label=method)
                    else:
                        ax2.scatter(method_data['sparsity'], method_data['neural_network_auc'], label=method, s=100)
                ax2.set_title('Neural Network AUC (WeMeM)')
                ax2.set_xlabel('Sparsity')
                ax2.set_ylabel('AUC')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # Confidence Attack Accuracy
            if 'confidence_accuracy' in mia_df.columns:
                ax3 = axes[1, 0]
                for method in mia_df['method'].unique():
                    method_data = mia_df[mia_df['method'] == method]
                    if len(method_data) > 1:
                        ax3.plot(method_data['sparsity'], method_data['confidence_accuracy'], 'o-', label=method)
                    else:
                        ax3.scatter(method_data['sparsity'], method_data['confidence_accuracy'], label=method, s=100)
                ax3.set_title('Confidence Attack Accuracy')
                ax3.set_xlabel('Sparsity')
                ax3.set_ylabel('Accuracy')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # SAMIA AUC
            if 'samia_auc' in mia_df.columns:
                ax4 = axes[1, 1]
                for method in mia_df['method'].unique():
                    method_data = mia_df[mia_df['method'] == method]
                    if len(method_data) > 1:
                        ax4.plot(method_data['sparsity'], method_data['samia_auc'], 'o-', label=method)
                    else:
                        ax4.scatter(method_data['sparsity'], method_data['samia_auc'], label=method, s=100)
                ax4.set_title('SAMIA AUC')
                ax4.set_xlabel('Sparsity')
                ax4.set_ylabel('AUC')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            wandb.log({"MIA_Attack_Comparison": wandb.Image(fig)})
            plt.close(fig)
        
        wandb.finish()
        print("✅ MIA results logged to Weights & Biases successfully!")
        
    except ImportError:
        print("⚠️ wandb not available, skipping MIA results logging")
    except Exception as e:
        print(f"❌ Error logging MIA results to wandb: {e}")

def reorganize_existing_models():
    """Reorganize existing model folders into new structure"""
    runs_dir = Path('./runs')
    if not runs_dir.exists():
        return
    
    print("Reorganizing existing model folders...")
    
    # Find folders to reorganize
    folders_to_move = []
    for folder in runs_dir.iterdir():
        if folder.is_dir():
            name = folder.name
            if name.startswith('dpf_sparsity'):
                sparsity = name.replace('dpf_sparsity', '')
                folders_to_move.append((folder, 'dpf', sparsity))
            elif name.startswith('static_sparsity'):
                sparsity = name.replace('static_sparsity', '')
                folders_to_move.append((folder, 'static', sparsity))
    
    # Move folders
    for old_folder, method, sparsity in folders_to_move:
        new_path = runs_dir / method / f'sparsity_{sparsity}'
        new_path.parent.mkdir(parents=True, exist_ok=True)
        
        if new_path.exists():
            print(f"Target path {new_path} already exists, skipping {old_folder}")
            continue
            
        print(f"Moving {old_folder} -> {new_path}")
        shutil.move(str(old_folder), str(new_path))

def main():
    parser = argparse.ArgumentParser(description='Train models and collect results')
    parser.add_argument('--reorganize-only', action='store_true',
                       help='Only reorganize existing folders, do not train')
    parser.add_argument('--methods', nargs='+', default=['dense', 'static', 'dpf'],
                       choices=['dense', 'static', 'dpf'],
                       help='Methods to train')
    parser.add_argument('--sparsities', nargs='+', type=float, 
                       default=[0.5, 0.7, 0.8, 0.9, 0.95],
                       help='Sparsity levels for pruned methods')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'],
                       help='Dataset to use')
    parser.add_argument('--arch', default='resnet', choices=['resnet', 'wideresnet'],
                       help='Architecture to use')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--freeze-epoch', type=int, default=180,
                       help='Epoch to freeze masks (default: 180)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip experiments if results already exist')
    
    # Wandb arguments
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', default='dcil-pytorch',
                       help='Wandb project name')
    parser.add_argument('--wandb-entity', default=None,
                       help='Wandb entity (username or team)')
    parser.add_argument('--wandb-tags', nargs='*', default=[],
                       help='Wandb tags for experiment')
    
    args = parser.parse_args()
    
    # Reorganize existing folders first
    reorganize_existing_models()
    
    if args.reorganize_only:
        print("Reorganization complete. Exiting.")
        return
    
    all_results = {}
    failed_experiments = []
    
    print(f"Starting training for methods: {args.methods}")
    print(f"Sparsities: {args.sparsities}")
    print(f"Dataset: {args.dataset}, Architecture: {args.arch}")
    
    for method in args.methods:
        if method == 'dense':
            sparsities = [None]  # Dense doesn't need sparsity
        else:
            sparsities = args.sparsities
            
        for sparsity in sparsities:
            exp_name = f"{method}"
            if sparsity is not None:
                exp_name += f"_sparsity_{sparsity}"
            
            # Add freeze indicator to experiment name
            if method == 'dpf':
                if args.freeze_epoch >= 0 and args.freeze_epoch < args.epochs:
                    exp_name += f"_freeze{args.freeze_epoch}"
                elif args.freeze_epoch < 0:
                    exp_name += "_nofreeze"
            
            exp_name += f"_{args.dataset}_{args.arch}"
            
            print(f"\n{'='*50}")
            print(f"Running experiment: {exp_name}")
            print(f"{'='*50}")
            
            # Create save path
            save_path = create_organized_save_path(method, sparsity, args.dataset, args.freeze_epoch, args.epochs)
            
            # Skip if results already exist
            if args.skip_existing and save_path.exists():
                best_model = save_path / 'best_model.pth'
                if best_model.exists():
                    print(f"Results already exist for {exp_name}, skipping...")
                    results = collect_results(save_path)
                    all_results[exp_name] = results
                    continue
            
            # Prepare training config
            config_params = {
                'name': exp_name,
                'save-dir': str(save_path),
                'dataset': args.dataset,
                'arch': args.arch,
                'epochs': args.epochs,
                'freeze-epoch': args.freeze_epoch,
                'seed': args.seed,
                'gpu': args.gpu,
            }
            
            # Add wandb config if enabled
            if args.wandb:
                config_params.update({
                    'wandb': True,
                    'wandb_project': args.wandb_project,
                    'wandb_entity': args.wandb_entity,
                    'wandb_name': exp_name,
                })
                if args.wandb_tags:
                    config_params['wandb_tags'] = ','.join(args.wandb_tags + [method, args.dataset, args.arch])
            
            if method != 'dense':
                config_params.update({
                    'prune': True,
                    'prune-method': method,
                    'sparsity': sparsity,
                })
            
            # Run training
            print(f"Starting training for {exp_name}...")
            success, output = run_training(config_params)
            
            if not success:
                print(f"Training failed for {exp_name}: {output}")
                failed_experiments.append((exp_name, 'training', output))
                continue
            
            print(f"Training completed for {exp_name}")
            
            # Collect results
            results = collect_results(save_path)
            all_results[exp_name] = results
            
            print(f"Experiment {exp_name} completed")
    
    # Create training results summary
    print(f"\n{'='*50}")
    print("Creating training summary...")
    print(f"{'='*50}")
    
    if all_results:
        # 실험 정보를 포함한 파일명으로 저장
        methods_info = f"{'-'.join(args.methods)}"
        summary_df = create_training_summary_csv(all_results, experiment_prefix=methods_info)
        print(f"Completed {len(all_results)} experiments")
        print(f"Training Summary:\n{summary_df.to_string()}")
    
    # Run MIA evaluation on all trained models
    # print(f"\n{'='*50}")
    # print("Running MIA evaluation...")
    # print(f"{'='*50}")
    
    # runs_dir = Path('./runs')
    # success, output = run_comprehensive_mia_evaluation(runs_dir)
    
    # if success:
    #     print("✅ MIA evaluation completed successfully!")
    #     print("📊 MIA results saved in: results/mia/")
    #     print("📁 Results: results/mia/comprehensive_mia_results.csv")
        
    #     # Log MIA results to wandb if available
    #     if args.wandb:
    #         log_mia_results_to_wandb(args)
    # else:
    #     print(f"❌ MIA evaluation failed: {output}")
    #     failed_experiments.append(('mia_evaluation', 'mia', output))
    
    # Report failed experiments
    if failed_experiments:
        print(f"\n{'='*20} FAILED EXPERIMENTS {'='*20}")
        for exp_name, stage, error in failed_experiments:
            print(f"FAILED: {exp_name} ({stage}) - {error}")
    
    print("\nAll experiments completed!")

if __name__ == '__main__':
    main()
