#!/usr/bin/env python3
"""
DWA (Dynamic Weight Adjustment) 실험 스크립트
3가지 실험:
1. Reactivation-only (살리기만)
2. Kill-active & plain-dead (죽이기만) 
3. Kill & Reactivate (둘 다)
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
    # 현재 파이썬 실행기 사용 (가상환경 안전)
    cmd = [sys.executable, 'run_experiment_dwa.py']
    
    # Add all config parameters to command
    for key, value in config_params.items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    cmd.append(f'--{key}')
            else:
                cmd.extend([f'--{key}', str(value)])
    
    print(f"Running training command: {' '.join(cmd)}")
    
    try:
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
    
    # 1. Advanced MIA
    cmd_advanced = [sys.executable, 'mia/mia_advanced.py', '--runs-dir', str(runs_dir), '--results-dir', str(results_dir / 'advanced')]
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
    
    # 2. WeMeM MIA
    cmd_wemem = [sys.executable, 'mia/mia_classic.py', '--runs-dir', str(runs_dir), '--results-dir', str(results_dir / 'wemem')]
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
            advanced_df = pd.read_csv(advanced_csv)
            for _, row in advanced_df.iterrows():
                entry = {
                    'experiment': row['Model'],
                    'method': row['Type'],
                    'sparsity': float(row['Sparsity'].replace('%', '')) / 100 if isinstance(row['Sparsity'], str) else row['Sparsity'],
                    'lira_accuracy': float(str(row.get('LIRA_Acc', 0)).replace('%', '')) if pd.notna(row.get('LIRA_Acc', None)) else 0,
                    'lira_auc': float(str(row.get('LIRA_AUC', 0)).replace('%', '')) if pd.notna(row.get('LIRA_AUC', None)) else 0,
                    'shokri_nn_accuracy': float(str(row.get('SHOKRI_NN_Acc', 0)).replace('%', '')) if pd.notna(row.get('SHOKRI_NN_Acc', None)) else 0,
                    'shokri_nn_auc': float(str(row.get('SHOKRI_NN_AUC', 0)).replace('%', '')) if pd.notna(row.get('SHOKRI_NN_AUC', None)) else 0,
                    'top3_nn_accuracy': float(str(row.get('TOP3_NN_Acc', 0)).replace('%', '')) if pd.notna(row.get('TOP3_NN_Acc', None)) else 0,
                    'top3_nn_auc': float(str(row.get('TOP3_NN_AUC', 0)).replace('%', '')) if pd.notna(row.get('TOP3_NN_AUC', None)) else 0,
                    'class_label_nn_accuracy': float(str(row.get('CLASS_LABEL_NN_Acc', 0)).replace('%', '')) if pd.notna(row.get('CLASS_LABEL_NN_Acc', None)) else 0,
                    'class_label_nn_auc': float(str(row.get('CLASS_LABEL_NN_AUC', 0)).replace('%', '')) if pd.notna(row.get('CLASS_LABEL_NN_AUC', None)) else 0,
                    'samia_accuracy': float(str(row.get('SAMIA_Acc', 0)).replace('%', '')) if pd.notna(row.get('SAMIA_Acc', None)) else 0,
                    'samia_auc': float(str(row.get('SAMIA_AUC', 0)).replace('%', '')) if pd.notna(row.get('SAMIA_AUC', None)) else 0,
                }
                combined_data.append(entry)
    
    # Load WeMeM MIA results  
    if wemem_success:
        wemem_csv = results_dir / 'wemem' / 'wemem_mia_summary.csv'
        if wemem_csv.exists():
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
                        matching_entry = {
                            'experiment': model_name,
                            'method': row.get('Type', None),
                            'sparsity': row.get('Sparsity', None),
                        }
                        combined_data.append(matching_entry)
                    
                    def safe_float(value):
                        try:
                            return float(value)
                        except (ValueError, TypeError):
                            return 0.0
                    
                    # 컬럼명 케이스/스페이스 다를 수 있어 안전하게 접근
                    matching_entry.update({
                        'confidence_accuracy': safe_float(row.get('Confidence_Accuracy') or row.get('confidence_accuracy')),
                        'confidence_f1': safe_float(row.get('Confidence_F1') or row.get('confidence_f1')),
                        'entropy_accuracy': safe_float(row.get('Entropy_Accuracy') or row.get('entropy_accuracy')),
                        'entropy_f1': safe_float(row.get('Entropy_F1') or row.get('entropy_f1')),
                        'modified_entropy_accuracy': safe_float(row.get('Modified_entropy_Accuracy') or row.get('Modified_Entropy_Accuracy') or row.get('modified_entropy_accuracy')),
                        'modified_entropy_f1': safe_float(row.get('Modified_entropy_F1') or row.get('Modified_Entropy_F1') or row.get('modified_entropy_f1')),
                        'neural_network_accuracy': safe_float(row.get('Neural_network_Accuracy') or row.get('Neural_Network_Accuracy') or row.get('neural_network_accuracy')),
                        'neural_network_f1': safe_float(row.get('Neural_network_F1') or row.get('Neural_Network_F1') or row.get('neural_network_f1')),
                        'neural_network_auc': safe_float(row.get('Neural_network_AUC') or row.get('Neural_Network_AUC') or row.get('neural_network_auc')),
                    })
    
    # Save combined results
    if combined_data:
        combined_df = pd.DataFrame(combined_data)
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
    
    config_file = results_dir / 'config.json'
    if config_file.exists():
        with open(config_file, 'r') as f:
            results['config'] = json.load(f)
    
    log_files = list(results_dir.glob('*.log'))
    if log_files:
        results['log_file'] = str(log_files[0])
    
    experiment_summary_file = results_dir / 'experiment_summary.json'
    if experiment_summary_file.exists():
        with open(experiment_summary_file, 'r') as f:
            results['training'] = json.load(f)
    
    val_history_file = results_dir / 'validation_history.json'
    if val_history_file.exists():
        with open(val_history_file, 'r') as f:
            results['validation_history'] = json.load(f)
    
    mia_results_file = results_dir / 'mia_results.json'
    if mia_results_file.exists():
        with open(mia_results_file, 'r') as f:
            results['mia'] = json.load(f)
    
    return results

def create_training_summary_csv(all_results, experiment_prefix='dwa_experiments'):
    """Create summary CSV with all results"""
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    summary_data = []
    
    for exp_name, results in all_results.items():
        row = {'experiment': exp_name}
        
        if 'config' in results:
            config = results['config']
            row.update({
                'method': config.get('pruning', {}).get('method', 'dpf'),
                'sparsity': config.get('pruning', {}).get('sparsity', 0.0),
                'dataset': config.get('data', {}).get('dataset', 'cifar10'),
                'arch': config.get('model', {}).get('arch', 'resnet'),
                'layers': config.get('model', {}).get('layers', 20),
                'epochs': config.get('training', {}).get('epochs', 200),
                'lr': config.get('training', {}).get('lr', 0.1),
            })
        
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
        
        if 'validation_history' in results:
            val_history = results['validation_history']
            if val_history:
                best_val_acc1 = max((epoch.get('acc1', 0) for epoch in val_history if 'acc1' in epoch), default=None)
                final_val_acc1 = val_history[-1].get('acc1', None) if val_history else None
                final_val_loss = val_history[-1].get('loss', None) if val_history else None
                
                row.update({
                    'val_best_acc1': best_val_acc1,
                    'val_final_acc1': final_val_acc1,
                    'val_final_loss': final_val_loss,
                })
        
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
            epochs = config.get('training', {}).get('epochs', 50)
            file_suffix = f"{dataset}_{arch}_e{epochs}_{timestamp}"
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
        mia_results_file = Path('results/mia/comprehensive_mia_results.csv')
        if not mia_results_file.exists():
            print("⚠️ MIA results file not found, skipping wandb logging")
            return
        
        mia_df = pd.read_csv(mia_results_file)
        print("📊 Logging MIA results to Weights & Biases...")
        
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            job_type="mia_evaluation",
            name=f"MIA_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags=args.wandb_tags + ['mia', 'evaluation'],
            reinit=True
        )
        
        wandb_table = wandb.Table(dataframe=mia_df)
        wandb.log({"MIA_Results_Table": wandb_table})
        
        for _, row in mia_df.iterrows():
            experiment_name = row['experiment']
            method = row['method']
            sparsity = row.get('sparsity', 0)
            mia_metrics = {}
            for metric in ['lira_accuracy', 'lira_auc', 'shokri_nn_accuracy', 'shokri_nn_auc', 
                           'top3_nn_accuracy', 'top3_nn_auc', 'class_label_nn_accuracy', 'class_label_nn_auc',
                           'samia_accuracy', 'samia_auc',
                           'confidence_accuracy', 'confidence_f1', 'entropy_accuracy', 'entropy_f1',
                           'modified_entropy_accuracy', 'modified_entropy_f1', 'neural_network_accuracy',
                           'neural_network_f1', 'neural_network_auc']:
                if metric in mia_df.columns and pd.notna(row.get(metric, None)):
                    mia_metrics[f"mia/{metric}"] = float(row[metric])
            mia_metrics.update({
                "mia/experiment": experiment_name,
                "mia/method": method,
                "mia/sparsity": float(sparsity) if sparsity else 0.0,
            })
            wandb.log(mia_metrics)
        
        if len(mia_df) > 1:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('MIA Attack Success Rates', fontsize=16)
            if 'lira_auc' in mia_df.columns:
                ax1 = axes[0, 0]
                for m in mia_df['method'].unique():
                    md = mia_df[mia_df['method'] == m]
                    if len(md) > 1:
                        ax1.plot(md['sparsity'], md['lira_auc'], 'o-', label=m)
                    else:
                        ax1.scatter(md['sparsity'], md['lira_auc'], label=m, s=100)
                ax1.set_title('LiRA AUC'); ax1.set_xlabel('Sparsity'); ax1.set_ylabel('AUC'); ax1.legend(); ax1.grid(True, alpha=0.3)
            if 'neural_network_auc' in mia_df.columns:
                ax2 = axes[0, 1]
                for m in mia_df['method'].unique():
                    md = mia_df[mia_df['method'] == m]
                    if len(md) > 1:
                        ax2.plot(md['sparsity'], md['neural_network_auc'], 'o-', label=m)
                    else:
                        ax2.scatter(md['sparsity'], md['neural_network_auc'], label=m, s=100)
                ax2.set_title('Neural Network AUC (WeMeM)'); ax2.set_xlabel('Sparsity'); ax2.set_ylabel('AUC'); ax2.legend(); ax2.grid(True, alpha=0.3)
            if 'confidence_accuracy' in mia_df.columns:
                ax3 = axes[1, 0]
                for m in mia_df['method'].unique():
                    md = mia_df[mia_df['method'] == m]
                    if len(md) > 1:
                        ax3.plot(md['sparsity'], md['confidence_accuracy'], 'o-', label=m)
                    else:
                        ax3.scatter(md['sparsity'], md['confidence_accuracy'], label=m, s=100)
                ax3.set_title('Confidence Attack Accuracy'); ax3.set_xlabel('Sparsity'); ax3.set_ylabel('Accuracy'); ax3.legend(); ax3.grid(True, alpha=0.3)
            if 'samia_auc' in mia_df.columns:
                ax4 = axes[1, 1]
                for m in mia_df['method'].unique():
                    md = mia_df[mia_df['method'] == m]
                    if len(md) > 1:
                        ax4.plot(md['sparsity'], md['samia_auc'], 'o-', label=m)
                    else:
                        ax4.scatter(md['sparsity'], md['samia_auc'], label=m, s=100)
                ax4.set_title('SAMIA AUC'); ax4.set_xlabel('Sparsity'); ax4.set_ylabel('AUC'); ax4.legend(); ax4.grid(True, alpha=0.3)
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
    for old_folder, method, sparsity in folders_to_move:
        new_path = runs_dir / method / f'sparsity_{sparsity}'
        new_path.parent.mkdir(parents=True, exist_ok=True)
        if new_path.exists():
            print(f"Target path {new_path} already exists, skipping {old_folder}")
            continue
        print(f"Moving {old_folder} -> {new_path}")
        shutil.move(str(old_folder), str(new_path))

def main():
    parser = argparse.ArgumentParser(description='DWA (Dynamic Weight Adjustment) Experiments')
    # DWA 실험 모드 - 3가지 중 선택
    parser.add_argument('--dwa-modes', nargs='+', 
                       default=['reactivate_only', 'kill_active_plain_dead', 'kill_and_reactivate'],
                       choices=['reactivate_only', 'kill_active_plain_dead', 'kill_and_reactivate'],
                       help='DWA modes to experiment with')
    # DWA 하이퍼파라미터
    parser.add_argument('--dwa-alphas', nargs='+', type=float, default=[1.0],
                       help='Alpha values for reactivation strength')
    parser.add_argument('--dwa-betas', nargs='+', type=float, default=[1.0], 
                       help='Beta values for kill strength')
    parser.add_argument('--dwa-threshold-percentile', type=int, default=50,
                       help='Percentile for threshold calculation')
    # 기본 실험 파라미터
    parser.add_argument('--sparsities', nargs='+', type=float, default=[0.5, 0.8, 0.9],
                       help='Sparsity levels for pruned methods')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'],
                       help='Dataset to use')
    parser.add_argument('--arch', default='resnet', choices=['resnet', 'wideresnet'],
                       help='Architecture to use')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (shorter for DWA experiments)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip experiments if results already exist')
    parser.add_argument('--reorganize-only', action='store_true',
                       help='Only reorganize existing runs folder and exit')
    # Wandb arguments
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', default='dwa-experiments',
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
    
    print(f"🚀 Starting DWA experiments")
    print(f"DWA Modes: {args.dwa_modes}")
    print(f"Alphas: {args.dwa_alphas}, Betas: {args.dwa_betas}")
    print(f"Sparsities: {args.sparsities}")
    print(f"Dataset: {args.dataset}, Architecture: {args.arch}")
    
    # DWA 실험 - 각 모드별로 실행
    for dwa_mode in args.dwa_modes:
        for alpha in args.dwa_alphas:
            for beta in args.dwa_betas:
                for sparsity in args.sparsities:
                    exp_name = f"dwa_{dwa_mode}"
                    if alpha != 1.0:
                        exp_name += f"_alpha{alpha}"
                    if beta != 1.0:
                        exp_name += f"_beta{beta}"
                    exp_name += f"_sparsity_{sparsity}_{args.dataset}_{args.arch}"
                    
                    print(f"\n{'='*50}")
                    print(f"Running experiment: {exp_name}")
                    print(f"DWA Mode: {dwa_mode}, Alpha: {alpha}, Beta: {beta}, Sparsity: {sparsity}")
                    print(f"{'='*50}")
                    
                    # Create save path (DWA 실험용 경로 구조)
                    save_path = Path('./runs/dwa') / dwa_mode / f'sparsity_{sparsity}' / args.dataset
                    if alpha != 1.0 or beta != 1.0:
                        save_path = save_path / f'alpha{alpha}_beta{beta}'
                    
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
                        # Pruning 설정 (DWA는 dpf 기반으로 작동)
                        'prune': True,
                        'prune-method': 'dpf',
                        'sparsity': sparsity,
                        'freeze-epoch': -1,  # DWA는 프리즈하지 않음
                        # DWA 파라미터 전달
                        'dwa-mode': dwa_mode,
                        'dwa-alpha': alpha,
                        'dwa-beta': beta,
                        'dwa-threshold-percentile': args.dwa_threshold_percentile,
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
                            config_params['wandb_tags'] = ','.join(args.wandb_tags + ['dwa', dwa_mode, args.dataset, args.arch])
                    
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
        # DWA 실험 정보를 포함한 파일명으로 저장
        dwa_info = f"dwa_{'-'.join(args.dwa_modes)}_alpha{'-'.join(map(str, args.dwa_alphas))}_beta{'-'.join(map(str, args.dwa_betas))}"
        summary_df = create_training_summary_csv(all_results, experiment_prefix=dwa_info)
        print(f"Completed {len(all_results)} experiments")
        print(f"Training Summary:\n{summary_df.to_string()}")
    
    # (선택) MIA 평가를 돌리고 싶으면 아래 주석 해제
    # runs_dir = Path('./runs')
    # success, output = run_comprehensive_mia_evaluation(runs_dir)
    # if success:
    #     print("✅ MIA evaluation completed successfully!")
    #     print("📊 MIA results saved in: results/mia/")
    #     print("📁 Results: results/mia/comprehensive_mia_results.csv")
    #     if args.wandb:
    #         log_mia_results_to_wandb(args)
    # else:
    #     print(f"❌ MIA evaluation failed: {output}")
    #     failed_experiments.append(('mia_evaluation', 'mia', output))
    
    if failed_experiments:
        print(f"\n{'='*20} FAILED EXPERIMENTS {'='*20}")
        for exp_name, stage, error in failed_experiments:
            print(f"FAILED: {exp_name} ({stage}) - {error}")
    
    print("\nAll experiments completed!")

if __name__ == '__main__':
    main()