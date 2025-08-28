#!/usr/bin/env python3
"""
Comprehensive Result Visualization: Dense vs Static vs DPF Analysis
Creates multiple comparative plots and analysis charts
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import ast
import argparse

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def parse_config_column(config_str):
    """Parse stringified config dictionary"""
    try:
        return ast.literal_eval(config_str)
    except:
        return {}

def load_experiment_data(csv_path):
    """Load and parse experiment results"""
    df = pd.read_csv(csv_path)
    
    # Parse pruning config to extract method and sparsity
    df['pruning_config'] = df['pruning'].apply(parse_config_column)
    df['method'] = df['pruning_config'].apply(lambda x: 'Dense' if not x.get('enabled', False) else x.get('method', 'unknown').upper())
    df['sparsity'] = df['pruning_config'].apply(lambda x: x.get('sparsity', 0.0) if x.get('enabled', False) else 0.0)
    df['sparsity_percent'] = df['sparsity'] * 100
    
    # Clean up method names
    df['method'] = df['method'].replace({'STATIC': 'Static', 'DPF': 'DPF', 'DENSE': 'Dense'})
    
    return df

def create_accuracy_comparison(df, save_dir):
    """Create accuracy vs sparsity comparison plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Best Accuracy vs Sparsity
    methods = ['Dense', 'Static', 'DPF']
    colors = {'Dense': '#2E8B57', 'Static': '#DC143C', 'DPF': '#4169E1'}
    markers = {'Dense': 'o', 'Static': 's', 'DPF': '^'}
    
    for method in methods:
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            ax1.scatter(method_data['sparsity_percent'], method_data['best_acc1'], 
                       c=colors[method], marker=markers[method], s=100, alpha=0.8, 
                       label=method, edgecolors='black', linewidth=1)
            
            # Add trend line for pruned methods
            if method != 'Dense' and len(method_data) > 1:
                sorted_data = method_data.sort_values('sparsity_percent')
                ax1.plot(sorted_data['sparsity_percent'], sorted_data['best_acc1'], 
                        color=colors[method], linestyle='--', alpha=0.6, linewidth=2)
    
    ax1.set_xlabel('Sparsity (%)', fontsize=12)
    ax1.set_ylabel('Best Accuracy (%)', fontsize=12)
    ax1.set_title('Model Accuracy vs Sparsity Level', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(80, 95)
    
    # Add annotations for key points
    dense_acc = df[df['method'] == 'Dense']['best_acc1'].mean()
    ax1.axhline(y=dense_acc, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax1.text(50, dense_acc + 0.5, f'Dense Baseline: {dense_acc:.1f}%', 
             fontsize=10, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
    
    # Plot 2: Performance Degradation
    dense_baseline = dense_acc
    
    for method in ['Static', 'DPF']:
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            degradation = dense_baseline - method_data['best_acc1']
            ax2.scatter(method_data['sparsity_percent'], degradation, 
                       c=colors[method], marker=markers[method], s=100, alpha=0.8,
                       label=method, edgecolors='black', linewidth=1)
            
            # Trend line
            if len(method_data) > 1:
                sorted_data = method_data.sort_values('sparsity_percent')
                sorted_degradation = dense_baseline - sorted_data['best_acc1']
                ax2.plot(sorted_data['sparsity_percent'], sorted_degradation, 
                        color=colors[method], linestyle='--', alpha=0.6, linewidth=2)
    
    ax2.set_xlabel('Sparsity (%)', fontsize=12)
    ax2.set_ylabel('Accuracy Drop from Dense (%)', fontsize=12)
    ax2.set_title('Performance Degradation Analysis', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, None)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_efficiency_analysis(df, save_dir):
    """Create efficiency analysis plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    methods = ['Dense', 'Static', 'DPF']
    colors = {'Dense': '#2E8B57', 'Static': '#DC143C', 'DPF': '#4169E1'}
    
    # 1. Training Time vs Sparsity
    for method in methods:
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            ax1.scatter(method_data['sparsity_percent'], method_data['total_duration_hours'], 
                       c=colors[method], label=method, s=80, alpha=0.8)
    
    ax1.set_xlabel('Sparsity (%)')
    ax1.set_ylabel('Training Time (hours)')
    ax1.set_title('Training Efficiency', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy per Training Hour
    df['acc_per_hour'] = df['best_acc1'] / df['total_duration_hours']
    
    for method in methods:
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            ax2.scatter(method_data['sparsity_percent'], method_data['acc_per_hour'], 
                       c=colors[method], label=method, s=80, alpha=0.8)
    
    ax2.set_xlabel('Sparsity (%)')
    ax2.set_ylabel('Accuracy per Training Hour')
    ax2.set_title('Training Efficiency Ratio', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Loss Comparison
    for method in methods:
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            ax3.scatter(method_data['sparsity_percent'], method_data['best_loss'], 
                       c=colors[method], label=method, s=80, alpha=0.8)
    
    ax3.set_xlabel('Sparsity (%)')
    ax3.set_ylabel('Best Loss')
    ax3.set_title('Training Loss Analysis', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Method Summary Statistics
    summary_stats = []
    for method in methods:
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            stats = {
                'Method': method,
                'Avg_Accuracy': method_data['best_acc1'].mean(),
                'Std_Accuracy': method_data['best_acc1'].std(),
                'Avg_Training_Time': method_data['total_duration_hours'].mean(),
                'Count': len(method_data)
            }
            summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Bar plot for average accuracy
    bars = ax4.bar(summary_df['Method'], summary_df['Avg_Accuracy'], 
                   color=[colors[m] for m in summary_df['Method']], alpha=0.8)
    
    # Add error bars
    ax4.errorbar(summary_df['Method'], summary_df['Avg_Accuracy'], 
                yerr=summary_df['Std_Accuracy'], fmt='none', color='black', capsize=5)
    
    # Add value labels
    for bar, acc in zip(bars, summary_df['Avg_Accuracy']):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax4.set_ylabel('Average Accuracy (%)')
    ax4.set_title('Method Performance Summary', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(80, 95)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'efficiency_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_sparsity_analysis(df, save_dir):
    """Create detailed sparsity level analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Filter only pruned methods
    pruned_df = df[df['method'].isin(['Static', 'DPF'])]
    
    # 1. Side-by-side comparison at each sparsity level
    sparsity_levels = sorted(pruned_df['sparsity_percent'].unique())
    x = np.arange(len(sparsity_levels))
    width = 0.35
    
    static_accs = []
    dpf_accs = []
    
    for sparsity in sparsity_levels:
        static_data = pruned_df[(pruned_df['method'] == 'Static') & (pruned_df['sparsity_percent'] == sparsity)]
        dpf_data = pruned_df[(pruned_df['method'] == 'DPF') & (pruned_df['sparsity_percent'] == sparsity)]
        
        static_accs.append(static_data['best_acc1'].mean() if len(static_data) > 0 else 0)
        dpf_accs.append(dpf_data['best_acc1'].mean() if len(dpf_data) > 0 else 0)
    
    bars1 = ax1.bar(x - width/2, static_accs, width, label='Static', color='#DC143C', alpha=0.8)
    bars2 = ax1.bar(x + width/2, dpf_accs, width, label='DPF', color='#4169E1', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Sparsity Level (%)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Static vs DPF Comparison by Sparsity', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{int(s)}%' for s in sparsity_levels])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(80, 95)
    
    # 2. DPF Advantage Analysis
    dpf_advantages = []
    for i, sparsity in enumerate(sparsity_levels):
        if static_accs[i] > 0 and dpf_accs[i] > 0:
            advantage = dpf_accs[i] - static_accs[i]
            dpf_advantages.append(advantage)
        else:
            dpf_advantages.append(0)
    
    colors_advantage = ['green' if x > 0 else 'red' for x in dpf_advantages]
    bars = ax2.bar(range(len(sparsity_levels)), dpf_advantages, color=colors_advantage, alpha=0.7)
    
    # Add value labels
    for bar, adv in zip(bars, dpf_advantages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height > 0 else -0.05),
                f'{adv:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Sparsity Level (%)')
    ax2.set_ylabel('DPF Advantage over Static (%)')
    ax2.set_title('DPF vs Static Performance Gap', fontweight='bold')
    ax2.set_xticks(range(len(sparsity_levels)))
    ax2.set_xticklabels([f'{int(s)}%' for s in sparsity_levels])
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sparsity_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_comprehensive_dashboard(df, save_dir):
    """Create a comprehensive dashboard with all key metrics"""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    methods = ['Dense', 'Static', 'DPF']
    colors = {'Dense': '#2E8B57', 'Static': '#DC143C', 'DPF': '#4169E1'}
    
    # Main accuracy plot (spans 2 columns)
    ax_main = fig.add_subplot(gs[0, :2])
    for method in methods:
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            ax_main.scatter(method_data['sparsity_percent'], method_data['best_acc1'], 
                          c=colors[method], s=120, alpha=0.8, label=method, edgecolors='black')
            if method != 'Dense' and len(method_data) > 1:
                sorted_data = method_data.sort_values('sparsity_percent')
                ax_main.plot(sorted_data['sparsity_percent'], sorted_data['best_acc1'], 
                           color=colors[method], linestyle='--', alpha=0.6, linewidth=2)
    
    ax_main.set_xlabel('Sparsity (%)', fontsize=12)
    ax_main.set_ylabel('Accuracy (%)', fontsize=12)
    ax_main.set_title('Dense vs Static vs DPF: Accuracy Comparison', fontsize=14, fontweight='bold')
    ax_main.legend(fontsize=12)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_ylim(80, 95)
    
    # Summary statistics table
    ax_table = fig.add_subplot(gs[0, 2:])
    ax_table.axis('off')
    
    table_data = []
    for method in methods:
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            table_data.append([
                method,
                f"{method_data['best_acc1'].mean():.1f}±{method_data['best_acc1'].std():.1f}",
                f"{method_data['total_duration_hours'].mean():.2f}h",
                len(method_data)
            ])
    
    table = ax_table.table(cellText=table_data,
                          colLabels=['Method', 'Accuracy (%)', 'Avg Training Time', 'Models'],
                          cellLoc='center',
                          loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style table
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
        else:
            method = table_data[i-1][0]
            cell.set_facecolor(colors.get(method, 'lightgray'))
            cell.set_alpha(0.7)
    
    ax_table.set_title('Performance Summary', fontsize=14, fontweight='bold')
    
    # Training time analysis
    ax_time = fig.add_subplot(gs[1, 0])
    method_times = []
    method_names = []
    method_colors = []
    
    for method in methods:
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            method_times.extend(method_data['total_duration_hours'])
            method_names.extend([method] * len(method_data))
            method_colors.extend([colors[method]] * len(method_data))
    
    bp = ax_time.boxplot([df[df['method'] == m]['total_duration_hours'] for m in methods if len(df[df['method'] == m]) > 0],
                        labels=[m for m in methods if len(df[df['method'] == m]) > 0],
                        patch_artist=True)
    
    for patch, method in zip(bp['boxes'], [m for m in methods if len(df[df['method'] == m]) > 0]):
        patch.set_facecolor(colors[method])
        patch.set_alpha(0.7)
    
    ax_time.set_ylabel('Training Time (hours)')
    ax_time.set_title('Training Time Distribution', fontweight='bold')
    ax_time.grid(True, alpha=0.3)
    
    # Loss analysis
    ax_loss = fig.add_subplot(gs[1, 1])
    for method in methods:
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            ax_loss.scatter(method_data['sparsity_percent'], method_data['best_loss'], 
                          c=colors[method], label=method, s=80, alpha=0.8)
    
    ax_loss.set_xlabel('Sparsity (%)')
    ax_loss.set_ylabel('Best Loss')
    ax_loss.set_title('Loss vs Sparsity', fontweight='bold')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    
    # Efficiency ratio
    ax_eff = fig.add_subplot(gs[1, 2])
    df['efficiency'] = df['best_acc1'] / df['total_duration_hours']
    
    for method in methods:
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            ax_eff.bar([method], [method_data['efficiency'].mean()], 
                      color=colors[method], alpha=0.8)
    
    ax_eff.set_ylabel('Accuracy per Hour')
    ax_eff.set_title('Training Efficiency', fontweight='bold')
    ax_eff.grid(True, alpha=0.3)
    
    # Parameter efficiency (approximate)
    ax_params = fig.add_subplot(gs[1, 3])
    
    # Estimate parameter reduction
    sparsity_data = []
    param_efficiency = []
    methods_sparse = []
    
    for _, row in df.iterrows():
        if row['method'] != 'Dense':
            remaining_params = 1 - row['sparsity']
            efficiency = row['best_acc1'] / remaining_params if remaining_params > 0 else 0
            sparsity_data.append(row['sparsity_percent'])
            param_efficiency.append(efficiency)
            methods_sparse.append(row['method'])
    
    scatter = ax_params.scatter(sparsity_data, param_efficiency, 
                              c=[colors[m] for m in methods_sparse], 
                              s=80, alpha=0.8, edgecolors='black')
    
    ax_params.set_xlabel('Sparsity (%)')
    ax_params.set_ylabel('Accuracy per Active Parameter')
    ax_params.set_title('Parameter Efficiency', fontweight='bold')
    ax_params.grid(True, alpha=0.3)
    
    # Bottom row: Detailed comparisons
    # DPF vs Static advantage
    ax_adv = fig.add_subplot(gs[2, :2])
    
    sparsity_levels = sorted(df[df['method'].isin(['Static', 'DPF'])]['sparsity_percent'].unique())
    dpf_advantages = []
    
    for sparsity in sparsity_levels:
        static_data = df[(df['method'] == 'Static') & (df['sparsity_percent'] == sparsity)]
        dpf_data = df[(df['method'] == 'DPF') & (df['sparsity_percent'] == sparsity)]
        
        static_acc = static_data['best_acc1'].mean() if len(static_data) > 0 else 0
        dpf_acc = dpf_data['best_acc1'].mean() if len(dpf_data) > 0 else 0
        
        if static_acc > 0 and dpf_acc > 0:
            dpf_advantages.append(dpf_acc - static_acc)
        else:
            dpf_advantages.append(0)
    
    colors_adv = ['green' if x > 0 else 'red' for x in dpf_advantages]
    bars = ax_adv.bar(range(len(sparsity_levels)), dpf_advantages, color=colors_adv, alpha=0.7)
    
    for bar, adv in zip(bars, dpf_advantages):
        height = bar.get_height()
        ax_adv.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height > 0 else -0.05),
                   f'{adv:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                   fontsize=10, fontweight='bold')
    
    ax_adv.set_xlabel('Sparsity Level (%)')
    ax_adv.set_ylabel('DPF Advantage (%)')
    ax_adv.set_title('DPF Performance Advantage over Static', fontweight='bold')
    ax_adv.set_xticks(range(len(sparsity_levels)))
    ax_adv.set_xticklabels([f'{int(s)}%' for s in sparsity_levels])
    ax_adv.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax_adv.grid(True, alpha=0.3)
    
    # Key insights text
    ax_insights = fig.add_subplot(gs[2, 2:])
    ax_insights.axis('off')
    
    # Calculate key metrics
    dense_acc = df[df['method'] == 'Dense']['best_acc1'].mean()
    best_static = df[df['method'] == 'Static']['best_acc1'].max()
    best_dpf = df[df['method'] == 'DPF']['best_acc1'].max()
    worst_static = df[df['method'] == 'Static']['best_acc1'].min()
    worst_dpf = df[df['method'] == 'DPF']['best_acc1'].min()
    
    insights_text = f"""
KEY FINDINGS:

• Dense Baseline: {dense_acc:.1f}% accuracy
• Best Static: {best_static:.1f}% accuracy
• Best DPF: {best_dpf:.1f}% accuracy

• Static degradation: {dense_acc - best_static:.1f}% → {dense_acc - worst_static:.1f}%
• DPF degradation: {dense_acc - best_dpf:.1f}% → {dense_acc - worst_dpf:.1f}%

• DPF shows {'better' if best_dpf > best_static else 'similar'} performance
• {'DPF maintains accuracy better' if worst_dpf > worst_static else 'Similar degradation pattern'} at high sparsity
    
RECOMMENDATIONS:
• Use DPF for better accuracy-sparsity trade-off
• Static acceptable for moderate sparsity (≤80%)
• Consider training time vs accuracy trade-offs
    """
    
    ax_insights.text(0.05, 0.95, insights_text, transform=ax_insights.transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Dense vs Static vs DPF: Comprehensive Analysis Dashboard', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(os.path.join(save_dir, 'comprehensive_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_report(df, save_dir):
    """Generate a detailed text summary report"""
    report_path = os.path.join(save_dir, 'analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DENSE vs STATIC vs DPF PRUNING: COMPREHENSIVE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL EXPERIMENT STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total experiments: {len(df)}\n")
        f.write(f"Methods tested: {', '.join(df['method'].unique())}\n")
        f.write(f"Sparsity levels: {', '.join([f'{int(s)}%' for s in sorted(df['sparsity_percent'].unique())])}\n")
        f.write(f"Total training time: {df['total_duration_hours'].sum():.2f} hours\n\n")
        
        # Method-wise analysis
        f.write("METHOD-WISE PERFORMANCE ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        
        for method in ['Dense', 'Static', 'DPF']:
            method_data = df[df['method'] == method]
            if len(method_data) > 0:
                f.write(f"\n{method} METHOD:\n")
                f.write(f"  Models trained: {len(method_data)}\n")
                f.write(f"  Average accuracy: {method_data['best_acc1'].mean():.2f}% ± {method_data['best_acc1'].std():.2f}%\n")
                f.write(f"  Best accuracy: {method_data['best_acc1'].max():.2f}%\n")
                f.write(f"  Worst accuracy: {method_data['best_acc1'].min():.2f}%\n")
                f.write(f"  Average training time: {method_data['total_duration_hours'].mean():.2f} hours\n")
                f.write(f"  Average loss: {method_data['best_loss'].mean():.4f}\n")
                
                if method != 'Dense':
                    f.write(f"  Sparsity range: {method_data['sparsity_percent'].min():.0f}% - {method_data['sparsity_percent'].max():.0f}%\n")
        
        # Comparative analysis
        f.write("\n\nCOMPARATIVE ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        
        dense_acc = df[df['method'] == 'Dense']['best_acc1'].mean()
        
        f.write(f"Dense baseline accuracy: {dense_acc:.2f}%\n\n")
        
        # Sparsity level comparison
        sparsity_levels = sorted(df[df['method'].isin(['Static', 'DPF'])]['sparsity_percent'].unique())
        
        f.write("SPARSITY LEVEL COMPARISON:\n")
        for sparsity in sparsity_levels:
            f.write(f"\nAt {int(sparsity)}% sparsity:\n")
            
            static_data = df[(df['method'] == 'Static') & (df['sparsity_percent'] == sparsity)]
            dpf_data = df[(df['method'] == 'DPF') & (df['sparsity_percent'] == sparsity)]
            
            if len(static_data) > 0:
                static_acc = static_data['best_acc1'].mean()
                static_drop = dense_acc - static_acc
                f.write(f"  Static: {static_acc:.2f}% (drop: {static_drop:.2f}%)\n")
            
            if len(dpf_data) > 0:
                dpf_acc = dpf_data['best_acc1'].mean()
                dpf_drop = dense_acc - dpf_acc
                f.write(f"  DPF: {dpf_acc:.2f}% (drop: {dpf_drop:.2f}%)\n")
                
                if len(static_data) > 0:
                    advantage = dpf_acc - static_acc
                    f.write(f"  DPF advantage: {advantage:+.2f}%\n")
        
        # Key insights
        f.write("\n\nKEY INSIGHTS:\n")
        f.write("-" * 40 + "\n")
        
        static_data = df[df['method'] == 'Static']
        dpf_data = df[df['method'] == 'DPF']
        
        if len(static_data) > 0 and len(dpf_data) > 0:
            f.write(f"1. DPF achieves {dpf_data['best_acc1'].mean() - static_data['best_acc1'].mean():+.2f}% better average accuracy than Static\n")
            f.write(f"2. Training time difference: DPF takes {dpf_data['total_duration_hours'].mean() - static_data['total_duration_hours'].mean():+.3f} hours more on average\n")
            
            # Find best performing method at each sparsity
            best_methods = []
            for sparsity in sparsity_levels:
                static_acc = df[(df['method'] == 'Static') & (df['sparsity_percent'] == sparsity)]['best_acc1'].mean()
                dpf_acc = df[(df['method'] == 'DPF') & (df['sparsity_percent'] == sparsity)]['best_acc1'].mean()
                
                if dpf_acc > static_acc:
                    best_methods.append('DPF')
                elif static_acc > dpf_acc:
                    best_methods.append('Static')
                else:
                    best_methods.append('Tie')
            
            dpf_wins = best_methods.count('DPF')
            static_wins = best_methods.count('Static')
            
            f.write(f"3. DPF outperforms Static at {dpf_wins}/{len(sparsity_levels)} sparsity levels\n")
            f.write(f"4. Maximum accuracy drop from dense: Static {dense_acc - static_data['best_acc1'].min():.2f}%, DPF {dense_acc - dpf_data['best_acc1'].min():.2f}%\n")
        
        f.write(f"\n5. Dense model serves as strong baseline with {dense_acc:.2f}% accuracy\n")
        f.write(f"6. Pruning enables significant parameter reduction with controlled accuracy loss\n")
        
        f.write("\n\nRECOMMENDations:\n")
        f.write("-" * 40 + "\n")
        f.write("1. Use DPF for applications requiring best accuracy-sparsity trade-off\n")
        f.write("2. Consider Static pruning for resource-constrained training environments\n")
        f.write("3. Dense models recommended when accuracy is paramount and resources allow\n")
        f.write("4. For sparsity >80%, carefully evaluate accuracy degradation\n")
        f.write("5. Consider training time constraints in method selection\n")

def main():
    parser = argparse.ArgumentParser(description='Visualize Dense vs Static vs DPF Results')
    parser.add_argument('--csv-path', default='./runs/final_report/experiments_comparison.csv',
                       help='Path to experiment results CSV')
    parser.add_argument('--output-dir', default='./results/visualization',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🎨 Dense vs Static vs DPF Visualization")
    print("=" * 50)
    
    # Load data
    print("📊 Loading experiment data...")
    df = load_experiment_data(args.csv_path)
    print(f"   Found {len(df)} experiments")
    print(f"   Methods: {', '.join(df['method'].unique())}")
    print(f"   Sparsity levels: {', '.join([f'{int(s)}%' for s in sorted(df['sparsity_percent'].unique())])}")
    
    # Generate plots
    print("\n🖼️ Generating visualizations...")
    
    print("   1. Accuracy comparison plots...")
    create_accuracy_comparison(df, args.output_dir)
    
    print("   2. Efficiency analysis...")
    create_efficiency_analysis(df, args.output_dir)
    
    print("   3. Sparsity analysis...")
    create_sparsity_analysis(df, args.output_dir)
    
    print("   4. Comprehensive dashboard...")
    create_comprehensive_dashboard(df, args.output_dir)
    
    # Generate report
    print("   5. Summary report...")
    generate_summary_report(df, args.output_dir)
    
    print(f"\n✅ Visualization complete!")
    print(f"📁 Results saved to: {args.output_dir}")
    print("\n📋 Generated files:")
    print("   - accuracy_comparison.png: Main accuracy vs sparsity plots")
    print("   - efficiency_analysis.png: Training efficiency and loss analysis")
    print("   - sparsity_analysis.png: Detailed sparsity level comparisons")
    print("   - comprehensive_dashboard.png: Complete analysis dashboard")
    print("   - analysis_report.txt: Detailed text summary")

if __name__ == '__main__':
    main()