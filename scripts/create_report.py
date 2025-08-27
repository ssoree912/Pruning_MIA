#!/usr/bin/env python3
"""
Create comprehensive comparison report and visualizations
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import ResultsCollector

def parse_args():
    parser = argparse.ArgumentParser(description='Create Comprehensive Report')
    parser.add_argument('--results-dir', required=True, type=str,
                       help='Directory containing experiment results')
    parser.add_argument('--output-dir', default='./report', type=str,
                       help='Output directory for report')
    parser.add_argument('--title', default='Dense/Static/DPF Comparison with MIA', type=str,
                       help='Report title')
    
    return parser.parse_args()

class ComprehensiveReporter:
    """Create comprehensive comparison report"""
    
    def __init__(self, results_dir: str, output_dir: str):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_experiment_data(self) -> pd.DataFrame:
        """Load all experiment data"""
        
        # Use ResultsCollector to get experiment summaries
        collector = ResultsCollector(str(self.results_dir))
        summaries = collector.collect_experiment_summaries()
        
        if not summaries:
            print("No experiment summaries found")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = collector.create_comparison_table(summaries)
        
        # Parse experiment names to extract model info
        model_info = []
        for _, row in df.iterrows():
            name = row.get('experiment_name', '')
            
            if 'dense' in name:
                model_type = 'Dense'
                sparsity = 0.0
            elif 'static' in name:
                model_type = 'Static'
                # Extract sparsity from name
                sparsity = self._extract_sparsity(name)
            elif 'dpf' in name:
                model_type = 'DPF'
                sparsity = self._extract_sparsity(name)
            else:
                model_type = 'Unknown'
                sparsity = 0.0
            
            # Extract seed
            seed = self._extract_seed(name)
            
            model_info.append({
                'model_type': model_type,
                'sparsity': sparsity,
                'seed': seed
            })
        
        # Add parsed info to DataFrame
        info_df = pd.DataFrame(model_info)
        df = pd.concat([df, info_df], axis=1)
        
        return df
    
    def _extract_sparsity(self, name: str) -> float:
        """Extract sparsity from experiment name"""
        import re
        match = re.search(r'sparsity([\d.]+)', name)
        return float(match.group(1)) if match else 0.0
    
    def _extract_seed(self, name: str) -> int:
        """Extract seed from experiment name"""
        import re
        match = re.search(r'seed(\d+)', name)
        return int(match.group(1)) if match else 42
    
    def load_mia_results(self) -> Optional[pd.DataFrame]:
        """Load MIA results if available"""
        
        mia_dir = self.results_dir / 'mia_results'
        if not mia_dir.exists():
            print("No MIA results found")
            return None
        
        # Look for comprehensive summary
        summary_file = mia_dir / 'comprehensive_summary.json'
        if not summary_file.exists():
            print("No comprehensive MIA summary found")
            return None
        
        with open(summary_file, 'r') as f:
            mia_data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(mia_data)
        
        return df
    
    def create_accuracy_plots(self, df: pd.DataFrame):
        """Create accuracy comparison plots"""
        
        if df.empty:
            return
        
        # Group by model type and sparsity
        grouped = df.groupby(['model_type', 'sparsity'])
        
        # Calculate statistics
        stats = grouped.agg({
            'best_acc1': ['mean', 'std', 'count'],
            'final_acc1': ['mean', 'std', 'count']
        }).round(4)
        
        # Flatten column names
        stats.columns = ['_'.join(col).strip() for col in stats.columns]
        stats = stats.reset_index()
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Best accuracy vs sparsity
        for model_type in ['Dense', 'Static', 'DPF']:
            data = stats[stats['model_type'] == model_type]
            if not data.empty:
                ax1.errorbar(data['sparsity'], data['best_acc1_mean'], 
                           yerr=data['best_acc1_std'], marker='o', label=model_type,
                           capsize=5, linewidth=2, markersize=8)
        
        ax1.set_xlabel('Sparsity')
        ax1.set_ylabel('Best Accuracy (%)')
        ax1.set_title('Best Accuracy vs Sparsity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Final accuracy vs sparsity  
        for model_type in ['Dense', 'Static', 'DPF']:
            data = stats[stats['model_type'] == model_type]
            if not data.empty:
                ax2.errorbar(data['sparsity'], data['final_acc1_mean'],
                           yerr=data['final_acc1_std'], marker='s', label=model_type,
                           capsize=5, linewidth=2, markersize=8)
        
        ax2.set_xlabel('Sparsity')
        ax2.set_ylabel('Final Accuracy (%)')
        ax2.set_title('Final Accuracy vs Sparsity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Accuracy distribution by model type
        if 'best_acc1' in df.columns:
            model_types = ['Dense', 'Static', 'DPF']
            accuracy_data = [df[df['model_type'] == mt]['best_acc1'].values 
                           for mt in model_types if mt in df['model_type'].values]
            labels = [mt for mt in model_types if mt in df['model_type'].values]
            
            ax3.boxplot(accuracy_data, labels=labels)
            ax3.set_ylabel('Best Accuracy (%)')
            ax3.set_title('Accuracy Distribution by Model Type')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Training time vs sparsity
        if 'total_duration_hours' in df.columns:
            for model_type in ['Dense', 'Static', 'DPF']:
                data = df[df['model_type'] == model_type]
                if not data.empty:
                    grouped_time = data.groupby('sparsity')['total_duration_hours'].mean()
                    ax4.plot(grouped_time.index, grouped_time.values, 
                           marker='o', label=model_type, linewidth=2, markersize=8)
        
            ax4.set_xlabel('Sparsity')
            ax4.set_ylabel('Training Time (hours)')
            ax4.set_title('Training Time vs Sparsity')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Accuracy plots saved to {self.output_dir / 'accuracy_comparison.png'}")
        
        return stats
    
    def create_mia_plots(self, mia_df: pd.DataFrame):
        """Create MIA comparison plots"""
        
        if mia_df.empty:
            return
        
        # Parse model names
        model_info = []
        for _, row in mia_df.iterrows():
            name = row.get('model_name', '')
            
            if 'dense' in name:
                model_type = 'Dense'
                sparsity = 0.0
            elif 'static' in name:
                model_type = 'Static'
                sparsity = self._extract_sparsity(name)
            elif 'dpf' in name:
                model_type = 'DPF'
                sparsity = self._extract_sparsity(name)
            else:
                model_type = 'Unknown'
                sparsity = 0.0
            
            model_info.append({
                'model_type': model_type,
                'sparsity': sparsity
            })
        
        info_df = pd.DataFrame(model_info)
        mia_df = pd.concat([mia_df, info_df], axis=1)
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: AUC vs Sparsity
        for model_type in ['Dense', 'Static', 'DPF']:
            data = mia_df[mia_df['model_type'] == model_type]
            if not data.empty:
                ax1.plot(data['sparsity'], data['auc'], marker='o', 
                        label=model_type, linewidth=2, markersize=8)
        
        ax1.set_xlabel('Sparsity')
        ax1.set_ylabel('AUC')
        ax1.set_title('MIA Attack AUC vs Sparsity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Random')
        
        # Plot 2: TPR@1% FPR vs Sparsity
        if 'TPR@0.01' in mia_df.columns:
            for model_type in ['Dense', 'Static', 'DPF']:
                data = mia_df[mia_df['model_type'] == model_type]
                if not data.empty:
                    ax2.plot(data['sparsity'], data['TPR@0.01'], marker='s',
                            label=model_type, linewidth=2, markersize=8)
            
            ax2.set_xlabel('Sparsity')
            ax2.set_ylabel('TPR @ 1% FPR')
            ax2.set_title('MIA Attack TPR@1% vs Sparsity')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: AUC distribution by model type
        model_types = ['Dense', 'Static', 'DPF']
        auc_data = [mia_df[mia_df['model_type'] == mt]['auc'].values 
                   for mt in model_types if mt in mia_df['model_type'].values]
        labels = [mt for mt in model_types if mt in mia_df['model_type'].values]
        
        if auc_data:
            ax3.boxplot(auc_data, labels=labels)
            ax3.set_ylabel('AUC')
            ax3.set_title('MIA AUC Distribution by Model Type')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        
        # Plot 4: Multiple TPR metrics
        tpr_columns = [col for col in mia_df.columns if col.startswith('TPR@')]
        if tpr_columns:
            x_pos = np.arange(len(model_types))
            width = 0.2
            
            for i, col in enumerate(tpr_columns):
                values = [mia_df[mia_df['model_type'] == mt][col].mean() 
                         for mt in model_types if mt in mia_df['model_type'].values]
                ax4.bar(x_pos + i * width, values, width, label=col)
            
            ax4.set_xlabel('Model Type')
            ax4.set_ylabel('TPR')
            ax4.set_title('MIA Attack TPR Comparison')
            ax4.set_xticks(x_pos + width)
            ax4.set_xticklabels(labels)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mia_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"MIA plots saved to {self.output_dir / 'mia_comparison.png'}")
        
        return mia_df
    
    def create_summary_table(self, df: pd.DataFrame, mia_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create comprehensive summary table"""
        
        if df.empty:
            return pd.DataFrame()
        
        # Group by model type and sparsity
        grouped = df.groupby(['model_type', 'sparsity'])
        
        # Calculate accuracy statistics
        summary_stats = grouped.agg({
            'best_acc1': ['mean', 'std'],
            'final_acc1': ['mean', 'std'],
            'total_duration_hours': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
        summary_stats = summary_stats.reset_index()
        
        # Add MIA results if available
        if mia_df is not None and not mia_df.empty:
            # Parse MIA data similarly
            mia_info = []
            for _, row in mia_df.iterrows():
                name = row.get('model_name', '')
                
                if 'dense' in name:
                    model_type = 'Dense'
                    sparsity = 0.0
                elif 'static' in name:
                    model_type = 'Static'
                    sparsity = self._extract_sparsity(name)
                elif 'dpf' in name:
                    model_type = 'DPF'
                    sparsity = self._extract_sparsity(name)
                else:
                    model_type = 'Unknown'
                    sparsity = 0.0
                
                mia_info.append({
                    'model_type': model_type,
                    'sparsity': sparsity,
                    'auc': row.get('auc', 0),
                    'tpr_at_1pct': row.get('TPR@0.01', 0)
                })
            
            mia_summary = pd.DataFrame(mia_info)
            mia_grouped = mia_summary.groupby(['model_type', 'sparsity']).agg({
                'auc': 'mean',
                'tpr_at_1pct': 'mean'
            }).round(4).reset_index()
            
            # Merge with accuracy data
            summary_stats = summary_stats.merge(
                mia_grouped, on=['model_type', 'sparsity'], how='left'
            )
        
        # Save summary table
        summary_stats.to_csv(self.output_dir / 'summary_table.csv', index=False)
        
        print(f"Summary table saved to {self.output_dir / 'summary_table.csv'}")
        
        return summary_stats
    
    def create_latex_table(self, summary_stats: pd.DataFrame):
        """Create LaTeX table for paper"""
        
        if summary_stats.empty:
            return
        
        # Format for LaTeX
        latex_lines = []
        latex_lines.append("\\begin{table}[h]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{Comparison of Dense, Static, and DPF Models}")
        latex_lines.append("\\label{tab:comparison}")
        
        # Determine number of columns
        has_mia = 'auc' in summary_stats.columns
        if has_mia:
            latex_lines.append("\\begin{tabular}{l|c|c|c|c|c}")
            latex_lines.append("\\hline")
            latex_lines.append("Model & Sparsity & Accuracy (\\%) & Training Time (h) & AUC & TPR@1\\% \\\\")
        else:
            latex_lines.append("\\begin{tabular}{l|c|c|c}")
            latex_lines.append("\\hline")
            latex_lines.append("Model & Sparsity & Accuracy (\\%) & Training Time (h) \\\\")
        
        latex_lines.append("\\hline")
        
        # Add data rows
        for _, row in summary_stats.iterrows():
            model_type = row['model_type']
            sparsity = f"{row['sparsity']:.1%}" if row['sparsity'] > 0 else "0\\%"
            
            acc_mean = row['best_acc1_mean']
            acc_std = row['best_acc1_std']
            acc_str = f"{acc_mean:.2f} $\\pm$ {acc_std:.2f}"
            
            time_mean = row['total_duration_hours_mean']
            time_std = row['total_duration_hours_std']
            time_str = f"{time_mean:.1f} $\\pm$ {time_std:.1f}"
            
            if has_mia:
                auc = row.get('auc', 0)
                tpr = row.get('tpr_at_1pct', 0)
                line = f"{model_type} & {sparsity} & {acc_str} & {time_str} & {auc:.3f} & {tpr:.3f} \\\\"
            else:
                line = f"{model_type} & {sparsity} & {acc_str} & {time_str} \\\\"
            
            latex_lines.append(line)
        
        latex_lines.append("\\hline")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")
        
        # Save LaTeX table
        with open(self.output_dir / 'comparison_table.tex', 'w') as f:
            f.write('\n'.join(latex_lines))
        
        print(f"LaTeX table saved to {self.output_dir / 'comparison_table.tex'}")
    
    def generate_report(self, title: str):
        """Generate comprehensive report"""
        
        print("Generating comprehensive report...")
        
        # Load data
        df = self.load_experiment_data()
        mia_df = self.load_mia_results()
        
        if df.empty:
            print("No experiment data found")
            return
        
        print(f"Loaded {len(df)} experiment results")
        if mia_df is not None:
            print(f"Loaded {len(mia_df)} MIA results")
        
        # Create accuracy plots
        print("Creating accuracy plots...")
        accuracy_stats = self.create_accuracy_plots(df)
        
        # Create MIA plots
        if mia_df is not None:
            print("Creating MIA plots...")
            mia_processed = self.create_mia_plots(mia_df)
        else:
            mia_processed = None
        
        # Create summary table
        print("Creating summary table...")
        summary_stats = self.create_summary_table(df, mia_processed)
        
        # Create LaTeX table
        print("Creating LaTeX table...")
        self.create_latex_table(summary_stats)
        
        # Create HTML report
        print("Creating HTML report...")
        self.create_html_report(title, summary_stats, df, mia_processed)
        
        print(f"\nReport generation completed!")
        print(f"Output directory: {self.output_dir}")
        print(f"Files created:")
        print(f"  - accuracy_comparison.png")
        if mia_processed is not None:
            print(f"  - mia_comparison.png")
        print(f"  - summary_table.csv")
        print(f"  - comparison_table.tex")
        print(f"  - report.html")
    
    def create_html_report(self, title: str, summary_stats: pd.DataFrame, 
                          df: pd.DataFrame, mia_df: Optional[pd.DataFrame]):
        """Create HTML report"""
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background-color: #f2f2f2; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        .section {{ margin: 40px 0; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    
    <div class="section">
        <h2>Experiment Overview</h2>
        <p><strong>Total Experiments:</strong> {len(df)}</p>
        <p><strong>Model Types:</strong> {', '.join(df['model_type'].unique())}</p>
        <p><strong>Sparsity Levels:</strong> {', '.join(map(str, sorted(df['sparsity'].unique())))}</p>
        {"<p><strong>MIA Evaluation:</strong> Available</p>" if mia_df is not None else "<p><strong>MIA Evaluation:</strong> Not Available</p>"}
    </div>
    
    <div class="section">
        <h2>Accuracy Comparison</h2>
        <img src="accuracy_comparison.png" alt="Accuracy Comparison Plots">
    </div>
"""
        
        if mia_df is not None:
            html_content += """
    <div class="section">
        <h2>MIA Evaluation Results</h2>
        <img src="mia_comparison.png" alt="MIA Comparison Plots">
    </div>
"""
        
        # Add summary table
        if not summary_stats.empty:
            html_content += """
    <div class="section">
        <h2>Summary Statistics</h2>
        <table>
"""
            # Table header
            html_content += "<tr>"
            for col in summary_stats.columns:
                html_content += f"<th>{col.replace('_', ' ').title()}</th>"
            html_content += "</tr>\n"
            
            # Table rows
            for _, row in summary_stats.iterrows():
                html_content += "<tr>"
                for col in summary_stats.columns:
                    value = row[col]
                    if isinstance(value, float):
                        html_content += f"<td>{value:.4f}</td>"
                    else:
                        html_content += f"<td>{value}</td>"
                html_content += "</tr>\n"
            
            html_content += """
        </table>
    </div>
"""
        
        html_content += """
    <div class="section">
        <h2>Files</h2>
        <ul>
            <li><a href="summary_table.csv">Summary Table (CSV)</a></li>
            <li><a href="comparison_table.tex">LaTeX Table</a></li>
            <li><a href="accuracy_comparison.png">Accuracy Plots</a></li>
"""
        
        if mia_df is not None:
            html_content += '            <li><a href="mia_comparison.png">MIA Plots</a></li>\n'
        
        html_content += """
        </ul>
    </div>
    
</body>
</html>
"""
        
        # Save HTML report
        with open(self.output_dir / 'report.html', 'w') as f:
            f.write(html_content)
        
        print(f"HTML report saved to {self.output_dir / 'report.html'}")

def main():
    args = parse_args()
    
    print(f"Creating comprehensive report...")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Create reporter
    reporter = ComprehensiveReporter(args.results_dir, args.output_dir)
    
    # Generate report
    reporter.generate_report(args.title)

if __name__ == '__main__':
    main()