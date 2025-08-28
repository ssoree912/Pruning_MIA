# Dense vs Static vs DPF Pruning: MIA Vulnerability Analysis

A comprehensive comparison of Dense, Static, and Dynamic Pruning with Feedback (DPF) methods and their vulnerability to Membership Inference Attacks (MIA).

## 🎯 Experiment Overview

This experiment evaluates **11 target models**:
- **1 Dense model** (baseline)
- **5 Static pruned models** (50%, 70%, 80%, 90%, 95% sparsity)
- **5 DPF pruned models** (50%, 70%, 80%, 90%, 95% sparsity)

**MIA Evaluation** using WeMeM methodology:
- Confidence-based attacks
- Entropy-based attacks  
- Modified entropy attacks
- Neural network-based attacks

## 🚀 Quick Start

### Run MIA Evaluation on Existing Models
```bash
./run_mia_evaluation.sh
```

### Run Complete Experiment (Training + MIA)
```bash
./full_experiment_with_mia.sh
```

### Single Model Training
```bash
# Dense model
python run_experiment.py --name dense_baseline --dataset cifar10 --epochs 200

# Static pruning (80% sparsity)
python run_experiment.py --name static_80 --prune --prune-method static --sparsity 0.8 --epochs 200

# DPF pruning (80% sparsity)  
python run_experiment.py --name dpf_80 --prune --prune-method dpf --sparsity 0.8 --epochs 200
```

## 📁 Project Structure

```
├── run_experiment.py              # Main experiment runner
├── mia_wemem.py                   # WeMeM MIA evaluation
├── full_experiment_with_mia.sh    # Complete experiment script
├── run_mia_evaluation.sh          # MIA evaluation only
│
├── pruning/                       # Pruning implementations
│   ├── dcil/mnn.py               # MaskerStatic & MaskerDynamic
│   └── models/resnet_mask.py     # Prunable ResNet
│
├── models/                        # Dense model definitions
├── mia/lira.py                    # LiRA implementation
├── experiments/                   # Training scripts
├── scripts/                       # Utility scripts
│
├── runs/                          # Trained models & results
├── results/                       # MIA evaluation results
└── final_results/                 # Comprehensive reports
```

## 🔬 Methods Comparison

### Static Pruning
- **Forward**: `output = x * mask` 
- **Backward**: `grad = grad * mask` (dead weights stay dead)
- **Characteristic**: Sparse gradient updates

### DPF (Dynamic Pruning with Feedback)
- **Forward**: `output = x * mask`
- **Backward**: `grad = grad` (full gradient, weights can reactivate)
- **Characteristic**: Dense gradient updates with selective reactivation

## 📊 Results

### Model Performance
Results saved in `runs/final_report/`:
- `experiments_comparison.csv`: Accuracy vs Sparsity comparison
- `comprehensive_report.json`: Detailed training metrics

### MIA Vulnerability  
Results saved in `results/wemem_mia/`:
- `wemem_mia_summary.csv`: Attack success rates comparison
- `wemem_mia_results.json`: Detailed attack results

## 🛠️ Dependencies

```bash
# Setup environment
conda env create -f environment.yml
conda activate dcil-mia
```

Required packages:
- PyTorch >= 1.9.0
- torchvision
- numpy, pandas
- scikit-learn
- matplotlib, seaborn

## 📈 Expected Results

**Utility (Accuracy)**:
- Dense > DPF > Static (at same sparsity)
- Higher sparsity → Lower accuracy

**Privacy (MIA Resistance)**:  
- Dense models most vulnerable
- Static vs DPF comparison varies by attack type
- Higher sparsity may reduce MIA vulnerability

## 🔍 Citation

```bibtex
@article{dcil_mia_2024,
  title={Membership Inference Vulnerability in Pruned Neural Networks: Dense vs Static vs Dynamic Comparison},
  author={Your Name},
  journal={Experiment Report},
  year={2024}
}
```

## 📞 Usage

For detailed usage and configuration options:
```bash
python run_experiment.py --help
python mia_wemem.py --help
```