# Dense/Static/Dynamic Pruning with MIA Evaluation

This repository implements a comprehensive comparison of Dense, Static, and Dynamic (DPF) pruning methods with Membership Inference Attack (MIA) evaluation using LiRA.

## Experiment Overview

### 🎯 Research Goals
- Compare **Dense**, **Static**, and **DPF (Dynamic Pruning with Feedback)** models across multiple sparsity levels
- Evaluate privacy vulnerabilities using **LiRA (Likelihood Ratio Attack)** 
- Analyze the relationship between model sparsity and privacy leakage

### 📊 Experimental Setup
- **Target Models**: 11 models total
  - 1 Dense baseline (ResNet-18)
  - 5 Static pruned models (50%, 70%, 80%, 90%, 95% sparsity)
  - 5 DPF pruned models (50%, 70%, 80%, 90%, 95% sparsity)
- **Dataset**: CIFAR-10/CIFAR-100
- **Architecture**: ResNet-18
- **MIA Method**: LiRA with 64 shadow models per configuration

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone repository
git clone <repository-url>
cd DCIL-pytorch

# Install dependencies
pip install torch torchvision
pip install numpy pandas matplotlib seaborn scikit-learn scipy tqdm
pip install pyyaml
```

### 2. Run Complete Experiment
```bash
# Run all models with MIA evaluation (multiple seeds)
python scripts/run_full_experiment.py \
    --dataset cifar10 \
    --seeds 42 123 456 \
    --gpus 0 1 2 3 \
    --run-mia \
    --max-parallel 4

# For a quick test (single seed, no MIA)
python scripts/run_full_experiment.py \
    --dataset cifar10 \
    --seeds 42 \
    --gpus 0 \
    --dry-run
```

### 3. Generate Report
```bash
# Create comprehensive report with visualizations
python scripts/create_report.py \
    --results-dir ./runs \
    --output-dir ./final_report
```

## 📁 Project Structure

```
DCIL-pytorch/
├── configs/                    # Experiment configurations
│   └── experiment_config.py   # Configuration management
├── experiments/               # Training scripts
│   ├── train_dense.py        # Dense baseline training
│   ├── train_static.py       # Static pruning training  
│   ├── train_dpf.py          # DPF training
│   ├── train_shadows.py      # Shadow model training
│   └── evaluate_lira.py      # LiRA evaluation
├── mia/                      # Membership inference attacks
│   ├── __init__.py
│   └── lira.py              # LiRA implementation
├── pruning/                  # Pruning implementations
│   └── dcil/
│       └── mnn.py           # Enhanced with Static/Dynamic maskers
├── scripts/                  # Automation scripts
│   ├── run_full_experiment.py # Full experiment runner
│   └── create_report.py      # Report generation
├── utils/                    # Utilities
│   └── logger.py            # Comprehensive logging
└── run_experiment.py         # Main experiment runner
```

## 🔬 Implementation Details

### Masking Strategies

#### Static Pruning
```python
class MaskerStatic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(mask)
        return x * mask
    
    @staticmethod 
    def backward(ctx, grad_out):
        (mask,) = ctx.saved_tensors
        return grad_out * mask, None  # Dead weights get zero gradients
```

#### Dynamic Pruning (DPF)
```python
class MaskerDynamic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        return x * mask  # Sparse forward pass
    
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None  # Dense backward pass - enables reactivation
```

### Key Differences
- **Static**: Dead weights stay dead (no gradient flow)
- **Dynamic**: Dead weights can be reactivated (gradient flow maintained)
- **Dense**: No pruning applied

## 🔍 LiRA Implementation

The LiRA attack implementation follows the methodology from "Membership inference attacks from first principles" by Carlini et al.:

1. **Shadow Model Training**: Train 64 shadow models for each target configuration
2. **Calibration**: Fit Gaussian distributions to IN/OUT loss statistics  
3. **Attack**: Calculate likelihood ratios for target model predictions
4. **Evaluation**: Report AUC and TPR@FPR metrics

## 📈 Usage Examples

### Individual Model Training
```bash
# Train dense baseline
python run_experiment.py \
    --name dense_baseline \
    --dataset cifar10 \
    --arch resnet \
    --layers 18 \
    --seed 42

# Train static pruned model
python run_experiment.py \
    --name static_90pct \
    --dataset cifar10 \
    --prune \
    --prune-method static \
    --sparsity 0.9 \
    --seed 42

# Train DPF model  
python run_experiment.py \
    --name dpf_90pct \
    --dataset cifar10 \
    --prune \
    --prune-method dpf \
    --sparsity 0.9 \
    --prune-freq 16 \
    --target-epoch 75 \
    --seed 42
```

### Configuration-Based Training
```bash
# Create preset configurations
python configs/experiment_config.py

# Train using configuration file
python run_experiment.py --config dense_baseline

# Train with custom config
python run_experiment.py --config my_config.yaml
```

### MIA Evaluation Only
```bash
# Evaluate pre-trained models with LiRA
python experiments/evaluate_lira.py \
    --target-models-dir ./runs \
    --shadow-models-dir ./runs/shadows \
    --results-dir ./mia_results \
    --dataset cifar10 \
    --num-shadows 64
```

## 📊 Expected Results

### Utility Metrics
- **Dense**: ~95% accuracy (baseline)
- **Static**: Accuracy decreases with sparsity, sharp drop at high sparsity
- **DPF**: Better accuracy retention at high sparsity due to weight reactivation

### Privacy Metrics (LiRA)
- **AUC**: Higher values indicate more vulnerable to MIA
- **TPR@FPR**: True Positive Rate at fixed False Positive Rate thresholds
- Expected trend: Privacy vulnerability may vary with sparsity and pruning method

## 🛠️ Customization

### Adding New Pruning Methods
1. Implement new masker in `pruning/dcil/mnn.py`
2. Add training logic in `run_experiment.py`  
3. Update configuration in `configs/experiment_config.py`

### Adding New MIA Methods
1. Implement attack in `mia/` directory
2. Add evaluation script in `experiments/`
3. Update report generation in `scripts/create_report.py`

## 📋 Requirements

### System Requirements
- Python 3.8+
- PyTorch 1.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM for shadow model training

### Package Dependencies
```
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.0
pandas>=1.3.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
scipy>=1.7.0
pyyaml>=5.4.0
tqdm>=4.60.0
psutil>=5.8.0
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch-size 64`
   - Use gradient checkpointing
   - Train fewer shadow models: `--num-shadows 32`

2. **Slow Training**
   - Use multiple GPUs: `--gpus 0 1 2 3`  
   - Increase parallel jobs: `--max-parallel 4`
   - Reduce print frequency: `--print-freq 200`

3. **Configuration Errors**
   - Check YAML syntax in config files
   - Verify file paths exist
   - Check GPU availability

### Performance Tips
- Use SSD storage for faster data loading
- Enable mixed precision training (add to config)
- Use DataParallel for multi-GPU training
- Monitor GPU utilization with `nvidia-smi`

## 📖 References

1. **DCIL**: Kim et al. "Dynamic collective intelligence learning: Finding efficient sparse model via refined gradients for pruned weights" (2021)
2. **LiRA**: Carlini et al. "Membership inference attacks from first principles" (2022)
3. **Pruning**: Zhu & Gupta "To prune, or not to prune: exploring the efficacy of pruning for model compression" (2017)

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review experiment logs in `./runs/<experiment>/experiment.log`
3. Examine configuration files for parameter validation
4. Check GPU memory and system resources

## 📄 License

This project extends the original DCIL implementation with comprehensive MIA evaluation capabilities. Please refer to the original DCIL license and cite appropriate papers when using this code.