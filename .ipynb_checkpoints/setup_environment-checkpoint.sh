#!/bin/bash

# DCIL-MIA Experiment Environment Setup Script
# This script sets up the conda environment for Dense/Static/DPF pruning experiments with MIA evaluation

set -e # Exit on any error

echo "🚀 Setting up DCIL-MIA Experiment Environment"
echo "=============================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Conda found: $(conda --version)"

# Check if CUDA is available (optional)
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️  No NVIDIA GPU detected. CPU-only installation will be used."
    echo "Note: Training will be significantly slower without GPU acceleration."
fi

# Remove existing environment if it exists
ENV_NAME="dcil-mia-experiment"
if conda env list | grep -q "^$ENV_NAME "; then
    echo "🗑️  Removing existing environment: $ENV_NAME"
    conda env remove -n $ENV_NAME -y
fi

# Create new environment
echo "📦 Creating new conda environment: $ENV_NAME"
conda env create -f environment_new.yml

echo "✅ Environment created successfully!"

# Activation instructions
echo ""
echo "🎯 Next Steps:"
echo "============="
echo "1. Activate the environment:"
echo "   conda activate $ENV_NAME"
echo ""
echo "2. Verify installation:"
# The line below has been modified to prevent quoting errors.
echo "   python -c \"import torch; print('PyTorch: ' + torch.__version__); print('CUDA available: ' + str(torch.cuda.is_available()))\""
echo ""
echo "3. Test the setup:"
echo "   python scripts/test_setup.py"
echo ""
echo "4. Run a quick experiment:"
echo "   python scripts/run_full_experiment.py --dataset cifar10 --seeds 42 --dry-run"
echo ""
echo "📚 Documentation:"
echo "   See README_EXPERIMENT.md for detailed usage instructions"
echo ""
echo "🎉 Setup completed successfully!"structions"
echo ""
echo "🎉 Setup completed successfully!"