#!/usr/bin/env python3
"""
Environment setup test script
Tests all dependencies and functionality for DCIL-MIA experiments
"""

import sys
import importlib
from typing import List, Tuple

def test_import(module_name: str, alias: str = None) -> Tuple[bool, str]:
    """Test if a module can be imported"""
    try:
        if alias:
            module = importlib.import_module(module_name)
            globals()[alias] = module
        else:
            importlib.import_module(module_name)
        return True, f"✅ {module_name}"
    except ImportError as e:
        return False, f"❌ {module_name}: {e}"

def test_torch_functionality():
    """Test PyTorch specific functionality"""
    try:
        import torch
        import torch.nn as nn
        
        # Test basic tensor operations
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        z = torch.mm(x, y)
        
        # Test neural network
        model = nn.Linear(10, 1)
        input_tensor = torch.randn(5, 10)
        output = model(input_tensor)
        
        # Test CUDA if available
        cuda_info = ""
        if torch.cuda.is_available():
            cuda_info = f" (CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPUs)"
            # Test CUDA tensor operations
            x_cuda = x.cuda()
            y_cuda = y.cuda()
            z_cuda = torch.mm(x_cuda, y_cuda)
        
        return True, f"✅ PyTorch functionality{cuda_info}"
        
    except Exception as e:
        return False, f"❌ PyTorch functionality: {e}"

def test_data_loading():
    """Test data loading functionality"""
    try:
        import torchvision
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader
        
        # Test CIFAR-10 dataset loading (without downloading)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Just test the dataset class creation (no actual download)
        dataset_class = torchvision.datasets.CIFAR10
        
        return True, "✅ Data loading functionality"
        
    except Exception as e:
        return False, f"❌ Data loading functionality: {e}"

def test_project_modules():
    """Test project-specific modules"""
    try:
        # Add project root to path
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Test config system
        from configs.experiment_config import ExperimentConfig
        config = ExperimentConfig()
        
        # Test logging system
        from utils.logger import ExperimentLogger
        
        # Test MIA module
        from mia import LiRAAttacker
        
        return True, "✅ Project modules"
        
    except Exception as e:
        return False, f"❌ Project modules: {e}"

def main():
    """Run all tests"""
    print("🧪 Testing DCIL-MIA Environment Setup")
    print("=" * 50)
    
    # Core dependencies
    core_modules = [
        ("sys", None),
        ("os", None),
        ("time", None),
        ("pathlib", None),
        ("json", None),
        ("pickle", None),
    ]
    
    # Scientific computing
    scientific_modules = [
        ("numpy", "np"),
        ("scipy", None),
        ("pandas", "pd"),
        ("sklearn", None),
    ]
    
    # PyTorch ecosystem
    pytorch_modules = [
        ("torch", None),
        ("torchvision", None),
        ("torchaudio", None),
    ]
    
    # Visualization
    viz_modules = [
        ("matplotlib", None),
        ("matplotlib.pyplot", "plt"),
        ("seaborn", "sns"),
        ("plotly", None),
    ]
    
    # Utilities
    util_modules = [
        ("tqdm", None),
        ("psutil", None),
        ("yaml", None),
        ("joblib", None),
    ]
    
    all_modules = [
        ("Core Python", core_modules),
        ("Scientific Computing", scientific_modules), 
        ("PyTorch Ecosystem", pytorch_modules),
        ("Visualization", viz_modules),
        ("Utilities", util_modules),
    ]
    
    total_tests = 0
    passed_tests = 0
    
    # Test module imports
    for category, modules in all_modules:
        print(f"\n📦 {category}:")
        for module_name, alias in modules:
            success, message = test_import(module_name, alias)
            print(f"   {message}")
            total_tests += 1
            if success:
                passed_tests += 1
    
    # Test functionality
    print(f"\n🔧 Functionality Tests:")
    
    # Test PyTorch functionality
    success, message = test_torch_functionality()
    print(f"   {message}")
    total_tests += 1
    if success:
        passed_tests += 1
    
    # Test data loading
    success, message = test_data_loading()
    print(f"   {message}")
    total_tests += 1
    if success:
        passed_tests += 1
    
    # Test project modules
    success, message = test_project_modules()
    print(f"   {message}")
    total_tests += 1
    if success:
        passed_tests += 1
    
    # System information
    print(f"\n🖥️  System Information:")
    print(f"   Python: {sys.version}")
    
    try:
        import torch
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"     GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    except:
        print("   PyTorch: Not available")
    
    try:
        import numpy as np
        print(f"   NumPy: {np.__version__}")
    except:
        print("   NumPy: Not available")
    
    try:
        import pandas as pd
        print(f"   Pandas: {pd.__version__}")
    except:
        print("   Pandas: Not available")
    
    # Summary
    print(f"\n📊 Test Summary:")
    print(f"   Passed: {passed_tests}/{total_tests} tests")
    print(f"   Success rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Environment is ready for experiments.")
        
        print(f"\n🚀 Quick Start Commands:")
        print(f"   # Run a quick test experiment")
        print(f"   python scripts/run_full_experiment.py --dataset cifar10 --seeds 42 --dry-run")
        print(f"   ")
        print(f"   # Train dense baseline")
        print(f"   python run_experiment.py --name dense_test --dataset cifar10 --epochs 5")
        print(f"   ")
        print(f"   # Full experiment with MIA")
        print(f"   python scripts/run_full_experiment.py --dataset cifar10 --run-mia")
        
        return 0
    else:
        failed_tests = total_tests - passed_tests
        print(f"\n❌ {failed_tests} tests failed. Please check the error messages above.")
        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Make sure you activated the conda environment:")
        print(f"      conda activate dcil-mia-experiment")
        print(f"   2. Try reinstalling the environment:")
        print(f"      bash setup_environment.sh")
        print(f"   3. Check that all dependencies are properly installed")
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)