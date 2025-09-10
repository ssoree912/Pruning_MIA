#!/usr/bin/env python3
"""
단일 모델에 대한 MIA 평가 실행 스크립트
DWA 훈련 결과를 WeMeM-main 구조로 변환하고 MIA 평가 실행
"""

import os
import sys
import subprocess
from pathlib import Path

def run_single_mia(dataset='cifar10', model='resnet18', mode=None, sparsity=None, gpu=0):
    """단일 모델에 대한 MIA 평가 실행"""
    
    print(f"🚀 Running MIA evaluation for {dataset}_{model}")
    if mode:
        print(f"   Mode: {mode}")
    if sparsity:
        print(f"   Sparsity: {sparsity}")
    
    # Step 1: DWA → WeMeM 구조 변환
    print("\n🔄 Step 1: Converting DWA results to WeMeM structure...")
    
    cmd = [
        'python', 'scripts/dwa_to_wemem_converter.py',
        '--dataset', dataset,
        '--model', model,
        '--runs_dir', './runs'
    ]
    
    if mode:
        cmd.extend(['--mode', mode])
    if sparsity:
        cmd.extend(['--sparsity', str(sparsity)])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Conversion successful!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    except FileNotFoundError:
        print("❌ Converter script not found. Make sure scripts/dwa_to_wemem_converter.py exists")
        return False
    
    # Step 2: 설정 파일 확인/생성
    config_path = f"configs/{dataset}_{model}.json"
    if not os.path.exists(config_path):
        print(f"\n⚠️ Config file {config_path} not found")
        print("Creating basic config file...")
        
        # 기본 설정 생성
        config = {
            "dataset_name": dataset,
            "model_name": model,
            "num_cls": 10 if dataset == 'cifar10' else 100,
            "input_dim": 3,
            "image_size": 32,
            "hidden_size": 128,
            "seed": 7,
            "early_stop": 5,
            "batch_size": 128,
            "pruner_name": "l1unstructure",
            "prune_sparsity": float(sparsity) if sparsity else 0.6,
            "shadow_num": 2,
            "attacks": "samia,threshold,nn,nn_top3,nn_cls",
            "original": False
        }
        
        os.makedirs("configs", exist_ok=True)
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Created {config_path}")
    
    # Step 3: MIA 평가 실행
    print("\n🎯 Step 2: Running MIA evaluation...")
    
    cmd = [
        'python', 'mia_modi.py',
        str(gpu),
        config_path,
        '--dataset_name', dataset,
        '--model_name', model,
        '--attacks', 'samia,threshold,nn,nn_top3,nn_cls'
    ]
    
    if sparsity:
        cmd.extend(['--prune_sparsity', str(sparsity)])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ MIA evaluation successful!")
        print(result.stdout)
        
        # 결과 파일 확인
        log_file = f"log/{dataset}_{model}/l1unstructure_{sparsity or '0.6'}_.txt"
        if os.path.exists(log_file):
            print(f"\n📊 Results saved to: {log_file}")
            print("\n📈 MIA Attack Results:")
            with open(log_file, 'r') as f:
                content = f.read()
                print(content)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ MIA evaluation failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    except FileNotFoundError:
        print("❌ mia_modi.py script not found")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run single model MIA evaluation')
    parser.add_argument('--dataset', default='cifar10', help='Dataset name (cifar10, cifar100)')
    parser.add_argument('--model', default='resnet18', help='Model name')
    parser.add_argument('--mode', help='DWA mode (kill_active_plain_dead, kill_and_reactivate, reactivate_only)')
    parser.add_argument('--sparsity', help='Sparsity level (0.5, 0.7, 0.8, 0.9, 0.95)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 DWA MIA Evaluation Pipeline")
    print("=" * 60)
    
    success = run_single_mia(
        dataset=args.dataset,
        model=args.model, 
        mode=args.mode,
        sparsity=args.sparsity,
        gpu=args.gpu
    )
    
    if success:
        print("\n✅ Pipeline completed successfully!")
    else:
        print("\n❌ Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()