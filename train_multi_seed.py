#!/usr/bin/env python3
"""
Multiple seed training script for MIA evaluation
기존 train.py 명령어를 여러 seed로 반복 실행하는 스크립트
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

def train_multi_seed(methods, sparsities, epochs=200, freeze_epoch=180, 
                    num_seeds=8, start_seed=43, base_seed=42, gpu=0, 
                    wandb_project="pruning_one_shot", dry_run=False):
    """Multiple seed로 훈련 실행 (shadow 모델들만)"""
    
    print(f"🎯 Training shadow models with multiple seeds")
    print(f"   Methods: {', '.join(methods)}")
    print(f"   Sparsities: {', '.join(map(str, sparsities))}")
    print(f"   Victim seed: {base_seed} (existing, not training)")
    print(f"   Shadow seeds: {start_seed}-{start_seed + num_seeds - 1} (new training)")
    print(f"   Epochs: {epochs}, Freeze: {freeze_epoch}")
    print(f"   GPU: {gpu}")
    
    # Shadow 모델들만 훈련 (victim은 이미 존재)
    shadow_seeds = list(range(start_seed, start_seed + num_seeds))
    
    if dry_run:
        print("🔍 DRY RUN: Would execute these commands:")
        for i, seed in enumerate(shadow_seeds):
            shadow_num = i + 1
            project_name = f"{wandb_project}_seed{seed}"
            
            cmd = [
                'python', 'train.py',
                '--methods'
            ] + methods + [
                '--sparsities'
            ] + [str(s) for s in sparsities] + [
                '--epochs', str(epochs),
                '--freeze-epoch', str(freeze_epoch),
                '--seed', str(seed),
                '--gpu', str(gpu),
                '--wandb',
                '--wandb-project', project_name
            ]
            print(f"   shadow{shadow_num:>2}: {' '.join(cmd)}")
        return True
    
    # 실제 훈련 실행
    print("\n🚀 Starting shadow model training...")
    
    failed_seeds = []
    successful_seeds = []
    
    for i, seed in enumerate(shadow_seeds):
        shadow_num = i + 1
        project_name = f"{wandb_project}_seed{seed}"
        
        print(f"\n{'='*80}")
        print(f"🤖 Training Shadow {shadow_num}/8 (seed={seed})")
        print(f"{'='*80}")
        
        cmd = [
            'python', 'train.py',
            '--methods'
        ] + methods + [
            '--sparsities'
        ] + [str(s) for s in sparsities] + [
            '--epochs', str(epochs),
            '--freeze-epoch', str(freeze_epoch),
            '--seed', str(seed),
            '--gpu', str(gpu),
            '--wandb',
            '--wandb-project', project_name
        ]
        
        print(f"🔥 Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True)
            print(f"✅ Shadow {shadow_num} training completed!")
            successful_seeds.append((seed, f"shadow{shadow_num}"))
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Shadow {shadow_num} training failed: {e}")
            failed_seeds.append((seed, f"shadow{shadow_num}", str(e)))
            
        except FileNotFoundError:
            print(f"❌ train.py not found. Please check if the script exists")
            return False
    
    # 결과 요약
    print(f"\n{'='*80}")
    print("📊 SHADOW MODEL TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successful: {len(successful_seeds)}/{len(shadow_seeds)}")
    print(f"❌ Failed: {len(failed_seeds)}/{len(shadow_seeds)}")
    
    if successful_seeds:
        print("\n🏆 Successful shadow trainings:")
        for seed, seed_type in successful_seeds:
            print(f"   {seed_type:>8} (seed={seed})")
    
    if failed_seeds:
        print("\n💥 Failed shadow trainings:")
        for seed, seed_type, error in failed_seeds:
            print(f"   {seed_type:>8} (seed={seed}): {error}")
    
    success_rate = len(successful_seeds) / len(shadow_seeds)
    print(f"\n📈 Success rate: {success_rate:.1%}")
    
    if success_rate >= 0.8:  # 80% 이상 성공
        print("🎉 Multi-seed training completed successfully!")
        return True
    else:
        print("⚠️ Multi-seed training completed with issues!")
        return False

def main():
    parser = argparse.ArgumentParser(description='Train multiple seeds for MIA evaluation')
    parser.add_argument('--methods', required=True, nargs='+', 
                       help='Pruning methods (dense, static, dpf, etc.)')
    parser.add_argument('--sparsities', required=True, nargs='+', type=float,
                       help='Sparsity levels (0.5, 0.7, 0.8, 0.9, 0.95)')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--freeze-epoch', type=int, default=180, help='Freeze epoch')
    parser.add_argument('--num-seeds', type=int, default=8, help='Number of shadow models')
    parser.add_argument('--start-seed', type=int, default=43, help='Starting seed for shadows')
    parser.add_argument('--base-seed', type=int, default=42, help='Base seed for victim model')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--wandb-project', default='pruning_one_shot', help='WandB project name')
    parser.add_argument('--dry-run', action='store_true', help='Show commands without execution')
    
    args = parser.parse_args()
    
    print("🤖 Multi-Seed Training Pipeline")
    print("=" * 60)
    
    success = train_multi_seed(
        methods=args.methods,
        sparsities=args.sparsities,
        epochs=args.epochs,
        freeze_epoch=args.freeze_epoch,
        num_seeds=args.num_seeds,
        start_seed=args.start_seed,
        base_seed=args.base_seed,
        gpu=args.gpu,
        wandb_project=args.wandb_project,
        dry_run=args.dry_run
    )
    
    if success:
        print("\n✅ Multi-seed training pipeline completed!")
        print("\n📝 Next steps:")
        print("1. Check trained models in runs/ directory")
        print("2. Update MIA converter to use seed-based models")
        print("3. Run MIA evaluation")
    else:
        print("\n❌ Multi-seed training completed with issues!")
        sys.exit(1)

if __name__ == "__main__":
    main()