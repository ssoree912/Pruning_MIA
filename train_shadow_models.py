#!/usr/bin/env python3
"""
DWA MIA를 위한 Shadow 모델 훈련 스크립트
Victim 모델(seed=42)과 동일한 설정으로 다른 seed값들을 사용해 shadow 모델들 훈련
"""

import os
import json
import subprocess
from pathlib import Path
import argparse

def create_shadow_config(base_config_path, shadow_seed, output_path):
    """Base config를 기반으로 다른 seed로 shadow config 생성"""
    
    with open(base_config_path, 'r') as f:
        config = json.load(f)
    
    # Seed 변경
    config['system']['seed'] = shadow_seed
    
    # Save directory 변경 
    original_save_dir = config['save_dir']
    base_dir = str(Path(original_save_dir).parent)
    config['save_dir'] = f"{base_dir}/seed{shadow_seed}"
    
    # WandB name 변경
    if config.get('wandb', {}).get('enabled', False):
        original_name = config['wandb']['name']
        config['wandb']['name'] = f"{original_name}_seed{shadow_seed}"
    
    # Config 저장
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created shadow config: {output_path} (seed={shadow_seed})")
    return config

def train_shadow_models(victim_config_path, num_shadows=8, start_seed=43, gpu=0, dry_run=False):
    """Shadow 모델들 훈련"""
    
    print(f"🎯 Training {num_shadows} shadow models")
    print(f"   Victim config: {victim_config_path}")
    print(f"   Shadow seeds: {start_seed} to {start_seed + num_shadows - 1}")
    print(f"   GPU: {gpu}")
    
    if not os.path.exists(victim_config_path):
        print(f"❌ Victim config not found: {victim_config_path}")
        return False
    
    # Shadow configs 생성
    shadow_configs = []
    configs_dir = "shadow_configs"
    os.makedirs(configs_dir, exist_ok=True)
    
    for i in range(num_shadows):
        shadow_seed = start_seed + i
        shadow_config_path = f"{configs_dir}/shadow_seed{shadow_seed}.json"
        
        config = create_shadow_config(victim_config_path, shadow_seed, shadow_config_path)
        shadow_configs.append((shadow_seed, shadow_config_path, config))
    
    if dry_run:
        print("🔍 DRY RUN: Shadow configs created but training skipped")
        return True
    
    # Shadow 모델들 훈련
    print("\n🚀 Starting shadow model training...")
    
    for i, (shadow_seed, config_path, config) in enumerate(shadow_configs):
        print(f"\n{'='*80}")
        print(f"🤖 Training Shadow Model {i+1}/{num_shadows} (seed={shadow_seed})")
        print(f"{'='*80}")
        
        # 훈련 명령어 (실제 DWA 훈련 스크립트에 맞게 조정 필요)
        cmd = [
            'python', 'train.py',  # 실제 DWA 훈련 스크립트 이름
            '--config', config_path,
            '--gpu', str(gpu)
        ]
        
        print(f"🔥 Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            print(f"✅ Shadow {i+1} training completed!")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Shadow {i+1} training failed: {e}")
            return False
        
        except FileNotFoundError:
            print(f"❌ Training script not found. Please check the command: {' '.join(cmd)}")
            print("💡 You may need to adjust the training command in this script")
            return False
    
    print(f"\n🎉 All {num_shadows} shadow models training completed!")
    return True

def main():
    parser = argparse.ArgumentParser(description='Train shadow models for DWA MIA')
    parser.add_argument('--victim_config', required=True, help='Path to victim model config')
    parser.add_argument('--num_shadows', type=int, default=8, help='Number of shadow models')
    parser.add_argument('--start_seed', type=int, default=43, help='Starting seed for shadows')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--dry_run', action='store_true', help='Only create configs, skip training')
    
    args = parser.parse_args()
    
    print("🤖 DWA Shadow Models Training")
    print("=" * 50)
    
    success = train_shadow_models(
        victim_config_path=args.victim_config,
        num_shadows=args.num_shadows,
        start_seed=args.start_seed,
        gpu=args.gpu,
        dry_run=args.dry_run
    )
    
    if success:
        print("\n✅ Shadow model training pipeline completed!")
        print("\n📝 Next steps:")
        print("1. Verify shadow models are saved properly")  
        print("2. Update MIA converter to use seed-based shadow models")
        print("3. Run MIA evaluation")
    else:
        print("\n❌ Shadow model training failed!")

if __name__ == "__main__":
    main()