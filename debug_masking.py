#!/usr/bin/env python3
"""
MaskConv2d 동작 검증 스크립트
type_value 설정과 그라디언트 전파를 확인
"""

import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pruning.dcil.mnn import MaskConv2d, MaskerDynamic, MaskerStatic

def test_masker_behavior():
    """MaskerDynamic vs MaskerStatic 그라디언트 동작 테스트"""
    print("🔍 Masker 그라디언트 동작 테스트")
    print("=" * 40)
    
    # 테스트 데이터 생성
    w = torch.nn.Parameter(torch.ones(4, requires_grad=True))
    mask = torch.tensor([1., 0., 1., 0.])  # 0번, 2번만 활성화
    
    print(f"원본 weight: {w.data}")
    print(f"mask: {mask}")
    
    # 1. Dynamic 테스트 (재활성화 가능)
    print("\n--- MaskerDynamic (DPF) ---")
    w.grad = None
    y_dyn = MaskerDynamic.apply(w, mask).sum()
    y_dyn.backward(retain_graph=True)
    print(f"Dynamic grad: {w.grad}")
    print(f"마스크=0 위치 grad 값: {w.grad[1].item():.6f}, {w.grad[3].item():.6f}")
    
    # 2. Static 테스트 (재활성화 불가)
    print("\n--- MaskerStatic (Static Pruning) ---")
    w.grad = None
    y_sta = MaskerStatic.apply(w, mask).sum()
    y_sta.backward()
    print(f"Static grad: {w.grad}")
    print(f"마스크=0 위치 grad 값: {w.grad[1].item():.6f}, {w.grad[3].item():.6f}")

def test_maskconv2d_typevalue():
    """MaskConv2d의 type_value별 동작 테스트"""
    print("\n🧪 MaskConv2d type_value 테스트")
    print("=" * 40)
    
    # 간단한 Conv 레이어 생성
    conv = MaskConv2d(3, 16, 3, padding=1)
    
    # 마스크 설정 (일부 가중치를 0으로)
    with torch.no_grad():
        conv.mask.data[:8] = 0  # 처음 8개 필터 마스킹
    
    # 테스트 입력
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    
    print(f"마스킹된 가중치 비율: {(conv.mask == 0).sum().item() / conv.mask.numel():.2%}")
    
    # 각 type_value별 테스트
    for type_val, name in [(0, "기본(Static)"), (5, "MaskerStatic"), (6, "MaskerDynamic")]:
        print(f"\n--- type_value={type_val} ({name}) ---")
        
        conv.type_value = type_val
        conv.zero_grad()
        if x.grad is not None:
            x.grad.zero_()
        
        y = conv(x)
        loss = y.sum()
        loss.backward(retain_graph=True)
        
        # 마스킹된 가중치의 그라디언트 확인
        masked_grad = conv.weight.grad[conv.mask == 0]
        active_grad = conv.weight.grad[conv.mask == 1]
        
        print(f"마스킹된 가중치 grad 합계: {masked_grad.sum().item():.6f}")
        print(f"활성 가중치 grad 합계: {active_grad.sum().item():.6f}")
        print(f"마스킹된 가중치 grad != 0 개수: {(masked_grad != 0).sum().item()}")

def create_training_monitor():
    """훈련 중 모니터링용 코드 생성"""
    monitor_code = '''
# 훈련 루프에 추가할 모니터링 코드
def monitor_masking_behavior(model, epoch, iteration):
    """모델의 마스킹 동작 모니터링"""
    if iteration % 100 == 0:  # 100 iteration마다 체크
        mask_layers = []
        for name, module in model.named_modules():
            if isinstance(module, MaskConv2d):
                mask_layers.append({
                    'name': name,
                    'type_value': module.type_value,
                    'mask_sparsity': (module.mask == 0).float().mean().item(),
                    'weight_grad_nonzero': (module.weight.grad != 0).float().mean().item() if module.weight.grad is not None else 0
                })
        
        if mask_layers:
            print(f"\\n[Epoch {epoch}, Iter {iteration}] Masking Status:")
            for layer_info in mask_layers[:3]:  # 처음 3개 레이어만 출력
                print(f"  {layer_info['name']}: type_value={layer_info['type_value']}, "
                      f"sparsity={layer_info['mask_sparsity']:.3f}, "
                      f"grad_nonzero={layer_info['weight_grad_nonzero']:.3f}")

# run_experiment.py의 train_one_epoch 함수에 추가:
# monitor_masking_behavior(model, epoch, iteration)
'''
    
    print("\n📝 훈련 모니터링 코드")
    print("=" * 40)
    print(monitor_code)
    
    return monitor_code

if __name__ == "__main__":
    print("🚀 MaskConv2d 동작 검증 시작")
    print("=" * 50)
    
    # 1. 기본 Masker 동작 테스트
    test_masker_behavior()
    
    # 2. MaskConv2d type_value 테스트
    test_maskconv2d_typevalue()
    
    # 3. 훈련 모니터링 코드 생성
    create_training_monitor()
    
    print(f"\n✅ 검증 완료!")
    print(f"💡 훈련 중 모니터링을 원한다면 위의 monitor_masking_behavior 함수를 run_experiment.py에 추가하세요.")