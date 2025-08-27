#!/bin/bash

# 전체 스파시티 레벨 테스트 스크립트
echo "🎯 DCIL-MIA 전체 스파시티 테스트"
echo "스파시티: 50%, 70%, 80%, 90%, 95%"
echo "====================================="

GPU=0
EPOCHS=5  # 테스트용으로 5 에포크
DATASET=cifar10
SEED=42
SPARSITIES=(0.5 0.7 0.8 0.9 0.95)

# 에러 발생 시 중단
set -e

echo ""
echo "1️⃣  Dense 베이스라인"
echo "=================="

echo "Dense 모델 훈련 중..."
python run_experiment.py \
    --name dense_baseline \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --seed $SEED \
    --gpu $GPU \
    --datapath ~/Datasets/CIFAR

echo "✅ Dense 모델 완료"

echo ""
echo "2️⃣  Static 프루닝 (모든 스파시티)"
echo "==============================="

for sparsity in "${SPARSITIES[@]}"; do
    echo "Static 프루닝 훈련 중: ${sparsity}0% 스파시티..."
    python run_experiment.py \
        --name static_sparsity${sparsity} \
        --dataset $DATASET \
        --prune \
        --prune-method static \
        --sparsity $sparsity \
        --epochs $EPOCHS \
        --seed $SEED \
        --gpu $GPU \
        --datapath ~/Datasets/CIFAR
    
    echo "✅ Static ${sparsity}0% 완료"
done

echo ""
echo "3️⃣  DPF 프루닝 (모든 스파시티)"
echo "============================"

for sparsity in "${SPARSITIES[@]}"; do
    echo "DPF 프루닝 훈련 중: ${sparsity}0% 스파시티..."
    python run_experiment.py \
        --name dpf_sparsity${sparsity} \
        --dataset $DATASET \
        --prune \
        --prune-method dpf \
        --sparsity $sparsity \
        --epochs $EPOCHS \
        --seed $SEED \
        --gpu $GPU \
        --datapath ~/Datasets/CIFAR
    
    echo "✅ DPF ${sparsity}0% 완료"
done

echo ""
echo "🎉 전체 11개 타겟 모델 훈련 완료!"
echo "================================"
echo "훈련된 모델:"
echo "  - Dense: 1개"
echo "  - Static: 5개 (50%, 70%, 80%, 90%, 95%)"
echo "  - DPF: 5개 (50%, 70%, 80%, 90%, 95%)"
echo "  - 총합: 11개 모델"
echo ""
echo "결과 확인: ls -la runs/"