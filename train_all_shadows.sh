#!/bin/bash

# 모든 타겟 모델 구성에 대한 섀도우 모델 훈련
echo "👥 전체 섀도우 모델 훈련"
echo "======================"

GPU=0
EPOCHS=5  # 테스트용
DATASET=cifar10
SPARSITIES=(0.5 0.7 0.8 0.9 0.95)
NUM_SHADOWS=8  # 테스트용으로 8개 (실제로는 64개)

# 에러 발생 시 중단
set -e

echo ""
echo "1️⃣  Dense 섀도우 모델들"
echo "====================="

echo "Dense 섀도우 모델 훈련 중 (${NUM_SHADOWS}개)..."
python experiments/train_shadows.py \
    --dataset $DATASET \
    --arch resnet \
    --layers 20 \
    --model-type dense \
    --epochs $EPOCHS \
    --num-shadows $NUM_SHADOWS \
    --gpu $GPU \
    --datapath ~/Datasets/CIFAR

echo "✅ Dense 섀도우 완료"

echo ""
echo "2️⃣  Static 섀도우 모델들"
echo "======================"

for sparsity in "${SPARSITIES[@]}"; do
    echo "Static 섀도우 모델 훈련 중: ${sparsity}0% 스파시티 (${NUM_SHADOWS}개)..."
    python experiments/train_shadows.py \
        --dataset $DATASET \
        --arch resnet \
        --layers 20 \
        --model-type static \
        --sparsity $sparsity \
        --epochs $EPOCHS \
        --num-shadows $NUM_SHADOWS \
        --gpu $GPU \
        --datapath ~/Datasets/CIFAR
    
    echo "✅ Static ${sparsity}0% 섀도우 완료"
done

echo ""
echo "3️⃣  DPF 섀도우 모델들"
echo "=================="

for sparsity in "${SPARSITIES[@]}"; do
    echo "DPF 섀도우 모델 훈련 중: ${sparsity}0% 스파시티 (${NUM_SHADOWS}개)..."
    python experiments/train_shadows.py \
        --dataset $DATASET \
        --arch resnet \
        --layers 20 \
        --model-type dpf \
        --sparsity $sparsity \
        --epochs $EPOCHS \
        --num-shadows $NUM_SHADOWS \
        --gpu $GPU \
        --datapath ~/Datasets/CIFAR
    
    echo "✅ DPF ${sparsity}0% 섀도우 완료"
done

echo ""
echo "🎉 전체 섀도우 모델 훈련 완료!"
echo "============================"
echo "훈련된 섀도우 모델:"
echo "  - Dense: ${NUM_SHADOWS}개"
echo "  - Static: $((${#SPARSITIES[@]} * NUM_SHADOWS))개 (각 스파시티별 ${NUM_SHADOWS}개)"
echo "  - DPF: $((${#SPARSITIES[@]} * NUM_SHADOWS))개 (각 스파시티별 ${NUM_SHADOWS}개)"
echo "  - 총합: $((1 * NUM_SHADOWS + 2 * ${#SPARSITIES[@]} * NUM_SHADOWS))개 섀도우 모델"