#!/bin/bash

# 완전한 DCIL-MIA 실험 (11개 타겟 모델 + MIA)
echo "🚀 완전한 DCIL-MIA 실험"
echo "======================="
echo "타겟 모델: Dense(1) + Static(5) + DPF(5) = 11개"
echo "스파시티: 50%, 70%, 80%, 90%, 95%"
echo "MIA: LiRA 평가 포함"
echo ""

# 설정
GPU=0
DATASET=cifar10
SEED=42
EPOCHS_QUICK=5    # 빠른 테스트용
EPOCHS_FULL=200   # 실제 실험용
NUM_SHADOWS=8     # 테스트용 (실제: 64)

# 사용자 선택
echo "실험 모드를 선택하세요:"
echo "1) 빠른 테스트 (5 에포크, 8 섀도우 모델)"
echo "2) 전체 실험 (200 에포크, 64 섀도우 모델)"
read -p "선택 [1/2]: " choice

if [ "$choice" = "2" ]; then
    EPOCHS=$EPOCHS_FULL
    NUM_SHADOWS=64
    echo "🎯 전체 실험 모드 선택"
else
    EPOCHS=$EPOCHS_QUICK
    NUM_SHADOWS=8
    echo "⚡ 빠른 테스트 모드 선택"
fi

echo "설정:"
echo "  - 에포크: $EPOCHS"
echo "  - 섀도우 모델 개수: $NUM_SHADOWS"
echo "  - GPU: $GPU"
echo ""

# 에러 발생 시 중단
set -e

echo ""
echo "📚 1단계: 타겟 모델 훈련 (11개)"
echo "=============================="

python scripts/run_full_experiment.py \
    --dataset $DATASET \
    --seeds $SEED \
    --epochs $EPOCHS \
    --gpus $GPU \
    --sparsity-levels 0.5 0.7 0.8 0.9 0.95 \
    --datapath ~/Datasets/CIFAR

echo "✅ 타겟 모델 훈련 완료"

echo ""
echo "👥 2단계: 섀도우 모델 훈련"
echo "========================="

# Dense 섀도우
echo "Dense 섀도우 모델 훈련 중..."
python experiments/train_shadows.py \
    --dataset $DATASET \
    --arch resnet \
    --layers 20 \
    --model-type dense \
    --epochs $EPOCHS \
    --num-shadows $NUM_SHADOWS \
    --gpu $GPU \
    --datapath ~/Datasets/CIFAR

# Static 섀도우
for sparsity in 0.5 0.7 0.8 0.9 0.95; do
    echo "Static 섀도우 모델 훈련 중: ${sparsity}..."
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
done

# DPF 섀도우
for sparsity in 0.5 0.7 0.8 0.9 0.95; do
    echo "DPF 섀도우 모델 훈련 중: ${sparsity}..."
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
done

echo "✅ 섀도우 모델 훈련 완료"

echo ""
echo "🔍 3단계: LiRA MIA 평가"
echo "====================="

python experiments/evaluate_lira.py \
    --dataset $DATASET \
    --arch resnet \
    --layers 20 \
    --target-models-dir ./runs \
    --shadow-models-dir ./runs/shadows \
    --results-dir ./results/lira \
    --num-shadows $NUM_SHADOWS \
    --gpu $GPU \
    --batch-size 128 \
    --datapath ~/Datasets/CIFAR

echo "✅ LiRA 평가 완료"

echo ""
echo "📊 4단계: 종합 리포트 생성"
echo "========================"

python scripts/create_report.py \
    --results-dir ./runs \
    --output-dir ./final_results \
    --title "Dense vs Static vs DPF: MIA Vulnerability Analysis"

echo "✅ 리포트 생성 완료"

echo ""
echo "🎉 실험 완료!"
echo "============"
echo ""
echo "📋 결과 요약:"
echo "  타겟 모델: 11개 (Dense 1개 + Static 5개 + DPF 5개)"
echo "  섀도우 모델: $((11 * NUM_SHADOWS))개 (타겟당 ${NUM_SHADOWS}개)"
echo "  스파시티 레벨: 50%, 70%, 80%, 90%, 95%"
echo ""
echo "📁 결과 위치:"
echo "  - 모델 체크포인트: ./runs/"
echo "  - MIA 결과: ./results/lira/"
echo "  - 종합 리포트: ./final_results/"
echo "  - HTML 리포트: ./final_results/report.html"
echo ""
echo "📈 주요 메트릭:"
echo "  - 유틸리티: 정확도 vs 스파시티"
echo "  - 프라이버시: LiRA AUC, TPR@FPR"
echo "  - 비교: Dense vs Static vs DPF"

# 결과 파일 크기 확인
echo ""
echo "💾 저장 공간 사용량:"
du -sh ./runs/ 2>/dev/null || echo "  ./runs/: 계산 중..."
du -sh ./results/ 2>/dev/null || echo "  ./results/: 계산 중..."
du -sh ./final_results/ 2>/dev/null || echo "  ./final_results/: 계산 중..."