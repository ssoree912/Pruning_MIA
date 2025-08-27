#!/bin/bash

# 전체 시스템 테스트 스크립트
echo "🧪 DCIL-MIA 전체 시스템 테스트 시작"
echo "=================================="

GPU=0
EPOCHS=2
DATASET=cifar10
SEED=42

# 에러 발생 시 중단
set -e

echo ""
echo "1️⃣  개별 모델 테스트 (각 2 에포크)"
echo "================================="

echo "Dense 모델 테스트 중..."
python run_experiment.py \
    --name test_dense \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --seed $SEED \
    --gpu $GPU \
    --datapath ~/Datasets/CIFAR

echo "✅ Dense 모델 완료"

echo "Static 프루닝 테스트 중..."
python run_experiment.py \
    --name test_static_80 \
    --dataset $DATASET \
    --prune \
    --prune-method static \
    --sparsity 0.8 \
    --epochs $EPOCHS \
    --seed $SEED \
    --gpu $GPU \
    --datapath ~/Datasets/CIFAR

echo "✅ Static 프루닝 완료"

echo "DPF 프루닝 테스트 중..."
python run_experiment.py \
    --name test_dpf_80 \
    --dataset $DATASET \
    --prune \
    --prune-method dpf \
    --sparsity 0.8 \
    --epochs $EPOCHS \
    --seed $SEED \
    --gpu $GPU \
    --datapath ~/Datasets/CIFAR

echo "✅ DPF 프루닝 완료"

echo ""
echo "2️⃣  전체 실험 파이프라인 테스트"
echo "============================="

echo "전체 실험 파이프라인 테스트 중 (5 에포크, 축소된 스파시티)..."
python scripts/run_full_experiment.py \
    --dataset $DATASET \
    --seeds $SEED \
    --epochs 3 \
    --gpus $GPU \
    --sparsity-levels 0.8 0.9 \
    --datapath ~/Datasets/CIFAR

echo "✅ 전체 실험 파이프라인 완료"

echo ""
echo "3️⃣  섀도우 모델 테스트"
echo "==================="

echo "Dense 섀도우 모델 테스트 중..."
python experiments/train_shadows.py \
    --dataset $DATASET \
    --arch resnet \
    --layers 20 \
    --model-type dense \
    --epochs 3 \
    --num-shadows 2 \
    --gpu $GPU \
    --datapath ~/Datasets/CIFAR

echo "✅ 섀도우 모델 테스트 완료"

echo ""
echo "4️⃣  MIA 평가 테스트"
echo "=================="

echo "LiRA 평가 테스트 중..."
python experiments/evaluate_lira.py \
    --dataset $DATASET \
    --arch resnet \
    --layers 20 \
    --target-models-dir ./runs \
    --shadow-models-dir ./runs/shadows \
    --results-dir ./test_mia_results \
    --num-shadows 2 \
    --gpu $GPU \
    --batch-size 128 \
    --datapath ~/Datasets/CIFAR

echo "✅ MIA 평가 테스트 완료"

echo ""
echo "5️⃣  리포트 생성 테스트"
echo "==================="

echo "결과 리포트 생성 중..."
python scripts/create_report.py \
    --results-dir ./runs \
    --output-dir ./test_report \
    --title "DCIL-MIA 시스템 테스트 리포트"

echo "✅ 리포트 생성 완료"

echo ""
echo "🎉 전체 시스템 테스트 완료!"
echo "========================="
echo "테스트 결과:"
echo "  - 개별 모델: 3개 완료"
echo "  - 전체 파이프라인: 완료"
echo "  - 섀도우 모델: 완료" 
echo "  - MIA 평가: 완료"
echo "  - 리포트: ./test_report/ 에서 확인"
echo ""
echo "📊 결과 확인:"
echo "  - 훈련 로그: ./runs/ 디렉토리"
echo "  - MIA 결과: ./test_mia_results/ 디렉토리" 
echo "  - 종합 리포트: ./test_report/report.html"