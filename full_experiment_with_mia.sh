#!/bin/bash

# 완전한 DCIL + WeMeM MIA 실험
echo "🚀 완전한 DCIL + WeMeM MIA 실험"
echo "================================"
echo "1단계: 11개 타겟 모델 훈련"
echo "2단계: WeMeM MIA 평가"
echo "3단계: 종합 결과 분석"
echo ""

# 설정
EPOCHS=5    # 테스트용 (실제는 200)
GPU=0

echo "🕐 실험 시작: $(date)"
echo ""

# 1단계: 타겟 모델 훈련
echo "📚 1단계: 타겟 모델 훈련"
echo "======================"

# Dense 모델
echo "Dense 모델 훈련 중..."
python run_experiment.py \
    --name dense_baseline \
    --dataset cifar10 \
    --epochs $EPOCHS \
    --gpu $GPU \
    --datapath ~/Datasets/CIFAR

# Static 모델들
for sparsity in 0.5 0.7 0.8 0.9 0.95; do
    echo "Static ${sparsity}0% 훈련 중..."
    python run_experiment.py \
        --name static_sparsity${sparsity} \
        --dataset cifar10 \
        --prune \
        --prune-method static \
        --sparsity $sparsity \
        --epochs $EPOCHS \
        --gpu $GPU \
        --datapath ~/Datasets/CIFAR
done

# DPF 모델들  
for sparsity in 0.5 0.7 0.8 0.9 0.95; do
    echo "DPF ${sparsity}0% 훈련 중..."
    python run_experiment.py \
        --name dpf_sparsity${sparsity} \
        --dataset cifar10 \
        --prune \
        --prune-method dpf \
        --sparsity $sparsity \
        --epochs $EPOCHS \
        --gpu $GPU \
        --datapath ~/Datasets/CIFAR
done

echo "✅ 타겟 모델 훈련 완료"

# 2단계: WeMeM MIA 평가
echo ""
echo "🔍 2단계: WeMeM MIA 평가"
echo "======================"

mkdir -p results/wemem_mia

python mia_wemem.py \
    --runs-dir ./runs \
    --results-dir ./results/wemem_mia

echo "✅ WeMeM MIA 평가 완료"

# 3단계: 결과 분석
echo ""
echo "📊 3단계: 결과 분석"
echo "=================="

echo "훈련된 모델 개수: $(ls -d runs/*/ | wc -l)"
echo ""

if [ -f "results/wemem_mia/wemem_mia_summary.csv" ]; then
    echo "📈 MIA 취약성 비교:"
    echo "=================="
    echo ""
    
    # Dense vs Static vs DPF 비교
    echo "🔵 Dense 모델:"
    grep "dense" results/wemem_mia/wemem_mia_summary.csv || echo "데이터 없음"
    echo ""
    
    echo "🔴 Static 모델들:"
    grep "static" results/wemem_mia/wemem_mia_summary.csv || echo "데이터 없음"
    echo ""
    
    echo "🟢 DPF 모델들:"
    grep "dpf" results/wemem_mia/wemem_mia_summary.csv || echo "데이터 없음"
    echo ""
    
    echo "📋 전체 요약:"
    head -20 results/wemem_mia/wemem_mia_summary.csv
else
    echo "❌ MIA 결과가 생성되지 않았습니다."
fi

echo ""
echo "🎉 전체 실험 완료!"
echo "================="
echo "완료 시간: $(date)"
echo ""
echo "📁 결과 위치:"
echo "  - 모델: ./runs/"
echo "  - MIA 결과: ./results/wemem_mia/"
echo ""
echo "📊 주요 결과:"
echo "  - 모델 성능: runs/*/experiment_summary.json"
echo "  - MIA 취약성: results/wemem_mia/wemem_mia_summary.csv"
echo "  - 상세 MIA: results/wemem_mia/wemem_mia_results.json"