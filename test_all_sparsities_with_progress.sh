#!/bin/bash

# 전체 스파시티 레벨 테스트 스크립트 (진행률 표시)
echo "🎯 DCIL-MIA 전체 스파시티 테스트 (진행률 모니터링)"
echo "스파시티: 50%, 70%, 80%, 90%, 95%"
echo "========================================================="

GPU=0
EPOCHS=5  # 테스트용으로 5 에포크
DATASET=cifar10
SEED=42
SPARSITIES=(0.5 0.7 0.8 0.9 0.95)
TOTAL_MODELS=11
CURRENT_MODEL=0

# 에러 발생 시 중단하지 않고 계속 진행
set +e

# 진행률 표시 함수
show_progress() {
    local current=$1
    local total=$2
    local model_name=$3
    local percentage=$((current * 100 / total))
    local filled=$((percentage / 5))
    local empty=$((20 - filled))
    
    printf "\r🚀 전체 진행률: ["
    printf "%${filled}s" | tr ' ' '█'
    printf "%${empty}s" | tr ' ' '░'
    printf "] %d%% (%d/%d) - %s" $percentage $current $total "$model_name"
}

# 실시간 로그 모니터링 함수 (백그라운드)
monitor_log() {
    local log_file=$1
    local model_name=$2
    
    if [ -f "$log_file" ]; then
        echo ""
        echo "📊 $model_name 실시간 로그:"
        echo "------------------------"
        
        # 로그 파일이 생성될 때까지 대기
        while [ ! -f "$log_file" ]; do
            sleep 1
        done
        
        # 실시간으로 에포크 진행 상황 표시
        tail -f "$log_file" | while read line; do
            if [[ "$line" == *"Epoch"* ]] || [[ "$line" == *"Loss"* ]] || [[ "$line" == *"Acc"* ]]; then
                echo "⏰ $(date '+%H:%M:%S') | $line"
            fi
        done &
        LOG_PID=$!
    fi
}

# 로그 모니터링 중지 함수
stop_monitor() {
    if [ ! -z "$LOG_PID" ]; then
        kill $LOG_PID 2>/dev/null
        wait $LOG_PID 2>/dev/null
    fi
}

# 훈련 함수
train_model() {
    local name=$1
    local extra_args="$2"
    local model_type=$3
    
    CURRENT_MODEL=$((CURRENT_MODEL + 1))
    show_progress $CURRENT_MODEL $TOTAL_MODELS "$model_type ($name)"
    
    echo ""
    echo ""
    echo "🔄 $model_type 모델 훈련 시작: $name"
    echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "예상 소요 시간: ~$(( EPOCHS * 2 ))분"
    echo "로그 파일: runs/$name/train.log"
    echo "----------------------------------------"
    
    # 로그 모니터링 시작
    monitor_log "runs/$name/train.log" "$model_type ($name)" &
    MONITOR_PID=$!
    
    # 모델 훈련 실행
    python run_experiment.py \
        --name $name \
        --dataset $DATASET \
        --epochs $EPOCHS \
        --seed $SEED \
        --gpu $GPU \
        --datapath ~/Datasets/CIFAR \
        $extra_args 2>&1 | tee -a "runs/$name/train.log"
    
    local exit_code=$?
    
    # 로그 모니터링 중지
    kill $MONITOR_PID 2>/dev/null
    wait $MONITOR_PID 2>/dev/null
    
    echo ""
    if [ $exit_code -eq 0 ]; then
        echo "✅ $model_type ($name) 훈련 완료 - $(date '+%H:%M:%S')"
    else
        echo "❌ $model_type ($name) 훈련 실패 - 코드: $exit_code - $(date '+%H:%M:%S')"
    fi
    
    # 현재까지 완료된 모델 요약
    echo "📋 현재까지 완료: $CURRENT_MODEL/$TOTAL_MODELS 모델"
    
    return $exit_code
}

echo ""
echo "🕐 실험 시작: $(date '+%Y-%m-%d %H:%M:%S')"
echo "예상 총 소요 시간: ~$(( TOTAL_MODELS * EPOCHS * 2 ))분"
echo ""

# 결과 디렉토리 생성
mkdir -p runs

echo ""
echo "1️⃣  Dense 베이스라인"
echo "=================="

train_model "dense_baseline" "" "Dense"

echo ""
echo "2️⃣  Static 프루닝 (모든 스파시티)"
echo "==============================="

static_count=0
for sparsity in "${SPARSITIES[@]}"; do
    static_count=$((static_count + 1))
    train_model "static_sparsity${sparsity}" "--prune --prune-method static --sparsity $sparsity" "Static ${sparsity}0%"
done

echo ""
echo "3️⃣  DPF 프루닝 (모든 스파시티)"
echo "============================"

dpf_count=0
for sparsity in "${SPARSITIES[@]}"; do
    dpf_count=$((dpf_count + 1))
    train_model "dpf_sparsity${sparsity}" "--prune --prune-method dpf --sparsity $sparsity" "DPF ${sparsity}0%"
done

echo ""
echo ""
echo "🎉 전체 11개 타겟 모델 훈련 완료!"
echo "================================"
echo "완료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "📊 훈련 결과 요약:"
echo "  - Dense: 1개"
echo "  - Static: 5개 (50%, 70%, 80%, 90%, 95%)"
echo "  - DPF: 5개 (50%, 70%, 80%, 90%, 95%)"
echo "  - 총합: 11개 모델"
echo ""

# 결과 요약 생성
echo "📁 결과 파일 위치:"
echo "결과 확인: ls -la runs/"
echo ""
echo "📈 각 모델별 최종 결과:"
echo "----------------------"

for dir in runs/*/; do
    if [ -d "$dir" ]; then
        model_name=$(basename "$dir")
        if [ -f "$dir/train.log" ]; then
            echo "🔍 $model_name:"
            # 마지막 에포크의 정확도 찾기
            last_acc=$(grep -E "(Test|Valid).*Acc" "$dir/train.log" | tail -1 | grep -oE "[0-9]+\.[0-9]+%" || echo "정보 없음")
            echo "   최종 정확도: $last_acc"
            
            # 모델 크기 확인
            if [ -f "$dir/best_model.pth" ]; then
                model_size=$(du -h "$dir/best_model.pth" | cut -f1)
                echo "   모델 크기: $model_size"
            fi
        else
            echo "🔍 $model_name: 로그 파일 없음"
        fi
        echo ""
    fi
done

echo "💾 전체 저장 공간 사용량:"
du -sh runs/ 2>/dev/null || echo "계산 중..."
echo ""

echo "🚀 다음 단계:"
echo "  1. 섀도우 모델 훈련: ./train_all_shadows.sh"
echo "  2. MIA 평가: python experiments/evaluate_lira.py"
echo "  3. 종합 리포트: python scripts/create_report.py"