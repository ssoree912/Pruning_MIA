#!/bin/bash

# 완전한 DCIL-MIA 실험 (200 에포크, 진행률 표시)
echo "🚀 완전한 DCIL-MIA 실험 (200 에포크)"
echo "======================================"
echo "타겟 모델: Dense(1) + Static(5) + DPF(5) = 11개"
echo "스파시티: 50%, 70%, 80%, 90%, 95%"
echo "에포크: 200개 (완전 훈련)"
echo ""

# 설정
GPU=0
DATASET=cifar10
SEED=42
EPOCHS=200        # 완전 실험
NUM_SHADOWS=64    # 완전 실험
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

# 실시간 로그 모니터링 함수
monitor_training() {
    local log_file=$1
    local model_name=$2
    local total_epochs=$3
    
    echo ""
    echo "📊 $model_name 실시간 진행:"
    echo "=========================="
    
    # 로그 파일이 생성될 때까지 대기
    local wait_count=0
    while [ ! -f "$log_file" ] && [ $wait_count -lt 30 ]; do
        sleep 2
        wait_count=$((wait_count + 1))
    done
    
    if [ -f "$log_file" ]; then
        # 실시간으로 에포크 진행 상황 표시
        tail -f "$log_file" | while read line; do
            # 에포크 정보 파싱
            if echo "$line" | grep -q "Epoch"; then
                epoch_info=$(echo "$line" | grep -oE "Epoch [0-9]+/[0-9]+" || echo "")
                if [ ! -z "$epoch_info" ]; then
                    current_epoch=$(echo "$epoch_info" | grep -oE "[0-9]+/[0-9]+" | cut -d'/' -f1)
                    epoch_progress=$((current_epoch * 100 / total_epochs))
                    printf "⏰ %s | %s [%d%%]\n" "$(date '+%H:%M:%S')" "$epoch_info" "$epoch_progress"
                fi
            fi
            
            # 손실/정확도 정보
            if echo "$line" | grep -qE "(Loss|Acc|loss|acc)"; then
                echo "📈 $(date '+%H:%M:%S') | $line"
            fi
        done &
        MONITOR_PID=$!
    fi
}

# 로그 모니터링 중지
stop_monitor() {
    if [ ! -z "$MONITOR_PID" ]; then
        kill $MONITOR_PID 2>/dev/null
        wait $MONITOR_PID 2>/dev/null
    fi
}

# 훈련 함수
train_model() {
    local name=$1
    local extra_args="$2"
    local model_type=$3
    
    CURRENT_MODEL=$((CURRENT_MODEL + 1))
    show_progress $CURRENT_MODEL $TOTAL_MODELS "$model_type"
    
    echo ""
    echo ""
    echo "🔄 $model_type 모델 훈련 시작: $name"
    echo "시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "에포크: $EPOCHS개"
    echo "예상 소요 시간: ~$(( EPOCHS * 3 ))분"
    echo "로그 파일: runs/$name/train.log"
    echo "----------------------------------------"
    
    # 결과 디렉토리 생성
    mkdir -p "runs/$name"
    
    # 로그 모니터링 시작
    monitor_training "runs/$name/train.log" "$model_type" $EPOCHS &
    MONITOR_PID=$!
    
    # 모델 훈련 실행
    python run_experiment.py \
        --name $name \
        --dataset $DATASET \
        --epochs $EPOCHS \
        --seed $SEED \
        --gpu $GPU \
        --datapath ~/Datasets/CIFAR \
        $extra_args > "runs/$name/train.log" 2>&1
    
    local exit_code=$?
    
    # 로그 모니터링 중지
    stop_monitor
    
    echo ""
    if [ $exit_code -eq 0 ]; then
        echo "✅ $model_type 훈련 완료 - $(date '+%H:%M:%S')"
        
        # 최종 결과 표시
        if [ -f "runs/$name/train.log" ]; then
            final_acc=$(grep -E "(Test|Valid).*[Aa]cc" "runs/$name/train.log" | tail -1 | grep -oE "[0-9]+\.[0-9]+%" || echo "정보 없음")
            echo "🎯 최종 정확도: $final_acc"
        fi
    else
        echo "❌ $model_type 훈련 실패 - 코드: $exit_code"
    fi
    
    echo "📋 진행 상황: $CURRENT_MODEL/$TOTAL_MODELS 모델 완료"
    echo ""
    
    return $exit_code
}

echo "🕐 실험 시작: $(date '+%Y-%m-%d %H:%M:%S')"
echo "예상 총 소요 시간: ~$(( TOTAL_MODELS * EPOCHS * 3 / 60 ))시간"
echo ""

# 1단계: 타겟 모델 훈련
echo ""
echo "📚 1단계: 타겟 모델 훈련 (11개)"
echo "=============================="

# Dense 베이스라인
echo ""
echo "1️⃣ Dense 베이스라인"
train_model "dense_baseline" "" "Dense"

# Static 프루닝
echo ""
echo "2️⃣ Static 프루닝 (모든 스파시티)"
for sparsity in "${SPARSITIES[@]}"; do
    train_model "static_sparsity${sparsity}" "--prune --prune-method static --sparsity $sparsity" "Static ${sparsity}0%"
done

# DPF 프루닝  
echo ""
echo "3️⃣ DPF 프루닝 (모든 스파시티)"
for sparsity in "${SPARSITIES[@]}"; do
    train_model "dpf_sparsity${sparsity}" "--prune --prune-method dpf --sparsity $sparsity" "DPF ${sparsity}0%"
done

echo ""
echo "✅ 타겟 모델 훈련 완료"

# 2단계: 섀도우 모델 훈련
echo ""
echo "👥 2단계: 섀도우 모델 훈련"
echo "========================="

echo "섀도우 모델 훈련 시작 - $(date '+%H:%M:%S')"
echo "각 타겟 모델당 $NUM_SHADOWS개 섀도우 모델 훈련"
echo "예상 소요 시간: ~$(( TOTAL_MODELS * NUM_SHADOWS * EPOCHS * 3 / 60 / 60 ))시간"

# Dense 섀도우
echo ""
echo "Dense 섀도우 모델 훈련 중 ($NUM_SHADOWS개)..."
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
for sparsity in "${SPARSITIES[@]}"; do
    echo ""
    echo "Static 섀도우 모델 훈련 중: ${sparsity}0% ($NUM_SHADOWS개)..."
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
for sparsity in "${SPARSITIES[@]}"; do
    echo ""
    echo "DPF 섀도우 모델 훈련 중: ${sparsity}0% ($NUM_SHADOWS개)..."
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

# 3단계: LiRA MIA 평가
echo ""
echo "🔍 3단계: LiRA MIA 평가"
echo "====================="

echo "MIA 평가 시작 - $(date '+%H:%M:%S')"
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

# 4단계: 종합 리포트
echo ""
echo "📊 4단계: 종합 리포트 생성"
echo "========================"

python scripts/create_report.py \
    --results-dir ./runs \
    --output-dir ./final_results \
    --title "Dense vs Static vs DPF: Complete MIA Vulnerability Analysis"

echo "✅ 리포트 생성 완료"

# 최종 요약
echo ""
echo "🎉 완전한 실험 완료!"
echo "===================="
echo "완료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "📋 실험 결과 요약:"
echo "  타겟 모델: 11개 (200 에포크 각각)"
echo "  섀도우 모델: $((TOTAL_MODELS * NUM_SHADOWS))개"
echo "  MIA 평가: LiRA 완료"
echo ""
echo "📁 결과 위치:"
echo "  - 타겟 모델: ./runs/"
echo "  - 섀도우 모델: ./runs/shadows/"
echo "  - MIA 결과: ./results/lira/"
echo "  - 종합 리포트: ./final_results/"
echo "  - HTML 리포트: ./final_results/report.html"
echo ""

# 최종 결과 요약
echo "📈 모델별 최종 성능:"
echo "------------------"
for dir in runs/*/; do
    if [ -d "$dir" ] && [ -f "$dir/train.log" ]; then
        model_name=$(basename "$dir")
        final_acc=$(grep -E "(Test|Valid).*[Aa]cc" "$dir/train.log" | tail -1 | grep -oE "[0-9]+\.[0-9]+%" || echo "N/A")
        printf "%-20s: %s\n" "$model_name" "$final_acc"
    fi
done

echo ""
echo "💾 저장 공간 사용량:"
du -sh ./runs/ ./results/ ./final_results/ 2>/dev/null

echo ""
echo "🏁 완전한 Dense vs Static vs DPF 실험이 완료되었습니다!"