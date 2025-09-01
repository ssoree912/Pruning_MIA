#!/bin/bash

# 훈련 전용 스크립트 (MIA 없이)
echo "🚀 Dense vs Static vs DPF 모델 훈련"
echo "=================================="
echo "타겟 모델: Dense(1) + Static(5) + DPF(5) = 11개"
echo "에포크: 200개 (완전 훈련)"
echo ""

# 설정
EPOCHS=200
GPU=0
DATASET=cifar10
SEED=42
SPARSITIES=(0.5 0.7 0.8 0.9 0.95)
TOTAL_MODELS=11
CURRENT_MODEL=0

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
    printf "] %d%% (%d/%d) - %s\n" $percentage $current $total "$model_name"
}

# 훈련 함수
train_model() {
    local name=$1
    local extra_args="$2"
    local model_type=$3
    
    CURRENT_MODEL=$((CURRENT_MODEL + 1))
    show_progress $CURRENT_MODEL $TOTAL_MODELS "$model_type"
    
    echo ""
    echo "🔄 $model_type 모델 훈련 시작: $name"
    echo "시작 시간: $(date '+%H:%M:%S')"
    echo "에포크: $EPOCHS개"
    echo "예상 소요 시간: ~30분"
    echo "----------------------------------------"
    
    # 모델 훈련 실행
    python run_experiment.py \
        --name $name \
        --dataset $DATASET \
        --epochs $EPOCHS \
        --seed $SEED \
        --gpu $GPU \
        --datapath ~/Datasets/CIFAR \
        $extra_args
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $model_type 훈련 완료 - $(date '+%H:%M:%S')"
    else
        echo "❌ $model_type 훈련 실패 - 코드: $exit_code"
    fi
    
    echo "📋 진행 상황: $CURRENT_MODEL/$TOTAL_MODELS 모델 완료"
    echo ""
    
    return $exit_code
}

echo "🕐 실험 시작: $(date '+%Y-%m-%d %H:%M:%S')"
echo "예상 총 소요 시간: ~6시간"
echo ""

# Dense 베이스라인 (2개 시드)
echo ""
echo "1️⃣ Dense 베이스라인"
train_model "dense_seed42" "" "Dense"
train_model "dense_seed123" "--seed 123" "Dense (seed 123)"

# Static 프루닝
echo ""
echo "2️⃣ Static 프루닝 (모든 스파시티)"
for sparsity in "${SPARSITIES[@]}"; do
    train_model "static_sparsity${sparsity}_seed42" "--prune --prune-method static --sparsity $sparsity" "Static ${sparsity}0%"
done

# DPF 프루닝
echo ""
echo "3️⃣ DPF 프루닝 (모든 스파시티)"
for sparsity in "${SPARSITIES[@]}"; do
    train_model "dpf_sparsity${sparsity}_seed42" "--prune --prune-method dpf --sparsity $sparsity" "DPF ${sparsity}0%"
done

echo ""
echo "✅ 모든 모델 훈련 완료!"
echo "완료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 훈련 결과 수집
echo "📊 훈련 결과 수집 중..."
python -c "
import os
import json
import pandas as pd
from pathlib import Path

runs_dir = Path('./runs')
results = []

for model_dir in runs_dir.iterdir():
    if model_dir.is_dir() and (model_dir / 'config.json').exists():
        config_path = model_dir / 'config.json'
        
        try:
            with open(config_path) as f:
                config = json.load(f)
            
            log_path = model_dir / 'experiment.log'
            best_acc = 0.0
            
            if log_path.exists():
                with open(log_path) as f:
                    for line in f:
                        if 'Best accuracy' in line:
                            try:
                                best_acc = float(line.split(':')[-1].strip())
                            except:
                                pass
            
            results.append({
                'name': config['name'],
                'method': config.get('pruning', {}).get('method', 'dense'),
                'sparsity': config.get('pruning', {}).get('sparsity', 0.0),
                'best_acc1': best_acc,
                'epochs': config.get('training', {}).get('epochs', 0)
            })
        except Exception as e:
            print(f'Error processing {model_dir}: {e}')

os.makedirs('./runs/final_report', exist_ok=True)
df = pd.DataFrame(results)
df.to_csv('./runs/final_report/experiments_comparison.csv', index=False)
print(f'✅ 결과 수집 완료: {len(results)}개 모델')
"

echo ""
echo "📈 모델별 최종 성능:"
echo "------------------"
if [ -f "runs/final_report/experiments_comparison.csv" ]; then
    echo "정확도 순위:"
    python -c "
import pandas as pd
import ast

df = pd.read_csv('runs/final_report/experiments_comparison.csv')
df['pruning_config'] = df['pruning'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else {})
df['method'] = df['pruning_config'].apply(lambda x: 'Dense' if not x.get('enabled', False) else x.get('method', 'unknown').upper())
df['sparsity'] = df['pruning_config'].apply(lambda x: x.get('sparsity', 0.0) if x.get('enabled', False) else 0.0)

print('🏆 Top 5 모델:')
top_models = df.nlargest(5, 'best_acc1')[['name', 'method', 'sparsity', 'best_acc1']]
for _, row in top_models.iterrows():
    sparsity_text = f'{row.sparsity*100:.0f}%' if row.sparsity > 0 else '0%'
    print(f'  {row.name}: {row.best_acc1:.2f}% ({row.method} {sparsity_text})')
"
fi

echo ""
echo "📁 결과 위치:"
echo "  - 모델 파일: ./runs/"
echo "  - 훈련 결과: ./runs/final_report/"
echo ""
echo "🎯 다음 단계:"
echo "  - MIA 평가: ./run_mia_evaluation.sh"
echo ""
echo "🏁 훈련 완료!"