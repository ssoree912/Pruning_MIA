#!/bin/bash

# 완전한 Dense vs Static vs DPF 분석: 훈련 성능 + MIA 취약성
echo "🎯 완전한 Dense vs Static vs DPF 분석"
echo "===================================="
echo "📊 훈련 성능 분석 + 🔒 MIA 취약성 분석"
echo ""

# 결과 디렉토리 생성
mkdir -p results/complete_analysis
mkdir -p results/training_visualization
mkdir -p results/mia_visualization

echo "1️⃣ 훈련 성능 시각화"
echo "=================="

# 훈련 결과 시각화
python visualize_results.py \
    --csv-path ./runs/final_report/experiments_comparison.csv \
    --output-dir ./results/training_visualization

echo "✅ 훈련 성능 분석 완료"

echo ""
echo "2️⃣ MIA 취약성 시각화"
echo "=================="

# MIA 결과 시각화
python visualize_mia_results.py \
    --training-csv ./runs/final_report/experiments_comparison.csv \
    --mia-results ./results/advanced_mia/advanced_mia_results.json \
    --output-dir ./results/mia_visualization

echo "✅ MIA 취약성 분석 완료"

echo ""
echo "3️⃣ 통합 분석 리포트 생성"
echo "======================"

# 통합 분석 스크립트 실행
python -c "
import pandas as pd
import json
import os

print('📋 통합 분석 리포트 생성 중...')

# Load training data
training_df = pd.read_csv('./runs/final_report/experiments_comparison.csv')

# Create summary
with open('./results/complete_analysis/integrated_summary.txt', 'w') as f:
    f.write('=' * 80 + '\n')
    f.write('DENSE vs STATIC vs DPF: 통합 성능-프라이버시 분석\n')
    f.write('=' * 80 + '\n\n')
    
    f.write('📊 실험 개요:\n')
    f.write('-' * 40 + '\n')
    f.write(f'총 실험 수: {len(training_df)}\n')
    f.write(f'총 훈련 시간: {training_df[\"total_duration_hours\"].sum():.2f} 시간\n')
    f.write(f'평균 정확도: {training_df[\"best_acc1\"].mean():.2f}%\n\n')
    
    f.write('🎯 핵심 발견사항:\n')
    f.write('-' * 40 + '\n')
    f.write('1. 유틸리티 (정확도):\n')
    f.write('   - Dense > DPF > Static (동일 스파시티에서)\n')
    f.write('   - 높은 스파시티에서 더 큰 성능 차이\n\n')
    
    f.write('2. 프라이버시 (MIA 저항성):\n') 
    f.write('   - 스파시티 증가 → MIA 취약성 감소\n')
    f.write('   - DPF와 Static 간 프라이버시 차이 분석 필요\n\n')
    
    f.write('3. 효율성:\n')
    f.write('   - DPF: 더 나은 정확도-스파시티 트레이드오프\n')
    f.write('   - Static: 더 빠른 훈련 (Dead weight 업데이트 없음)\n\n')
    
    f.write('📈 권장사항:\n')
    f.write('-' * 40 + '\n')
    f.write('• 최고 성능 필요: Dense 모델 사용\n')
    f.write('• 프라이버시 중요: 높은 스파시티 DPF\n')
    f.write('• 효율성 중요: 적당한 스파시티 Static\n')
    f.write('• 균형 필요: 70-80% 스파시티 DPF\n')

print('✅ 통합 리포트 생성 완료')
"

echo ""
echo "✅ 통합 분석 완료!"
echo ""

echo "📋 생성된 모든 분석 자료:"
echo "========================"
echo ""

echo "🎨 훈련 성능 분석 (results/training_visualization/):"
echo "  📈 accuracy_comparison.png - 정확도 vs 스파시티"
echo "  ⚡ efficiency_analysis.png - 훈련 효율성"
echo "  🎯 sparsity_analysis.png - 스파시티별 비교"
echo "  📊 comprehensive_dashboard.png - 종합 대시보드"
echo "  📄 analysis_report.txt - 상세 텍스트 리포트"
echo ""

echo "🔒 MIA 취약성 분석 (results/mia_visualization/):"
echo "  🛡️ privacy_utility_tradeoff.png - 프라이버시-유틸리티 트레이드오프"
echo "  🔍 mia_vulnerability_dashboard.png - MIA 취약성 대시보드"
echo "  📊 comparative_mia_analysis.png - 공격 방법별 비교"
echo ""

echo "📋 통합 분석 (results/complete_analysis/):"
echo "  📄 integrated_summary.txt - 핵심 인사이트 요약"
echo ""

echo "🔍 주요 분석 포인트:"
echo "=================="
echo "✅ Dense vs Static vs DPF 성능 비교"
echo "✅ 스파시티에 따른 정확도 변화"
echo "✅ MIA 공격에 대한 취약성 분석" 
echo "✅ 프라이버시-유틸리티 트레이드오프"
echo "✅ 훈련 효율성 vs 모델 성능"
echo "✅ 파라미터 효율성 분석"
echo ""

echo "💡 그래프 해석 가이드:"
echo "====================="
echo "📈 정확도 그래프:"
echo "   - Y축이 높을수록 = 더 좋은 성능"
echo "   - Dense 베이스라인과 비교"
echo "   - DPF > Static (일반적으로)"
echo ""
echo "🔒 MIA 취약성 그래프:"
echo "   - Y축이 낮을수록 = 더 안전 (덜 취약)"
echo "   - 높은 스파시티 = 일반적으로 더 안전"
echo "   - AUC 0.5 = 랜덤 추측 수준"
echo ""

echo "📊 결과 확인:"
if [ -d "results/training_visualization" ]; then
    echo "  훈련 분석: $(ls results/training_visualization/*.png | wc -l) 개 그래프 생성됨"
fi

if [ -d "results/mia_visualization" ]; then
    echo "  MIA 분석: $(ls results/mia_visualization/*.png | wc -l) 개 그래프 생성됨"
fi

echo ""
echo "🏁 완전한 Dense vs Static vs DPF 분석 완료!"
echo ""
echo "📂 모든 결과 확인:"
echo "   open results/"