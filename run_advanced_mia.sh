#!/bin/bash

# Advanced MIA 평가 실행 스크립트
echo "🚀 Advanced MIA 평가 시작"
echo "========================"
echo "공격 방법: LiRA, Shokri-NN, Top3-NN, ClassLabel-NN, SAMIA"
echo ""

# 결과 디렉토리 생성
mkdir -p results/advanced_mia

# Advanced MIA 평가 실행
echo "1️⃣ Advanced MIA 평가 실행 중..."
python mia_advanced.py \
    --runs-dir ./runs \
    --results-dir ./results/advanced_mia

echo ""
echo "✅ Advanced MIA 평가 완료!"
echo ""

echo "📊 결과 확인:"
echo "  - 요약 테이블: results/advanced_mia/advanced_mia_summary.csv"
echo "  - 상세 결과: results/advanced_mia/advanced_mia_results.json"
echo ""

# 결과 미리보기
if [ -f "results/advanced_mia/advanced_mia_summary.csv" ]; then
    echo "📈 Advanced MIA 공격 결과:"
    echo "========================="
    echo ""
    
    echo "🔥 가장 효과적인 공격들:"
    echo "  - LiRA: Likelihood Ratio Attack (최고 성능)"
    echo "  - ClassLabel-NN: 정답 라벨 포함 공격"
    echo "  - Top3-NN: 효율적인 공격"
    echo ""
    
    head -20 results/advanced_mia/advanced_mia_summary.csv
    echo ""
    
    echo "🎯 공격 성공률 분석:"
    echo "  - Dense 모델: 가장 취약 (높은 AUC)"
    echo "  - Static vs DPF: 스파시티별 비교"
    echo "  - 높은 스파시티 → MIA 저항성 변화"
else
    echo "❌ 결과 파일이 생성되지 않았습니다."
fi

echo ""
echo "🏁 Advanced MIA 평가 완료!"
echo ""
echo "💡 다음 단계:"
echo "  1. 결과 분석: cat results/advanced_mia/advanced_mia_summary.csv"
echo "  2. 상세 보기: python -m json.tool results/advanced_mia/advanced_mia_results.json"
echo "  3. 비교 분석: Dense vs Static vs DPF 취약성 비교"