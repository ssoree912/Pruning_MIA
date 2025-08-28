#!/bin/bash

# WeMeM-style MIA 평가 실행 스크립트
echo "🔍 WeMeM-style MIA 평가 시작"
echo "=========================="

# 결과 디렉토리 생성
mkdir -p results/wemem_mia

# WeMeM MIA 평가 실행
echo "1️⃣ WeMeM MIA 평가 실행 중..."
python mia_wemem.py \
    --runs-dir ./runs \
    --results-dir ./results/wemem_mia

echo ""
echo "✅ MIA 평가 완료!"
echo ""

echo "📊 결과 확인:"
echo "  - 요약 테이블: results/wemem_mia/wemem_mia_summary.csv"
echo "  - 상세 결과: results/wemem_mia/wemem_mia_results.json"
echo ""

# 결과 미리보기
if [ -f "results/wemem_mia/wemem_mia_summary.csv" ]; then
    echo "📈 MIA 취약성 요약:"
    echo "===================="
    head -20 results/wemem_mia/wemem_mia_summary.csv
    echo ""
else
    echo "❌ 결과 파일이 생성되지 않았습니다."
fi

echo "🏁 MIA 평가 완료!"