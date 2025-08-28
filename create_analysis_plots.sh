#!/bin/bash

# Dense vs Static vs DPF 결과 분석 및 시각화
echo "🎨 Dense vs Static vs DPF 결과 분석 및 시각화"
echo "=============================================="

# 결과 디렉토리 생성
mkdir -p results/visualization

# Python 시각화 스크립트 실행
echo "📊 실험 결과 시각화 중..."
python visualize_results.py \
    --csv-path ./runs/final_report/experiments_comparison.csv \
    --output-dir ./results/visualization

echo ""
echo "✅ 시각화 완료!"
echo ""

echo "📋 생성된 분석 자료:"
echo "=================="
echo ""

if [ -d "results/visualization" ]; then
    echo "📈 그래프들:"
    echo "  1. accuracy_comparison.png - 정확도 vs 스파시티 비교"
    echo "  2. efficiency_analysis.png - 훈련 효율성 분석"  
    echo "  3. sparsity_analysis.png - 스파시티별 상세 비교"
    echo "  4. comprehensive_dashboard.png - 종합 대시보드"
    echo ""
    
    echo "📄 리포트:"
    echo "  5. analysis_report.txt - 상세 텍스트 분석 리포트"
    echo ""
    
    echo "🔍 주요 분석 내용:"
    echo "  ✓ Dense 베이스라인 vs 프루닝 방법 성능 비교"
    echo "  ✓ Static vs DPF 정확도 비교 (스파시티별)"
    echo "  ✓ 훈련 시간 vs 성능 효율성 분석"
    echo "  ✓ 파라미터 효율성 분석"
    echo "  ✓ DPF의 Static 대비 장점 분석"
    echo ""
    
    echo "📊 결과 미리보기:"
    if [ -f "results/visualization/analysis_report.txt" ]; then
        echo "=========================================="
        head -20 results/visualization/analysis_report.txt
        echo ""
        echo "... (전체 리포트는 results/visualization/analysis_report.txt 참조)"
        echo "=========================================="
    fi
    
    echo ""
    echo "🖼️  생성된 그래프들:"
    ls -la results/visualization/*.png 2>/dev/null || echo "  그래프 파일들 확인 중..."
    
else
    echo "❌ 시각화 디렉토리가 생성되지 않았습니다."
fi

echo ""
echo "💡 다음 단계:"
echo "  1. 그래프 확인: open results/visualization/"
echo "  2. 리포트 읽기: cat results/visualization/analysis_report.txt"
echo "  3. MIA 결과와 함께 분석: 성능 vs 프라이버시 트레이드오프"
echo ""
echo "🏁 분석 완료!"