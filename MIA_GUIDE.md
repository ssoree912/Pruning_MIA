# DWA 프루닝된 모델에 대한 MIA 평가 가이드

이 가이드는 train_dwa.py로 훈련된 DWA 프루닝 모델들에 대해 Membership Inference Attack (MIA) 평가를 수행하는 방법을 설명합니다.

## 📋 개요

**MIA (Membership Inference Attack)**는 특정 데이터 포인트가 모델 훈련에 사용되었는지 추론하는 공격입니다. 이 파이프라인은 DWA (Dynamic Weight Adjustment) 프루닝이 모델의 프라이버시 누출에 미치는 영향을 평가합니다.

## 🚀 빠른 시작

### 1. 전체 파이프라인 한 번에 실행

```bash
# DWA 모델들 훈련 (먼저 필요)
python train_dwa.py --dwa-modes reactivate_only kill_active_plain_dead kill_and_reactivate \
    --sparsities 0.5 0.8 0.9 --dataset cifar10 --epochs 50

# MIA 평가 전체 파이프라인 실행 (기존 스타일)
python run_mia_pipeline.py --dataset cifar10 --device cuda:0

# 또는 WeMeM-main 스타일로 실행
python evaluate_dwa_mia_wemem.py 0 configs/cifar10_resnet18.json \
    --dataset_name cifar10 --model_name resnet18 \
    --attacks samia,threshold,nn,nn_top3,nn_cls \
    --runs_dir ./runs --output_dir ./mia_results_dwa
```

### 2. WeMeM-main 스타일 개별 실행

```bash
# Step 1: DWA 모델 훈련 결과 확인
ls runs/dwa/

# Step 2: WeMeM-main 스타일 MIA 평가
python evaluate_dwa_mia_wemem.py 0 configs/cifar10_resnet18.json \
    --dataset_name cifar10 --model_name resnet18 \
    --attacks samia,threshold,nn,nn_top3,nn_cls \
    --shadow_num 2

# Step 3: 결과 확인  
ls ./mia_results_dwa/
```

### 3. 기존 스타일 개별 실행

```bash
# Step 1: MIA용 데이터 분할 준비
python scripts/prepare_mia_data.py --dataset cifar10 --output_dir ./mia_data

# Step 2: DWA 모델들에 대해 MIA 평가 
python evaluate_dwa_mia.py --runs_dir ./runs --output_dir ./mia_results --dataset cifar10

# Step 3: 결과 확인
ls ./mia_results/
```

## 📁 파일 구조

```
prunning/
├── train_dwa.py                 # DWA 모델 훈련 (기존)
├── run_mia_pipeline.py          # 전체 MIA 파이프라인
├── evaluate_dwa_mia.py          # 메인 MIA 평가 스크립트
├── mia_utils.py                 # MIA 관련 유틸리티
├── attacker_threshold.py        # 임계값 기반 MIA 공격
├── base_model.py               # 모델 기본 클래스 (기존)
├── scripts/
│   └── prepare_mia_data.py     # MIA용 데이터 분할
├── runs/                       # DWA 훈련 결과 (자동 생성)
│   └── dwa/
│       ├── reactivate_only/
│       ├── kill_active_plain_dead/
│       └── kill_and_reactivate/
├── mia_data/                   # MIA용 데이터 분할 (자동 생성)
└── mia_results/                # MIA 평가 결과 (자동 생성)
```

## 🔬 MIA 공격 방법들

이 파이프라인은 다음 4가지 MIA 공격을 구현합니다:

1. **Confidence (GT)**: 정답 클래스에 대한 모델 확신도 기반
2. **Entropy**: 예측 분포의 엔트로피 기반  
3. **Modified Entropy**: Song et al.의 개선된 엔트로피 메트릭
4. **Confidence (Top-1)**: 최대 확신도 기반

## 📊 결과 해석

### 출력 파일들

- `dwa_mia_results_YYYYMMDD_HHMMSS.csv`: 전체 결과 CSV
- `dwa_mia_results_YYYYMMDD_HHMMSS.json`: 전체 결과 JSON
- `runs/dwa/.../mia_evaluation.json`: 개별 모델별 결과

### 주요 메트릭

- **attack_conf_gt**: 정답 확신도 기반 공격 정확도
- **attack_entropy**: 엔트로피 기반 공격 정확도  
- **attack_modified_entropy**: 수정된 엔트로피 기반 공격 정확도
- **attack_conf_top1**: Top-1 확신도 기반 공격 정확도
- **confidence_gap**: Member와 Non-member 간 확신도 차이
- **sparsity_actual**: 실제 측정된 모델 희소성

### 해석 가이드

- **공격 정확도 > 0.5**: 모델이 멤버십 정보를 누출함 (높을수록 위험)
- **공격 정확도 ≈ 0.5**: 랜덤 추측 수준 (안전)
- **confidence_gap > 0**: Member 데이터에 대해 더 높은 확신도 (위험)

## ⚙️ 고급 사용법

### 1. 커스텀 데이터 분할

```bash
python scripts/prepare_mia_data.py \
    --dataset cifar10 \
    --victim_ratio 0.3 \
    --shadow_ratio 0.5 \
    --test_ratio 0.1 \
    --seed 12345
```

### 2. 특정 DWA 모드만 평가

runs 디렉토리에서 원하지 않는 모드 폴더를 임시로 이동하고 평가를 수행할 수 있습니다.

### 3. 배치 크기 및 디바이스 조정

```bash
python evaluate_dwa_mia.py \
    --device cuda:1 \
    --batch_size 64 \
    --runs_dir ./runs \
    --output_dir ./custom_results
```

## 🐛 문제 해결

### 1. "No DWA models found" 오류

```bash
# DWA 모델이 없는 경우
python train_dwa.py --dwa-modes reactivate_only --sparsities 0.5 --epochs 10
```

### 2. CUDA 메모리 부족

```bash
# 배치 크기 줄이기
python evaluate_dwa_mia.py --batch_size 32 --device cuda:0
```

### 3. 데이터셋 경로 오류

```bash
# 데이터셋 경로 명시
python evaluate_dwa_mia.py --datapath /path/to/your/datasets
```

### 4. 임포트 오류

```bash
# 필요한 패키지 설치
pip install torch torchvision numpy pandas
```

## 📈 결과 분석 예시

```python
import pandas as pd
import matplotlib.pyplot as plt

# 결과 로드
df = pd.read_csv('mia_results/dwa_mia_results_20241201_123456.csv')

# DWA 모드별 공격 성공률 비교
attack_cols = ['attack_conf_gt', 'attack_entropy', 'attack_modified_entropy', 'attack_conf_top1']
df['best_attack'] = df[attack_cols].max(axis=1)

# 시각화
df.boxplot(column='best_attack', by='dwa_mode', figsize=(12, 6))
plt.title('MIA Attack Success by DWA Mode')
plt.ylabel('Attack Accuracy')
plt.show()

# Sparsity vs Privacy 관계
plt.figure(figsize=(10, 6))
for mode in df['dwa_mode'].unique():
    subset = df[df['dwa_mode'] == mode]
    plt.scatter(subset['sparsity_actual'], subset['best_attack'], 
               label=mode, alpha=0.7)
plt.xlabel('Model Sparsity')
plt.ylabel('MIA Attack Accuracy')
plt.legend()
plt.title('Sparsity vs Privacy Trade-off')
plt.show()
```

## 🔗 참고 자료

- **DWA 논문**: Dynamic Weight Adjustment in Neural Network Pruning
- **MIA 관련 논문**: 
  - Shokri et al., "Membership Inference Attacks against Machine Learning Models"
  - Song et al., "Systematic Evaluation of Privacy Risks of Machine Learning Models"

## 💡 팁

1. **재현성**: 모든 실험에서 동일한 시드 사용
2. **통계적 유의성**: 여러 시드로 실험 반복 후 평균내기
3. **베이스라인**: Dense 모델과 비교하여 프루닝 효과 측정
4. **메모리 관리**: GPU 메모리가 부족하면 배치 크기 조정
